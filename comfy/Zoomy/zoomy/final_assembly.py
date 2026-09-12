"""Python-side assembly of segmented finalize outputs with ffmpeg.

ComfyUI renders one video pair per segment (twins carry the concatenated
pixels; FLAC stems carry the full-length music and effects mixes), and this
module stitches them into the final artifacts: the silent concatenated video
plus the ``-audio`` twin with the full soundtrack. Every diffusion job stays
bounded by the segment size; only this cheap CPU assembly ever sees the
whole sequence, which is what makes thousand-frame finalizes possible.

Audio coherence between segments comes from the same idea as FILM frame
interpolation: adjacent renders overlap and blend. Music joins with a long
triangular crossfade (shared tags, tempo, and key keep the takes compatible),
while sound effects join with a short one so transient sync survives. The
blend is placed with explicit fades and delays from the known segment
durations — never with ``acrossfade``, which collapses on short tails
(verified live: a 0.84 s tail truncated a whole join to silence).

Every ffmpeg call runs quiet (banner and progress off) and every step
verifies its outputs are present and non-empty, so a silent filter failure
becomes an :class:`AssemblyError` at the step that caused it.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, cast

from zoomy.errors import AssemblyError

if TYPE_CHECKING:
    from pathlib import Path

MUSIC_CROSSFADE_SECONDS = 1.0
SOUND_EFFECT_CROSSFADE_SECONDS = 0.25
SOUND_EFFECT_RELATIVE_VOLUME = 0.501187  # -6 dB, matches soften_sound_effects
OUTPUT_SAMPLE_RATE = 44100
OUTPUT_AUDIO_BITRATE = "192k"

CommandRunner = Callable[[tuple[str, ...]], None]


@dataclass(frozen=True, slots=True)
class SegmentSoundtrack:
    """One rendered segment's artifacts, in sequence order.

    Attributes:
        music_video_path: The twin whose video stream concatenates.
        music_stem_path: The full-length music stem (FLAC keeps the overlap
            tail the twin trims away).
        sound_effect_stem_path: The full-length effects stem.
        music_seconds: Nominal music stem duration placing the join fades.
        sound_effect_seconds: Nominal effects stem duration.
    """

    music_video_path: Path
    music_stem_path: Path
    sound_effect_stem_path: Path
    music_seconds: float
    sound_effect_seconds: float


@dataclass(frozen=True, slots=True)
class AssemblyRequest:
    """Everything the final assembly needs.

    Attributes:
        segments: Per-segment twins in playback order.
        work_directory: Scratch directory for extracted waves and the concat
            list; removed after a successful assembly.
        output_video_path: The silent concatenated video (the main file).
        output_audio_video_path: The muxed twin carrying the soundtrack.
    """

    segments: tuple[SegmentSoundtrack, ...]
    work_directory: Path
    output_video_path: Path
    output_audio_video_path: Path


@dataclass(frozen=True, slots=True)
class AssemblyOutputs:
    """Paths the assembly wrote: the silent main video and its audio twin."""

    main_video_path: Path
    audio_video_path: Path


def ffmpeg_binary() -> str:
    """Return the ffmpeg binary path, importing its provider lazily.

    The import stays here (not at module top) so the builder and its tests
    load without the ffmpeg package installed; only assembly needs it.
    """
    try:
        # Lazy so builders and tests load without the ffmpeg package installed.
        import imageio_ffmpeg  # noqa: PLC0415
    except ImportError as failure:
        message = "Assembling a segmented video needs the imageio-ffmpeg package"
        raise AssemblyError(message) from failure
    return cast("str", imageio_ffmpeg.get_ffmpeg_exe())


def resolve_crossfade_seconds(
    requested_seconds: float, first_seconds: float, second_seconds: float
) -> float:
    """Clamp one boundary overlap so it fits inside both neighbors.

    An overlap can never exceed half of either input; short floor-length
    tails simply blend over a shorter window instead of failing.
    """
    return min(requested_seconds, first_seconds / 2, second_seconds / 2)


def _command_head(ffmpeg: str) -> tuple[str, ...]:
    """Start every ffmpeg call quiet: no banner, errors only, overwrite."""
    return (ffmpeg, "-hide_banner", "-loglevel", "error", "-y")


def build_concat_list_file(segment_paths: tuple[Path, ...], list_path: Path) -> None:
    """Write a concat-demuxer list file, keeping quoted names on one line."""
    lines = [f"file '{_escape_concat_path(path)}'\n" for path in segment_paths]
    list_path.write_text("".join(lines))


def _escape_concat_path(path: Path) -> str:
    """Escape a path for single-quoted concat-demuxer syntax."""
    return str(path).replace("'", "'\\''")


def build_concat_command(ffmpeg: str, list_path: Path, destination: Path) -> tuple[str, ...]:
    """Concatenate segment videos without re-encoding (identical codecs)."""
    return (
        *_command_head(ffmpeg),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-map",
        "0:v",
        "-c",
        "copy",
        str(destination),
    )


def build_extract_command(
    ffmpeg: str, source_video: Path, destination_wave: Path
) -> tuple[str, ...]:
    """Extract a twin's audio track as normalized 44.1 kHz stereo wave audio."""
    return (
        *_command_head(ffmpeg),
        "-i",
        str(source_video),
        "-map",
        "0:a",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(OUTPUT_SAMPLE_RATE),
        "-ac",
        "2",
        str(destination_wave),
    )


def build_join_command(
    ffmpeg: str,
    sources: tuple[Path, ...],
    destination: Path,
    *,
    crossfade_seconds: float,
    durations: tuple[float, ...],
) -> tuple[str, ...]:
    """Join segment waves with deterministic crossfades (the audio FILM analog).

    One source needs no blending — just a normalized transcode. Several
    sources overlap by the (per-boundary clamped) crossfade: each fades out
    over the overlap while the next, delayed into place, fades in, and one
    mix sums the timeline. Durations are the known segment lengths, so every
    fade and delay is placed exactly.

    Raises:
        ValueError: If a duration is missing for any source.
    """
    if len(sources) != len(durations):
        message = (
            "Join needs one duration per source, "
            f"got {len(sources)} sources and {len(durations)} durations"
        )
        raise ValueError(message)
    if len(sources) == 1:
        return (*_command_head(ffmpeg), "-i", str(sources[0]), str(destination))
    overlaps = [
        resolve_crossfade_seconds(crossfade_seconds, first, second)
        for first, second in pairwise(durations)
    ]
    starts: list[float] = [0.0]
    for position, overlap in enumerate(overlaps):
        starts.append(starts[-1] + durations[position] - overlap)
    filter_parts = []
    for index in range(len(sources)):
        stages = [f"[{index}:a]aresample={OUTPUT_SAMPLE_RATE},aformat=channel_layouts=stereo"]
        if index > 0:
            stages.append(f"afade=t=in:st=0:d={overlaps[index - 1]}:curve=tri")
        if index < len(sources) - 1:
            fade_start = durations[index] - overlaps[index]
            stages.append(f"afade=t=out:st={fade_start}:d={overlaps[index]}:curve=tri")
        if index > 0:
            delay_milliseconds = round(starts[index] * 1000)
            stages.append(f"adelay={delay_milliseconds}:all=1")
        filter_parts.append(",".join(stages) + f"[track{index}]")
    mixed_inputs = "".join(f"[track{index}]" for index in range(len(sources)))
    filter_parts.append(
        f"{mixed_inputs}amix=inputs={len(sources)}"
        ":duration=longest:dropout_transition=0:normalize=0[mixed]"
    )
    inputs: tuple[str, ...] = ()
    for source in sources:
        inputs += ("-i", str(source))
    return (
        *_command_head(ffmpeg),
        *inputs,
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[mixed]",
        "-c:a",
        "pcm_s16le",
        str(destination),
    )


def build_mix_command(
    ffmpeg: str, music_wave: Path, effects_wave: Path, destination: Path
) -> tuple[str, ...]:
    """Mix the joined music and effects at the graph's balance (0/-6 dB)."""
    audio_filter = (
        f"[1:a]volume={SOUND_EFFECT_RELATIVE_VOLUME}[effects];"
        "[0:a][effects]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0"
    )
    return (
        *_command_head(ffmpeg),
        "-i",
        str(music_wave),
        "-i",
        str(effects_wave),
        "-filter_complex",
        audio_filter,
        "-c:a",
        "pcm_s16le",
        str(destination),
    )


def build_mux_command(
    ffmpeg: str, silent_video: Path, mixed_audio: Path, destination: Path
) -> tuple[str, ...]:
    """Mux the concatenated pixels with the mixed soundtrack (the twin)."""
    return (
        *_command_head(ffmpeg),
        "-i",
        str(silent_video),
        "-i",
        str(mixed_audio),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        OUTPUT_AUDIO_BITRATE,
        "-shortest",
        str(destination),
    )


def assemble_final_video(
    request: AssemblyRequest,
    *,
    ffmpeg_path: str | None = None,
    command_runner: CommandRunner | None = None,
) -> AssemblyOutputs:
    """Assemble segment twins into the final video pair, cleaning up after.

    Every ffmpeg step verifies its outputs are present and non-empty, so a
    silent filter failure surfaces at the step that caused it instead of
    poisoning the mix downstream.

    Raises:
        AssemblyError: No segments were given, an artifact is missing, an
            ffmpeg step failed, or a step wrote nothing. The scratch
            directory survives failures for inspection and is removed on
            success.
    """
    if not request.segments:
        raise AssemblyError("Cannot assemble a video with no segments")
    for segment in request.segments:
        for artifact in (
            segment.music_video_path,
            segment.music_stem_path,
            segment.sound_effect_stem_path,
        ):
            if not artifact.is_file():
                raise AssemblyError(f"Cannot assemble: segment file is missing: {artifact}")
    ffmpeg = ffmpeg_path if ffmpeg_path is not None else ffmpeg_binary()
    runner = command_runner if command_runner is not None else _run_command
    work_directory = request.work_directory
    work_directory.mkdir(parents=True, exist_ok=True)
    try:
        music_waves = _extract_stems(
            runner,
            ffmpeg,
            tuple(segment.music_stem_path for segment in request.segments),
            work_directory,
            stem="music",
        )
        effects_waves = _extract_stems(
            runner,
            ffmpeg,
            tuple(segment.sound_effect_stem_path for segment in request.segments),
            work_directory,
            stem="effects",
        )
        joined_music = work_directory / "music_joined.wav"
        _run_and_verify(
            runner,
            build_join_command(
                ffmpeg,
                music_waves,
                joined_music,
                crossfade_seconds=MUSIC_CROSSFADE_SECONDS,
                durations=tuple(segment.music_seconds for segment in request.segments),
            ),
            (joined_music,),
        )
        joined_effects = work_directory / "effects_joined.wav"
        _run_and_verify(
            runner,
            build_join_command(
                ffmpeg,
                effects_waves,
                joined_effects,
                crossfade_seconds=SOUND_EFFECT_CROSSFADE_SECONDS,
                durations=tuple(segment.sound_effect_seconds for segment in request.segments),
            ),
            (joined_effects,),
        )
        mixed_audio = work_directory / "mix.wav"
        _run_and_verify(
            runner,
            build_mix_command(ffmpeg, joined_music, joined_effects, mixed_audio),
            (mixed_audio,),
        )
        music_twins = tuple(segment.music_video_path for segment in request.segments)
        list_path = work_directory / "concat.txt"
        build_concat_list_file(music_twins, list_path)
        _run_and_verify(
            runner,
            build_concat_command(ffmpeg, list_path, request.output_video_path),
            (request.output_video_path,),
        )
        _run_and_verify(
            runner,
            build_mux_command(
                ffmpeg, request.output_video_path, mixed_audio, request.output_audio_video_path
            ),
            (request.output_audio_video_path,),
        )
    except AssemblyError:
        raise
    except (OSError, subprocess.CalledProcessError) as failure:
        message = f"Final video assembly failed with ffmpeg: {failure}"
        raise AssemblyError(message) from failure
    shutil.rmtree(work_directory, ignore_errors=True)
    return AssemblyOutputs(
        main_video_path=request.output_video_path,
        audio_video_path=request.output_audio_video_path,
    )


def _extract_stems(
    runner: CommandRunner,
    ffmpeg: str,
    stem_paths: tuple[Path, ...],
    work_directory: Path,
    *,
    stem: str,
) -> tuple[Path, ...]:
    """Extract one stem (music or effects) from every segment stem file."""
    waves: list[Path] = []
    for stem_path in stem_paths:
        destination = work_directory / f"{stem}_{len(waves):03d}.wav"
        command = build_extract_command(ffmpeg, stem_path, destination)
        _run_and_verify(runner, command, (destination,))
        waves.append(destination)
    return tuple(waves)


def _run_and_verify(
    runner: CommandRunner, command: tuple[str, ...], expected_outputs: tuple[Path, ...]
) -> None:
    """Run one assembly step and reject missing or empty outputs at once."""
    runner(command)
    for output in expected_outputs:
        try:
            output_bytes = output.stat().st_size
        except OSError as failure:
            message = f"Assembly step produced no output file: {output}"
            raise AssemblyError(message) from failure
        if output_bytes == 0:
            message = f"Assembly step produced an empty output file: {output}"
            raise AssemblyError(message)


def _run_command(arguments: tuple[str, ...]) -> None:
    """Run one ffmpeg command, raising when it fails."""
    # Tuple argv, no shell: every argument comes from our own command builders.
    subprocess.run(list(arguments), check=True)  # noqa: S603
