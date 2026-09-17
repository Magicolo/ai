"""Tests for the Python-side assembly of segmented finalize outputs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from zoomy.errors import AssemblyError
from zoomy.final_assembly import (
    MUSIC_CROSSFADE_SECONDS,
    SOUND_EFFECT_CROSSFADE_SECONDS,
    AssemblyRequest,
    CommandRunner,
    SegmentSoundtrack,
    assemble_final_video,
    build_concat_command,
    build_concat_list_file,
    build_extract_command,
    build_join_command,
    build_mix_command,
    build_mux_command,
    resolve_crossfade_seconds,
)


def _recording_runner() -> tuple[list[tuple[str, ...]], CommandRunner]:
    """Return a fake command runner plus the argument list it appends to."""
    recorded: list[tuple[str, ...]] = []

    def record(arguments: tuple[str, ...]) -> None:
        recorded.append(arguments)

    return recorded, record


def _stub_runner(
    recorded: list[tuple[str, ...]],
) -> CommandRunner:
    """Return a fake runner that also stubs every command's destination file.

    Every builder puts its destination last, so writing stub bytes there
    satisfies the assembly's per-step output verification.
    """

    def stub(arguments: tuple[str, ...]) -> None:
        recorded.append(arguments)
        Path(arguments[-1]).write_bytes(b"stub")

    return stub


def _assembly_request(work_directory: Path) -> AssemblyRequest:
    """Build a two-segment request pointing at files the test creates."""
    soundtracks = []
    for index in range(2):
        music_twin = work_directory / f"seg{index:03d}_music-audio.mp4"
        music_stem = work_directory / f"seg{index:03d}_music_stem.flac"
        effects_stem = work_directory / f"seg{index:03d}_sfx_stem.flac"
        music_twin.touch()
        music_stem.touch()
        effects_stem.touch()
        soundtracks.append(
            SegmentSoundtrack(
                music_video_path=music_twin,
                music_stem_path=music_stem,
                sound_effect_stem_path=effects_stem,
                music_seconds=6.90625,
                sound_effect_seconds=6.15625,
            )
        )
    return AssemblyRequest(
        segments=tuple(soundtracks),
        work_directory=work_directory / "assembly_work",
        output_video_path=work_directory / "z_image_00001.mp4",
        output_audio_video_path=work_directory / "z_image_00001-audio.mp4",
    )


def test_concat_list_file_escapes_single_quotes(tmp_path: Path) -> None:
    """Paths with quotes stay on one line so the demuxer reads them whole."""
    tricky = tmp_path / "o'clock.mp4"
    tricky.touch()
    list_path = tmp_path / "concat.txt"
    build_concat_list_file((tricky,), list_path)
    escaped = str(tricky).replace("'", "'\\''")
    assert list_path.read_text() == f"file '{escaped}'\n"


def test_commands_run_quietly() -> None:
    """Every ffmpeg call hides its banner and Banach log flood."""
    commands = [
        build_concat_command("/usr/bin/ffmpeg", Path("/work/list.txt"), Path("/out.mp4")),
        build_extract_command("/usr/bin/ffmpeg", Path("/seg.mp4"), Path("/seg.wav")),
        build_join_command(
            "/usr/bin/ffmpeg",
            (Path("/a.wav"),),
            Path("/joined.wav"),
            crossfade_seconds=1.0,
            durations=(5.89,),
        ),
        build_mix_command(
            "/usr/bin/ffmpeg", Path("/music.wav"), Path("/sfx.wav"), Path("/mix.wav")
        ),
        build_mux_command(
            "/usr/bin/ffmpeg", Path("/main.mp4"), Path("/mix.wav"), Path("/twin.mp4")
        ),
    ]
    for command in commands:
        assert command[:5] == ("/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y")


def test_concat_command_copies_the_video_stream() -> None:
    """Concatenation never re-encodes: same codec settings, stream copy."""
    command = build_concat_command("/usr/bin/ffmpeg", Path("/work/list.txt"), Path("/out.mp4"))
    assert command[command.index("-f") + 1] == "concat"
    assert "-safe" in command
    assert "0" in command
    assert command[command.index("-i") + 1] == "/work/list.txt"
    assert command[command.index("-map") + 1] == "0:v"
    assert command[command.index("-c") + 1] == "copy"
    assert command[-1] == "/out.mp4"


def test_extract_command_normalizes_audio() -> None:
    """Extraction maps the twin's audio track to 44.1 kHz stereo wave audio."""
    command = build_extract_command("/usr/bin/ffmpeg", Path("/seg-audio.mp4"), Path("/seg.wav"))
    assert command[command.index("-i") + 1] == "/seg-audio.mp4"
    assert command[command.index("-map") + 1] == "0:a"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert command[command.index("-ar") + 1] == "44100"
    assert command[command.index("-ac") + 1] == "2"
    assert command[-1] == "/seg.wav"


def test_join_command_transcodes_a_lone_source() -> None:
    """One segment needs no crossfade, just a normalized transcode."""
    command = build_join_command(
        "/usr/bin/ffmpeg",
        (Path("/only.wav"),),
        Path("/joined.wav"),
        crossfade_seconds=1.0,
        durations=(5.89,),
    )
    assert "filter_complex" not in command
    assert command[command.index("-i") + 1] == "/only.wav"
    assert command[-1] == "/joined.wav"


def test_join_command_rejects_mismatched_durations() -> None:
    """Every source needs a known duration to place its fades and delays."""
    with pytest.raises(ValueError, match="durations"):
        build_join_command(
            "/usr/bin/ffmpeg",
            (Path("/a.wav"), Path("/b.wav")),
            Path("/joined.wav"),
            crossfade_seconds=1.0,
            durations=(5.89,),
        )


def test_join_command_blends_boundaries_with_fades_and_delays() -> None:
    """Segments overlap by the crossfade: fade out, delayed fade in, mix.

    Acrossfade filters collapse on short tails (verified live: a 0.84 s tail
    truncated the whole join), so the join places deterministic fades and
    delays from the known segment durations instead.
    """
    command = build_join_command(
        "/usr/bin/ffmpeg",
        (Path("/a.wav"), Path("/b.wav"), Path("/c.wav")),
        Path("/joined.wav"),
        crossfade_seconds=1.0,
        durations=(5.89, 5.89, 0.77),
    )
    audio_filter = command[command.index("-filter_complex") + 1]
    assert "acrossfade" not in audio_filter
    assert "afade=t=out:st=4.89:d=1.0:curve=tri" in audio_filter
    assert "adelay=4890:all=1" in audio_filter
    assert "afade=t=out:st=5.505:d=0.385:curve=tri" in audio_filter
    assert "adelay=10395:all=1" in audio_filter
    assert "afade=t=in:st=0:d=0.385:curve=tri" in audio_filter
    assert "amix=inputs=3" in audio_filter
    assert command[-1] == "/joined.wav"


def test_join_command_clamps_each_boundary_overlap() -> None:
    """No overlap exceeds half of either neighbor it blends."""
    assert resolve_crossfade_seconds(1.0, 5.89, 5.89) == 1.0
    assert resolve_crossfade_seconds(1.0, 5.89, 0.77) == 0.385
    assert resolve_crossfade_seconds(0.25, 0.77, 0.77) == 0.25


def test_mix_command_attenuates_effects_like_the_graph() -> None:
    """The Python mix preserves the graph's balance: music 0 dB, SFX -6 dB."""
    command = build_mix_command(
        "/usr/bin/ffmpeg", Path("/music.wav"), Path("/sfx.wav"), Path("/mix.wav")
    )
    audio_filter = command[command.index("-filter_complex") + 1]
    assert "volume=0.501187" in audio_filter
    assert "amix=inputs=2" in audio_filter
    assert "normalize=0" in audio_filter
    assert command[-1] == "/mix.wav"


def test_mux_command_copies_video_and_encodes_audio() -> None:
    """The twin keeps the concatenated pixels and gains an AAC soundtrack."""
    command = build_mux_command(
        "/usr/bin/ffmpeg", Path("/main.mp4"), Path("/mix.wav"), Path("/twin.mp4")
    )
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "aac"
    assert "-shortest" in command
    assert command[-1] == "/twin.mp4"


def test_assembly_rejects_empty_segments(tmp_path: Path) -> None:
    """Assembling nothing is a programming error, not an empty video."""
    request = AssemblyRequest(
        segments=(),
        work_directory=tmp_path,
        output_video_path=tmp_path / "main.mp4",
        output_audio_video_path=tmp_path / "twin.mp4",
    )
    with pytest.raises(AssemblyError, match="no segments"):
        assemble_final_video(request)


def test_assembly_rejects_missing_twins(tmp_path: Path) -> None:
    """A segment whose twin never landed fails fast naming the file."""
    missing = tmp_path / "seg000_music-audio.mp4"
    request = AssemblyRequest(
        segments=(
            SegmentSoundtrack(
                music_video_path=missing,
                music_stem_path=tmp_path / "music.flac",
                sound_effect_stem_path=tmp_path / "sfx.flac",
                music_seconds=6.90625,
                sound_effect_seconds=6.15625,
            ),
        ),
        work_directory=tmp_path,
        output_video_path=tmp_path / "main.mp4",
        output_audio_video_path=tmp_path / "twin.mp4",
    )
    with pytest.raises(AssemblyError, match="segment file is missing"):
        assemble_final_video(request)


def test_assembly_runs_the_full_pipeline_and_cleans_up(tmp_path: Path) -> None:
    """Two segments run extract, join, mix, concat, mux — then work files go."""
    recorded: list[tuple[str, ...]] = []
    request = _assembly_request(tmp_path)
    outputs = assemble_final_video(
        request, ffmpeg_path="/usr/bin/ffmpeg", command_runner=_stub_runner(recorded)
    )
    assert outputs.main_video_path == request.output_video_path
    assert outputs.audio_video_path == request.output_audio_video_path
    assert len(recorded) == 2 * 2 + 2 + 1 + 1 + 1  # extracts, joins, mix, concat, mux
    assert recorded[0][0] == "/usr/bin/ffmpeg"
    filter_commands = [args for args in recorded if "-filter_complex" in args]
    assert len(filter_commands) == 3  # music join, effects join, mix
    assert not request.work_directory.exists()


def test_assembly_rejects_missing_step_output(tmp_path: Path) -> None:
    """A step that writes nothing fails naming the absent file."""
    recorded, runner = _recording_runner()
    request = _assembly_request(tmp_path)
    with pytest.raises(AssemblyError, match="no output file"):
        assemble_final_video(request, ffmpeg_path="/usr/bin/ffmpeg", command_runner=runner)
    assert recorded


def test_assembly_rejects_empty_step_output(tmp_path: Path) -> None:
    """A step that writes zero bytes fails instead of poisoning later steps."""

    def empty_runner(arguments: tuple[str, ...]) -> None:
        Path(arguments[-1]).write_bytes(b"")

    request = _assembly_request(tmp_path)
    with pytest.raises(AssemblyError, match="empty output file"):
        assemble_final_video(request, ffmpeg_path="/usr/bin/ffmpeg", command_runner=empty_runner)


def test_assembly_failure_surfaces_as_assembly_error(tmp_path: Path) -> None:
    """A dying ffmpeg becomes a ZoomyError the interface already handles."""

    def failing_runner(arguments: tuple[str, ...]) -> None:
        raise subprocess.CalledProcessError(1, arguments)

    request = _assembly_request(tmp_path)
    with pytest.raises(AssemblyError, match="ffmpeg"):
        assemble_final_video(request, ffmpeg_path="/usr/bin/ffmpeg", command_runner=failing_runner)


def test_crossfade_constants_favor_music_coherence() -> None:
    """Music blends over a full second; effects cut tight to stay synced."""
    assert MUSIC_CROSSFADE_SECONDS == 1.0
    assert SOUND_EFFECT_CROSSFADE_SECONDS == 0.25
