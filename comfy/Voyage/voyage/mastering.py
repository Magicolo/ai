"""SonicMaster mastering pass for finalize assembly (DESIGN §56 final assembly).

Why: the shipped final should carry a mastered soundtrack (consistent
loudness and tone from the first second to the last), but the durable
mix/bed caches must keep pre-master bytes — mastering re-runs after
every cache hit, so a no-change resume still reuses the cached mix/bed
and only pays the mastering render. This module owns the mastering
runner plus the single publish-time choke both finalize paths share.

Pipeline (`master_fn`): resample the input WAV to the mastering-native
44.1 kHz, slice into 30 s windows with 10 s overlap, render each window
through the mastering backend with the fixed prompt/sampler contract,
join with a linear crossfade over the overlap (stdlib `wave`, never
`acrossfade`), then resample back to the run's sample rate/channels.
Duration-preserving within the 0.6 s A/V budget (`MediaError` past it).

Backend seam: `_render_mastered_chunk` is the only model touchpoint
(offline passthrough copy today; the SonicMaster weights land behind
this seam in a later track without touching callers). The fixed
prompt/sampler constants below define that future call — tests pin
them, and `master_fn` threads them through on every window.

Routing (`maybe_master_ship_audio`): the ship video's audio track IS
the mixed bed when dubbed, else the music-only/silent `final_audio`
content — so one demux → master → remux covers the dubbed,
music-only, and silent paths with a single call site per finalize
path. Off unless the caller passes `enabled=True` (production threads
the effective `config.audio.mastering and not no_master` flag;
`MASTERING_ENABLED = False` is only the fallback for direct callers):
the choke returns the ship untouched when off and fingerprints ignore
the knob, so toggling it never invalidates the mix/bed slots — only
the mastering render+remux runs.

Errors: `ValueError` for invalid arguments (rates/channels/geometry),
`MediaError` for any ffmpeg/probe/render/duration failure. Nothing
else escapes. All intermediates live under a `TemporaryDirectory`
rooted at the output parent (a finalize tmpdir in production), so the
run directory never gains mastering scratch — only `output_wav`
(resp. the remuxed ship) is written outside it.
"""

from __future__ import annotations

import array
import shutil
import tempfile
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from voyage.console import optional_bar
from voyage.errors import MediaError
from voyage.media_audio import (
    AV_ALIGNMENT_TOLERANCE_SECONDS,
    _audio_duration_seconds,
    run_capture,
)

if TYPE_CHECKING:
    from voyage.console import VoyageConsole

MASTERING_VERSION = 1
"""Bump when the mastering render changes (fingerprints miss, never lie)."""

MASTERING_ENABLED = False
"""Publish-time knob fallback (Track B default off — safe no-op wire until provisioned).

Only applies when `maybe_master_ship_audio` gets `enabled=None`;
production callers pass the effective flag explicitly. The mix/bed
fingerprints ignore the knob, so toggling it never invalidates slots.
"""

MASTERING_SAMPLE_RATE = 44100
"""Mastering-native rate: resample in on entry, back to settings on exit."""

MASTERING_CHUNK_SECONDS = 30.0
"""Window length for the chunked render (seconds)."""

MASTERING_OVERLAP_SECONDS = 10.0
"""Overlap between consecutive windows, killed by linear crossfade (seconds)."""

MASTERING_PROMPT = "Master this track for me, please!"
"""Fixed mastering prompt (no per-segment text — one mastering intent)."""

MASTERING_SAMPLER = "euler"
"""Sampler for the mastering backend."""

MASTERING_STEPS = 10
"""Denoising steps for the mastering backend."""

MASTERING_GUIDANCE = 1.0
"""Guidance scale for the mastering backend."""

MASTERING_SEED = 0
"""Fixed seed for the mastering backend (deterministic re-finalize)."""


def master_chunk_windows(
    duration_seconds: float,
    chunk_seconds: float = MASTERING_CHUNK_SECONDS,
    overlap_seconds: float = MASTERING_OVERLAP_SECONDS,
) -> list[tuple[float, float]]:
    """Slice `duration_seconds` into (start, duration) mastering windows (pure).

    Stride is `chunk - overlap`, so consecutive windows share exactly
    the overlap; the tail window clamps to the timeline end (possibly
    shorter than a full chunk — still rendered, never dropped). A
    timeline at or below one chunk is a single window.
    """
    if not isinstance(duration_seconds, (int, float)) or not duration_seconds > 0.0:
        raise ValueError(f"duration_seconds must be positive (got {duration_seconds})")
    try:
        chunk = float(chunk_seconds)
        overlap = float(overlap_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"chunk/overlap must be numeric: {exc}") from exc
    if not chunk > 0.0:
        raise ValueError(f"chunk_seconds must be positive (got {chunk_seconds})")
    if not 0.0 <= overlap < chunk:
        raise ValueError(
            f"overlap_seconds must satisfy 0 <= overlap < chunk (got {overlap_seconds})"
        )
    duration = float(duration_seconds)
    if duration <= chunk:
        return [(0.0, duration)]
    stride = chunk - overlap
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = start + chunk
        if end >= duration:
            windows.append((start, duration - start))
            break
        windows.append((start, chunk))
        start += stride
    return windows


def _check_tail_window(windows: list[tuple[float, float]]) -> None:
    """Fail loud when the tail window cannot carry the overlap (M5).

    Inner-function home for the `master_fn` tail gate (TRY301): raising
    directly inside the `TemporaryDirectory` try would be caught by its
    own `except MediaError` re-raise arm — abstracting here keeps the
    raise out of the try body.
    """
    if len(windows) > 1 and windows[-1][1] <= MASTERING_OVERLAP_SECONDS:
        raise MediaError(
            f"mastering tail window {windows[-1][1]:.3f}s "
            f"must exceed overlap {MASTERING_OVERLAP_SECONDS:.3f}s"
        )


def _resample_wav(source: Path, dest: Path, sample_rate: int, channels: int) -> Path:
    """Resample `source` to (`sample_rate`, `channels`) s16le WAV (ffmpeg arg-list)."""
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        # Unbounded: resamples full-timeline inputs (whole mix / joined windows).
        timeout=None,
    )
    if proc.returncode != 0:
        raise MediaError(f"mastering resample failed for {source}: {proc.stderr[-2000:]}")
    try:
        if dest.stat().st_size == 0:
            raise MediaError(f"mastering resample produced empty output for {dest}")
    except OSError as exc:
        raise MediaError(f"mastering resample produced no output for {dest}: {exc}") from exc
    return dest


def _slice_wav(
    source: Path, dest: Path, start: float, duration: float, sample_rate: int, channels: int
) -> Path:
    """Cut one mastering window out of `source` (post-input seek for accuracy)."""
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"mastering slice failed for {source}: {proc.stderr[-2000:]}")
    try:
        if dest.stat().st_size == 0:
            raise MediaError(f"mastering slice produced empty output for {dest}")
    except OSError as exc:
        raise MediaError(f"mastering slice produced no output for {dest}: {exc}") from exc
    return dest


def _render_mastered_chunk(
    chunk_in: Path,
    chunk_out: Path,
    *,
    prompt: str = MASTERING_PROMPT,
    sampler: str = MASTERING_SAMPLER,
    steps: int = MASTERING_STEPS,
    guidance: float = MASTERING_GUIDANCE,
    seed: int = MASTERING_SEED,
) -> Path:
    """Render one mastering window (sole model seam — offline passthrough today).

    The fixed prompt/sampler contract arrives as explicit kwargs so
    `master_fn` threads the pinned constants on every window (tests
    assert the values at this seam). Production replaces this body
    with the SonicMaster backend call — same signature, same
    duration-preserving guarantee — without touching callers.
    """
    del prompt, sampler, steps, guidance, seed
    try:
        shutil.copyfile(chunk_in, chunk_out)
    except OSError as exc:
        raise MediaError(f"mastering chunk render failed for {chunk_in}: {exc}") from exc
    return chunk_out


def _join_mastered_chunks(
    chunks: list[Path],
    dest: Path,
    *,
    channels: int,
    overlap_seconds: float = MASTERING_OVERLAP_SECONDS,
) -> Path:
    """Join mastered windows with a linear crossfade over the overlap (stdlib wave).

    Manual sample-domain crossfade (never `acrossfade` — same short-tail
    doctrine as the music/SFX joins). A single chunk copies through;
    otherwise each joint blends the tail/head over exactly the overlap
    with a per-frame linear ramp, so no click lands on a window edge.

    Streaming-join ceiling (M5): all windows load into RAM before the
    blend (`~5 MB/min` stereo at 44.1 kHz s16le — a 30-min timeline holds
    ~150 MB, an hour ~300 MB). Past hour-scale timelines prefer a
    streaming join (window-at-a-time); this pass documents the ceiling
    instead of implementing it — the `master_fn` tail gate bounds the
    window count, not the RAM.
    """
    if not chunks:
        raise MediaError("mastering join needs at least one chunk (got none)")
    if len(chunks) == 1:
        try:
            shutil.copyfile(chunks[0], dest)
        except OSError as exc:
            raise MediaError(f"mastering join failed for {chunks[0]}: {exc}") from exc
        return dest
    try:
        overlap = float(overlap_seconds)
    except (TypeError, ValueError) as exc:
        raise MediaError(f"mastering join needs a numeric overlap: {exc}") from exc
    if not overlap >= 0.0:
        raise MediaError(f"mastering join needs overlap >= 0 (got {overlap_seconds})")
    overlap_samples = int(round(overlap * MASTERING_SAMPLE_RATE))
    bodies: list[array.array[int]] = []
    for chunk in chunks:
        try:
            with wave.open(str(chunk), "rb") as handle:
                in_channels = handle.getnchannels()
                sampwidth = handle.getsampwidth()
                framerate = handle.getframerate()
                nframes = handle.getnframes()
                raw = handle.readframes(nframes)
        except (OSError, wave.Error) as exc:
            raise MediaError(f"mastering join cannot read {chunk}: {exc}") from exc
        if in_channels != channels or sampwidth != 2 or framerate != MASTERING_SAMPLE_RATE:
            raise MediaError(
                f"mastering join shape mismatch in {chunk}: "
                f"expected {channels}ch/s16le/{MASTERING_SAMPLE_RATE}Hz"
            )
        samples: array.array[int] = array.array("h")
        samples.frombytes(raw)
        bodies.append(samples)
    overlap_values = overlap_samples * channels
    if overlap_values > 0:
        for samples in bodies:
            if len(samples) < overlap_values:
                raise MediaError(
                    "mastering join overlap exceeds a window "
                    f"({overlap}s over {len(samples) // max(channels, 1)} frames)"
                )
        mixed: array.array[int] = array.array("h", bodies[0])
        for samples in bodies[1:]:
            base = len(mixed) - overlap_values
            for index in range(overlap_values):
                frame = index // channels
                gain = frame / overlap_samples if overlap_samples > 0 else 1.0
                old = mixed[base + index]
                new = samples[index]
                blended = int(round(old * (1.0 - gain) + new * gain))
                if blended > 32767:
                    blended = 32767
                elif blended < -32768:
                    blended = -32768
                mixed[base + index] = blended
            mixed.extend(samples[overlap_values:])
    else:
        mixed = array.array("h")
        for samples in bodies:
            mixed.extend(samples)
    try:
        with wave.open(str(dest), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(MASTERING_SAMPLE_RATE)
            handle.writeframes(mixed.tobytes())
    except (OSError, wave.Error) as exc:
        raise MediaError(f"mastering join cannot write {dest}: {exc}") from exc
    try:
        if dest.stat().st_size == 0:
            raise MediaError(f"mastering join produced empty output for {dest}")
    except OSError as exc:
        raise MediaError(f"mastering join produced no output for {dest}: {exc}") from exc
    return dest


def master_fn(
    input_wav: Path,
    output_wav: Path,
    sample_rate: int,
    channels: int,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
) -> Path:
    """Master `input_wav` to `output_wav` (44.1 kHz chunked pass, tmpdir-only).

    Resamples in to the mastering-native 44.1 kHz, renders each
    30 s / 10 s-overlap window with the fixed prompt/sampler contract,
    joins with a linear crossfade, and resamples back to
    (`sample_rate`, `channels`). Output duration must match the input
    within `AV_ALIGNMENT_TOLERANCE_SECONDS` (0.6 s), else `MediaError`.

    M5 progress + tail gate: per-window renders report through the
    shared `optional_bar` pattern (`mastering chunks`, silent when
    `progress` is None), and the tail window must exceed the overlap —
    a tail at or below the overlap would make the streaming join read
    past the window (fail-loud `MediaError`, never a silent short join).
    Join ceiling: `_join_mastered_chunks` holds all windows in RAM
    (`array("h")` bodies, ~5 MB/min stereo at 44.1 kHz — a 30-min
    timeline holds ~150 MB); a streaming join is the documented next
    step past hour-scale timelines, not this pass.

    Raises: `TypeError` for mistyped rates/channels; `ValueError` for
    invalid rates/channels/geometry; `MediaError` for missing/unprobable
    input and any ffmpeg/probe/render/join/duration failure. Nothing
    else escapes.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError(f"sample_rate must be int (got {sample_rate!r})")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive (got {sample_rate})")
    if not isinstance(channels, int) or isinstance(channels, bool):
        raise TypeError(f"channels must be int (got {channels!r})")
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2 (got {channels})")
    try:
        input_duration = _audio_duration_seconds(input_wav)
    except MediaError:
        raise
    except (OSError, ValueError) as exc:
        raise MediaError(f"mastering cannot probe {input_wav}: {exc}") from exc
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress.info(f"mastering {input_wav.name} ({input_duration:.1f}s)")
    overall_start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(
            prefix="voyage-master-", dir=str(output_wav.parent)
        ) as tmp:
            tmpdir = Path(tmp)
            stage_start = time.monotonic()
            native_in = tmpdir / "master_in_44100.wav"
            _resample_wav(input_wav, native_in, MASTERING_SAMPLE_RATE, channels)
            resampled_duration = _audio_duration_seconds(native_in)
            if timings is not None:
                timings["master_resample_in_s"] = time.monotonic() - stage_start
            stage_start = time.monotonic()
            windows = master_chunk_windows(
                resampled_duration, MASTERING_CHUNK_SECONDS, MASTERING_OVERLAP_SECONDS
            )
            _check_tail_window(windows)
            rendered: list[Path] = []
            with optional_bar(progress, "mastering chunks", total=len(windows)) as tracker:
                for index, (start, duration) in enumerate(windows):
                    sliced = tmpdir / f"master_chunk_{index:04d}_in.wav"
                    _slice_wav(native_in, sliced, start, duration, MASTERING_SAMPLE_RATE, channels)
                    mastered = tmpdir / f"master_chunk_{index:04d}_out.wav"
                    _render_mastered_chunk(
                        sliced,
                        mastered,
                        prompt=MASTERING_PROMPT,
                        sampler=MASTERING_SAMPLER,
                        steps=MASTERING_STEPS,
                        guidance=MASTERING_GUIDANCE,
                        seed=MASTERING_SEED,
                    )
                    rendered.append(mastered)
                    if tracker is not None:
                        tracker.update()
            if timings is not None:
                timings["master_render_s"] = time.monotonic() - stage_start
            stage_start = time.monotonic()
            joined = tmpdir / "master_joined_44100.wav"
            _join_mastered_chunks(
                rendered,
                joined,
                channels=channels,
                overlap_seconds=MASTERING_OVERLAP_SECONDS,
            )
            if timings is not None:
                timings["master_join_s"] = time.monotonic() - stage_start
            stage_start = time.monotonic()
            _resample_wav(joined, output_wav, sample_rate, channels)
            if timings is not None:
                timings["master_resample_out_s"] = time.monotonic() - stage_start
    except MediaError:
        raise
    except (OSError, ValueError, wave.Error) as exc:
        raise MediaError(f"mastering failed for {input_wav}: {exc}") from exc
    try:
        output_duration = _audio_duration_seconds(output_wav)
    except MediaError:
        raise
    except (OSError, ValueError) as exc:
        raise MediaError(f"mastering cannot probe {output_wav}: {exc}") from exc
    drift = abs(output_duration - input_duration)
    if drift > AV_ALIGNMENT_TOLERANCE_SECONDS:
        raise MediaError(f"mastered {output_duration:.2f}s drifts from input {input_duration:.2f}s")
    if timings is not None:
        timings["master_s"] = time.monotonic() - overall_start
    if progress is not None:
        progress.info(f"mastered {output_wav.name} ({output_duration:.1f}s)")
    return output_wav


def select_master_source(dubbed_mixed: Path | None, final_audio: Path) -> Path:
    """Pick the mastering input (pure routing rule).

    A dubbed finalize masters the mixed bed (`dubbed_mixed`); a
    music-only or silent finalize masters `final_audio` (which carries
    the music mix resp. the sized silence). Presence of the dubbed mix
    decides — its content already encodes which path rendered.
    """
    if dubbed_mixed is not None:
        return dubbed_mixed
    return final_audio


def maybe_master_ship_audio(
    ship: Path,
    tmpdir: Path,
    sample_rate: int,
    channels: int,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
    enabled: bool | None = None,
) -> Path:
    """Master the ship video's audio track once (single publish-time choke).

    The ship's audio IS the mixed bed when dubbed, else the
    music-only/silent `final_audio` content — so one demux → master →
    remux covers the dubbed, music-only, and silent paths. Off (default)
    returns `ship` untouched; on demuxes to tmpdir, runs `master_fn`,
    and remuxes video-copy + mastered AAC to `tmpdir/final_mastered.mp4`.
    Fail-loud `MediaError` (never a silent unmastered ship when on).
    Tmpdir-only besides the returned remux. `enabled=None` falls back to
    `MASTERING_ENABLED` (legacy direct callers stay off).
    """
    run_enabled = MASTERING_ENABLED if enabled is None else enabled
    if not run_enabled:
        return ship
    if progress is not None:
        progress.info("mastering final audio")
    tmpdir.mkdir(parents=True, exist_ok=True)
    demuxed = tmpdir / "master_source.wav"
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(ship),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            str(demuxed),
        ],
        # Unbounded: demuxes the full shipped timeline.
        timeout=None,
    )
    if proc.returncode != 0:
        raise MediaError(f"mastering demux failed for {ship}: {proc.stderr[-2000:]}")
    mastered = tmpdir / "master_source_mastered.wav"
    master_fn(demuxed, mastered, sample_rate, channels, timings=timings, progress=None)
    remuxed = tmpdir / "final_mastered.mp4"
    mux = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(ship),
            "-i",
            str(mastered),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(remuxed),
        ],
        # Unbounded: remuxes the full shipped timeline.
        timeout=None,
    )
    if mux.returncode != 0:
        raise MediaError(f"mastering remux failed for {ship}: {mux.stderr[-2000:]}")
    try:
        if remuxed.stat().st_size == 0:
            raise MediaError(f"mastering remux produced empty output for {remuxed}")
    except OSError as exc:
        raise MediaError(f"mastering remux produced no output for {remuxed}: {exc}") from exc
    return remuxed
