"""ffmpeg/ffprobe wrappers + validation + finalizer (DESIGN §§54-57, I).

All invocations use argument lists — never shell strings. The finalizer
never mutates source segment files; it publishes the final path
atomically.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from voyage import paths
from voyage.atomic import atomic_copy, atomic_write_json, read_json
from voyage.augment import CRF_MAXIMUM as _AUGMENT_CRF_MAXIMUM
from voyage.augment import CRF_MINIMUM as _AUGMENT_CRF_MINIMUM
from voyage.augment import interpolated_frame_count as interpolated_frame_count
from voyage.errors import DiskSpaceError, MediaError
from voyage.hashing import sha256_file
from voyage.segment_manifest import load_segment_manifest

#: Max |video duration − audio duration| per segment, seconds (DESIGN §56
#: step 6). Same budget the commit path enforces, so anything committed
#: stays finalizable.
AV_ALIGNMENT_TOLERANCE_SECONDS = 0.6

#: Probed-vs-target fps tolerance, fps. ffprobe reports fractional container
#: rates (30000/1001 ≈ 29.97) for nominally integer sources, so exact
#: equality would reject healthy segments; 0.5 absorbs that rounding
#: without masking a real off-by-one-fps mismatch.
FPS_MATCH_TOLERANCE = 0.5

#: Fade length below which assembly skips the filter graph, seconds. A
#: sub-0.1 s fade is inaudible as a blend — plain concat carries the same
#: bytes without an ffmpeg filter pass.
MIN_FADE_GRAPH_SECONDS = 0.1

#: Overlap below which the finalize blend falls back to concat, seconds. A
#: sub-50 ms overlap cannot carry an audible fade, so building the overlap
#: graph would only add ffmpeg spawns for zero blend.
MIN_OVERLAP_BLEND_SECONDS = 0.05

#: Fade absorption below which tail compensation is skipped, seconds. The
#: `-ss`/`-t` slice grid carries millisecond precision, so sub-10 ms of
#: absorbed fade is rounding noise, not drift worth extending the tail for.
ABSORPTION_EPSILON_SECONDS = 0.01

#: Smallest legitimate take-slice piece, seconds (issue 104). Takes render
#: at >= 1 s (the ACE render layer rejects anything below), and take joints
#: land on segment boundaries — so a sub-50 ms piece is never real music
#: coverage, only a degenerate ledger sliver the `-t` floor would amplify
#: (0.1 s of audio per sliver) into an A/V monster.
MIN_SLICE_PIECE_SECONDS = 0.05

#: Bound on take slices per window walk (issue 104). The common case is one
#: slice (two at a take joint); 128 is orders of magnitude above legitimate
#: while capping ffmpeg spawns when the ledger degrades.
MAX_SLICES_PER_WINDOW = 128

#: Tolerance when estimating frame counts from duration, frames (issue 096).
#: Container durations round to milliseconds, so a 29-frame @32fps file can
#: probe as 28.99 estimated frames — one frame of slack keeps the
#: `nb_frames == 0` fallback from false-failing healthy files while still
#: rejecting genuinely short ones.
DURATION_FRAME_ESTIMATE_SLACK_FRAMES = 1.0

#: Inputs in one pairwise blend (issue 152). Two inputs delegate to
#: `_blend_pair` (identical single spawn); three or more take the staged
#: single-graph path with per-stage s32 barriers.
PAIR_BLEND_INPUT_COUNT = 2


def av_drift_seconds(video_duration: float, audio_duration: float) -> float:
    """Absolute A/V duration drift in seconds (issue 003 helper).

    Pure math over already-measured durations — no ffprobe here, so the
    commit path and `validate` can share the exact budget the finalizer
    enforces without re-probing media.
    """
    return abs(float(video_duration) - float(audio_duration))


def check_av_alignment(
    video_duration: float, audio_duration: float, segment_name: str = "segment"
) -> float:
    """Enforce the A/V budget; return the drift for the metrics event.

    Raises MediaError past the tolerance. HOOK FOR THE SUPERVISOR TRACK
    (supervisor.py is out of this change's scope): in `commit_one_segment`
    step 4, replace the video-vs-expected check with
    `drift = check_av_alignment(float(video_info["duration"]),
    float(audio_info["duration"]), segment_id)` (keeping the expected-
    duration check alongside), and record `"av_drift_seconds": drift` in
    the `segment_committed` metric event so scoreboard/soak can trend it.
    """
    drift = av_drift_seconds(video_duration, audio_duration)
    if drift > AV_ALIGNMENT_TOLERANCE_SECONDS:
        raise MediaError(
            f"segment {segment_name} A/V alignment drift {drift:.3f}s "
            f"exceeds {AV_ALIGNMENT_TOLERANCE_SECONDS:.1f}s"
        )
    return drift


def check_free_space(run_dir: Path, min_free_gib: float) -> float:
    """Free GiB under `run_dir`; raise DiskSpaceError below the reserve.

    Shared by the commit precheck and the finalize preflight (DESIGN §53):
    a final metadata write must never be the thing that discovers a full
    disk. Moved here from supervisor (import direction is supervisor →
    media, never the reverse).
    """
    free_gib = shutil.disk_usage(run_dir).free / (1024**3)
    if free_gib < min_free_gib:
        raise DiskSpaceError(f"free space {free_gib:.1f} GiB below reserve {min_free_gib:.1f} GiB")
    return free_gib


#: Default bound for every local ffmpeg/ffprobe spawn (issue 019). Mirrors
#: `rpc.DEFAULT_RPC_TIMEOUT_SECONDS` and `[voyage] rpc_timeout_seconds` —
#: the local-subprocess layer gets the same budget as the worker RPC layer.
FFMPEG_TIMEOUT_SECONDS = 600.0


def run_capture(
    argv: list[str], timeout: float = FFMPEG_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run argv capturing output; a wedged child maps to MediaError (issue 019).

    `timeout` mirrors `[voyage] rpc_timeout_seconds` / RPC 600 s default —
    callers may thread the configured value through; the default keeps
    existing call sites bounded without a config round-trip.
    """
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{argv[0] if argv else 'subprocess'} timed out after {timeout}s") from exc


def probe(path: Path) -> dict[str, Any]:
    proc = run_capture(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"ffprobe failed for {path}: {proc.stderr[-2000:]}")
    try:
        data: Any = json.loads(proc.stdout or "{}")
    except ValueError as exc:
        raise MediaError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MediaError(f"ffprobe returned non-object for {path}")
    return data


def validate_video(
    path: Path, width: int, height: int, fps: int, min_frames: int = 1
) -> dict[str, Any]:
    info = probe(path)
    streams = [s for s in info.get("streams", []) if isinstance(s, dict)]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream in {path}")
    if int(video.get("width", -1)) != width or int(video.get("height", -1)) != height:
        raise MediaError(
            f"resolution mismatch in {path}: "
            f"{video.get('width')}x{video.get('height')} != {width}x{height}"
        )
    rate = str(video.get("avg_frame_rate", "0/1"))
    num, _, den = rate.partition("/")
    try:
        actual_fps = float(num) / float(den or 1) if num else 0.0
    except (ValueError, ZeroDivisionError):
        raise MediaError(f"unparseable fps in {path}: {rate}") from None
    if abs(actual_fps - fps) > FPS_MATCH_TOLERANCE:
        raise MediaError(f"fps mismatch in {path}: {actual_fps} != {fps}")
    try:
        frames = int(video.get("nb_frames", 0) or 0)
    except (ValueError, TypeError):
        frames = 0  # containers that omit the count (mkv "N/A") read as unknown
    duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0:
        raise MediaError(f"non-positive duration in {path}")
    if frames == 0:
        # Unknown count: gate on the duration-derived estimate instead of
        # skipping the check (issue 096) — a genuinely long file passes, a
        # truncated one still fails.
        estimate = duration * actual_fps
        if estimate < min_frames - DURATION_FRAME_ESTIMATE_SLACK_FRAMES:
            raise MediaError(f"too few frames in {path}: {frames}")
    elif frames < min_frames:
        raise MediaError(f"too few frames in {path}: {frames}")
    return {"frames": frames, "duration": duration, "fps": actual_fps}


def _probe_video_fps(info: dict[str, Any]) -> float:
    """Parse the video stream's avg_frame_rate; 0.0 when absent/unparseable."""
    streams = info.get("streams", [])
    video = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"),
        None,
    )
    if video is None:
        return 0.0
    num, _, den = str(video.get("avg_frame_rate", "0/1")).partition("/")
    try:
        return float(num) / float(den or 1) if num else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def validate_audio(path: Path, sample_rate: int, channels: int) -> dict[str, Any]:
    info = probe(path)
    streams = [s for s in info.get("streams", []) if isinstance(s, dict)]
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise MediaError(f"no audio stream in {path}")
    if int(audio.get("sample_rate", -1)) != sample_rate:
        raise MediaError(f"sample-rate mismatch in {path}")
    if int(audio.get("channels", -1)) != channels:
        raise MediaError(f"channel mismatch in {path}")
    duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0:
        raise MediaError(f"non-positive duration in {path}")
    return {"duration": duration}


def slice_take(
    take_path: Path,
    start_seconds: float,
    duration_seconds: float,
    dest: Path,
    sample_rate: int,
    channels: int,
) -> Path:
    """Cut one segment-sized slice out of a music take (§35).

    Output is canonical segment audio (WAV s16le at the run's sample
    rate/channels) so slices from different takes always share the format
    the assembly crossfade requires.

    Rejects non-finite/non-positive/sub-50 ms durations (issue 104): the
    `max(duration, 0.1)` floor exists for ffmpeg's sake, but any caller
    passing a sliver is a bug — fail loud here instead of amplifying it
    into 0.1 s of audio per sliver.
    """
    import math

    if (
        not math.isfinite(start_seconds)
        or not math.isfinite(duration_seconds)
        or duration_seconds < MIN_SLICE_PIECE_SECONDS
    ):
        raise MediaError(
            f"take slice of {take_path} has degenerate window "
            f"(start={start_seconds}, duration={duration_seconds})"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-ss",
            f"{max(start_seconds, 0.0):.6f}",
            "-t",
            f"{max(duration_seconds, 0.1):.6f}",
            "-i",
            str(take_path),
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
        raise MediaError(f"take slice failed for {take_path}: {proc.stderr[-2000:]}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise MediaError(
            f"take slice produced empty output for {take_path} "
            f"(start={start_seconds:.6f}, duration={duration_seconds:.6f})"
        )
    return dest


def _take_joint_fade(min_piece_seconds: float, crossfade_seconds: float) -> float:
    """Crossfade length for one take joint (issues 094/095: single formula).

    Both the window loop (which compensates the absorption up front) and
    `assemble_segment_audio` (which consumes it) must agree exactly, so the
    formula lives here instead of inline in two places.
    """
    return min(crossfade_seconds, min_piece_seconds / 2.0)


def probed_take_seconds(take_file: Path) -> float:
    """Measured duration of a rendered take file (issue 094).

    ACE-Step renders are not sample-exact vs the requested duration; the
    ledger must be clamped to the file, not the request. Raises MediaError
    when the file probes empty so a missing render fails loud instead of
    covering zero seconds silently.
    """
    try:
        actual = float(probe(take_file).get("format", {}).get("duration", 0.0) or 0.0)
    except (OSError, ValueError) as exc:
        raise MediaError(f"take file {take_file} is unprobable: {exc}") from exc
    if actual <= 0.0:
        raise MediaError(f"take file {take_file} probed empty")
    return actual


def assemble_segment_audio(
    slices: list[Path],
    dest: Path,
    crossfade_seconds: float,
    joint_fade: float | None = None,
    *,
    blend_timings: list[float] | None = None,
) -> Path:
    """Join take slices into one segment audio.wav (§35).

    Consecutive slices (a take boundary falls inside the segment) are
    joined with a manual-fade pairwise blend; a single slice is copied
    through. The fade length is clamped to half the shortest slice so
    short segments can never collapse the filter (zoomy lesson: manual
    fades, never acrossfade on short tails — here takes are 30-60s but
    slices may be ~2s, hence the clamp).

    `joint_fade` carries the fade the caller compensated for (issue 095):
    `build_final_audio` extends the tail slice by the total absorption and
    passes the fade it used, so assembly consumes exactly what was added
    and the window tiles its nominal range. Without it the fade is derived
    from the slices as before.
    """
    if not slices:
        raise MediaError("no slices to assemble")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(slices) == 1:
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(slices[0]),
                "-c:a",
                "pcm_s16le",
                str(dest),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"slice copy failed: {proc.stderr[-2000:]}")
        if not dest.exists() or dest.stat().st_size == 0:
            raise MediaError(f"slice copy produced empty output for {slices[0]}")
        return dest
    durations = [float(probe(s).get("format", {}).get("duration", 0.0) or 0.0) for s in slices]
    if any(d <= 0 for d in durations):
        raise MediaError("slice with non-positive duration")
    fade = (
        joint_fade
        if joint_fade is not None
        else _take_joint_fade(min(durations), crossfade_seconds)
    )
    if fade < MIN_FADE_GRAPH_SECONDS:
        argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
        for s in slices:
            argv += ["-i", str(s)]
        filter_graph = "".join(f"[{i}:a]" for i in range(len(slices)))
        filter_graph += f"concat=n={len(slices)}:v=0:a=1[aout]"
        argv += ["-filter_complex", filter_graph, "-map", "[aout]", "-c:a", "pcm_s16le", str(dest)]
        proc = run_capture(argv)
        if proc.returncode != 0:
            raise MediaError(f"segment audio assembly failed: {proc.stderr[-2000:]}")
        return dest
    # Single-graph staged join (issue 152): chained pairwise stages with
    # s32 barriers replay the fold byte-for-byte in one spawn (each slice
    # decoded once). The old left-fold pairwise loop is gone; durations
    # probed above thread through so the join adds zero re-probes.
    with tempfile.TemporaryDirectory(prefix="voyage-assemble-") as staging:
        joined = Path(staging) / "joined.wav"
        _join_audio_single_graph(
            slices,
            joined,
            fade,
            durations=durations,
            timing_ms=blend_timings,
        )
        accum = joined
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(accum),
                "-c:a",
                "pcm_s16le",
                str(dest),
            ]
        )
    if proc.returncode != 0:
        raise MediaError(f"segment audio assembly failed: {proc.stderr[-2000:]}")
    return dest


def _sha256_file(path: Path) -> str:
    """Chunked SHA-256 (constant memory — takes can be multi-GB).

    Delegates to :func:`voyage.hashing.sha256_file` (issue 021); kept under
    the private name so the segment verifier below is untouched.
    """
    return sha256_file(path)


def _verify_segment(segment: Path) -> tuple[int, float, float]:
    """DESIGN §56 steps 4-6: checksums, frame ranges, A/V alignment.

    Returns (frames, video_duration, audio_duration). Raises MediaError
    on any mismatch — fail loud, never finalize corrupt media silently.
    """
    name = segment.name
    try:
        manifest = load_segment_manifest(segment)
    except MediaError as exc:
        raise MediaError(f"segment {name} has unreadable manifest: {exc}") from exc
    expected_any = manifest.get("checksums")
    expected = dict(expected_any) if isinstance(expected_any, dict) else {}
    if not expected:
        if (segment / "sha256.json").exists():
            raise MediaError(f"segment {name} has unreadable sha256.json")
        raise MediaError(f"segment {name} missing manifest.json")
    for artifact in ("video.mp4", "audio.wav"):
        recorded = expected.get(artifact)
        if not isinstance(recorded, str) or not recorded:
            raise MediaError(f"segment {name} manifest missing {artifact}")
        try:
            actual = _sha256_file(segment / artifact)
        except OSError as exc:
            raise MediaError(f"segment {name} missing {artifact}: {exc}") from exc
        if actual != recorded:
            raise MediaError(f"segment {name} checksum mismatch for {artifact}")
    for artifact, recorded in sorted(expected.items()):
        if artifact in ("video.mp4", "audio.wav"):
            continue
        if not isinstance(recorded, str) or not recorded:
            continue  # legacy manifest: media only means "not covered"
        target = segment / artifact
        if not target.exists():
            raise MediaError(f"segment {name} manifest lists missing {artifact}")
        if _sha256_file(target) != recorded:
            raise MediaError(f"segment {name} checksum mismatch for {artifact}")
    metrics_any = manifest.get("metrics")
    metrics = dict(metrics_any) if isinstance(metrics_any, dict) else {}
    try:
        frames = int(metrics.get("frames", 0))
    except (ValueError, TypeError) as exc:
        raise MediaError(f"segment {name} has unreadable manifest metrics: {exc}") from exc
    if frames <= 0:
        raise MediaError(f"segment {name} has non-positive frame count {frames}")
    video_duration = float(probe(segment / "video.mp4").get("format", {}).get("duration", 0.0))
    audio_duration = float(probe(segment / "audio.wav").get("format", {}).get("duration", 0.0))
    if video_duration <= 0 or audio_duration <= 0:
        raise MediaError(f"segment {name} has non-positive media duration")
    check_av_alignment(video_duration, audio_duration, name)
    return frames, video_duration, audio_duration


def _check_segment_committed(segment: Path) -> None:
    """Existence probe + full verification for one committed segment.

    Split out so the `skip_bad` triage loop below never raises inside its
    own `try` (the loop catches `MediaError` to skip — a raise in the
    `try` body would read as self-caught).
    """
    for artifact in ("video.mp4", "audio.wav"):
        if not (segment / artifact).exists():
            raise MediaError(f"segment {segment.name} missing {artifact}")
    _verify_segment(segment)


def _segment_timeline(usable: list[Path], fps: int) -> tuple[list[float], list[float], float]:
    """Per-segment [start, end) video-times from committed metrics + total.

    The timeline follows frame counts (the supervisor's truthful
    accounting), not container durations, so the blended audio matches
    the concatenated video sample-exactly.
    """
    if fps <= 0:
        raise MediaError(f"segment timeline needs positive fps (got {fps})")
    starts: list[float] = []
    ends: list[float] = []
    cursor = 0.0

    for segment in usable:
        try:
            manifest = load_segment_manifest(segment)
            metrics_any = manifest.get("metrics")
            metrics = dict(metrics_any) if isinstance(metrics_any, dict) else {}
            frames = int(metrics.get("frames", 0))
        except (ValueError, TypeError, MediaError, RecursionError) as exc:
            raise MediaError(
                f"segment {segment.name} has unreadable manifest metrics: {exc}"
            ) from exc
        if frames <= 0:
            raise MediaError(f"segment {segment.name} has non-positive frame count {frames}")
        starts.append(cursor)
        cursor += frames / fps
        ends.append(cursor)
    return starts, ends, cursor


def _slice_cache_key(
    take_path: Path, start_seconds: float, duration_seconds: float
) -> tuple[str, str, str]:
    """Cache key for one take slice (issue 031): take identity + the
    take-relative window at ffmpeg `.6f` precision, so adjacent segment
    windows re-slicing identical bytes hit instead of re-spawning ffmpeg
    on multi-GB takes."""
    return (str(take_path), f"{start_seconds:.6f}", f"{duration_seconds:.6f}")


def _cached_slice_take(
    slice_cache: dict[tuple[str, str, str], Path],
    take_path: Path,
    start_seconds: float,
    duration_seconds: float,
    dest: Path,
    sample_rate: int,
    channels: int,
) -> Path:
    """`slice_take` with a per-finalize memo (issue 031).

    Overlapping segment windows re-slice identical (take, start, duration)
    bytes when a take joint falls inside two windows; the repeat slice is a
    file copy of the first instead of another ffmpeg `-ss/-t` spawn
    reopening the take. Misses without an existing cached file re-slice and
    populate the cache (a moved-away entry simply misses again — the
    `exists` check keeps the cache honest after `replace` moves).
    """
    key = _slice_cache_key(take_path, start_seconds, duration_seconds)
    cached = slice_cache.get(key)
    if cached is not None and cached.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dest)
        return dest
    slice_take(take_path, start_seconds, duration_seconds, dest, sample_rate, channels)
    slice_cache[key] = dest
    return dest


def _concat_fallback_audio(inputs: list[Path], dest: Path) -> Path:
    """Join segment audio.wav files with a plain concat (no blend)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for source in inputs:
        argv += ["-i", str(source)]
    filter_graph = "".join(f"[{i}:a]" for i in range(len(inputs)))
    filter_graph += f"concat=n={len(inputs)}:v=0:a=1[aout]"
    argv += ["-filter_complex", filter_graph, "-map", "[aout]", "-c:a", "pcm_s16le", str(dest)]
    proc = run_capture(argv)
    if proc.returncode != 0:
        raise MediaError(f"final audio concat failed: {proc.stderr[-2000:]}")
    return dest


def _audio_duration_seconds(path: Path) -> float:
    """Probed audio duration; fail loud on unreadable/empty files."""
    duration = float(probe(path).get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0:
        raise MediaError(f"audio file has non-positive duration: {path}")
    return duration


def _blend_fade_seconds(first_seconds: float, second_seconds: float, overlap: float) -> float:
    """One crossfade length both blends agree on (issue 152: single source).

    `_blend_pair` clamps each pair to half the shortest input; fold callers
    (`assemble_segment_audio`, `build_final_audio`, `render_sfx_bed`) use the
    same formula to track the growing accum length arithmetically, so each
    stem/window is probed once instead of twice per blend.
    """
    fade = min(float(overlap), first_seconds / 2.0, second_seconds / 2.0)
    if fade <= 0:
        raise MediaError(
            f"cannot blend with non-positive overlap ({first_seconds:.3f}s + {second_seconds:.3f}s)"
        )
    return fade


def _blend_pair(
    first: Path,
    second: Path,
    dest: Path,
    overlap: float,
    *,
    first_seconds: float | None = None,
    second_seconds: float | None = None,
    timing_ms: list[float] | None = None,
) -> Path:
    """Crossfade-blend two audio files with manual fades (never acrossfade).

    acrossfade is unusable here in two independent ways (live incidents on
    ffmpeg 7.1.5): a 31-input chain deadlocks the filter scheduler (futex
    wait, zero bytes out, stuck 3+ days), and even a 2-input graph
    collapses whenever the FIRST input is much longer than the second
    (86s+1s at d=0.4 came out as ~0.2s; poulah window 21's 3.49s+0.91s
    slices came out as ~1-3s instead of ~4s). Manual fades (afade out on
    the first tail + afade in on the second head + adelay + amix with
    normalize=0) are sample-passthrough filters with no scheduler
    pathology — the recipe the codebase already prescribes for short
    tails. Intermediates stay s32le (no generational 16-bit loss); the
    caller converts to s16le at the end.

    `first_seconds`/`second_seconds` thread already-known durations (issue
    152): fold callers probe each stem/window once and pass the values, so
    the N-1 re-probes of the growing accum disappear; absent (None) probes
    as before. `timing_ms` collects one wall-millisecond entry per call so
    soak can trend fold cost vs timeline length.
    """
    start = time.monotonic()
    try:
        if first_seconds is None:
            first_seconds = _audio_duration_seconds(first)
        if second_seconds is None:
            second_seconds = _audio_duration_seconds(second)
        fade = _blend_fade_seconds(first_seconds, second_seconds, overlap)
        fade_start = first_seconds - fade
        delay_ms = int(round(fade_start * 1000))
        filter_graph = (
            f"[0:a]afade=t=out:st={fade_start:.3f}:d={fade:.3f}[a0];"
            f"[1:a]afade=t=in:st=0:d={fade:.3f},adelay={delay_ms}:all=1[a1];"
            "[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[aout]"
        )
        argv: list[str] = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(first),
            "-i",
            str(second),
            "-filter_complex",
            filter_graph,
            "-map",
            "[aout]",
            "-c:a",
            "pcm_s32le",
            str(dest),
        ]
        proc = run_capture(argv)
        if proc.returncode != 0:
            raise MediaError(f"final audio pairwise blend failed: {proc.stderr[-2000:]}")
        return dest
    finally:
        if timing_ms is not None:
            timing_ms.append((time.monotonic() - start) * 1000.0)


def _join_audio_single_graph(
    inputs: list[Path],
    dest: Path,
    overlap: float,
    *,
    durations: list[float] | None = None,
    timing_ms: list[float] | None = None,
) -> Path:
    """Join N audio files in one ffmpeg spawn, fold-identical (issue 152).

    Chained pairwise stages with an `aformat=sample_fmts=s32` barrier per
    stage replay the left-fold's per-blend s32 quantization inside a single
    graph: each stage's `afade` negotiates s32 (as when fed the fold's s32
    intermediates) instead of fltp, so the output is byte-identical to the
    `accum -> _blend_pair -> step` fold while each stem is decoded once
    (O(N) I/O, one spawn, not N-1). Proven at N=31 CPU-only (byte-identical,
    staged ~0.7 s vs fold ~1.9 s on 4 s sine stems) — the incident-scale
    no-hang proof the `test_finalize_fastpath` pin required.

    Never `acrossfade` (the 31-input deadlocked filter class): only the
    manual recipe (`afade` out/in + `adelay` + `amix inputs=2 normalize=0`)
    plus the s32 barriers. Fade arithmetic mirrors the fold exactly (same
    `_blend_fade_seconds` per pair, same `%.3f` fades, same integer-ms
    delays). Two inputs delegate to `_blend_pair` (identical single spawn).
    `durations` threads already-probed lengths (O(N) probes when None);
    `timing_ms` collects one wall-millisecond entry for the single spawn.
    Output stays s32le; the caller converts to s16le at the end.
    """
    start = time.monotonic()
    try:
        if len(inputs) < PAIR_BLEND_INPUT_COUNT:
            raise MediaError("single-graph join needs at least 2 inputs")
        if durations is None:
            resolved = [_audio_duration_seconds(item) for item in inputs]
        else:
            if len(durations) != len(inputs):
                raise MediaError(
                    f"single-graph join needs {len(inputs)} durations (got {len(durations)})"
                )
            resolved = [float(value) for value in durations]
            if any(value <= 0 for value in resolved):
                raise MediaError("single-graph join needs positive durations")
        if len(inputs) == PAIR_BLEND_INPUT_COUNT:
            return _blend_pair(
                inputs[0],
                inputs[1],
                dest,
                overlap,
                first_seconds=resolved[0],
                second_seconds=resolved[1],
                timing_ms=timing_ms,
            )
        fades: list[float] = []
        starts = [0.0]
        accum = resolved[0]
        for index in range(1, len(inputs)):
            fade = _blend_fade_seconds(accum, resolved[index], overlap)
            fades.append(fade)
            starts.append(accum - fade)
            accum = accum + resolved[index] - fade
        parts: list[str] = []
        for index in range(1, len(inputs)):
            if index == 1:
                left = (
                    f"[0:a]afade=t=out:st={starts[index]:.3f}:d={fades[index - 1]:.3f}[m{index}a]"
                )
            else:
                left = (
                    f"[m{index - 1}q]afade=t=out:st={starts[index]:.3f}:"
                    f"d={fades[index - 1]:.3f}[m{index}a]"
                )
            right = (
                f"[{index}:a]afade=t=in:st=0:d={fades[index - 1]:.3f},"
                f"adelay={int(round(starts[index] * 1000))}:all=1[m{index}b]"
            )
            mix = (
                f"[m{index}a][m{index}b]amix=inputs=2:duration=longest:"
                f"dropout_transition=0:normalize=0[m{index}]"
            )
            parts += [left, right, mix]
            if index < len(inputs) - 1:
                parts.append(f"[m{index}]aformat=sample_fmts=s32[m{index}q]")
        dest.parent.mkdir(parents=True, exist_ok=True)
        argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
        for item in inputs:
            argv += ["-i", str(item)]
        argv += [
            "-filter_complex",
            ";".join(parts),
            "-map",
            f"[m{len(inputs) - 1}]",
            "-c:a",
            "pcm_s32le",
            str(dest),
        ]
        proc = run_capture(argv)
        if proc.returncode != 0:
            raise MediaError(f"single-graph audio join failed: {proc.stderr[-2000:]}")
        return dest
    finally:
        if timing_ms is not None and len(inputs) > PAIR_BLEND_INPUT_COUNT:
            timing_ms.append((time.monotonic() - start) * 1000.0)


def build_final_audio(
    run_dir: Path,
    usable: list[Path],
    tmpdir: Path,
    fps: int,
    sample_rate: int,
    channels: int,
    overlap_fraction: float = 0.10,
    overlap_cap_seconds: float = 0.5,
    *,
    blend_timings: list[float] | None = None,
) -> Path:
    """Blend committed segments into one timeline-exact final mix (§56).

    Each segment boundary gets an overlap crossfade: segment windows are
    extended by half the overlap on each side and re-sliced from the
    takes ledger (takes are continuous, so the extension is real musical
    content — not time-stretched), then blended pairwise with manual
    fades (afade out/in + adelay + amix). Total
    length stays exactly the video timeline, so no A/V drift.

    Falls back to a plain concat of the per-segment audio.wav previews
    when the takes ledger is unavailable (pre-take runs) — the old
    hard-splice behavior, clicks included. Per-segment previews are
    never rewritten: the blend exists only in the returned final mix.
    """
    from voyage.audio.planner import AudioPlanner, load_takes

    dest = tmpdir / "final_audio.wav"
    if len(usable) == 1:
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(usable[0] / "audio.wav"),
                "-c:a",
                "pcm_s16le",
                str(dest),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"final audio copy failed: {proc.stderr[-2000:]}")
        return dest
    starts, ends, timeline = _segment_timeline(usable, fps)
    durations = [end - start for start, end in zip(starts, ends, strict=True)]
    overlap = min(overlap_fraction * min(durations), overlap_cap_seconds)
    ledger = run_dir / "audio" / "takes.jsonl"
    takes: list[Any] = load_takes(ledger) if ledger.exists() else []
    planner = AudioPlanner(takes=takes)
    if not takes or overlap < MIN_OVERLAP_BLEND_SECONDS:
        return _concat_fallback_audio([s / "audio.wav" for s in usable], dest)
    half = overlap / 2.0
    windows: list[Path] = []
    # Issue 031: memo take slices across windows (take joints inside two
    # windows re-slice identical bytes). Scoped to this build — tmpdir paths
    # die with the finalize, so the cache must never outlive the call.
    slice_cache: dict[tuple[str, str, str], Path] = {}
    for index, segment in enumerate(usable):
        window_start = max(starts[index] - (half if index > 0 else 0.0), 0.0)
        window_end = min(ends[index] + (half if index < len(usable) - 1 else 0.0), timeline)
        # Slice the takes covering this (possibly extended) window; a take
        # joint inside the window yields two slices joined as usual.
        # Pieces record their take-coverage bounds so the joint
        # compensation below can clamp extensions to real content.
        slices: list[Path] = []
        piece_bounds: list[tuple[float, float]] = []
        take_bounds: list[tuple[float, float]] = []
        tail_serving_path: Path | None = None
        cursor = window_start
        piece = 0
        while cursor < window_end - 1e-6:
            if piece >= MAX_SLICES_PER_WINDOW:
                return _concat_fallback_audio([s / "audio.wav" for s in usable], dest)
            serving = planner.take_for_time(cursor)
            if serving is None or not serving.path:
                return _concat_fallback_audio([s / "audio.wav" for s in usable], dest)
            # Issue 016 consumer side: ledger entries may be run-relative
            # or legacy absolute — resolve the same way at every use site.
            serving_path = paths.resolve_stored_path(run_dir, serving.path)
            if not serving_path.exists():
                return _concat_fallback_audio([s / "audio.wav" for s in usable], dest)
            piece_end = min(serving.covers_until(), window_end)
            if piece_end <= cursor:
                return _concat_fallback_audio([s / "audio.wav" for s in usable], dest)
            if piece_end - cursor < MIN_SLICE_PIECE_SECONDS:
                return _concat_fallback_audio([s / "audio.wav" for s in usable], dest)
            slice_path = tmpdir / f"{segment.name}_w{piece:02d}.wav"
            _cached_slice_take(
                slice_cache,
                serving_path,
                cursor - serving.covers_from,
                piece_end - cursor,
                slice_path,
                sample_rate,
                channels,
            )
            slices.append(slice_path)
            piece_bounds.append((cursor, piece_end))
            take_bounds.append((serving.covers_from, serving.covers_until()))
            tail_serving_path = serving_path
            cursor = piece_end
            piece += 1
        window_path = tmpdir / f"{segment.name}_window.wav"
        if len(slices) == 1:
            slices[0].replace(window_path)
        else:
            # Issue 095: a crossfade overlaps unique content, so joining N
            # abutting slices absorbs fade*(N-1) seconds and the window
            # comes out short of its nominal range (which then shortens the
            # whole mix, and `-shortest` trims video frames to match). Extend
            # the tail piece by exactly the absorption — takes are
            # continuous, so the extra content is real music — clamped to
            # the take file and the timeline; pass the fade explicitly so
            # assembly consumes exactly what was added.
            if tail_serving_path is None:
                raise MediaError(f"segment {segment.name} assembled no slices")
            piece_durations = [end - start for start, end in piece_bounds]
            fade = _take_joint_fade(min(piece_durations), overlap)
            absorption = fade * (len(slices) - 1)
            if absorption >= ABSORPTION_EPSILON_SECONDS:
                tail_start, tail_end = piece_bounds[-1]
                take_start, _take_end = take_bounds[-1]
                # Clamp to the take file only — never to the timeline: the
                # extension restores the nominal range at most, so the
                # window cannot overshoot it (and the final `-shortest` mux
                # guards any residual). Clamping to the timeline instead
                # would forbid compensation exactly where the last window
                # needs it while its take continues past the end.
                take_file_end = take_start + probed_take_seconds(tail_serving_path)
                compensated_end = min(tail_end + absorption, take_file_end)
                if compensated_end > tail_end + 1e-6:
                    tail_slice = tmpdir / f"{segment.name}_w{piece:02d}.wav"
                    _cached_slice_take(
                        slice_cache,
                        tail_serving_path,
                        tail_start - take_start,
                        compensated_end - tail_start,
                        tail_slice,
                        sample_rate,
                        channels,
                    )
                    slices[-1] = tail_slice
            assemble_segment_audio(slices, window_path, overlap, joint_fade=fade)
        windows.append(window_path)
    # Single-graph staged join (issue 152): chained pairwise stages with
    # s32 barriers replay the fold byte-for-byte in one spawn (N=31 proven,
    # no acrossfade anywhere). Each window is probed once here and threaded
    # through, so the join adds zero re-probes.
    window_seconds = [_audio_duration_seconds(window) for window in windows]
    joined = tmpdir / "final_joined.wav"
    _join_audio_single_graph(
        windows,
        joined,
        overlap,
        durations=window_seconds,
        timing_ms=blend_timings,
    )
    accum = joined
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(accum),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"final audio blend failed: {proc.stderr[-2000:]}")
    return dest


# Floor for the final presentation frame rate (user decision 2026-09-25:
# the shipped video is always >= 24fps). Sub-24fps sources (CausVid native
# 16fps) are motion-interpolated up; sources already at/above the floor
# keep the plain fps filter (no behavior change).
PRESENTATION_MIN_FPS = 24

#: Presentation floors for the Track B augment path (coming config): the
#: shipped video is always >= 32fps and covers 1280x720. `finalize_run`
#: and `FinalizeOptions` default to these so headless/legacy callers get
#: the same presentation without a config round-trip.
AUGMENT_DEFAULT_MIN_FPS = 32
AUGMENT_DEFAULT_MIN_WIDTH = 1280
AUGMENT_DEFAULT_MIN_HEIGHT = 720


@dataclass
class AugmentPlan:
    """Presentation geometry/fps the finalizer must produce (Track B).

    Why this exists: segment videos render at backend-native geometry
    (CausVid 832x480@16, LTXV 768x512@24, fake 768x432@24) but the
    shipped video must always cover the presentation floors. The plan is
    pure math over probed source + requested target + floors, so unit
    tests pin it without ffmpeg and `finalize_run` just renders it.
    """

    out_w: int
    out_h: int
    out_fps: int
    needs_reencode: bool
    needs_minterpolate: bool


def plan_augmentation(
    source_w: int,
    source_h: int,
    source_fps: float,
    target_w: int,
    target_h: int,
    requested_fps: int,
    min_fps: int | None,
    min_width: int | None,
    min_height: int | None,
) -> AugmentPlan:
    """Compute the presentation box/fps for one finalize (pure, Track B).

    `effective_fps = max(requested, min_fps or 0, PRESENTATION_MIN_FPS)`;
    geometry is `max(target, min)` per axis — the output box always
    covers both the requested target and the floors, preserving aspect
    downstream via the scale-to-fit + pad vf (never stretched). Floors
    only ever upscale: a target already above them is kept as-is
    ("minimal upscale"), and a 0/None floor disables that axis (the
    24fps `PRESENTATION_MIN_FPS` still applies — 0 disables the new
    32fps floor, not the shipped-video guarantee).

    `needs_minterpolate` is True only for an fps lift (source + 0.5 <
    out — motion interpolation); an fps drop uses the plain fps filter.
    `needs_reencode` covers any pixel/timing change (dims differ, fps
    differs past 0.5 either way, lift, or unknown source fps) and gates
    the stream-copy fast path off.
    """
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"target geometry must be positive (got {target_w}x{target_h})")
    if requested_fps <= 0:
        raise ValueError(f"requested fps must be positive (got {requested_fps})")
    floor_fps = int(min_fps or 0)
    floor_w = int(min_width or 0)
    floor_h = int(min_height or 0)
    if floor_fps < 0 or floor_w < 0 or floor_h < 0:
        raise ValueError(f"augment floors must be >= 0 (got {min_fps}/{min_width}/{min_height})")
    out_fps = max(int(requested_fps), floor_fps, PRESENTATION_MIN_FPS)
    out_w = max(int(target_w), floor_w)
    out_h = max(int(target_h), floor_h)
    source_fps_value = float(source_fps)
    needs_minterpolate = source_fps_value > 0 and out_fps > source_fps_value + FPS_MATCH_TOLERANCE
    fps_mismatch = source_fps_value <= 0 or abs(out_fps - source_fps_value) > FPS_MATCH_TOLERANCE
    needs_reencode = bool(
        needs_minterpolate or fps_mismatch or int(source_w) != out_w or int(source_h) != out_h
    )
    return AugmentPlan(
        out_w=out_w,
        out_h=out_h,
        out_fps=out_fps,
        needs_reencode=needs_reencode,
        needs_minterpolate=needs_minterpolate,
    )


def presentation_setup_facts(
    plan: AugmentPlan,
    *,
    min_fps: int | None,
    min_width: int | None,
    min_height: int | None,
) -> dict[str, object]:
    """§104 setup facts for the finalize presentation floors (issue 194).

    Pure: the floor triple as given plus the resolved presentation plan,
    so benchmark/soak reports can record the dominant finalize variable
    instead of leaving re-encode-vs-stream-copy unexplained. HOOK FOR THE
    OBSERVE TRACK (`voyage/cli_observe.py` is out of this change's scope):
    spread these facts into `_benchmark_env()` (or alongside
    `_video_geometry_setup()`) at the `cmd_benchmark`/`cmd_soak` setup
    sites so every §104 setup block carries them.
    """
    return {
        "min_fps": min_fps,
        "min_width": min_width,
        "min_height": min_height,
        "out_w": plan.out_w,
        "out_h": plan.out_h,
        "out_fps": plan.out_fps,
        "needs_reencode": plan.needs_reencode,
        "needs_minterpolate": plan.needs_minterpolate,
    }


def _probe_video_geometry(info: dict[str, Any]) -> tuple[int, int]:
    """Source WxH from ffprobe info; (0, 0) when absent/unparseable.

    Unknown geometry forces the augment re-encode path (the plan treats
    0 as "differs from any positive out box"), so callers never
    stream-copy blind.
    """
    streams = info.get("streams", [])
    video = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"),
        None,
    )
    if video is None:
        return (0, 0)
    try:
        return (int(video.get("width", 0) or 0), int(video.get("height", 0) or 0))
    except (ValueError, TypeError):
        return (0, 0)


JointStyle = Literal["blend", "hard-splice"]
"""Audio-joint rendering for the final mix (issues 045, 046).

`blend` re-slices the takes ledger with a proportional overlap crossfade
(the default); `hard-splice` concatenates the per-segment previews with
no blend (the legacy behavior, clicks included). An explicit style
replaces the old `overlap_fraction=0` encoding, so call sites state the
intent instead of smuggling it through a zero.
"""

#: Default h264 quality for the finalize encode (issues 050). Matches the
#: validated Comfy `video_export.json` recipe (crf 15) and the chunk
#: encoder in `voyage.augment.ffmpeg_encode_chunk` (same default), so the
#: shipped video never silently uses ffmpeg's default CRF 23.
FINALIZE_CRF_DEFAULT = 15

#: Lowest/highest h264 CRF (issues 050, 083). Single ladder home is
#: `voyage.augment` (`CRF_MINIMUM`/`CRF_MAXIMUM`); these aliases keep the
#: `FINALIZE_CRF_*` names for existing imports (the old "stated here so
#: `media` stays stdlib-only without importing" comment is stale —
#: `voyage.augment` is stdlib-only too, so the import is free).
FINALIZE_CRF_MINIMUM = _AUGMENT_CRF_MINIMUM
FINALIZE_CRF_MAXIMUM = _AUGMENT_CRF_MAXIMUM

#: Default x264 speed/quality trade-off (issues 050). Keeps the validated
#: `veryfast` recipe; slower presets are opt-in via `FinalizeOptions`.
FINALIZE_PRESET_DEFAULT = "veryfast"

#: Allowed x264 presets (issues 050). The full ffmpeg `-preset` vocabulary
#: for libx264, so validation rejects typos before an ffmpeg spawn fails.
FINALIZE_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
)


def validate_crf(value: int) -> int:
    """Validate a finalize CRF (issues 050).

    Ints only (bools rejected — `True` is `1` but never a quality knob);
    range is the h264 0..51 ladder. Returns the value for `__post_init__`
    and scalar-override paths to share.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"crf must be an int (got {value!r})")
    if value < FINALIZE_CRF_MINIMUM or value > FINALIZE_CRF_MAXIMUM:
        raise ValueError(
            f"crf must be within [{FINALIZE_CRF_MINIMUM}, {FINALIZE_CRF_MAXIMUM}] (got {value})"
        )
    return value


def validate_preset(value: str) -> str:
    """Validate a finalize x264 preset (issues 050)."""
    if not isinstance(value, str):
        raise TypeError(f"preset must be a str (got {value!r})")
    if value not in FINALIZE_PRESETS:
        raise ValueError(f"preset must be one of {sorted(FINALIZE_PRESETS)} (got {value!r})")
    return value


def escape_concat_path(path: Path | str) -> str:
    """Escape a path for an ffmpeg concat-demuxer `file '...'` line (053).

    A single quote inside the single-quoted value closes the quoting, so
    it becomes `'\\''` (close, escaped literal, reopen) per the ffmpeg
    concat-demuxer docs. Spaces/`$`/double quotes need no escaping inside
    the single quotes with `-safe 0`; arg-lists already keep them safe on
    the supervisor side.
    """
    return str(path).replace("'", "'\\''")


def write_concat_list(entries: list[Path], dest: Path) -> Path:
    """Write a concat-demuxer list with quoting-safe entries (053).

    Every entry goes through `escape_concat_path`, so adversarial run
    directories (`o'brien`, spaces) produce a parseable list instead of a
    truncated `file '...'` line. Returns `dest` for call-site chaining.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "".join(f"file '{escape_concat_path(entry)}'\n" for entry in entries),
        encoding="utf-8",
    )
    return dest


@dataclass
class FinalizeOptions:
    """Explicit finalize knobs (issue 045).

    Twelve positional scalars made `finalize_run`'s semantics depend on
    hand-audited call sites; this dataclass groups the five semantic
    knobs (skip policy, audio shape, joint rendering) under names. New
    callers should pass `options=`; the legacy scalars stay as the
    default path and build an equivalent instance internally, so existing
    callers are untouched.

    Track B augment fields (`min_fps`/`min_width`/`min_height`): the
    presentation floors `finalize_run` enforces via `plan_augmentation`
    (defaults 32/1280/720 to match the coming config; 0 disables that
    axis — the 24fps `PRESENTATION_MIN_FPS` still applies).

    Encode fields (`crf`/`preset`, issues 050): the single vf encode
    quality (defaults crf 15 + veryfast match the validated Comfy
    `video_export.json` recipe and `augment.ffmpeg_encode_chunk`).

    Model-pass field (`use_model_pass`, issue 166): opt-in Real-ESRGAN +
    FILM pass when provisioned (default False — ffmpeg floors only; absent
    legs fall back to the same vf path, so off == on-absent byte-for-byte).
    """

    skip_bad: bool = False
    sample_rate: int = 48000
    channels: int = 2
    overlap_fraction: float = 0.10
    overlap_cap_seconds: float = 0.5
    joint_style: JointStyle = "blend"
    min_fps: int = 32
    min_width: int = 1280
    min_height: int = 720
    crf: int = FINALIZE_CRF_DEFAULT
    preset: str = FINALIZE_PRESET_DEFAULT
    use_model_pass: bool = False

    def __post_init__(self) -> None:
        if self.joint_style not in ("blend", "hard-splice"):
            raise ValueError(
                f"joint_style must be 'blend' or 'hard-splice' (got {self.joint_style!r})"
            )
        if self.overlap_fraction < 0:
            raise ValueError(f"overlap_fraction must be >= 0 (got {self.overlap_fraction})")
        if self.overlap_cap_seconds < 0:
            raise ValueError(f"overlap_cap_seconds must be >= 0 (got {self.overlap_cap_seconds})")
        if self.min_fps < 0:
            raise ValueError(f"min_fps must be >= 0 (got {self.min_fps})")
        if self.min_width < 0:
            raise ValueError(f"min_width must be >= 0 (got {self.min_width})")
        if self.min_height < 0:
            raise ValueError(f"min_height must be >= 0 (got {self.min_height})")
        if not isinstance(self.use_model_pass, bool):
            raise TypeError(f"use_model_pass must be a bool (got {self.use_model_pass!r})")
        validate_crf(self.crf)
        validate_preset(self.preset)

    def effective_overlap_fraction(self) -> float:
        """Overlap the mixer actually uses: hard-splice forces zero."""
        return 0.0 if self.joint_style == "hard-splice" else self.overlap_fraction


def _segment_video_matches_target(segment: Path, width: int, height: int, fps: int) -> bool:
    """True when a committed segment video can stream-copy into the final (031).

    Probe-only, never raises: any mismatch or probe failure falls back to
    the re-encode path. Requires h264 + yuv420p + matching WxH/fps so
    `ffmpeg -f concat -c copy` yields a valid final without touching pixels.
    """
    try:
        info = probe(segment / "video.mp4")
        streams = [s for s in info.get("streams", []) if isinstance(s, dict)]
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        if video is None:
            return False
        if int(video.get("width", -1)) != width or int(video.get("height", -1)) != height:
            return False
        if str(video.get("codec_name", "")) != "h264":
            return False
        if str(video.get("pix_fmt", "")) != "yuv420p":
            return False
        return abs(_probe_video_fps(info) - fps) <= FPS_MATCH_TOLERANCE
    except (MediaError, ValueError, TypeError, KeyError):
        return False


@dataclass(frozen=True)
class ResolvedFinalizeSettings:
    """One contract for `finalize_run` knob resolution (issue 083).

    The twelve positional scalars are the legacy shim; `options=` is the
    canonical knob. Both spell the same thing — this struct is what
    `finalize_run` actually consumes — so scalar-built and options-built
    calls with matching values resolve identically (pinned by
    `tests/test_media_augment_unified_083.py`). `settings` carries the
    audio/joint/skip policy; the `effective_*` fields carry the
    presentation floors + encode quality after the
    "explicit scalar wins over `options`, `None` means use `options`"
    rule (0 disables a floor axis — the 24fps `PRESENTATION_MIN_FPS`
    still applies downstream in `plan_augmentation`).
    """

    settings: FinalizeOptions
    min_fps: int
    min_width: int
    min_height: int
    crf: int
    preset: str
    use_model_pass: bool


def resolve_finalize_settings(
    *,
    options: FinalizeOptions | None,
    skip_bad: bool | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    overlap_fraction: float | None = None,
    overlap_cap_seconds: float | None = None,
    min_fps: int | None,
    min_width: int | None,
    min_height: int | None,
    crf: int | None,
    preset: str | None,
    use_model_pass: bool | None = None,
) -> ResolvedFinalizeSettings:
    """Resolve the scalar/`options=` split into one settings struct (pure).

    One uniform rule for every knob (issue 190): an explicit scalar wins
    over `options`, `None` means "use the `options` value" (0 disables a
    floor axis — the 24fps `PRESENTATION_MIN_FPS` still applies downstream
    in `plan_augmentation`). No filesystem, no ffmpeg: `finalize_run`
    calls this first, then runs the §53 preflight + encode off the result.
    Extracted (not duplicated) so the shim and the canonical path can
    never drift.
    """
    if options is None:
        effective_overlap = (
            FinalizeOptions.overlap_fraction if overlap_fraction is None else overlap_fraction
        )
        settings = FinalizeOptions(
            skip_bad=False if skip_bad is None else skip_bad,
            sample_rate=48000 if sample_rate is None else sample_rate,
            channels=2 if channels is None else channels,
            overlap_fraction=effective_overlap,
            overlap_cap_seconds=0.5 if overlap_cap_seconds is None else overlap_cap_seconds,
            joint_style="hard-splice" if effective_overlap <= 0 else "blend",
            min_fps=AUGMENT_DEFAULT_MIN_FPS if min_fps is None else min_fps,
            min_width=AUGMENT_DEFAULT_MIN_WIDTH if min_width is None else min_width,
            min_height=AUGMENT_DEFAULT_MIN_HEIGHT if min_height is None else min_height,
            crf=FINALIZE_CRF_DEFAULT if crf is None else crf,
            preset=FINALIZE_PRESET_DEFAULT if preset is None else preset,
            use_model_pass=False if use_model_pass is None else use_model_pass,
        )
    else:
        settings = options
        if skip_bad is not None:
            settings = replace(settings, skip_bad=skip_bad)
        if sample_rate is not None:
            settings = replace(settings, sample_rate=sample_rate)
        if channels is not None:
            settings = replace(settings, channels=channels)
        if overlap_fraction is not None:
            settings = replace(settings, overlap_fraction=overlap_fraction)
        if overlap_cap_seconds is not None:
            settings = replace(settings, overlap_cap_seconds=overlap_cap_seconds)
        if use_model_pass is not None:
            settings = replace(settings, use_model_pass=use_model_pass)
    return ResolvedFinalizeSettings(
        settings=settings,
        min_fps=min_fps if min_fps is not None else settings.min_fps,
        min_width=min_width if min_width is not None else settings.min_width,
        min_height=min_height if min_height is not None else settings.min_height,
        crf=validate_crf(crf if crf is not None else settings.crf),
        preset=validate_preset(preset if preset is not None else settings.preset),
        use_model_pass=settings.use_model_pass,
    )


def _record_final_geometry(
    run_dir: Path,
    width: int,
    height: int,
    fps: int,
    min_fps: int,
    min_width: int,
    min_height: int,
) -> bool:
    """Best-effort provenance write-back (issue 141): stamp the shipped box.

    `build_manifest` records the run's `[augment]` floors plus a null
    `final_geometry` at init; the first finalize overwrites both with the
    effective floors and the validated output box, so the manifest never
    claims the stale 768x432 source hint as shipped geometry. Returns False
    (never raises — a provenance write must not fail a finalize) when the
    run has no manifest, e.g. throwaway/legacy dirs.
    """
    try:
        manifest_path = run_dir / paths.MANIFEST_FILENAME
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            return False
        manifest["presentation"] = {
            "min_fps": min_fps,
            "min_width": min_width,
            "min_height": min_height,
        }
        manifest["final_geometry"] = {"width": width, "height": height, "fps": fps}
        atomic_write_json(manifest_path, manifest)
    except (OSError, ValueError):
        return False
    return True


def finalize_run(
    run_dir: Path,
    output_path: Path,
    width: int = 768,
    height: int = 432,
    fps: int = 24,
    skip_bad: bool | None = None,
    min_free_space_gib: float = 0.0,
    sample_rate: int | None = None,
    channels: int | None = None,
    overlap_fraction: float | None = None,
    overlap_cap_seconds: float | None = None,
    min_fps: int | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
    crf: int | None = None,
    preset: str | None = None,
    use_model_pass: bool | None = None,
    models_dir: Path | str | None = None,
    options: FinalizeOptions | None = None,
) -> Path:
    """Concat committed segments → single normalized MP4 (DESIGN §56).

    The presentation box/fps come from `plan_augmentation` (Track B):
    `max(requested, floors, 24fps)` for fps and `max(target, floors)`
    per axis for geometry, so backend-native segments (CausVid
    832x480@16, LTXV 768x512@24) ship at >= 1280x720@32 by default.
    One uniform knob rule (issue 190): an explicit scalar wins over
    `options`, `None` means "use the `options` value" (which defaults to
    32/1280/720 floors, crf 15 + veryfast, blend joints at 0.10 overlap);
    pass 0 to disable a floor axis (the 24fps `PRESENTATION_MIN_FPS`
    still applies). An explicit `overlap_fraction=0` behaves as a hard
    splice (the blend falls back to concat below the audibility floor).

    Native-geometry runs (presentation already matches) stream-copy the
    committed videos with zero video re-encodes (issue 031 fast path);
    anything the plan flags (`needs_reencode`) takes a single
    concat-demuxer + vf encode (issues 050: minterpolate-when-lifting +
    scale/pad/fps, one libx264 pass over the originals — no intermediate
    per-segment parts). Then mux audio, validate against the presentation
    box/fps, atomically publish, and append a `finalize_completed` event
    (effective crf/preset + `parts_encode_ms`/`audio_blend_ms`/
    `final_encode_ms`) to `logs/metrics.jsonl` for soak trending. With
    skip_bad, corrupt segments are skipped with a warning instead of
    aborting the whole finalize — input triage only (issues 138/188):
    missing artifacts, checksum/metrics/alignment failures, and numbering
    gaps each print a `finalize: skipping ...` line and continue, while
    the post-assembly `validate_video` stays strict (a corrupt stage
    still aborts even under skip_bad). A positive `min_free_space_gib`
    runs the §53 preflight first so a full disk fails fast instead of
    mid-encode.

    Native-geometry runs (presentation already matches) stream-copy the
    committed videos with zero video re-encodes (issue 031 fast path);
    anything the plan flags (`needs_reencode`) takes a single
    concat-demuxer + vf encode (issues 050: minterpolate-when-lifting +
    scale/pad/fps, one libx264 pass over the originals — no intermediate
    per-segment parts). Then mux audio, validate against the presentation
    box/fps, atomically publish, and append a `finalize_completed` event
    (effective crf/preset + `parts_encode_ms`/`audio_blend_ms`/
    `final_encode_ms`) to `logs/metrics.jsonl` for soak trending. With
    skip_bad, corrupt segments are skipped with a warning instead of
    aborting the whole finalize. A positive `min_free_space_gib` runs
    the §53 preflight first so a full disk fails fast instead of
    mid-encode.

    Audio joints get a proportional overlap crossfade (re-sliced from
    the takes ledger — previews untouched); pass overlap_fraction=0 to
    keep the legacy hard splice. The audio timeline stays on the source
    fps (frame counts / requested fps = seconds) — the fps lift touches
    video only, never the mix.

    Knob contract (issue 083): `options=` is canonical; the scalars are a
    tested shim resolved by `resolve_finalize_settings` (scalar wins over
    `options`, `None` means use `options`). The `width`/`height`/`fps`
    defaults (768/432/24) are the legacy fake-native fallback — kept (not
    raised to the presentation floors) because zero-floor callers rely on
    them for the stream-copy fast path (see `test_finalize_fastpath`);
    default-floor callers are lifted to 1280x720@32 by `plan_augmentation`
    either way, so either default ships the same presentation.

    Model pass (issue 166): `use_model_pass=True` consults
    `resolve_augment_weights(models_dir)` — absent legs (or no `models_dir`)
    read as ffmpeg fallback, never an error, so knob-off == knob-on-absent
    byte-for-byte. Present legs still encode via the ffmpeg vf path below:
    the tensor pipeline lands on a GPU box (see the issue residual).
    """
    resolved = resolve_finalize_settings(
        options=options,
        skip_bad=skip_bad,
        sample_rate=sample_rate,
        channels=channels,
        overlap_fraction=overlap_fraction,
        overlap_cap_seconds=overlap_cap_seconds,
        min_fps=min_fps,
        min_width=min_width,
        min_height=min_height,
        crf=crf,
        preset=preset,
        use_model_pass=use_model_pass,
    )
    settings = resolved.settings
    effective_min_fps = resolved.min_fps
    effective_min_width = resolved.min_width
    effective_min_height = resolved.min_height
    effective_crf = resolved.crf
    effective_preset = resolved.preset
    effective_use_model_pass = resolved.use_model_pass
    if effective_use_model_pass and models_dir is not None:
        from voyage.augment import resolve_augment_weights

        # Threading proof: consult the seam so absent legs fall back below;
        # the discarded result keeps the ffmpeg output in every CPU-provable
        # case (the tensor encode is the GPU-box residual).
        resolve_augment_weights(models_dir)
    if min_free_space_gib > 0:
        check_free_space(run_dir, min_free_space_gib)
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    segment_dirs = (
        sorted(p for p in segments_root.iterdir() if p.is_dir()) if segments_root.exists() else []
    )
    committed = [d for d in segment_dirs if (d / paths.DONE_MARKER).exists()]
    if not committed:
        raise MediaError(f"no committed segments in {run_dir}")

    # §56 steps 4-6 per segment, before any encoding work. skip_bad is
    # input triage (issues 138/188): missing artifacts fold into the same
    # skippable loop as checksum/metrics/alignment failures, and numbering
    # gaps warn instead of vanishing silently — while the post-assembly
    # validate_video below stays strict under both settings.
    if settings.skip_bad:
        for position, segment in enumerate(committed):
            if segment.name != f"{position:06d}":
                print(
                    "finalize: skipping segment numbering gap: "
                    f"expected {position:06d}, found {segment.name}"
                )
                break
    else:
        for position, segment in enumerate(committed):
            if segment.name != f"{position:06d}":
                raise MediaError(
                    f"segment numbering gap: expected {position:06d}, found {segment.name}"
                )
    usable: list[Path] = []
    for segment in committed:
        try:
            _check_segment_committed(segment)
        except MediaError as exc:
            if not settings.skip_bad:
                raise
            print(f"finalize: skipping {segment.name} ({exc})")
            continue
        usable.append(segment)
    if not usable:
        raise MediaError(f"no usable segments in {run_dir}")

    # Presentation box/fps via the pure augment plan (Track B): sources
    # below the floors (CausVid 16fps, sub-720p natives) are lifted with
    # motion interpolation + upscale; the audio timeline stays on the
    # requested (== source) fps — frame counts / source fps = seconds.
    source_info = probe(usable[0] / "video.mp4")
    source_fps = _probe_video_fps(source_info)
    source_w, source_h = _probe_video_geometry(source_info)
    plan = plan_augmentation(
        source_w,
        source_h,
        source_fps,
        width,
        height,
        fps,
        effective_min_fps,
        effective_min_width,
        effective_min_height,
    )
    out_w, out_h, out_fps = plan.out_w, plan.out_h, plan.out_fps
    lift = ""
    if plan.needs_minterpolate:
        lift = f"minterpolate=fps={out_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"

    with tempfile.TemporaryDirectory(prefix="voyage-final-", dir=run_dir) as tmp:
        tmpdir = Path(tmp)
        # Blended final mix (overlap re-sliced from takes; previews untouched).
        audio_start = time.monotonic()
        final_audio = build_final_audio(
            run_dir,
            usable,
            tmpdir,
            fps,
            settings.sample_rate,
            settings.channels,
            settings.effective_overlap_fraction(),
            settings.overlap_cap_seconds,
        )
        audio_blend_ms = (time.monotonic() - audio_start) * 1000.0
        # Single-pass shape (issues 050): no intermediate per-segment
        # libx264 parts — the concat demuxer feeds one vf encode over the
        # originals. `parts_encode_ms` stays 0.0 so soak trending keeps a
        # stable schema across the old double-encode and the new path.
        parts_encode_ms = 0.0
        staged = tmpdir / "final.mp4"
        # Issue 031 fast path: every committed video already matches the
        # presentation geometry/pix_fmt/fps, so concat the originals with a
        # stream copy and mux the final audio — zero video re-encodes. The
        # augment plan gates it off whenever an upscale or fps lift is
        # required (needs_reencode covers both, plus any fps mismatch).
        native = (
            lift == ""
            and not plan.needs_reencode
            and source_fps > 0
            and abs(out_fps - source_fps) <= FPS_MATCH_TOLERANCE
            and all(
                _segment_video_matches_target(segment, out_w, out_h, out_fps) for segment in usable
            )
        )
        if native:
            concat_list = write_concat_list(
                [segment / "video.mp4" for segment in usable], tmpdir / "concat.txt"
            )
            final_start = time.monotonic()
            proc = run_capture(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    "-i",
                    str(final_audio),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                    "-shortest",
                    str(staged),
                ]
            )
            final_encode_ms = (time.monotonic() - final_start) * 1000.0
            if proc.returncode != 0:
                raise MediaError(f"final concat copy failed: {proc.stderr[-2000:]}")
        else:
            # Single vf encode over the concat demuxer (issues 050): the
            # originals feed `scale/pad/fps/minterpolate` once — the old
            # per-segment `libx264/veryfast/-an` parts were a wasted first
            # pass with no vf. Concat entries go through the quoting-safe
            # helper (issues 053) so adversarial paths stay parseable.
            concat_list = write_concat_list(
                [segment / "video.mp4" for segment in usable], tmpdir / "concat.txt"
            )
            vf = (
                f"{lift}scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={out_fps}"
            )
            final_start = time.monotonic()
            proc = run_capture(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    "-i",
                    str(final_audio),
                    "-vf",
                    vf,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    effective_preset,
                    "-crf",
                    str(effective_crf),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                    "-shortest",
                    str(staged),
                ]
            )
            final_encode_ms = (time.monotonic() - final_start) * 1000.0
            if proc.returncode != 0:
                raise MediaError(f"final encode failed: {proc.stderr[-2000:]}")
        validate_video(staged, out_w, out_h, out_fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(staged, output_path)
        _record_final_geometry(
            run_dir,
            out_w,
            out_h,
            out_fps,
            effective_min_fps,
            effective_min_width,
            effective_min_height,
        )
        from voyage.logrotate import append_line

        append_line(
            run_dir / paths.LOGS_DIRNAME / "metrics.jsonl",
            json.dumps(
                {
                    "ts": time.time(),
                    "event": "finalize_completed",
                    "segments": len(usable),
                    "out_w": out_w,
                    "out_h": out_h,
                    "out_fps": out_fps,
                    "crf": effective_crf,
                    "preset": effective_preset,
                    "parts_encode_ms": round(parts_encode_ms, 1),
                    "audio_blend_ms": round(audio_blend_ms, 1),
                    "final_encode_ms": round(final_encode_ms, 1),
                    "fast_path": native,
                }
            ),
        )
    return output_path
