"""ffmpeg probe/slice/blend/join audio group (issue 036 split, DESIGN §§35, 53-56).

Move-verbatim from `voyage.media` (A/V alignment + free-space preflight +
ffmpeg wrappers + take slicing + segment assembly + verification + the
issue-152 single-graph join + final mix): ffmpeg-bound, GPU-free, zero
outward cross-refs as a block. `media` re-exports every name via
explicit-`as` facade so existing importers hold.

Six test modules patch the `run_capture`/`probe`/`slice_take`/
`_cached_slice_take` seams live on THIS module (re-homed from
`voyage.media` in the same pass - `media` carries only facades, so
patches must target the defining module).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from voyage import paths
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

#: |stretch − 1| at or below which the final mix skips retiming, ratio.
#: Stretch factors arrive as float math (source_fps * multiplier / out_fps),
#: so an exact `== 1.0` check would atempo a 1.0000000001x mix for zero
#: benefit. Mirrors `media.SLOWMO_STRETCH_TOLERANCE` (defined here because
#: `media` imports this module — a back-import would cycle).
STRETCH_IDENTITY_TOLERANCE = 1e-9

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
    argv: list[str], timeout: float | None = FFMPEG_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run argv capturing output; a wedged child maps to MediaError (issue 019).

    `timeout` mirrors `[voyage] rpc_timeout_seconds` / RPC 600 s default —
    callers may thread the configured value through; the default keeps
    existing call sites bounded without a config round-trip. `None` means
    unbounded (`subprocess.run` waits forever) — reserved for ops whose
    wall time scales with the total timeline (final publish encodes,
    full-timeline mixes/convert/demux/remux/concats), the same way the
    already-unbounded augment chunk path does; probes, slices, windows,
    and per-segment assembly keep the 600 s bound.
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


_AUDIO_DURATION_CACHE: dict[str, tuple[int, int, float]] = {}
"""Probed audio durations keyed by path -> (mtime_ns, size, seconds).

Issue 250: finalize re-probed the same takes/windows/stems/beds on
every pass, retry, and stage (music, SFX hit + render paths, mix
gates). The (mtime_ns, size) identity keeps the cache honest across
re-renders — a replaced file stats differently and re-probes.
Positive results only; failures re-probe (a transient ffprobe error
must never pin a file as broken). Per-process only.
"""


def _duration_identity(path: Path) -> tuple[int, int] | None:
    """(mtime_ns, size) for a path, None when it cannot be stated."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _cached_duration(path: Path) -> float | None:
    """Cached probed duration for an unchanged file, else None."""
    identity = _duration_identity(path)
    if identity is None:
        return None
    cached = _AUDIO_DURATION_CACHE.get(str(path))
    if cached is not None and (cached[0], cached[1]) == identity:
        return cached[2]
    return None


def _store_duration(path: Path, seconds: float) -> None:
    """Cache a positive probed duration under the file's identity."""
    identity = _duration_identity(path)
    if identity is not None:
        _AUDIO_DURATION_CACHE[str(path)] = (identity[0], identity[1], seconds)


def probed_take_seconds(take_file: Path) -> float:
    """Measured duration of a rendered take file (issue 094).

    ACE-Step renders are not sample-exact vs the requested duration; the
    ledger must be clamped to the file, not the request. Raises MediaError
    when the file probes empty so a missing render fails loud instead of
    covering zero seconds silently. Shares the issue-250 duration cache
    (positive results only) with `_audio_duration_seconds`.
    """
    cached = _cached_duration(take_file)
    if cached is not None:
        return cached
    try:
        actual = float(probe(take_file).get("format", {}).get("duration", 0.0) or 0.0)
    except (OSError, ValueError) as exc:
        raise MediaError(f"take file {take_file} is unprobable: {exc}") from exc
    if actual <= 0.0:
        raise MediaError(f"take file {take_file} probed empty")
    _store_duration(take_file, actual)
    return actual


def assemble_segment_audio(
    slices: list[Path],
    dest: Path,
    crossfade_seconds: float,
    joint_fade: float | None = None,
    *,
    blend_timings: list[float] | None = None,
    staging_parent: Path | None = None,
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

    The two-stage join's intermediate lands under `staging_parent`
    (`build_final_audio` passes its run-scoped tmpdir — never bare /tmp,
    boba /tmp-quota incident); None keeps the TMPDIR default.
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
    with tempfile.TemporaryDirectory(prefix="voyage-assemble-", dir=staging_parent) as staging:
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


def _verify_segment(segment: Path) -> tuple[int, float]:
    """DESIGN §56 steps 4-6, video-only: checksum, frame range, duration.

    Every backend commits video only (always-deferred finalize — no
    `audio.wav` previews), so verification covers `video.mp4` alone:
    manifest checksum, non-positive frame count, non-positive probed
    duration. `audio.wav` is neither required nor verified, and there is
    no A/V gate — the takes ledger owns the audio timeline at finalize.
    Returns (frames, video_duration). Raises MediaError on any mismatch
    — fail loud, never finalize corrupt media silently.
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
    recorded = expected.get("video.mp4")
    if not isinstance(recorded, str) or not recorded:
        raise MediaError(f"segment {name} manifest missing video.mp4")
    try:
        actual = _sha256_file(segment / "video.mp4")
    except OSError as exc:
        raise MediaError(f"segment {name} missing video.mp4: {exc}") from exc
    if actual != recorded:
        raise MediaError(f"segment {name} checksum mismatch for video.mp4")
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
    if video_duration <= 0:
        raise MediaError(f"segment {name} has non-positive media duration")
    return frames, video_duration


def _check_segment_committed(segment: Path) -> None:
    """Existence probe + full verification for one committed segment.

    Split out so the `skip_bad` triage loop below never raises inside its
    own `try` (the loop catches `MediaError` to skip — a raise in the
    `try` body would read as self-caught).
    """
    if not (segment / "video.mp4").exists():
        raise MediaError(f"segment {segment.name} missing video.mp4")
    _verify_segment(segment)


def _segment_timeline(usable: list[Path], fps: float) -> tuple[list[float], list[float], float]:
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
    take-relative window at millisecond precision, so adjacent segment
    windows re-slicing identical bytes hit instead of re-spawning ffmpeg
    on multi-GB takes. Millisecond (not microsecond): the `-ss`/`-t`
    grid carries ms precision and float dust below 1 ms must hit, not
    miss (LOW slice-cache key)."""
    return (str(take_path), f"{start_seconds:.3f}", f"{duration_seconds:.3f}")


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


def _audio_duration_seconds(path: Path) -> float:
    """Probed audio duration; fail loud on unreadable/empty files.

    Shares the issue-250 cache with `probed_take_seconds`: unchanged
    files probe once per process no matter how many stages ask.
    """
    cached = _cached_duration(path)
    if cached is not None:
        return cached
    duration = float(probe(path).get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0:
        raise MediaError(f"audio file has non-positive duration: {path}")
    _store_duration(path, duration)
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
        # Unbounded: the fold accumulator grows to the full timeline.
        proc = run_capture(argv, timeout=None)
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
        # Unbounded: one spawn over the full timeline.
        proc = run_capture(argv, timeout=None)
        if proc.returncode != 0:
            raise MediaError(f"single-graph audio join failed: {proc.stderr[-2000:]}")
        return dest
    finally:
        if timing_ms is not None and len(inputs) > PAIR_BLEND_INPUT_COUNT:
            timing_ms.append((time.monotonic() - start) * 1000.0)


def build_final_audio_with_metrics(
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
    stretch: float = 1.0,
) -> tuple[Path, float]:
    """Blend committed segments into one timeline-exact final mix (§56, H3).

    Same contract as `build_final_audio` but returns `(dest, mix_seconds)`
    where `mix_seconds` is the probed output duration for the caller's
    `music_mix_seconds` metric hook (Track A/C emits it — this module never
    touches supervisor/media). H3 mix gate: the probed mix must agree with
    the stretched timeline within `AV_ALIGNMENT_TOLERANCE_SECONDS` (0.6 s,
    mirroring the SFX `mix_music_and_sfx` gate); past it raises `MediaError`
    instead of shipping a short mix the dub would then fail on.
    """
    from voyage.audio.planner import AudioPlanner, load_takes

    if stretch <= 0:
        raise MediaError(f"final audio stretch must be positive (got {stretch})")
    dest = tmpdir / "final_audio.wav"

    def _fail(reason: str) -> MediaError:
        return MediaError(f"final audio has no rendered takes: {reason}")

    def _convert_window_to_dest(window: Path, label: str) -> Path:
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(window),
                "-c:a",
                "pcm_s16le",
                str(dest),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"final audio {label} failed: {proc.stderr[-2000:]}")
        return dest

    starts, ends, timeline = _segment_timeline(usable, fps / stretch)
    durations = [end - start for start, end in zip(starts, ends, strict=True)]
    overlap = min(overlap_fraction * min(durations), overlap_cap_seconds)
    ledger = run_dir / "audio" / "takes.jsonl"
    takes: list[Any] = load_takes(ledger) if ledger.exists() else []
    planner = AudioPlanner(takes=takes)
    if not takes:
        raise _fail("no takes ledger")
    if overlap < MIN_OVERLAP_BLEND_SECONDS:
        raise _fail("tiny overlap")
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
                raise _fail("too many slices")
            serving = planner.take_for_time(cursor, int(segment.name))
            if serving is None or not serving.path:
                raise _fail("take gap in window")
            # Issue 016 consumer side: ledger entries may be run-relative
            # or legacy absolute — resolve the same way at every use site.
            serving_path = paths.resolve_stored_path(run_dir, serving.path)
            if not serving_path.exists():
                raise _fail(f"missing take file {serving.path}")
            piece_end = min(serving.covers_until(), window_end)
            if piece_end <= cursor:
                raise _fail("degenerate take window")
            if piece_end - cursor < MIN_SLICE_PIECE_SECONDS:
                raise _fail("sliver take piece")
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
            assemble_segment_audio(
                slices, window_path, overlap, joint_fade=fade, staging_parent=tmpdir
            )
        windows.append(window_path)
    if len(windows) == 1:
        _convert_window_to_dest(windows[0], "single-window copy")
    else:
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
            ],
            # Unbounded: the joined mix spans the full timeline.
            timeout=None,
        )
        if proc.returncode != 0:
            raise MediaError(f"final audio blend failed: {proc.stderr[-2000:]}")
    mix_seconds = _audio_duration_seconds(dest)
    if abs(mix_seconds - timeline) > AV_ALIGNMENT_TOLERANCE_SECONDS:
        raise MediaError(f"final music mix {mix_seconds:.2f}s drifts from timeline {timeline:.2f}s")
    return (dest, mix_seconds)


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
    stretch: float = 1.0,
) -> Path:
    """Blend committed segments into one timeline-exact final mix (§56).

    Ledger-only (always-deferred finalize): every segment boundary gets
    an overlap crossfade — segment windows are extended by half the
    overlap on each side and re-sliced from the takes ledger (takes are
    continuous, so the extension is real musical content — not
    time-stretched), then blended pairwise with manual fades (afade
    out/in + adelay + amix). Total length stays exactly the video
    timeline, so no A/V drift. Segments commit video only, so there are
    no per-segment `audio.wav` previews to fall back to: a missing takes
    ledger, a take gap, or any degenerate window raises `MediaError`
    instead of shipping silence — takes must have rendered via
    `ensure_deferred_takes` first. `stretch` (>1 for slow motion)
    divides the timeline fps so the mix covers the stretched video.

    H3 mix gate rides along (via `build_final_audio_with_metrics`): the
    probed mix must agree with the timeline within 0.6 s or `MediaError`.
    Callers needing the `music_mix_seconds` metric hook use
    `build_final_audio_with_metrics` directly (Track A/C wiring) — this
    wrapper keeps the historical `Path` return for existing callers.
    """
    dest, _mix_seconds = build_final_audio_with_metrics(
        run_dir,
        usable,
        tmpdir,
        fps,
        sample_rate,
        channels,
        overlap_fraction,
        overlap_cap_seconds,
        blend_timings=blend_timings,
        stretch=stretch,
    )
    return dest
