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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from voyage import paths
from voyage.atomic import atomic_write_bytes
from voyage.errors import DiskSpaceError, MediaError
from voyage.hashing import sha256_file

#: Max |video duration − audio duration| per segment, seconds (DESIGN §56
#: step 6). Same budget the commit path enforces, so anything committed
#: stays finalizable.
AV_ALIGNMENT_TOLERANCE_SECONDS = 0.6


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


def run_capture(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


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
    data: Any = json.loads(proc.stdout or "{}")
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
    actual_fps = float(num) / float(den or 1) if num else 0.0
    if abs(actual_fps - fps) > 0.5:
        raise MediaError(f"fps mismatch in {path}: {actual_fps} != {fps}")
    frames = int(video.get("nb_frames", 0) or 0)
    if frames < min_frames and frames != 0:
        raise MediaError(f"too few frames in {path}: {frames}")
    duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0:
        raise MediaError(f"non-positive duration in {path}")
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
    """
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
        return dest
    durations = [float(probe(s).get("format", {}).get("duration", 0.0) or 0.0) for s in slices]
    if any(d <= 0 for d in durations):
        raise MediaError("slice with non-positive duration")
    fade = (
        joint_fade
        if joint_fade is not None
        else _take_joint_fade(min(durations), crossfade_seconds)
    )
    if fade < 0.1:
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
    # Left-fold pairwise blends through _blend_pair: the old inline
    # N-input acrossfade chain shared both acrossfade failure modes
    # (scheduler deadlock at scale, long-first collapse at take joints).
    with tempfile.TemporaryDirectory(prefix="voyage-assemble-") as staging:
        accum = slices[0]
        for index, following in enumerate(slices[1:]):
            step = Path(staging) / f"blend_{index:02d}.wav"
            _blend_pair(accum, following, step, fade)
            accum = step
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
    checksums_path = segment / "sha256.json"
    if not checksums_path.exists():
        raise MediaError(f"segment {name} missing sha256.json")
    try:
        expected = json.loads(checksums_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise MediaError(f"segment {name} has unreadable sha256.json: {exc}") from exc
    if not isinstance(expected, dict):
        raise MediaError(f"segment {name} has malformed sha256.json")
    for artifact in ("video.mp4", "audio.wav"):
        recorded = expected.get(artifact)
        if not isinstance(recorded, str) or not recorded:
            raise MediaError(f"segment {name} sha256.json missing {artifact}")
        actual = _sha256_file(segment / artifact)
        if actual != recorded:
            raise MediaError(f"segment {name} checksum mismatch for {artifact}")
    metrics_path = segment / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        frames = int(metrics.get("frames", 0))
    except (ValueError, KeyError, AttributeError) as exc:
        raise MediaError(f"segment {name} has unreadable metrics.json: {exc}") from exc
    if frames <= 0:
        raise MediaError(f"segment {name} has non-positive frame count {frames}")
    video_duration = float(probe(segment / "video.mp4").get("format", {}).get("duration", 0.0))
    audio_duration = float(probe(segment / "audio.wav").get("format", {}).get("duration", 0.0))
    if video_duration <= 0 or audio_duration <= 0:
        raise MediaError(f"segment {name} has non-positive media duration")
    check_av_alignment(video_duration, audio_duration, name)
    return frames, video_duration, audio_duration


def _segment_timeline(usable: list[Path], fps: int) -> tuple[list[float], list[float], float]:
    """Per-segment [start, end) video-times from committed metrics + total.

    The timeline follows frame counts (the supervisor's truthful
    accounting), not container durations, so the blended audio matches
    the concatenated video sample-exactly.
    """
    starts: list[float] = []
    ends: list[float] = []
    cursor = 0.0
    for segment in usable:
        try:
            metrics = json.loads((segment / "metrics.json").read_text(encoding="utf-8"))
            frames = int(metrics.get("frames", 0))
        except (ValueError, KeyError, AttributeError) as exc:
            raise MediaError(f"segment {segment.name} has unreadable metrics.json: {exc}") from exc
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


def _blend_pair(first: Path, second: Path, dest: Path, overlap: float) -> Path:
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
    """
    first_seconds = _audio_duration_seconds(first)
    second_seconds = _audio_duration_seconds(second)
    fade = min(float(overlap), first_seconds / 2.0, second_seconds / 2.0)
    if fade <= 0:
        raise MediaError(f"cannot blend with non-positive overlap for {first} + {second}")
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


def build_final_audio(
    run_dir: Path,
    usable: list[Path],
    tmpdir: Path,
    fps: int,
    sample_rate: int,
    channels: int,
    overlap_fraction: float = 0.10,
    overlap_cap_seconds: float = 0.5,
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
    if not takes or overlap < 0.05:
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
            if absorption >= 0.01:
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
    # Pairwise reduction through 2-input manual-fade graphs only (see
    # _blend_pair: a single N-input acrossfade chain deadlocks the ffmpeg
    # scheduler on long runs, and acrossfade collapses long-first pairs).
    accum = windows[0]
    for index in range(1, len(windows)):
        step = tmpdir / f"final_blend_{index:02d}.wav"
        _blend_pair(accum, windows[index], step, overlap)
        accum = step
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

JointStyle = Literal["blend", "hard-splice"]
"""Audio-joint rendering for the final mix (issues 045, 046).

`blend` re-slices the takes ledger with a proportional overlap crossfade
(the default); `hard-splice` concatenates the per-segment previews with
no blend (the legacy behavior, clicks included). An explicit style
replaces the old `overlap_fraction=0` encoding, so call sites state the
intent instead of smuggling it through a zero.
"""


@dataclass
class FinalizeOptions:
    """Explicit finalize knobs (issue 045).

    Twelve positional scalars made `finalize_run`'s semantics depend on
    hand-audited call sites; this dataclass groups the five semantic
    knobs (skip policy, audio shape, joint rendering) under names. New
    callers should pass `options=`; the legacy scalars stay as the
    default path and build an equivalent instance internally, so existing
    callers are untouched.
    """

    skip_bad: bool = False
    sample_rate: int = 48000
    channels: int = 2
    overlap_fraction: float = 0.10
    overlap_cap_seconds: float = 0.5
    joint_style: JointStyle = "blend"

    def __post_init__(self) -> None:
        if self.joint_style not in ("blend", "hard-splice"):
            raise ValueError(
                f"joint_style must be 'blend' or 'hard-splice' (got {self.joint_style!r})"
            )
        if self.overlap_fraction < 0:
            raise ValueError(f"overlap_fraction must be >= 0 (got {self.overlap_fraction})")
        if self.overlap_cap_seconds < 0:
            raise ValueError(f"overlap_cap_seconds must be >= 0 (got {self.overlap_cap_seconds})")

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
        return abs(_probe_video_fps(info) - fps) <= 0.5
    except (MediaError, ValueError, TypeError, KeyError):
        return False


def finalize_run(
    run_dir: Path,
    output_path: Path,
    width: int = 768,
    height: int = 432,
    fps: int = 24,
    skip_bad: bool = False,
    min_free_space_gib: float = 0.0,
    sample_rate: int = 48000,
    channels: int = 2,
    overlap_fraction: float = 0.10,
    overlap_cap_seconds: float = 0.5,
    options: FinalizeOptions | None = None,
) -> Path:
    """Concat committed segments → single normalized MP4 (DESIGN §56).

    Native-geometry runs (the default — generation WxH/fps already match)
    stream-copy the committed videos with zero video re-encodes (issue 031
    fast path); anything else takes the single scale/pad/fps re-encode.
    Then mux audio, validate, atomically publish. With skip_bad, corrupt
    segments are skipped with a warning instead of aborting the whole
    finalize. A positive `min_free_space_gib` runs the §53 preflight first
    so a full disk fails fast instead of mid-encode.

    Audio joints get a proportional overlap crossfade (re-sliced from
    the takes ledger — previews untouched); pass overlap_fraction=0 to
    keep the legacy hard splice.
    """
    settings = (
        options
        if options is not None
        else FinalizeOptions(
            skip_bad=skip_bad,
            sample_rate=sample_rate,
            channels=channels,
            overlap_fraction=overlap_fraction,
            overlap_cap_seconds=overlap_cap_seconds,
            joint_style="hard-splice" if overlap_fraction <= 0 else "blend",
        )
    )
    if min_free_space_gib > 0:
        check_free_space(run_dir, min_free_space_gib)
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    segment_dirs = (
        sorted(p for p in segments_root.iterdir() if p.is_dir()) if segments_root.exists() else []
    )
    committed = [d for d in segment_dirs if (d / paths.DONE_MARKER).exists()]
    if not committed:
        raise MediaError(f"no committed segments in {run_dir}")
    for segment in committed:
        if not (segment / "video.mp4").exists():
            raise MediaError(f"segment {segment.name} missing video.mp4")
        if not (segment / "audio.wav").exists():
            raise MediaError(f"segment {segment.name} missing audio.wav")

    # §56 steps 4-6 per segment, before any encoding work.
    if not settings.skip_bad:
        for position, segment in enumerate(committed):
            if segment.name != f"{position:06d}":
                raise MediaError(
                    f"segment numbering gap: expected {position:06d}, found {segment.name}"
                )
    usable: list[Path] = []
    for segment in committed:
        try:
            _verify_segment(segment)
        except MediaError as exc:
            if not settings.skip_bad:
                raise
            print(f"finalize: skipping {segment.name} ({exc})")
            continue
        usable.append(segment)
    if not usable:
        raise MediaError(f"no usable segments in {run_dir}")

    # Presentation frame rate: sources below the floor (CausVid 16fps) are
    # lifted with motion interpolation; the audio timeline stays on source
    # fps (frame counts / source fps = seconds either way).
    source_info = probe(usable[0] / "video.mp4")
    source_fps = _probe_video_fps(source_info)
    presentation_fps = max(fps, PRESENTATION_MIN_FPS)
    lift = ""
    if source_fps > 0 and presentation_fps > source_fps + 0.5:
        lift = (
            f"minterpolate=fps={presentation_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # Blended final mix (overlap re-sliced from takes; previews untouched).
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
        staged = tmpdir / "final.mp4"
        # Issue 031 fast path: every committed video already matches the
        # target geometry/pix_fmt/fps, so concat the originals with a stream
        # copy and mux the final audio — zero video re-encodes. The per-part
        # re-encode below is pure waste on native runs.
        native = (
            lift == ""
            and source_fps > 0
            and abs(presentation_fps - source_fps) <= 0.5
            and all(
                _segment_video_matches_target(segment, width, height, presentation_fps)
                for segment in usable
            )
        )
        if native:
            concat_list = tmpdir / "concat.txt"
            concat_list.write_text(
                "".join(f"file '{segment / 'video.mp4'}'\n" for segment in usable),
                encoding="utf-8",
            )
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
            if proc.returncode != 0:
                raise MediaError(f"final concat copy failed: {proc.stderr[-2000:]}")
            validate_video(staged, width, height, presentation_fps)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(output_path, staged.read_bytes())
            return output_path
        # Per-segment video-only parts, then concat the parts.
        parts: list[Path] = []
        for segment in usable:
            part = tmpdir / f"{segment.name}.mp4"
            proc = run_capture(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(segment / "video.mp4"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "veryfast",
                    "-an",
                    str(part),
                ]
            )
            if proc.returncode != 0:
                raise MediaError(f"segment mux failed for {segment.name}: {proc.stderr[-2000:]}")
            parts.append(part)
        concat_list = tmpdir / "concat.txt"
        concat_list.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
        vf = (
            f"{lift}scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={presentation_fps}"
        )
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
                "veryfast",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-shortest",
                str(staged),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"final encode failed: {proc.stderr[-2000:]}")
        validate_video(staged, width, height, presentation_fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(output_path, staged.read_bytes())
    return output_path
