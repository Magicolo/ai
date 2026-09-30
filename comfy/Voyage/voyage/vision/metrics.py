"""Visual inspector: deterministic metrics live here (DESIGN §§43-44, 100)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from voyage.errors import MediaError
from voyage.media import probe

HISTOGRAM_BINS = 8
"""Bins per channel for the compact RGB distribution descriptor."""

MOTION_THRESHOLD = 0.05
"""A pixel counts as changed when its gray level moves by at least this."""

EDGE_THRESHOLD = 0.15
"""A pixel counts as edge when its Sobel magnitude exceeds this."""

Frame: TypeAlias = NDArray[np.uint8]
Histogram: TypeAlias = NDArray[np.float64]


def select_frame_indices(total_frames: int, count: int) -> list[int]:
    """Evenly spaced frame indices over `total_frames` (pure helper).

    `count == 1` picks the middle frame (the VLM inspect view) —
    the same choice `sample_frames` documents.
    """
    if total_frames < 1:
        raise MediaError(f"frame selection needs total_frames >= 1 (got {total_frames})")
    if count < 1:
        raise MediaError(f"frame selection needs count >= 1 (got {count})")
    if count == 1:
        return [total_frames // 2]
    return [round(index * (total_frames - 1) / (count - 1)) for index in range(count)]


def select_filter_expression(indices: list[int]) -> str:
    """ffmpeg `select` filter decoding only `indices` (issue 032).

    `n` counts input frames from 0, so `eq(n\\,X)` keeps exactly input
    frame X; callers pair this with `-vsync 0` so no frames are
    duplicated to fill a constant rate.
    """
    if not indices:
        raise MediaError("select filter needs at least one frame index")
    if any(index < 0 for index in indices):
        raise MediaError(f"select filter needs non-negative indices (got {indices})")
    terms = "+".join(f"eq(n\\,{index})" for index in indices)
    return f"select='{terms}'"


def estimate_frame_total(
    video_stream: dict[str, Any], container_format: dict[str, Any]
) -> int | None:
    """Best-effort decoded-frame count from probe data (issue 032).

    Prefers the stream's `nb_frames`; falls back to
    duration × `avg_frame_rate`. Returns None when neither yields a
    positive count — the caller then takes the full-decode path.
    """
    raw_count = video_stream.get("nb_frames")
    try:
        direct = int(str(raw_count))
    except (TypeError, ValueError):
        direct = 0
    if direct > 0:
        return direct
    raw_duration = video_stream.get("duration", container_format.get("duration"))
    raw_rate = video_stream.get("avg_frame_rate", "")
    try:
        duration = float(str(raw_duration))
        numerator, _, denominator = str(raw_rate).partition("/")
        rate = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    estimated = round(duration * rate)
    return estimated if estimated > 0 else None


def _to_gray(frame: Frame) -> NDArray[np.float64]:
    rgb = frame.astype(np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


#: Hard ceiling on frames the estimate-drift fallback decodes (issue 044).
#: Real segments are < 200 frames; only a pathological clip (or a wildly
#: wrong probe estimate) ever reaches this — instead of full-decoding a
#: multi-GB file to pick 3 frames, the fallback serves evenly spaced
#: picks from the capped prefix.
FALLBACK_MAX_FRAMES = 2048


def _read_frame_bytes(stdout: Any, stride: int) -> bytes | None:
    """Read exactly one frame (`stride` bytes); None on clean EOF.

    Raises MediaError on a truncated tail (some bytes, then EOF) — a
    corrupt stream must fail loud, never serve a short frame silently.
    """
    chunks: list[bytes] = []
    remaining = stride
    while remaining > 0:
        piece = stdout.read(remaining)
        if not piece:
            if chunks:
                raise MediaError(
                    f"frame sampling hit a truncated stream "
                    f"({stride - remaining} of {stride} bytes)"
                )
            return None
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def _decode_frames(
    video_path: Path,
    width: int,
    height: int,
    video_filter: str,
    *,
    frame_limit: int | None = None,
) -> list[Frame]:
    """Stream ffmpeg rawvideo frames with constant memory (issue 044).

    The old `subprocess.run(..., capture_output=True)` held the entire
    decoded byte stream in `proc.stdout` plus per-frame views plus the
    callers' `.copy()` fan-out (~2-3x transient RAM). This reads one
    frame at a time from `Popen.stdout` and returns owned arrays (one
    copy at decode, none after) — peak is the retained frames, never
    the whole stream. `frame_limit` caps emitted frames server-side
    (`-frames:v`) and stops the read early (the process is killed once
    the cap is reached), so a drifted select estimate cannot emit the
    whole clip. Kills are not errors: only EOF-before-limit checks the
    exit status.
    """
    import subprocess

    if frame_limit is not None and frame_limit < 1:
        raise MediaError(f"frame decode needs frame_limit >= 1 (got {frame_limit})")
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        video_filter,
        "-vsync",
        "0",
        "-pix_fmt",
        "rgb24",
    ]
    if frame_limit is not None:
        argv += ["-frames:v", str(frame_limit)]
    argv += ["-f", "rawvideo", "-"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None or proc.stderr is None:
        proc.kill()
        proc.wait()
        raise MediaError(f"frame sampling could not capture ffmpeg pipes for {video_path}")
    stride = width * height * 3
    frames: list[Frame] = []
    truncated = False
    try:
        while frame_limit is None or len(frames) < int(frame_limit):
            try:
                chunk = _read_frame_bytes(proc.stdout, stride)
            except MediaError:
                truncated = True
                raise
            if chunk is None:
                break
            frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3).copy())
    finally:
        if proc.poll() is None:
            proc.kill()
        _, stderr = proc.communicate()
    if truncated or not frames:
        raise MediaError(f"frame sampling yielded no whole frames for {video_path}")
    if proc.returncode != 0 and (frame_limit is None or len(frames) < int(frame_limit)):
        raise MediaError(f"frame sampling failed for {video_path}: {stderr.decode()[-2000:]!r}")
    return frames


def sample_frames(video_path: Path, count: int = 3, width: int = 160) -> list[Frame]:
    """Decode a segment clip and return `count` evenly spaced RGB frames.

    `count == 1` returns the middle frame (the VLM inspect view). Frames
    are downscaled to `width` (aspect kept, even height) — regulation and
    drift signals, never pixels for show. Raises MediaError when ffmpeg
    yields no decodable frames.

    Issue 032: decodes only the needed frames via a `select` filter
    when the probe yields a frame total (the common mp4 case) — the
    capped-prefix decode below stays as the fallback when the estimate
    is missing or drifts, so the contract never changes.

    Issue 044: the decode streams (`_decode_frames` reads one frame at
    a time and stops at the cap), each returned frame already owns its
    bytes, and the fallback decodes at most `FALLBACK_MAX_FRAMES`
    instead of the whole clip.
    """
    if count < 1:
        raise MediaError(f"sample_frames needs count >= 1 (got {count})")
    info = probe(video_path)
    streams = [s for s in info.get("streams", []) if isinstance(s, dict)]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream in {video_path}")
    source_width = int(video.get("width", 0) or 0)
    source_height = int(video.get("height", 0) or 0)
    if source_width <= 0 or source_height <= 0:
        raise MediaError(f"unreadable dimensions in {video_path}")
    height = max(2, (source_height * width // source_width) // 2 * 2)
    container = info.get("format", {})
    container_format: dict[str, Any] = container if isinstance(container, dict) else {}
    estimated_total = estimate_frame_total(video, container_format)
    if estimated_total is not None:
        picks = select_frame_indices(estimated_total, count)
        selected = _decode_frames(
            video_path,
            width,
            height,
            f"{select_filter_expression(picks)},scale={width}:{height}",
            frame_limit=len(picks),
        )
        if len(selected) == len(picks):
            return selected
        # Estimate drifted (VFR / wrong nb_frames): fall through to the
        # capped-prefix decode.
    frames = _decode_frames(
        video_path, width, height, f"scale={width}:{height}", frame_limit=FALLBACK_MAX_FRAMES
    )
    total = len(frames)
    if count == 1:
        return [frames[total // 2]]
    picks = select_frame_indices(total, count)
    return [frames[pick] for pick in picks]


class SegmentZeroAnchor:
    """Caches the segment-0 palette histogram across commits (issue 032).

    The supervisor re-decoded all of seg0 on every segment when the
    inspector is on; seg0 never changes after its commit, so one
    decode serves the whole run. Construct once per run and call
    `reference(seg0_video)` per commit: the cached histogram serves
    while the path + mtime match, otherwise it re-reads (a
    re-rendered seg0 heals instead of serving a stale anchor).

    Supervisor hook (not wired here — `supervisor.py` is out of scope
    for this change): keep one `SegmentZeroAnchor` on the supervisor
    across `run_segments` and replace the per-commit
    `frame_histogram(sample_frames(seg0_video, 1)[0])` in
    `_run_previous_inspect` with `anchor.reference(seg0_video)`.
    """

    def __init__(self) -> None:
        self._source: Path | None = None
        self._source_mtime: float = 0.0
        self._reference: Histogram | None = None

    def reference(self, segment_zero_video: Path) -> Histogram:
        """Segment-0 palette histogram, decoded at most once per render."""
        try:
            modified = segment_zero_video.stat().st_mtime
        except OSError as exc:
            raise MediaError(f"segment-0 anchor unreadable: {segment_zero_video}") from exc
        if (
            self._reference is not None
            and self._source == segment_zero_video
            and self._source_mtime == modified
        ):
            return self._reference.copy()
        frames = sample_frames(segment_zero_video, 1)
        reference = frame_histogram(frames[0])
        self._source = segment_zero_video
        self._source_mtime = modified
        self._reference = reference
        return reference.copy()


def frame_histogram(frame: Frame, bins: int = HISTOGRAM_BINS) -> Histogram:
    """Compact RGB distribution descriptor: each channel sums to 1."""
    # Issue 067: an empty frame divides 0/0 into a NaN array (warning
    # only) that then poisons drift/style decisions — fail loudly instead.
    if frame.size == 0:
        raise MediaError(f"frame_histogram needs a non-empty frame (got shape {frame.shape})")
    parts = [
        np.histogram(frame[:, :, channel], bins=bins, range=(0, 256))[0].astype(np.float64)
        for channel in range(3)
    ]
    hist = np.concatenate(parts)
    return hist / hist.reshape(3, bins).sum(axis=1, keepdims=True).repeat(bins, axis=0).reshape(-1)


def histogram_intersection(first: Histogram, second: Histogram) -> float:
    """Distribution overlap 0..1 (1 = identical palettes)."""
    return float(np.minimum(first, second).sum() / 3.0)


def histogram_distance(first: Histogram, second: Histogram) -> float:
    """L1 distribution distance 0..1 (0 = identical palettes)."""
    return float(np.abs(first - second).sum() / 6.0)


def motion_energy(frames: list[Frame]) -> float:
    """Share of pixels that visibly change between consecutive frames."""
    if len(frames) < 2:
        return 0.0
    gray = [_to_gray(frame) for frame in frames]
    changed = [
        float((np.abs(later - earlier) > MOTION_THRESHOLD).mean())
        for earlier, later in zip(gray, gray[1:], strict=False)
    ]
    return float(sum(changed) / len(changed))


def visual_complexity(frames: list[Frame]) -> float:
    """Share of edge pixels (Sobel magnitude over threshold), averaged."""
    # Issue 067: empty input returns 0.0 like the motion/palette/boundary
    # siblings — no ZeroDivisionError, uniform contract for the supervisor.
    if not frames:
        return 0.0
    scored = []
    for frame in frames:
        gray = _to_gray(frame)
        grad_y, grad_x = np.gradient(gray)
        magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        scored.append(float((magnitude > EDGE_THRESHOLD).mean()))
    return float(sum(scored) / len(scored))


def semantic_change_rate(frames: list[Frame]) -> float:
    """Palette drift inside the segment: 1 minus first/last overlap."""
    if len(frames) < 2:
        return 0.0
    return 1.0 - histogram_intersection(frame_histogram(frames[0]), frame_histogram(frames[-1]))


def palette_distance(frames: list[Frame]) -> float:
    """Mean-color walk inside the segment (tint shift, not distribution)."""
    if len(frames) < 2:
        return 0.0
    means = [frame.astype(np.float64).mean(axis=(0, 1)) / 255.0 for frame in frames]
    walk = [
        float(np.linalg.norm(later - earlier) / np.sqrt(3.0))
        for earlier, later in zip(means, means[1:], strict=False)
    ]
    return float(sum(walk) / len(walk))


def style_similarity(frames: list[Frame], reference: Histogram | None) -> float:
    """Segment-mid palette overlap with the segment-0 anchor (§100).

    The drift-vs-segment-0 proxy the Q&A chose: 1.0 means the current
    look still matches the voyage's opening frame. No anchor yet
    (segment 0 itself) → identity.
    """
    if reference is None:
        return 1.0
    # Issue 067: no frames means no overlap (not an IndexError) — keeps
    # `summarize_segment([])` total instead of raising halfway through.
    if not frames:
        return 0.0
    mid = frames[len(frames) // 2]
    return histogram_intersection(frame_histogram(mid), reference)


def scene_boundary_strength(frames: list[Frame]) -> float:
    """Strongest single pair cut inside the segment (cut detector)."""
    if len(frames) < 2:
        return 0.0
    hists = [frame_histogram(frame) for frame in frames]
    drops = [
        1.0 - histogram_intersection(earlier, later)
        for earlier, later in zip(hists, hists[1:], strict=False)
    ]
    return max(drops)


def summarize_segment(frames: list[Frame], reference: Histogram | None) -> dict[str, float]:
    """The six §43 metrics in spec order, every value normalized 0..1."""
    return {
        "motion_energy": motion_energy(frames),
        "visual_complexity": visual_complexity(frames),
        "semantic_change_rate": semantic_change_rate(frames),
        "palette_distance": palette_distance(frames),
        "style_similarity": style_similarity(frames, reference),
        "scene_boundary_strength": scene_boundary_strength(frames),
    }
