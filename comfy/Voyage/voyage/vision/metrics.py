"""Visual inspector: deterministic metrics live here (DESIGN §§43-44, 100)."""

from __future__ import annotations

from pathlib import Path

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

Frame = NDArray[np.uint8]
Histogram = NDArray[np.float64]


def _to_gray(frame: Frame) -> NDArray[np.float64]:
    rgb = frame.astype(np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def sample_frames(video_path: Path, count: int = 3, width: int = 160) -> list[Frame]:
    """Decode a segment clip and return `count` evenly spaced RGB frames.

    `count == 1` returns the middle frame (the VLM inspect view). Frames
    are downscaled to `width` (aspect kept, even height) — regulation and
    drift signals, never pixels for show. Raises MediaError when ffmpeg
    yields no decodable frames.
    """
    import subprocess

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
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"scale={width}:{height}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise MediaError(f"frame sampling failed for {video_path}: {proc.stderr[-2000:]!r}")
    raw = proc.stdout if isinstance(proc.stdout, bytes) else b""
    stride = width * height * 3
    total, leftover = divmod(len(raw), stride)
    if total == 0 or leftover:
        raise MediaError(f"frame sampling yielded no whole frames for {video_path}")
    frames = [
        np.frombuffer(raw[i * stride : (i + 1) * stride], dtype=np.uint8).reshape(height, width, 3)
        for i in range(total)
    ]
    if count == 1:
        return [frames[total // 2].copy()]
    picks = [round(i * (total - 1) / (count - 1)) for i in range(count)]
    return [frames[pick].copy() for pick in picks]


def frame_histogram(frame: Frame, bins: int = HISTOGRAM_BINS) -> Histogram:
    """Compact RGB distribution descriptor: each channel sums to 1."""
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
