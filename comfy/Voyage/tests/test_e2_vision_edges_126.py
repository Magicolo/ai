"""Vision-metrics edge inputs: fail loud or degrade, never raw-crash (126).

Pure-logic tests (no ffmpeg): `estimate_frame_total` non-finite probes,
`select_frame_indices` over-requests, and `sample_frames` width guards.
The end-to-end short-clip case uses a real testsrc mp4 like the existing
vision tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from voyage.errors import MediaError
from voyage.vision.metrics import (
    estimate_frame_total,
    sample_frames,
    select_frame_indices,
)


def test_estimate_frame_total_inf_probe_returns_none() -> None:
    """126: `duration="inf"` (broken moov) takes the full-decode fallback."""
    assert estimate_frame_total({"avg_frame_rate": "24/1"}, {"duration": "inf"}) is None


def test_estimate_frame_total_nan_probe_returns_none() -> None:
    assert estimate_frame_total({"avg_frame_rate": "24/1"}, {"duration": "nan"}) is None


def test_estimate_frame_total_inf_rate_returns_none() -> None:
    assert estimate_frame_total({"avg_frame_rate": "inf/1", "duration": "2.0"}, {}) is None


def test_estimate_frame_total_healthy_estimate_unchanged() -> None:
    assert estimate_frame_total({"avg_frame_rate": "24/1"}, {"duration": "2.0"}) == 48
    assert estimate_frame_total({"nb_frames": "12"}, {}) == 12


def test_select_frame_indices_clamps_over_request() -> None:
    """126: `count > total` serves distinct frames, never `[0, 0, 0]`."""
    assert select_frame_indices(2, 5) == [0, 1]
    assert select_frame_indices(1, 3) == [0]


def test_select_frame_indices_normal_spacing_unchanged() -> None:
    assert select_frame_indices(10, 3) == [0, 4, 9]
    assert select_frame_indices(2, 2) == [0, 1]


def test_sample_frames_rejects_zero_width_without_ffmpeg(tmp_path: Path) -> None:
    """126: `width=0` raises MediaError before any probe/divmod."""
    with pytest.raises(MediaError, match="width"):
        sample_frames(tmp_path / "missing.mp4", 3, width=0)


def _render_testsrc_clip(destination: Path, frames: int = 5) -> Path:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x64:rate=24:duration={frames / 24.0}",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"testsrc render failed: {proc.stderr.strip()}"
    return destination


def test_sample_frames_short_clip_serves_distinct_frames(tmp_path: Path) -> None:
    """126: over-requesting frames on a short clip never duplicates."""
    clip = _render_testsrc_clip(tmp_path / "tiny.mp4", frames=5)
    frames = sample_frames(clip, 8)
    assert len(frames) == 5
    assert all(frame.shape == frames[0].shape for frame in frames)
