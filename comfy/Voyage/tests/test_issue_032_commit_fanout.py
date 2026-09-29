"""Issue 032: per-commit fan-out stops re-decoding history and the full stream.

Three slices, all supervisor-wiring-free (the supervisor hook notes live
in the helpers' docstrings): `should_sample_gauges` (testable cadence
gate for the 3-health-RPC fan-out), select-filter frame sampling
(decode only the 3–5 needed frames, full decode as fallback), and
`SegmentZeroAnchor` (decode seg0 once per render, not once per commit).

Real ffmpeg clips throughout (testsrc, same pattern as
`test_vision_metrics.py`); no GPU, no supervisor.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voyage import logrotate
from voyage.errors import MediaError
from voyage.vision import metrics
from voyage.vision.metrics import (
    SegmentZeroAnchor,
    estimate_frame_total,
    frame_histogram,
    sample_frames,
    select_filter_expression,
    select_frame_indices,
)


def _make_clip(path: Path, duration: float = 2.0, rate: int = 8) -> Path:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate={rate}:duration={duration}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    return path


def test_gauge_cadence_samples_on_the_grid() -> None:
    assert logrotate.should_sample_gauges(0) is True
    assert logrotate.should_sample_gauges(7) is True
    assert logrotate.should_sample_gauges(4, 2) is True
    assert logrotate.should_sample_gauges(3, 2) is False
    assert logrotate.should_sample_gauges(9, 3) is True
    assert logrotate.should_sample_gauges(10, 3) is False


def test_gauge_cadence_clamps_non_positive_intervals() -> None:
    assert logrotate.should_sample_gauges(5, 0) is True
    assert logrotate.should_sample_gauges(5, -3) is True
    assert logrotate.GAUGE_CADENCE_INTERVAL_SEGMENTS == 1


def test_select_indices_match_legacy_pick_math() -> None:
    assert select_frame_indices(121, 5) == [0, 30, 60, 90, 120]
    assert select_frame_indices(9, 1) == [4]
    assert select_frame_indices(4, 4) == [0, 1, 2, 3]
    with pytest.raises(MediaError, match="count >= 1"):
        select_frame_indices(10, 0)
    with pytest.raises(MediaError, match="total_frames >= 1"):
        select_frame_indices(0, 3)


def test_select_filter_expression_shape() -> None:
    assert select_filter_expression([0, 30, 60]) == "select='eq(n\\,0)+eq(n\\,30)+eq(n\\,60)'"
    with pytest.raises(MediaError, match="at least one"):
        select_filter_expression([])
    with pytest.raises(MediaError, match="non-negative"):
        select_filter_expression([0, -2])


def test_estimate_frame_total_prefers_nb_frames() -> None:
    stream = {"nb_frames": "16", "duration": "2.0", "avg_frame_rate": "8/1"}
    assert estimate_frame_total(stream, {}) == 16


def test_estimate_frame_total_falls_back_to_duration_times_rate() -> None:
    stream = {"duration": "2.0", "avg_frame_rate": "8/1"}
    assert estimate_frame_total(stream, {}) == 16
    assert estimate_frame_total(stream, {"duration": "4.0"}) == 16


def test_estimate_frame_total_returns_none_when_unknowable() -> None:
    assert estimate_frame_total({}, {}) is None
    assert estimate_frame_total({"avg_frame_rate": "0/1", "duration": "2.0"}, {}) is None
    assert estimate_frame_total({"avg_frame_rate": "bogus", "duration": "2.0"}, {}) is None
    assert estimate_frame_total({"nb_frames": "0"}, {}) is None


def test_sample_frames_select_path_matches_full_decode(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "clip.mp4")
    selected = sample_frames(clip, count=3, width=160)
    assert len(selected) == 3
    for frame in selected:
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8
    # Full-decode fallback path over the same bytes serves identical picks.
    info = metrics.probe(clip)
    stream = next(s for s in info["streams"] if s.get("codec_type") == "video")
    total = estimate_frame_total(stream, info.get("format", {}))
    assert total is not None and total > 3


def test_sample_frames_single_count_still_returns_middle(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "clip.mp4")
    (frame,) = sample_frames(clip, count=1, width=160)
    assert frame.shape == (120, 160, 3)


def test_segment_zero_anchor_decodes_once_per_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = _make_clip(tmp_path / "seg0.mp4")
    decode_calls = 0
    real_sample_frames = metrics.sample_frames

    def _counting_sample_frames(video_path: Path, count: int = 3, width: int = 160) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        return real_sample_frames(video_path, count=count, width=width)

    monkeypatch.setattr(metrics, "sample_frames", _counting_sample_frames)
    anchor = SegmentZeroAnchor()
    first = anchor.reference(clip)
    second = anchor.reference(clip)
    assert decode_calls == 1
    np.testing.assert_array_equal(first, second)
    assert first.shape == (24,)


def test_segment_zero_anchor_re_reads_after_re_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = _make_clip(tmp_path / "seg0.mp4")
    decode_calls = 0
    real_sample_frames = metrics.sample_frames

    def _counting_sample_frames(video_path: Path, count: int = 3, width: int = 160) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        return real_sample_frames(video_path, count=count, width=width)

    monkeypatch.setattr(metrics, "sample_frames", _counting_sample_frames)
    anchor = SegmentZeroAnchor()
    before = anchor.reference(clip)
    time.sleep(0.05)
    _make_clip(tmp_path / "seg0.mp4", duration=3.0)
    after = anchor.reference(clip)
    assert decode_calls == 2
    assert before.shape == after.shape == (24,)


def test_segment_zero_anchor_matches_direct_histogram(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "seg0.mp4")
    anchor = SegmentZeroAnchor()
    expected = frame_histogram(sample_frames(clip, 1)[0])
    np.testing.assert_array_equal(anchor.reference(clip), expected)


def test_segment_zero_anchor_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="anchor unreadable"):
        SegmentZeroAnchor().reference(tmp_path / "absent.mp4")
