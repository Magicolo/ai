"""Rhythm grid tests: adaptive beat math, take quantization, BPM ledger.

Pure logic over voyage.audio.beat + voyage.audio.planner — no workers,
no GPU, no ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.audio.beat import (
    MIN_BPM,
    beats_for_segment,
    quantize_take_seconds,
    segment_progress_info,
)
from voyage.audio.planner import AudioPlanner, AudioTake


@pytest.mark.parametrize(
    ("segment_seconds", "expected_beats", "expected_bpm"),
    [
        (4.0, 4, 60.0),  # steady-state LTXV 96f @24fps: exactly 60 BPM
        (2.0, 4, 120.0),  # fake 48f @24fps: 120 BPM
        (1.0, 4, 240.0),  # short segment, fast but exact
        (0.5, 4, 480.0),  # degenerate short: math holds, music may not
        (5.04, 8, pytest.approx(8 * 60.0 / 5.04)),  # cold-start 121f: doubles to 8
        (8.0, 8, 60.0),  # 4 beats would be 30 BPM: doubles to 8
        (16.0, 16, 60.0),  # doubles twice
    ],
)
def test_beats_for_segment_adaptive_k(
    segment_seconds: float, expected_beats: int, expected_bpm: float
) -> None:
    beats, bpm = beats_for_segment(segment_seconds, 4)
    assert beats == expected_beats
    assert bpm == pytest.approx(expected_bpm)
    assert bpm >= MIN_BPM


def test_beats_for_segment_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        beats_for_segment(0.0, 4)
    with pytest.raises(ValueError):
        beats_for_segment(-2.0, 4)
    with pytest.raises(ValueError):
        beats_for_segment(2.0, 0)


def test_beats_for_segment_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        beats_for_segment(float("nan"), 4)
    with pytest.raises(ValueError):
        beats_for_segment(float("inf"), 4)
    with pytest.raises(ValueError):
        beats_for_segment(2.0, 4, float("nan"))
    with pytest.raises(ValueError):
        beats_for_segment(2.0, 4, float("inf"))


@pytest.mark.parametrize(
    ("take_seconds", "segment_seconds", "expected"),
    [
        (45.0, 2.0, 44.0),  # 22.5 segments rounds to 22
        (45.0, 4.0, 44.0),  # 11.25 segments rounds to 11
        (45.0, 5.04, 45.36),  # 8.93 rounds to 9 cold-start segments
        (1.0, 4.0, 4.0),  # min one segment
        (4.0, 2.0, 4.0),  # already aligned
    ],
)
def test_quantize_take_seconds_snaps_to_grid(
    take_seconds: float, segment_seconds: float, expected: float
) -> None:
    assert quantize_take_seconds(take_seconds, segment_seconds) == pytest.approx(expected)


def test_quantize_take_seconds_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        quantize_take_seconds(0.0, 2.0)
    with pytest.raises(ValueError):
        quantize_take_seconds(45.0, 0.0)


def test_quantize_take_seconds_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        quantize_take_seconds(float("nan"), 4.0)
    with pytest.raises(ValueError):
        quantize_take_seconds(float("inf"), 4.0)
    with pytest.raises(ValueError):
        quantize_take_seconds(45.0, float("nan"))
    with pytest.raises(ValueError):
        quantize_take_seconds(45.0, float("inf"))


def test_planner_quantizes_fresh_takes_to_segment_grid() -> None:
    planner = AudioPlanner(take_seconds=45.0, ahead_seconds=20.0, segment_seconds=2.0)
    plan = planner.plan(0.0, "ambient", 11, 0)
    assert plan.action == "render"
    assert plan.take is not None
    assert plan.take.duration == pytest.approx(44.0)


def test_planner_without_segment_seconds_keeps_legacy_length() -> None:
    planner = AudioPlanner(take_seconds=45.0, ahead_seconds=20.0)
    plan = planner.plan(0.0, "ambient", 11, 0)
    assert plan.action == "render"
    assert plan.take is not None
    assert plan.take.duration == pytest.approx(45.0)


def test_take_bpm_ledger_roundtrip(tmp_path: Path) -> None:
    take_file = tmp_path / "take_0000.wav"
    take_file.write_bytes(b"fake-take")
    take = AudioTake(
        take_id="take_0000",
        path=str(take_file),
        caption="ambient",
        seed=1,
        covers_from=0.0,
        duration=44.0,
        segment_index=0,
        bpm=120.0,
    )
    assert AudioTake.from_dict(take.to_dict()).bpm == pytest.approx(120.0)


def test_take_bpm_absent_on_legacy_lines(tmp_path: Path) -> None:
    take_file = tmp_path / "take_0000.wav"
    take_file.write_bytes(b"fake-take")
    take = AudioTake(
        take_id="take_0000",
        path=str(take_file),
        caption="ambient",
        seed=1,
        covers_from=0.0,
        duration=45.0,
        segment_index=0,
    )
    raw = take.to_dict()
    assert "bpm" not in raw
    assert AudioTake.from_dict(raw).bpm is None


def test_segment_progress_info_reports_beats_and_bpm() -> None:
    """Progress-only display numbers (issue 046): same grid as the mixer."""
    info = segment_progress_info(4.0, 4)
    assert info == {"beats": 4.0, "grid_bpm": pytest.approx(60.0)}
    doubled = segment_progress_info(5.04, 4)
    assert doubled["beats"] == 8.0
    assert doubled["grid_bpm"] == pytest.approx(8 * 60.0 / 5.04)


def test_segment_progress_info_rejects_bad_durations() -> None:
    with pytest.raises(ValueError):
        segment_progress_info(0.0, 4)
    with pytest.raises(ValueError):
        segment_progress_info(float("nan"), 4)
