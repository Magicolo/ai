"""Rank-2 observability tests (issues 059/051/062/027/063/028).

Covers only the observability half owned by this track:
- bench.py gauge VRAM trending + timing percentiles (059, 051-benchmark-half)
- scoreboard.py fail-closed hardening (062, folds 027)
- console.py stream/timing/tracker contracts (063, folds 028)

CPU-only, fake data, no GPU/network. Failing-first per TDD.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from voyage import bench, scoreboard
from voyage.console import ParallelDownloadTracker, VoyageConsole


def _gauge_event(
    segment: str = "000000",
    rss: float = 100.0,
    disk: float = 10.0,
    video_free: Any = 12.0,
    video_total: Any = 15.0,
) -> dict[str, Any]:
    return {
        "event": "resource_gauges",
        "segment_id": segment,
        "rss_peak_mb": rss,
        "disk_free_gib": disk,
        "video_vram_free_gib": video_free,
        "video_vram_total_gib": video_total,
    }


def test_summarize_gauges_trends_vram_per_worker() -> None:
    """059: per-worker VRAM free series reaches the summary (min-free = OOM risk)."""
    events = [
        _gauge_event("000000", video_free=12.0),
        _gauge_event("000001", video_free=10.5),
        _gauge_event("000002", video_free=11.0),
    ]
    summary = bench.summarize_gauges(events)
    assert summary["video_vram_free_first_gib"] == pytest.approx(12.0)
    assert summary["video_vram_free_last_gib"] == pytest.approx(11.0)
    assert summary["video_vram_free_min_gib"] == pytest.approx(10.5)
    assert "video" in summary["vram_workers_reporting"]


def test_summarize_gauges_skips_unknown_strings() -> None:
    """059/051: string-typed holes never break aggregation (issue 071 precedent)."""
    events = [
        _gauge_event("000000", video_free="unknown", video_total="unknown"),
        _gauge_event("000001", video_free=10.0, video_total=15.0),
    ]
    summary = bench.summarize_gauges(events)
    # Only the numeric sample trends; the string sample is skipped, not fatal.
    assert summary["video_vram_free_first_gib"] == pytest.approx(10.0)
    assert summary["video_vram_free_last_gib"] == pytest.approx(10.0)
    # Core rss/disk contract from the existing gauge tests still holds.
    assert summary["segments"] == 2
    assert summary["rss_first_mb"] == pytest.approx(100.0)


def test_timing_stats_ex_reports_percentiles() -> None:
    """051-benchmark-half: extended timing stats carry p50/p95/std."""
    stats = bench.timing_stats_ex([1.0, 2.0, 3.0, 4.0])
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(4.0)
    assert stats["p50"] == pytest.approx(2.5)
    assert stats["p95"] >= stats["p50"]
    assert stats["std"] >= 0.0


def test_timing_stats_stays_backward_compatible() -> None:
    """051: legacy timing_stats shape is frozen (existing tests pin it)."""
    assert bench.timing_stats([2.0, 1.0, 3.0]) == {
        "count": 3,
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
    }


def _write_committed_segment(
    run_dir: Path,
    segment_id: str,
    metrics: dict[str, Any],
    *,
    with_media: bool = False,
) -> Path:
    from voyage.segment_manifest import write_segment_manifest

    segment = run_dir / "segments" / segment_id
    segment.mkdir(parents=True, exist_ok=True)
    (segment / "DONE").write_text("ok", encoding="utf-8")
    write_segment_manifest(
        segment,
        {
            "metrics": metrics,
            "transition": {
                "destination": {"canonical_name": "probe harbor"},
                "phase": "HOLD",
            },
            "prompt_plan": {},
            "audio_state": {"take_ids": ["take-1"]},
            "world_state": {},
            "checksums": {},
        },
    )
    if with_media:
        (segment / "video.mp4").write_bytes(b"\x00")
    return segment


def _visual_metrics(value: float = 1.0) -> dict[str, Any]:
    return {
        "video": {"frames": 48},
        "visual": {
            "metrics": dict.fromkeys(scoreboard.METRIC_KEYS, value),
        },
    }


def test_stages_by_segment_skips_bad_values(tmp_path: Path) -> None:
    """062/027: one bad stage value degrades the cell, never the table."""
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "event": "segment_committed",
                "segment_id": "000001",
                "stages": {"video": "unknown", "audio": 3.0, "validate": None},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stages = scoreboard._stages_by_segment(run_dir)
    assert stages["000001"] == {"audio": pytest.approx(3.0)}


def test_scoreboard_rows_survives_bad_metric(tmp_path: Path) -> None:
    """062/027: a hand-edited metric string yields an error cell, not zero rows."""
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "metrics.jsonl").write_text("", encoding="utf-8")
    _write_committed_segment(
        run_dir,
        "000001",
        {"video": {"frames": 48}, "visual": {"metrics": {"motion_energy": "NaN-string"}}},
    )
    rows = scoreboard.scoreboard_rows(run_dir)
    assert len(rows) == 1
    assert rows[0]["segment_id"] == "000001"
    # Bad cell is dropped; the row still renders with an errors note.
    assert rows[0]["errors"]


def test_scoreboard_rows_marks_missing_paths(tmp_path: Path) -> None:
    """062/027: synthesized view paths carry existence flags (no dead links).

    Video-only: scoreboard dropped the audio_path/audio_exists columns
    (no per-segment audio artifact); only video_exists is asserted.
    """
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "metrics.jsonl").write_text("", encoding="utf-8")
    _write_committed_segment(run_dir, "000001", _visual_metrics(), with_media=False)
    (row,) = scoreboard.scoreboard_rows(run_dir)
    assert row["video_exists"] is False
    assert "audio_exists" not in row
    assert "audio_path" not in row


def test_scoreboard_rows_marks_present_paths(tmp_path: Path) -> None:
    """Existence flags are True when the artifacts are on disk (video-only)."""
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "metrics.jsonl").write_text("", encoding="utf-8")
    _write_committed_segment(run_dir, "000001", _visual_metrics(), with_media=True)
    (row,) = scoreboard.scoreboard_rows(run_dir)
    assert row["video_exists"] is True
    assert "audio_exists" not in row
    assert "audio_path" not in row


def test_scoreboard_rows_records_baseline_id(tmp_path: Path) -> None:
    """062/027: deltas name their baseline so gaps cannot misattribute."""
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "metrics.jsonl").write_text("", encoding="utf-8")
    _write_committed_segment(run_dir, "000000", _visual_metrics(1.0))
    _write_committed_segment(run_dir, "000001", {"video": {"frames": 48}})
    _write_committed_segment(run_dir, "000002", _visual_metrics(2.0))
    rows = scoreboard.scoreboard_rows(run_dir)
    assert len(rows) == 3
    assert rows[0]["baseline_segment_id"] is None
    assert rows[1]["metrics"] is None
    assert rows[2]["baseline_segment_id"] == "000000"


def test_scoreboard_rows_validates_frames(tmp_path: Path) -> None:
    """062/027: non-integer frame counts degrade to None, never a raw string."""
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "metrics.jsonl").write_text("", encoding="utf-8")
    _write_committed_segment(run_dir, "000001", {"video": {"frames": "many"}}, with_media=False)
    (row,) = scoreboard.scoreboard_rows(run_dir)
    assert row["frames"] is None


def test_partial_segments_lists_non_done_dirs(tmp_path: Path) -> None:
    """062: stalled partials are visible via a helper (CLI renders the line)."""
    run_dir = tmp_path / "run"
    (run_dir / "segments" / "000000").mkdir(parents=True)
    (run_dir / "segments" / "000001").mkdir(parents=True)
    (run_dir / "segments" / "000000" / "DONE").write_text("ok", encoding="utf-8")
    partials = scoreboard.partial_segment_ids(run_dir)
    assert partials == ["000001"]


def test_console_error_goes_to_stream() -> None:
    """063/028: error() honors the injected stream like every sibling."""
    stream = io.StringIO()
    console = VoyageConsole(verbose=False, no_color=True, stream=stream)
    console.error("boom")
    assert "boom" in stream.getvalue()


def test_console_stage_failure_includes_elapsed() -> None:
    """063: plain failure path reports elapsed like the success path."""
    stream = io.StringIO()
    console = VoyageConsole(verbose=False, no_color=True, stream=stream)
    with pytest.raises(RuntimeError, match="boom"), console.stage("video"):
        msg = "boom"
        raise RuntimeError(msg)
    out = stream.getvalue()
    assert "video" in out
    assert "failed after" in out


def test_console_stage_tears_down_on_base_exception() -> None:
    """063: KeyboardInterrupt still tears down the stage and propagates."""
    stream = io.StringIO()
    console = VoyageConsole(verbose=False, no_color=True, stream=stream)
    with pytest.raises(KeyboardInterrupt), console.stage("video"):
        raise KeyboardInterrupt
    assert "video" in stream.getvalue()
    assert "failed" in stream.getvalue()


def test_tracker_unknown_label_warns_without_crash() -> None:
    """063/028: typo'd tracker labels warn instead of raising KeyError."""
    stream = io.StringIO()
    console = VoyageConsole(verbose=False, no_color=True, stream=stream)
    tracker = ParallelDownloadTracker(console, ["a"])
    tracker._begin()
    tracker.succeed("typo-label")
    tracker.fail("other-typo", "detail")
    out = stream.getvalue()
    assert "unknown" in out.lower()
