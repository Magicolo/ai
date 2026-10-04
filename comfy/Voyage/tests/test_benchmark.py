"""Benchmark + soak tests: worker `benchmark` ops, §104 reports, per-segment
gauges, soak CLI, endurance marker (Phase 6 slice E).

Fake backends (real media, no GPU) in-container; GPU-only report fields
degrade to "unknown" and are asserted as such here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.bench import format_report, summarize_gauges, timing_stats
from voyage.cli import main
from voyage.persistence import read_effective_config, read_state
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "benchmark") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _gauge_events(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    return [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if '"event": "resource_gauges"' in line
    ]


def test_fake_video_benchmark_op_reports_math(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        result: dict[str, Any] = supervisor._video.call(
            "benchmark",
            {"warmup": 1, "measured": 2, "width": 320, "height": 180, "fps": 24, "frames": 24},
        )
    finally:
        supervisor.stop_workers()
    assert result["backend"] == "fake"
    assert result["warmup_blocks"] == 1
    assert result["measured_blocks"] == 2
    assert len(result["block_wall_seconds"]) == 2
    assert result["blocks_per_second"] > 0
    assert result["fps_equivalent"] == pytest.approx(result["blocks_per_second"] * 24, rel=1e-2)
    assert result["vram_peak_gib"] == "unknown"


def test_fake_audio_benchmark_op_reports_math(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        result: dict[str, Any] = supervisor._audio.call(
            "benchmark",
            {"warmup": 1, "measured": 2, "duration_seconds": 1.0},
        )
    finally:
        supervisor.stop_workers()
    assert result["backend"] == "fake"
    assert result["measured_takes"] == 2
    assert result["takes_per_second"] > 0
    assert result["audio_seconds_per_wall_second"] > 0


def test_director_benchmark_op_times_decisions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        result: dict[str, Any] = supervisor._director.call(
            "benchmark", {"warmup": 1, "measured": 2}
        )
    finally:
        supervisor.stop_workers()
    assert result["backend"] == "deterministic"
    assert result["measured_decisions"] == 2
    assert result["decisions_per_second"] > 0


def test_timing_stats_and_report_format() -> None:
    stats = timing_stats([2.0, 1.0, 3.0])
    assert stats == {"count": 3, "mean": 2.0, "min": 1.0, "max": 3.0}
    report = format_report(
        "video",
        {"backend": "fake", "warmup": 1, "measured": 2},
        {"blocks_per_second": 4.0},
    )
    assert "blocks_per_second" in report
    assert "warmup" in report


def test_benchmark_cli_video_prints_report(tmp_path: Path, capsys: object) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    assert main(["benchmark", "video", "--run", str(run_dir)]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "blocks_per_second" in out


def test_benchmark_cli_end_to_end_uses_temp_dir(tmp_path: Path, capsys: object) -> None:
    assert main(["benchmark", "end-to-end", "--segments", "1"]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "segments" in out


def test_per_segment_gauges_logged(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = read_effective_config(run_dir)
    assert Supervisor(run_dir, config).run_segments(2) == ["000000", "000001"]
    events = _gauge_events(run_dir)
    assert len(events) == 2
    assert [event["segment_id"] for event in events] == ["000000", "000001"]
    for event in events:
        assert event["disk_free_gib"] > 0
        assert event["rss_peak_mb"] > 0
        assert "run_id" in event
    summary = summarize_gauges(events)
    assert summary["segments"] == 2
    # Symmetric flatness budget (issue 089): peak RSS can legitimately
    # *fall* between segments after GC, so a one-sided `>= 0` flakes.
    assert abs(summary["rss_delta_mb"]) < 500


def test_soak_cli_reports_trend(tmp_path: Path, capsys: object) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    assert main(["soak", "--run", str(run_dir), "--segments", "2"]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "rss_delta_mb" in out
    assert read_state(run_dir).committed_segments == 2


# Always-on stability gate (issue 089 decision): `gates.sh`/`test.sh` pass
# no `-m` filter, so this `endurance` test runs on every gate by design —
# small (3 fake segments, seconds) and bounded the same 500 MB budget as
# the gauge test above. Exclude deliberately with `-m "not endurance"`.
@pytest.mark.endurance
def test_endurance_segments_stay_flat(tmp_path: Path) -> None:
    """Small always-on soak: gauges every segment, bounded RSS, valid run."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = read_effective_config(run_dir)
    assert Supervisor(run_dir, config).run_segments(3) == ["000000", "000001", "000002"]
    events = _gauge_events(run_dir)
    assert len(events) == 3
    summary = summarize_gauges(events)
    assert summary["rss_delta_mb"] < 500
    assert main(["validate", "--run", str(run_dir)]) == 0


# --- 088 fold: test_stage_timings.py (slice 2) ---
# """Per-stage wall-time breakdown in segment_committed metrics (slice 2).
#
# Each committed segment records how long its stages took (inspect,
# director, video, audio, validate, commit) so experiment runs reveal the
# real bottleneck instead of one opaque elapsed number.
# """


def _committed_events(run_dir: Path) -> list[dict[str, object]]:
    lines = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in lines.splitlines() if line.strip()]
    return [e for e in events if e.get("event") == "segment_committed"]


def test_segment_committed_carries_stage_breakdown(tmp_path: Path) -> None:
    """segment_committed includes per-stage seconds that add up to elapsed."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="timings")
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
    finally:
        supervisor.stop_workers()
    assert read_state(run_dir).committed_segments == 1

    events = _committed_events(run_dir)
    assert len(events) == 1
    stages = events[0].get("stages")
    assert isinstance(stages, dict)
    assert set(stages) == {"inspect", "director", "video", "audio", "validate", "commit"}
    elapsed = events[0]["elapsed_seconds"]
    assert isinstance(elapsed, (int, float))
    total = 0.0
    for name, seconds in stages.items():
        assert isinstance(seconds, (int, float)), name
        assert seconds >= 0.0, name
        assert seconds <= elapsed, name
        total += float(seconds)
    assert total <= elapsed + 0.01  # +10ms: six round(x, 3) stages can sum 3ms over
