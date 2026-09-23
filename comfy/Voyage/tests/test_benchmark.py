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

from voyage import paths
from voyage.bench import format_report, summarize_gauges, timing_stats
from voyage.cli import main
from voyage.config import default_config_toml, load_config
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_state,
    write_manifest,
    write_state,
)
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "benchmark") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.CONFIG_FILENAME).write_text(
        default_config_toml(run_id, "pastel neon line-art, peaceful", 11),
        encoding="utf-8",
    )
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


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
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        result = supervisor._video.call(
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
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        result = supervisor._audio.call(
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
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        result = supervisor._director.call("benchmark", {"warmup": 1, "measured": 2})
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
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
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
    assert summary["rss_delta_mb"] >= 0


def test_soak_cli_reports_trend(tmp_path: Path, capsys: object) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    assert main(["soak", "--run", str(run_dir), "--segments", "2"]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "rss_delta_mb" in out
    assert read_state(run_dir).committed_segments == 2


@pytest.mark.endurance
def test_endurance_segments_stay_flat(tmp_path: Path) -> None:
    """Small always-on soak: gauges every segment, bounded RSS, valid run."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert Supervisor(run_dir, config).run_segments(3) == ["000000", "000001", "000002"]
    events = _gauge_events(run_dir)
    assert len(events) == 3
    summary = summarize_gauges(events)
    assert summary["rss_delta_mb"] < 500
    assert main(["validate", "--run", str(run_dir)]) == 0
