"""Observability tests: daily log rotation + retention, run_id-enriched
metric events, §59 status output (Phase 6 slice D).

All run against fake backends (real media, no GPU) in-container.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.cli import _last_commit_stages, _read_all_metric_events, cmd_status
from voyage.config import load_config
from voyage.logrotate import append_line, iter_metric_files
from voyage.rpc import SubprocessWorker
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "observability") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _backdate(path: Path, days_ago: int) -> None:
    aged = time.time() - days_ago * 86400 - 60
    os.utime(path, (aged, aged))


def _day_ago(days_ago: int) -> str:
    today = datetime.datetime.now(datetime.timezone.utc).date()  # noqa: UP017
    return (today - datetime.timedelta(days=days_ago)).isoformat()


def test_rotation_rolls_stale_log(tmp_path: Path) -> None:
    """A log last written on a previous day rolls to a dated sibling."""
    log = tmp_path / "metrics.jsonl"
    log.write_text('{"event": "old"}\n', encoding="utf-8")
    _backdate(log, 1)
    append_line(log, '{"event": "new"}')
    rotated = tmp_path / f"metrics-{_day_ago(1)}.jsonl"
    assert rotated.exists()
    assert rotated.read_text(encoding="utf-8") == '{"event": "old"}\n'
    assert log.read_text(encoding="utf-8") == '{"event": "new"}\n'


def test_same_day_appends_without_rotation(tmp_path: Path) -> None:
    """A current log just grows; no dated siblings appear."""
    log = tmp_path / "metrics.jsonl"
    append_line(log, '{"event": "one"}')
    append_line(log, '{"event": "two"}')
    assert log.read_text(encoding="utf-8") == '{"event": "one"}\n{"event": "two"}\n'
    assert list(tmp_path.glob("metrics-*.jsonl")) == []


def test_prune_keeps_recent_drops_old(tmp_path: Path) -> None:
    """Retention keeps recent daily logs and deletes older ones."""
    recent = tmp_path / f"metrics-{_day_ago(5)}.jsonl"
    old = tmp_path / f"metrics-{_day_ago(40)}.jsonl"
    recent.write_text("", encoding="utf-8")
    old.write_text("", encoding="utf-8")
    log = tmp_path / "metrics.jsonl"
    log.write_text("", encoding="utf-8")
    _backdate(log, 1)
    append_line(log, '{"event": "new"}', keep_days=30)
    assert recent.exists()
    assert not old.exists()


def test_worker_start_rotates_stale_log(tmp_path: Path) -> None:
    """Spawning a worker rolls its stale stderr log before appending."""
    log = tmp_path / "w.log"
    log.write_text("old worker output\n", encoding="utf-8")
    _backdate(log, 2)
    worker = SubprocessWorker("definitely_not_a_voyage_module", tmp_path, log, init_op=None)
    worker.start()
    worker.stop()
    rotated = list(tmp_path.glob("w-*.log"))
    assert len(rotated) == 1
    assert rotated[0].read_text(encoding="utf-8") == "old worker output\n"
    assert "old worker output" not in log.read_text(encoding="utf-8")


def test_metric_file_iterator_lists_live_only_without_rotation(tmp_path: Path) -> None:
    """No rotation yet → the iterator yields just the live file."""
    run_dir = tmp_path / "run"
    logs_dir = run_dir / paths.LOGS_DIRNAME
    logs_dir.mkdir(parents=True)
    live = logs_dir / "metrics.jsonl"
    live.write_text('{"event": "one"}\n', encoding="utf-8")
    assert iter_metric_files(run_dir) == [live]


def test_metric_file_iterator_spans_rotation_oldest_first(tmp_path: Path) -> None:
    """Rotate-then-read: the dated sibling sorts before the live file."""
    run_dir = tmp_path / "run"
    logs_dir = run_dir / paths.LOGS_DIRNAME
    logs_dir.mkdir(parents=True)
    live = logs_dir / "metrics.jsonl"
    live.write_text('{"event": "old"}\n', encoding="utf-8")
    _backdate(live, 2)
    append_line(live, '{"event": "new"}')
    rotated = logs_dir / f"metrics-{_day_ago(2)}.jsonl"
    assert iter_metric_files(run_dir) == [rotated, live]


def test_metric_file_iterator_ignores_non_dated_siblings(tmp_path: Path) -> None:
    """Undated lookalikes never leak into the history stream."""
    run_dir = tmp_path / "run"
    logs_dir = run_dir / paths.LOGS_DIRNAME
    logs_dir.mkdir(parents=True)
    live = logs_dir / "metrics.jsonl"
    live.write_text("", encoding="utf-8")
    stray = logs_dir / "metrics-backup.jsonl"
    stray.write_text("", encoding="utf-8")
    assert iter_metric_files(run_dir) == [live]


def test_metric_file_iterator_missing_logs_dir(tmp_path: Path) -> None:
    """A run without logs yet yields no files instead of raising."""
    assert iter_metric_files(tmp_path / "absent-run") == []


def test_metric_events_carry_run_id(tmp_path: Path) -> None:
    """Every metric event names its run (multi-run log spelunking)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir, run_id="run-id-probe")
    assert Supervisor(run_dir, load_config(run_dir / paths.CONFIG_FILENAME)[0]).run_segments(1) == [
        "000000"
    ]
    metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        assert json.loads(line)["run_id"] == "run-id-probe"


def test_status_shows_section_layout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`voyage status` renders the §59 sections plus last-commit stages."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    assert Supervisor(run_dir, load_config(run_dir / paths.CONFIG_FILENAME)[0]).run_segments(1) == [
        "000000"
    ]
    args = type("Args", (), {"run": str(run_dir)})()
    assert cmd_status(args) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    for expected in (
        "Voyage: observability",
        "Uptime:",
        "Video",
        "Backend: fake",
        "768",
        "432",
        "World",
        "Audio",
        "Music: ambient electronic",
        "Workers",
        "video: idle",
        "Stages (last commit 000000)",
        "video:",
        "Storage",
        "Free:",
    ):
        assert expected in out, expected


def test_last_commit_stages_spans_rotation(tmp_path: Path) -> None:
    """Day-after rotation: the commit in the dated sibling is still found (049)."""
    run_dir = tmp_path / "run"
    logs_dir = run_dir / paths.LOGS_DIRNAME
    logs_dir.mkdir(parents=True)
    rotated = logs_dir / "metrics-2026-01-01.jsonl"
    rotated.write_text(
        '{"event": "segment_committed", "segment_id": "000000", "stages": {"video": 1.0}}\n',
        encoding="utf-8",
    )
    live = logs_dir / "metrics.jsonl"
    live.write_text('{"event": "resource_gauges"}\n', encoding="utf-8")
    assert _last_commit_stages(run_dir) == ("000000", {"video": 1.0})
    live.write_text(
        '{"event": "segment_committed", "segment_id": "000001", "stages": {"video": 2.0}}\n',
        encoding="utf-8",
    )
    assert _last_commit_stages(run_dir) == ("000001", {"video": 2.0})


def test_read_all_metric_events_concatenates_oldest_first(tmp_path: Path) -> None:
    """Soak/benchmark averages see rotated + live events in order, torn lines skipped."""
    run_dir = tmp_path / "run"
    logs_dir = run_dir / paths.LOGS_DIRNAME
    logs_dir.mkdir(parents=True)
    rotated = logs_dir / "metrics-2026-01-01.jsonl"
    rotated.write_text(
        '{"event": "segment_committed", "segment_id": "000000"}\nnot-json\n',
        encoding="utf-8",
    )
    live = logs_dir / "metrics.jsonl"
    live.write_text('{"event": "resource_gauges"}\n', encoding="utf-8")
    events = _read_all_metric_events(run_dir)
    assert [event["event"] for event in events] == ["segment_committed", "resource_gauges"]
    assert _read_all_metric_events(tmp_path / "absent-run") == []
