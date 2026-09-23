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

from voyage import paths
from voyage.cli import cmd_status
from voyage.config import default_config_toml, load_config
from voyage.logrotate import append_line
from voyage.persistence import (
    build_manifest,
    initial_state,
    write_manifest,
    write_state,
)
from voyage.rpc import SubprocessWorker
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "observability") -> None:
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
