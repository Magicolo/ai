"""Supervisor metrics ship the versioned schema (issue 224).

`Supervisor._log_metric` routes through `format_metric_line`, so every
commit-path event carries `ts`/`ts_iso`/`run_id`/`schema` with the 16
KiB truncation cap; schema-less v0 lines stay readable (legacy
fallback). Against tmp runs, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.logrotate import (
    MAX_METRIC_LINE_BYTES,
    METRICS_SCHEMA_VERSION,
    parse_metric_lines,
)
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _supervisor_for(run_dir: Path) -> Supervisor:
    initialize_run_directory(run_dir)
    return Supervisor(run_dir, read_effective_config(run_dir))


def _last_event(run_dir: Path) -> dict[str, object]:
    lines = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    parsed = json.loads([line for line in lines.splitlines() if line.strip()][-1])
    assert isinstance(parsed, dict)
    return parsed


def test_log_metric_lines_carry_schema_and_iso_ts(tmp_path: Path) -> None:
    """Supervisor-emitted lines are v1: schema + ts_iso + run_id."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_for(run_dir)
    supervisor._log_metric({"event": "probe_event", "segment_id": "000000"})
    event = _last_event(run_dir)
    assert event["schema"] == METRICS_SCHEMA_VERSION
    assert event["run_id"] == supervisor._config.name
    assert isinstance(event["ts"], float)
    assert isinstance(event["ts_iso"], str)
    assert event["event"] == "probe_event"
    assert event["segment_id"] == "000000"


def test_log_metric_caps_oversized_event(tmp_path: Path) -> None:
    """An oversized event truncates to the 16 KiB cap instead of clogging."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_for(run_dir)
    supervisor._log_metric({"event": "probe_event", "blob": "x" * 100000})
    raw = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    assert len(raw.strip().encode("utf-8")) <= MAX_METRIC_LINE_BYTES
    assert _last_event(run_dir).get("truncated") is True


def test_log_metric_event_cannot_spoof_base_fields(tmp_path: Path) -> None:
    """Caller-supplied ts/run_id never override the stamped base fields."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_for(run_dir)
    supervisor._log_metric({"event": "probe_event", "run_id": "evil", "ts": 0})
    event = _last_event(run_dir)
    assert event["run_id"] == supervisor._config.name
    assert event["ts"] != 0


def test_legacy_schema_less_lines_still_parse() -> None:
    """v0 lines (pre-224 writers) remain readable: tolerance is the fallback."""
    events, torn = parse_metric_lines(['{"event": "legacy", "run_id": "run-1"}'])
    assert torn == 0
    assert events[0].get("schema") is None
    assert events[0]["event"] == "legacy"
