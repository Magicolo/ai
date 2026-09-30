"""`inspect metrics` rendering agreement (issue 036 extraction from `cli_observe`).

`render_inspect_metrics` is the verbatim `cmd_inspect -- metrics` branch,
moved so `cli_observe.py` (above the §12 ~500-line signal) shrinks toward
it. Behavior contract: identical stdout/exit codes to the pre-split
branch — no events prints `no metrics yet`, events print the
`<n> metric events across <m> files` header plus the last five rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage.cli_inspect_metrics import render_inspect_metrics


def _write_event(log_path: Path, event: str, segment_id: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "segment_id": segment_id}) + "\n")


def test_render_empty_run_reports_no_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No metric events at all still prints the empty-run line (exit 0)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="metrics-empty")
    assert render_inspect_metrics(run_dir) == 0
    assert "no metrics yet" in capsys.readouterr().out


def test_render_counts_events_across_rotated_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live + rotated events count together; header names the file span."""
    from voyage import paths

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="metrics")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    _write_event(logs_dir / "metrics-2026-09-28.jsonl", "segment_committed", "000000")
    _write_event(logs_dir / "metrics.jsonl", "segment_committed", "000001")
    assert render_inspect_metrics(run_dir) == 0
    out = capsys.readouterr().out
    assert "2 metric events" in out
    assert "2 files" in out
    assert "000000" in out
    assert "000001" in out


def test_render_lists_only_last_five(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Seven events render the trailing five rows (verbatim `events[-5:]`)."""
    from voyage import paths

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="metrics-trim")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    for index in range(7):
        _write_event(logs_dir / "metrics.jsonl", "segment_committed", f"{index:06d}")
    assert render_inspect_metrics(run_dir) == 0
    out = capsys.readouterr().out
    assert "7 metric events" in out
    assert "000000" not in out
    assert "000001" not in out
    assert "000006" in out


def test_cmd_inspect_metrics_delegates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`cmd_inspect -- metrics` delegates to the extracted renderer (seam)."""
    from voyage.cli_observe import cmd_inspect

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="metrics-seam")
    args = argparse.Namespace(run=str(run_dir), inspect_target="metrics")
    assert cmd_inspect(args) == 0
    assert "no metrics yet" in capsys.readouterr().out
