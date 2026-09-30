"""Rotation-aware inspect metrics + fail-loud fps (issues 029/055).

``inspect metrics`` opened only the live ``metrics.jsonl`` while every
sibling reader spans live + rotated files via ``iter_metric_files`` —
after a daily rotation the count dropped and pre-rotation history
vanished (055 is the same reader, superseded by 029 here). Separately,
``RunState`` tolerates legacy ``fps=0`` while ``status`` shows a
zero-length timeline and ``validate`` silently substituted 24 for the
SFX-ledger check. These tests pin the shared reader plus the
fail-loud ``fps <= 0`` validate error (the 24 fallback stays for the
SFX math only after reporting). CPU-only: synthetic logs, no workers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.cli import cmd_inspect, validate_run
from voyage.persistence import read_state, write_state


def _write_metric_event(log_path: Path, segment_id: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "segment_committed", "segment_id": segment_id}) + "\n")


def test_inspect_metrics_spans_rotated_siblings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live + rotated events count together; both segments are listed."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="metrics")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    _write_metric_event(logs_dir / "metrics-2026-09-28.jsonl", "000000")
    _write_metric_event(logs_dir / "metrics.jsonl", "000001")
    args = argparse.Namespace(run=str(run_dir), inspect_target="metrics")
    assert cmd_inspect(args) == 0
    output = capsys.readouterr().out
    assert "2 metric events" in output
    assert "000000" in output
    assert "000001" in output


def test_inspect_metrics_reports_file_span(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The header names the file count so rotation stays visible."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="metrics")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    _write_metric_event(logs_dir / "metrics-2026-09-28.jsonl", "000000")
    _write_metric_event(logs_dir / "metrics.jsonl", "000001")
    args = argparse.Namespace(run=str(run_dir), inspect_target="metrics")
    assert cmd_inspect(args) == 0
    output = capsys.readouterr().out
    assert "2 files" in output


def test_validate_reports_nonpositive_fps(tmp_path: Path) -> None:
    """A legacy fps=0 state fails loud naming fps (was: silent VALID)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="fps")
    state = read_state(run_dir)
    state.fps = 0
    state.timeline_frames = 48
    write_state(run_dir, state)
    errors = validate_run(run_dir)
    assert any("fps" in error for error in errors)


def test_validate_keeps_sfx_fallback_after_reporting(tmp_path: Path) -> None:
    """Remaining checks still run after the fps error (no early return)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="fps")
    state = read_state(run_dir)
    state.fps = 0
    state.timeline_frames = 48
    write_state(run_dir, state)
    errors = validate_run(run_dir)
    assert any("fps" in error for error in errors)
    assert any("timeline frames 48 != sum of segment frames 0" in error for error in errors)
