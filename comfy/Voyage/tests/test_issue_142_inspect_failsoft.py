"""Issue 142: `inspect` verbs are fail-soft best-effort views (OPERATIONS contract).

Read-only verbs never traceback, even racing a commit: a torn concept store
degrades to `unknown` (exit 0, like `_latest_novelty`), and a scoreboard row
with unformattable cells degrades to `no-visual` instead of raising.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


def _args(run_dir: Path, target: str) -> argparse.Namespace:
    return argparse.Namespace(run=str(run_dir), inspect_target=target)


def test_inspect_concepts_tolerates_torn_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Valid line + truncated tail prints best-effort output, exit 0."""
    from tests.conftest import initialize_run_directory
    from voyage.cli_observe import cmd_inspect

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    novelty = run_dir / "novelty"
    novelty.mkdir(parents=True, exist_ok=True)
    (novelty / "concepts.jsonl").write_text(
        json.dumps({"id": "0", "canonical_name": "glass dunes"}) + "\n{truncated",
        encoding="utf-8",
    )
    assert cmd_inspect(_args(run_dir, "concepts")) == 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


def test_inspect_concepts_missing_store_is_unknown(tmp_path: Path) -> None:
    """An unreadable store degrades instead of raising."""
    from tests.conftest import initialize_run_directory
    from voyage.cli_observe import cmd_inspect

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    novelty = run_dir / "novelty"
    novelty.mkdir(parents=True, exist_ok=True)
    (novelty / "concepts.jsonl").write_text("\x00 not json \x00", encoding="utf-8")
    assert cmd_inspect(_args(run_dir, "concepts")) == 0


def test_inspect_scoreboard_tolerates_unformattable_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A row with non-numeric metrics renders `no-visual`, exit 0."""
    from tests.conftest import initialize_run_directory
    from voyage.scoreboard import METRIC_KEYS

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)

    import voyage.cli_observe as observe_module

    row = {
        "segment_id": "000000",
        "frames": 96,
        "stages": {},
        "metrics": dict.fromkeys(METRIC_KEYS, "not-a-float"),
        "deltas": dict.fromkeys(METRIC_KEYS, "not-a-float"),
        "destination": "glass dunes",
        "phase": "build",
        "take_ids": [],
        "video_path": "v",
        "audio_path": "a",
    }
    monkeypatch.setattr(
        "voyage.scoreboard.scoreboard_rows",
        lambda _run: [row],
    )
    assert observe_module.cmd_inspect(_args(run_dir, "scoreboard")) == 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "no-visual" in captured.out
