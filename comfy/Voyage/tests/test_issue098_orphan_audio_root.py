"""Orphan scan covers audio/ + run root (issue 098).

Why this module exists: `validate_run` orphan-scanned only `segments/` +
`novelty/` while `atomic_write_*` stages `*.partial` in every
`destination.parent` — including `audio/` takes and the run root
(`state.json`). A crashed take or state advance left residue the
canonical "is this run clean?" tool never reported. The root pass is
scoped to top-level `*.partial` only so the recursive `segments/` +
`novelty/` + `audio/` hits are not double-counted; `logs/` rotation
siblings (`metrics-YYYY-MM-DD.jsonl`) never match the globs.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage.cli_validate import validate_run


def test_validate_flags_audio_partial(tmp_path: Path) -> None:
    """A crashed take staging file under audio/ must be reported."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    orphan = audio_dir / "take_0000.wav.partial"
    orphan.write_text("torn", encoding="utf-8")
    errors = validate_run(run_dir)
    assert any("orphan" in error and "take_0000" in error for error in errors)


def test_validate_flags_root_partial(tmp_path: Path) -> None:
    """A crashed state advance (`state.json.*.partial`) must be reported."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    orphan = run_dir / "state.json.abc123.partial"
    orphan.write_text("torn", encoding="utf-8")
    errors = validate_run(run_dir)
    assert any("orphan" in error and "state.json" in error for error in errors)


def test_validate_ignores_logs_rotation_sibling(tmp_path: Path) -> None:
    """Dated `logs/` rotation siblings are legitimate, never orphans."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    sibling = run_dir / "logs" / "metrics-2026-01-01.jsonl"
    sibling.write_text("{}\n", encoding="utf-8")
    errors = validate_run(run_dir)
    assert validate_run(run_dir) == [] or not any("orphan" in error for error in errors)
    assert errors == []
