"""Scoreboard table rendering (issue 036 extraction from `cli_observe`).

`render_scoreboard` is the verbatim `cmd_inspect -- scoreboard` branch,
moved so `cli_observe.py` (above the §12 ~500-line signal) shrinks toward
it. Behavior contract: identical stdout/exit codes to the pre-split
branch — empty runs print `no committed segments`, committed segments
print one header + one row block each, and a present `final.mp4` prints
its trailing line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import paths
from voyage.cli_scoreboard import render_scoreboard
from voyage.scoreboard import METRIC_KEYS
from voyage.segment_manifest import write_segment_manifest


def _write_segment(
    run_dir: Path, segment_id: str, *, visual: dict[str, float] | None = None
) -> None:
    """One DONE segment with frames/transition/take ids (+ optional visual)."""
    segment = run_dir / paths.SEGMENTS_DIRNAME / segment_id
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_text("", encoding="utf-8")
    metrics: dict[str, object] = {"video": {"frames": 48}}
    if visual is not None:
        metrics["visual"] = {"metrics": dict(visual)}
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


def test_render_empty_run_reports_no_segments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No segments dir at all still prints the empty-run line (exit 0)."""
    assert render_scoreboard(tmp_path / "run") == 0
    assert "no committed segments" in capsys.readouterr().out


def test_render_one_segment_without_visual(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Committed segment without visual metrics renders the no-visual row."""
    run_dir = tmp_path / "run"
    _write_segment(run_dir, "000000")
    assert render_scoreboard(run_dir) == 0
    out = capsys.readouterr().out
    assert "000000" in out
    assert "no-visual" in out
    assert "probe harbor" in out
    assert "take-1" in out


def test_render_one_segment_with_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Visual metrics render as `value(+delta)` cells, first row delta 0."""
    run_dir = tmp_path / "run"
    _write_segment(run_dir, "000000", visual=dict.fromkeys(METRIC_KEYS, 0.5))
    assert render_scoreboard(run_dir) == 0
    out = capsys.readouterr().out
    assert "0.500(+0.000)" in out
    assert "no-visual" not in out


def test_render_reports_final_mp4(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A present final.mp4 prints its trailing line after the rows."""
    run_dir = tmp_path / "run"
    _write_segment(run_dir, "000000")
    (run_dir / "final.mp4").write_bytes(b"fake-video")
    assert render_scoreboard(run_dir) == 0
    out = capsys.readouterr().out
    assert "final:" in out
