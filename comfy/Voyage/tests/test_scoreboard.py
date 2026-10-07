"""Scoreboard view: per-segment rows with metric deltas (slice 3)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.logrotate import append_line
from voyage.persistence import read_effective_config
from voyage.scoreboard import main, scoreboard_rows
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "score") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _commit(run_dir: Path, count: int) -> None:
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.run_segments(count) == [f"{n:06d}" for n in range(count)]
    finally:
        supervisor.stop_workers()


def test_scoreboard_two_segments_without_visual(tmp_path: Path) -> None:
    """No piggyback inspector: rows carry frames/stages/plan, metrics None.

    Video-only commit (all backends deferred): no audio_path/audio_exists
    columns; take_ids is [] on fresh deferred commits (finalize takes
    live under run/audio/takes.jsonl).
    """
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 2)
    rows = scoreboard_rows(run_dir)
    assert [row["segment_id"] for row in rows] == ["000000", "000001"]
    assert all(row["done"] for row in rows)
    assert all(row["frames"] == 48 for row in rows)
    stages = cast(dict[str, object], rows[0]["stages"])
    assert set(stages) == {"director", "video", "audio", "validate", "commit"}
    assert rows[0]["metrics"] is None
    assert rows[0]["deltas"] is None
    assert rows[0]["destination"]
    assert rows[0]["phase"]
    assert rows[0]["take_ids"] == []
    assert cast(str, rows[0]["video_path"]).endswith("000000/video.mp4")
    assert "audio_path" not in rows[0]
    assert "audio_exists" not in rows[0]


def test_scoreboard_missing_visual(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    (rows,) = scoreboard_rows(run_dir)
    assert rows["metrics"] is None
    assert rows["deltas"] is None
    assert rows["frames"] == 48


def test_scoreboard_skips_partial(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 2)
    (run_dir / "segments" / "000002").mkdir(parents=True)
    rows = scoreboard_rows(run_dir)
    assert [row["segment_id"] for row in rows] == ["000000", "000001"]


def _write_committed_segment(run_dir: Path, segment_id: str) -> None:
    from voyage.segment_manifest import write_segment_manifest

    segment = run_dir / paths.SEGMENTS_DIRNAME / segment_id
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_text("", encoding="utf-8")
    write_segment_manifest(
        segment,
        {
            "metrics": {"video": {"frames": 48}},
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


def test_scoreboard_reads_stages_past_rotation(tmp_path: Path) -> None:
    """Rotate-then-read: stages committed before rotation still score."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _write_committed_segment(run_dir, "000000")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    live = logs_dir / "metrics.jsonl"
    live.write_text(
        json.dumps({"event": "segment_committed", "segment_id": "000000", "stages": {}}) + "\n",
        encoding="utf-8",
    )
    aged = time.time() - 86400 - 60
    os.utime(live, (aged, aged))
    rotated_stages = {"video": 12.5, "audio": 3.0}
    append_line(
        live,
        json.dumps(
            {
                "event": "segment_committed",
                "segment_id": "000001",
                "stages": {"video": 1.0},
            }
        ),
    )
    _write_committed_segment(run_dir, "000001")
    # Rewrite the rotated sibling with the real stages (it holds the
    # pre-rotation commit; the live file now holds only the new one).
    rotated = next(logs_dir.glob("metrics-*.jsonl"))
    rotated.write_text(
        json.dumps(
            {
                "event": "segment_committed",
                "segment_id": "000000",
                "stages": rotated_stages,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = scoreboard_rows(run_dir)
    by_id = {row["segment_id"]: row for row in rows}
    assert by_id["000000"]["stages"] == rotated_stages
    assert by_id["000001"]["stages"] == {"video": 1.0}


def test_scoreboard_main_text_lists_rows_and_partial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`python -m voyage.scoreboard` renders rows + the partial trailer (253)."""
    run_dir = tmp_path / "run"
    _write_committed_segment(run_dir, "000000")
    (run_dir / paths.SEGMENTS_DIRNAME / "000001").mkdir(parents=True)
    assert main(["--run", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "segments: 1" in out
    assert "000000" in out
    assert "partial: ['000001']" in out


def test_scoreboard_main_json_lists_rows_and_partial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--json` prints the raw rows document (253)."""
    run_dir = tmp_path / "run"
    _write_committed_segment(run_dir, "000000")
    assert main(["--run", str(run_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"][0]["segment_id"] == "000000"
    assert payload["partial"] == []


def test_scoreboard_main_is_read_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The viewer writes nothing into the run."""
    run_dir = tmp_path / "run"
    _write_committed_segment(run_dir, "000000")
    before = sorted(str(path) for path in run_dir.rglob("*"))
    assert main(["--run", str(run_dir)]) == 0
    capsys.readouterr()
    assert main(["--run", str(run_dir), "--json"]) == 0
    capsys.readouterr()
    assert sorted(str(path) for path in run_dir.rglob("*")) == before
