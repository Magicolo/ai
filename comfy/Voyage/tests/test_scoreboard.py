"""Scoreboard view: per-segment rows with metric deltas (slice 3)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.logrotate import append_line
from voyage.persistence import read_effective_config
from voyage.scoreboard import scoreboard_rows
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
    """No piggyback inspector: rows carry frames/stages/plan, metrics None."""
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
    assert rows[0]["take_ids"]
    assert cast(str, rows[0]["video_path"]).endswith("000000/video.mp4")
    assert cast(str, rows[0]["audio_path"]).endswith("000000/audio.wav")


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
