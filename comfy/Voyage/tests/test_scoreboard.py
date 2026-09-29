"""Scoreboard view: per-segment rows with metric deltas (slice 3)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import load_config
from voyage.logrotate import append_line
from voyage.scoreboard import METRIC_KEYS, scoreboard_rows
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, *, inspector: bool, run_id: str = "score") -> None:
    initialize_run_directory(run_dir, run_id=run_id, seed=7, visual_inspector=inspector)


def _commit(run_dir: Path, count: int) -> None:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.run_segments(count) == [f"{n:06d}" for n in range(count)]
    finally:
        supervisor.stop_workers()


def test_scoreboard_two_segments_with_deltas(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir, inspector=True)
    # Three commits: the piggyback inspector always covers the PREVIOUS
    # segment, so seg0/seg1 carry visual metrics and seg2 does not yet.
    _commit(run_dir, 3)
    rows = scoreboard_rows(run_dir)
    assert [row["segment_id"] for row in rows] == ["000000", "000001", "000002"]
    assert all(row["done"] for row in rows)
    assert all(row["frames"] == 48 for row in rows)
    assert set(rows[0]["stages"]) == {"inspect", "director", "video", "audio", "validate", "commit"}
    assert set(rows[0]["metrics"]) == set(METRIC_KEYS)
    assert rows[0]["deltas"] == dict.fromkeys(METRIC_KEYS, 0.0)
    expected = {
        key: round(rows[1]["metrics"][key] - rows[0]["metrics"][key], 3) for key in METRIC_KEYS
    }
    assert rows[1]["deltas"] == expected
    assert rows[0]["destination"]
    assert rows[0]["phase"]
    assert rows[0]["take_ids"]
    assert rows[0]["video_path"].endswith("000000/video.mp4")
    assert rows[0]["audio_path"].endswith("000000/audio.wav")


def test_scoreboard_missing_visual(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir, inspector=False)
    _commit(run_dir, 1)
    (rows,) = scoreboard_rows(run_dir)
    assert rows["metrics"] is None
    assert rows["deltas"] is None
    assert rows["frames"] == 48


def test_scoreboard_skips_partial(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir, inspector=True)
    _commit(run_dir, 2)
    (run_dir / "segments" / "000002").mkdir(parents=True)
    rows = scoreboard_rows(run_dir)
    assert [row["segment_id"] for row in rows] == ["000000", "000001"]


def _write_committed_segment(run_dir: Path, segment_id: str) -> None:
    segment = run_dir / paths.SEGMENTS_DIRNAME / segment_id
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_text("", encoding="utf-8")
    (segment / "metrics.json").write_text(json.dumps({"video": {"frames": 48}}), encoding="utf-8")
    (segment / "transition.json").write_text(
        json.dumps({"destination": {"canonical_name": "probe harbor"}, "phase": "HOLD"}),
        encoding="utf-8",
    )
    (segment / "audio_state.json").write_text(
        json.dumps({"take_ids": ["take-1"]}), encoding="utf-8"
    )


def test_scoreboard_reads_stages_past_rotation(tmp_path: Path) -> None:
    """Rotate-then-read: stages committed before rotation still score."""
    run_dir = tmp_path / "run"
    _init_run(run_dir, inspector=False)
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
