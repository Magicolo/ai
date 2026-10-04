"""Recovery tests: supervisor restart, worker crash, pause (DESIGN §§27, 69).

All run against fake backends (real media, no GPU) in-container.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.cli import main
from voyage.persistence import (
    read_effective_config,
    read_state,
    write_state,
)
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "recovery") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def test_supervisor_restart_continues(tmp_path: Path) -> None:
    """A new Supervisor picks up where the old one stopped (Phase 1 exit)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    assert Supervisor(run_dir, config).run_segments(1) == ["000001"]
    state = read_state(run_dir)
    assert state.committed_segments == 2
    assert state.next_segment_number == 2
    assert main(["validate", "--run", str(run_dir)]) == 0


def test_killed_video_worker_recovers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.inject_worker_crash("video")
        segment_id = supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert segment_id == "000000"
    assert (paths.segment_dir(run_dir, segment_id) / paths.DONE_MARKER).exists()
    metrics = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "worker_restart"' in metrics
    assert '"worker": "video"' in metrics


def test_pause_before_start_exits_cleanly(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    state = read_state(run_dir)
    state.status = "PAUSE_REQUESTED"
    write_state(run_dir, state)
    assert Supervisor(run_dir, config).run_segments(2) == []
    assert read_state(run_dir).status == "PAUSED"


def test_pause_mid_run_stops_at_boundary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)

    def _ask_pause() -> None:
        # Event poll with a hard deadline (issue 089): the DONE marker is
        # the event, the deadline keeps a wedged commit from silently
        # falling through to a confusing zero-segment assertion below.
        deadline = time.monotonic() + 120.0
        while not (run_dir / "segments" / "000000" / "DONE").exists():
            assert time.monotonic() < deadline, "first segment never committed"
            time.sleep(0.05)
        state = read_state(run_dir)
        state.status = "PAUSE_REQUESTED"
        write_state(run_dir, state)

    thread = threading.Thread(target=_ask_pause, daemon=True)
    thread.start()
    committed = Supervisor(run_dir, config).run_segments(10)
    thread.join()
    assert 1 <= len(committed) <= 2
    assert read_state(run_dir).status == "PAUSED"


# ---------------------------------------------------------------------------
# Phase 2 remainder: tape lookup, restart hook (issue 088
# fold — moved verbatim from tests/test_phase2.py so recovery owns the
# whole restart/resume surface; all against fake backends, no GPU).
# (The scene-cut prompt-prefix helper lived on a removed video
# backend — scene cuts are boolean payload flags on the live backends.)
# ---------------------------------------------------------------------------


def _tape_run(run_dir: Path) -> Supervisor:
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    return Supervisor(run_dir, config)


def test_latest_recovery_tape_picks_newest_done(tmp_path: Path) -> None:
    supervisor = _tape_run(tmp_path / "run")
    assert supervisor._latest_recovery_tape() is None
    root = tmp_path / "run" / paths.SEGMENTS_DIRNAME
    (root / "000000").mkdir(parents=True)
    (root / "000000" / paths.DONE_MARKER).write_text("")
    (root / "000000" / "recovery.pt").write_text("old")
    (root / "000001").mkdir()
    (root / "000001" / paths.DONE_MARKER).write_text("")
    (root / "000002").mkdir()
    (root / "000002" / "recovery.pt").write_text("uncommitted")
    # 000001 is DONE but has no tape → falls back to 000000; uncommitted
    # 000002 is never eligible.
    assert supervisor._latest_recovery_tape() == root / "000000" / "recovery.pt"
    (root / "000001" / "recovery.pt").write_text("new")
    assert supervisor._latest_recovery_tape() == root / "000001" / "recovery.pt"


def test_restart_hook_runs_after_video_crash(tmp_path: Path) -> None:
    supervisor = _tape_run(tmp_path / "run")
    supervisor.start_workers()
    try:
        supervisor.inject_worker_crash("video")
        fired: list[tuple[str, str]] = []
        result = supervisor._call_with_restart(
            supervisor._video,
            "video",
            "000000",
            "health",
            {},
            restart_hook=lambda segment_id: fired.append(("resume", segment_id)),
        )
        assert result["status"] == "READY"
        assert fired == [("resume", "000000")]
    finally:
        supervisor.stop_workers()
