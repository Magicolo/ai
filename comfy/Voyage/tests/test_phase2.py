"""Phase 2 remainder tests: scene-cut prefix, tape lookup, restart hook.

All against fake backends (real media, no GPU) in-container. GPU replay
correctness is proven by E2E (coherent frames post-resume); these pin the
protocol and state machine deterministically (DESIGN §27.2).
"""

from __future__ import annotations

from pathlib import Path

from tests.test_recovery import _init_run
from voyage import paths
from voyage.config import load_config
from voyage.supervisor import Supervisor
from voyage.workers.video_longlive import SCENE_CUT_PREFIX, apply_scene_cut_prefix


def test_scene_cut_prefix_applied_once(tmp_path: Path) -> None:
    del tmp_path
    assert apply_scene_cut_prefix("a meadow", True) == SCENE_CUT_PREFIX + "a meadow"
    assert apply_scene_cut_prefix("a meadow", False) == "a meadow"
    already = SCENE_CUT_PREFIX + "a meadow"
    assert apply_scene_cut_prefix(already, True) == already


def _tape_run(run_dir: Path) -> Supervisor:
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
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
