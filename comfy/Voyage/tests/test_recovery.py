"""Recovery tests: supervisor restart, worker crash, pause (DESIGN §§27, 69).

All run against fake backends (real media, no GPU) in-container.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from voyage import paths
from voyage.cli import main
from voyage.config import default_config_toml, load_config
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_state,
    write_manifest,
    write_state,
)
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "recovery") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.CONFIG_FILENAME).write_text(
        default_config_toml(run_id, "pastel neon line-art, peaceful", 11),
        encoding="utf-8",
    )
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


def test_supervisor_restart_continues(tmp_path: Path) -> None:
    """A new Supervisor picks up where the old one stopped (Phase 1 exit)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    assert Supervisor(run_dir, config).run_segments(1) == ["000001"]
    state = read_state(run_dir)
    assert state.committed_segments == 2
    assert state.next_segment_number == 2
    assert main(["validate", "--run", str(run_dir)]) == 0


def test_killed_video_worker_recovers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
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
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    state = read_state(run_dir)
    state.status = "PAUSE_REQUESTED"
    write_state(run_dir, state)
    assert Supervisor(run_dir, config).run_segments(2) == []
    assert read_state(run_dir).status == "PAUSED"


def test_pause_mid_run_stops_at_boundary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)

    def _ask_pause() -> None:
        for _ in range(400):
            if (run_dir / "segments" / "000000" / "DONE").exists():
                break
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
