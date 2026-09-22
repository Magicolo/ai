"""Integration tests: workers over RPC + full segment commit + validate.

Fake backends render real media with ffmpeg, so validation, checksums,
DONE markers, state advance, and finalizer concat are all exercised.
"""

from __future__ import annotations

from pathlib import Path

from voyage import paths
from voyage.config import default_config_toml, load_config
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_state,
    write_manifest,
    write_state,
)
from voyage.rpc import SubprocessWorker
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.CONFIG_FILENAME).write_text(
        default_config_toml("itest", "pastel neon line-art, peaceful", 7),
        encoding="utf-8",
    )
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


def test_workers_answer_health(tmp_path: Path) -> None:
    worker = SubprocessWorker("voyage.workers.video", tmp_path, tmp_path / "v.log")
    worker.start()
    try:
        assert worker.health()["status"] == "READY"
    finally:
        worker.stop()


def test_director_worker_decides(tmp_path: Path) -> None:
    worker = SubprocessWorker("voyage.workers.director", tmp_path, tmp_path / "d.log")
    worker.start()
    try:
        result = worker.call(
            "decide",
            {
                "decision_index": 0,
                "current_concept": "a",
                "destination_concept": "b",
                "phase": "ESTABLISH",
                "style": "s",
            },
        )
        assert result["destination"]["canonical_name"] == "b"
        assert result["fallback"] is True
    finally:
        worker.stop()


def test_commit_one_segment_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        segment_id = supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert segment_id == "000000"
    segment = paths.segment_dir(run_dir, segment_id)
    for name in ("video.mp4", "audio.wav", "DONE", "sha256.json", "prompt_plan.json"):
        assert (segment / name).exists(), name
    state = read_state(run_dir)
    assert state.committed_segments == 1
    assert state.next_segment_number == 1
    assert state.timeline_frames == config.video.segment_frames
