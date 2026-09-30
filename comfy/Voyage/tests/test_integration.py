"""Integration tests: workers over RPC + full segment commit + validate.

Fake backends render real media with ffmpeg, so validation, checksums,
DONE markers, state advance, and finalizer concat are all exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import load_config
from voyage.media import FinalizeOptions, finalize_run, validate_video
from voyage.persistence import read_state
from voyage.rpc import SubprocessWorker
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path) -> None:
    initialize_run_directory(run_dir, run_id="itest", seed=7)


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
                # Explicit opt-out: the worker default is qwen since
                # 2026-09-29, so the deterministic path must say so.
                "backend": "deterministic",
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


def _committed_run(tmp_path: Path, segment_count: int) -> Path:
    """Commit `segment_count` fake segments; caller owns no workers after return."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        for _ in range(segment_count):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    return run_dir


def test_finalize_end_to_end_after_commit(tmp_path: Path) -> None:
    """Commit → finalize → valid presentation MP4 (issue 038 contract path)."""
    run_dir = _committed_run(tmp_path, 2)
    output_path = tmp_path / "final.mp4"
    assert finalize_run(run_dir, output_path) == output_path
    assert output_path.exists()
    info = validate_video(output_path, 1280, 720, 32)
    assert info["duration"] > 0


def test_finalize_options_explicit_joint_style(tmp_path: Path) -> None:
    """The `FinalizeOptions` path (issue 045) finalizes identically."""
    run_dir = _committed_run(tmp_path, 2)
    for joint_style in ("blend", "hard-splice"):
        output_path = tmp_path / f"final-{joint_style}.mp4"
        options = FinalizeOptions(joint_style=joint_style)  # type: ignore[arg-type]
        assert finalize_run(run_dir, output_path, options=options) == output_path
        info = validate_video(output_path, 1280, 720, 32)
        assert info["duration"] > 0


def test_finalize_options_rejects_unknown_joint_style() -> None:
    with pytest.raises(ValueError, match="joint_style"):
        FinalizeOptions(joint_style="crossfade-everything")  # type: ignore[arg-type]
