"""Crash-matrix tests: kill each worker, crash mid-op, repeated crashes, and
supervisor-SIGKILL mid-commit states (Phase 6 slice C, DESIGN §69).

All run against fake backends (real media, no GPU) in-container. SIGKILL of
the supervisor itself cannot be tested in-process, so the mid-commit states
a dead supervisor would leave behind (partial dir without DONE; DONE without
a state advance) are crafted on disk and the retry path must heal them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voyage import paths
from voyage.cli import validate_run
from voyage.config import default_config_toml, load_config
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_state,
    write_manifest,
    write_state,
)
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "crash-matrix") -> None:
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


def _restart_events(run_dir: Path, worker: str) -> int:
    metrics = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    return sum(
        1
        for line in metrics.splitlines()
        if '"event": "worker_restart"' in line and f'"worker": "{worker}"' in line
    )


def test_killed_audio_worker_recovers(tmp_path: Path) -> None:
    """SIGKILLed audio worker restarts and the segment still commits."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.inject_worker_crash("audio")
        segment_id = supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert segment_id == "000000"
    assert (paths.segment_dir(run_dir, segment_id) / paths.DONE_MARKER).exists()
    assert _restart_events(run_dir, "audio") == 1
    assert validate_run(run_dir) == []


def test_killed_director_worker_recovers(tmp_path: Path) -> None:
    """SIGKILLed director worker restarts and the segment still commits."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.inject_worker_crash("director")
        segment_id = supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert segment_id == "000000"
    assert (paths.segment_dir(run_dir, segment_id) / paths.DONE_MARKER).exists()
    assert _restart_events(run_dir, "director") == 1
    assert validate_run(run_dir) == []


def test_crash_during_video_generate_recovers(tmp_path: Path) -> None:
    """A SIGKILL landing mid-`generate_blocks` restarts and retries the op."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        orig_call = supervisor._video.call
        crashed: list[str] = []

        def _crash_once(op: str, payload: dict[str, Any]) -> dict[str, Any]:
            if op == "generate_blocks" and not crashed:
                crashed.append(op)
                supervisor.inject_worker_crash("video")
            return orig_call(op, payload)

        supervisor._video.call = _crash_once  # type: ignore[method-assign]
        segment_id = supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert crashed == ["generate_blocks"]
    assert segment_id == "000000"
    assert (paths.segment_dir(run_dir, segment_id) / paths.DONE_MARKER).exists()
    assert _restart_events(run_dir, "video") == 1
    assert validate_run(run_dir) == []


def test_repeated_crashes_recover_without_state_leak(tmp_path: Path) -> None:
    """Two straight SIGKILLs recover; every segment stays valid."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.inject_worker_crash("video")
        assert supervisor.commit_one_segment() == "000000"
        supervisor.inject_worker_crash("video")
        assert supervisor.commit_one_segment() == "000001"
    finally:
        supervisor.stop_workers()
    assert _restart_events(run_dir, "video") == 2
    assert validate_run(run_dir) == []


def test_partial_segment_without_done_is_reused(tmp_path: Path) -> None:
    """A dead supervisor's partial dir (truncated media, no DONE) is healed."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    segment = paths.segment_dir(run_dir, "000000")
    segment.mkdir(parents=True, exist_ok=True)
    (segment / "video.mp4").write_bytes(b"truncated mid-ffmpeg")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    assert (segment / paths.DONE_MARKER).exists()
    assert validate_run(run_dir) == []


def test_done_without_state_advance_heals_on_retry(tmp_path: Path) -> None:
    """DONE renamed but state.json not advanced: the retry re-commits cleanly."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    segment = paths.segment_dir(run_dir, "000000")
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_bytes(b"")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    state = read_state(run_dir)
    assert state.committed_segments == 1
    assert state.next_segment_number == 1
    assert validate_run(run_dir) == []


def test_director_embed_degrades_after_crash(tmp_path: Path) -> None:
    """A dead director worker makes novelty embedding fall back, never raise."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.inject_worker_crash("director")
        assert supervisor._embed_texts(["a misty harbor"]) is None
    finally:
        supervisor.stop_workers()
