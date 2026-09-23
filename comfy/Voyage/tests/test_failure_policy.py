"""Failure-policy tests: restart budget/circuit-breaker, RPC timeout,
PAUSED_DISK_FULL, resume retry, finalize space preflight (Phase 6 slice B).

All run against fake backends (real media, no GPU) in-container.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voyage import paths
from voyage.config import default_config_toml, load_config
from voyage.errors import (
    DiskSpaceError,
    FatalWorkerError,
    RecoverableWorkerError,
    VoyageError,
)
from voyage.media import finalize_run
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_state,
    write_manifest,
    write_state,
)
from voyage.rpc import SubprocessWorker
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "failure-policy") -> None:
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


def _always_fail(op: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    if op == "init":
        return {}
    raise RecoverableWorkerError(f"boom in {op}")


def test_restart_budget_exhaustion_opens_circuit_breaker(tmp_path: Path) -> None:
    """Beyond max_worker_restarts the breaker opens: Fatal, no more restarts."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config.voyage.max_worker_restarts = 2
    supervisor = Supervisor(run_dir, config)

    calls: list[str] = []

    def _flaky(op: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
        calls.append(op)
        raise RecoverableWorkerError("always fails")

    supervisor._video.call = _flaky  # type: ignore[method-assign]
    supervisor._video.restart = lambda: calls.append("restart")  # type: ignore[method-assign]
    with pytest.raises(FatalWorkerError, match="circuit breaker"):
        supervisor._call_with_restart(supervisor._video, "video", "000000", "generate_blocks", {})
    assert calls.count("restart") == 2
    assert calls.count("generate_blocks") == 3  # 1 initial + 2 retries
    metrics = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    assert metrics.count('"event": "worker_restart"') == 2
    assert '"event": "circuit_breaker_open"' in metrics


def test_zero_budget_fails_fast_without_restart(tmp_path: Path) -> None:
    """max_worker_restarts=0: first recoverable failure trips the breaker."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config.voyage.max_worker_restarts = 0
    supervisor = Supervisor(run_dir, config)

    restarts: list[str] = []
    supervisor._video.call = _always_fail  # type: ignore[method-assign]
    supervisor._video.restart = lambda: restarts.append("restart")  # type: ignore[method-assign]
    with pytest.raises(FatalWorkerError, match="circuit breaker"):
        supervisor._call_with_restart(supervisor._video, "video", "000000", "generate_blocks", {})
    assert restarts == []


def test_repeated_failure_aborts_failed_not_running(tmp_path: Path) -> None:
    """A run killed by repeated worker failure rests at FAILED (never RUNNING)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config.voyage.max_worker_restarts = 0
    supervisor = Supervisor(run_dir, config)
    supervisor._video.call = _always_fail  # type: ignore[method-assign]
    supervisor._audio.call = _always_fail  # type: ignore[method-assign]
    supervisor._director.call = _always_fail  # type: ignore[method-assign]
    with pytest.raises(VoyageError):
        supervisor.run_segments(1)
    state = read_state(run_dir)
    assert state.status == "FAILED"
    assert state.last_error
    assert "circuit breaker" in state.last_error


def test_disk_full_pauses_run_and_resumes(tmp_path: Path) -> None:
    """DiskSpaceError rests the run at PAUSED_DISK_FULL; freeing space resumes."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config.min_free_space_gib = 1e12
    with pytest.raises(DiskSpaceError):
        Supervisor(run_dir, config).run_segments(1)
    paused = read_state(run_dir)
    assert paused.status == "PAUSED_DISK_FULL"
    assert paused.last_error

    config.min_free_space_gib = 5.0
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    assert read_state(run_dir).status == "PAUSED"


def test_worker_call_times_out_on_silent_worker() -> None:
    """No worker output within `timeout` → RecoverableWorkerError (restart-class)."""
    read_fd, write_fd = os.pipe()
    try:
        worker = SubprocessWorker.__new__(SubprocessWorker)
        worker._module = "silent-test-worker"
        worker._proc = SimpleNamespace(  # type: ignore[assignment]
            stdin=SimpleNamespace(write=lambda data: len(data), flush=lambda: None),
            stdout=read_fd,
        )
        worker._counter = 0
        worker._timeout = 600.0
        started = time.monotonic()
        with pytest.raises(RecoverableWorkerError, match="timed out"):
            worker.call("health", {}, timeout=0.2)
        assert time.monotonic() - started < 10.0
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_resume_failure_gets_second_chance(tmp_path: Path) -> None:
    """A failed video resume retries after restart instead of aborting at once."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    segment = paths.segment_dir(run_dir, "000000")
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_bytes(b"done")
    (segment / "recovery.pt").write_bytes(b"tape")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)

    calls: list[str] = []

    def _flaky_resume(op: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
        calls.append(op)
        if op == "resume" and calls.count("resume") == 1:
            raise RecoverableWorkerError("resume boom")
        return {}

    supervisor._video.call = _flaky_resume  # type: ignore[method-assign]
    supervisor._video.restart = lambda: calls.append("restart")  # type: ignore[method-assign]
    supervisor._resume_video_worker("000001")
    assert calls.count("resume") == 2
    assert calls.count("restart") == 1
    metrics = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "video_resumed"' in metrics


def test_resume_failures_consume_the_same_budget(tmp_path: Path) -> None:
    """Resume retries share the per-worker budget, then the breaker opens."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    segment = paths.segment_dir(run_dir, "000000")
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_bytes(b"done")
    (segment / "recovery.pt").write_bytes(b"tape")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config.voyage.max_worker_restarts = 2
    supervisor = Supervisor(run_dir, config)

    calls: list[str] = []

    def _dead_resume(op: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
        calls.append(op)
        raise RecoverableWorkerError("resume always fails")

    supervisor._video.call = _dead_resume  # type: ignore[method-assign]
    supervisor._video.restart = lambda: calls.append("restart")  # type: ignore[method-assign]
    with pytest.raises(FatalWorkerError, match="circuit breaker"):
        supervisor._resume_video_worker("000001")
    assert calls.count("resume") == 3  # 1 initial + 2 budget restarts
    assert calls.count("restart") == 2


def test_finalize_space_preflight(tmp_path: Path) -> None:
    """finalize_run refuses to start when the free-space reserve is crossed."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    output = tmp_path / "final.mp4"
    with pytest.raises(DiskSpaceError):
        finalize_run(run_dir, output, min_free_space_gib=1e12)
    assert not output.exists()
    assert finalize_run(run_dir, output).exists()


def test_failure_policy_config_plumbing(tmp_path: Path) -> None:
    """TOML carries the new knobs; Supervisor hands the timeout to workers."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert config.voyage.max_worker_restarts == 3
    assert config.voyage.rpc_timeout_seconds == 600.0
    supervisor = Supervisor(run_dir, config)
    assert supervisor._video._timeout == 600.0
    assert supervisor._audio._timeout == 600.0
    assert supervisor._director._timeout == 600.0
