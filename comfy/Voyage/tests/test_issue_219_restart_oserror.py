"""Issue 219: OSError from worker.restart() routes through the restart budget.

Popen failures (ENOENT/EACCES/ENOMEM) must consume the same budget as
RecoverableWorkerError, emit worker_restart_failed, and trip
circuit_breaker_open — never escape raw. FatalWorkerError from the hook
still propagates untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.errors import FatalWorkerError, RecoverableWorkerError
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path) -> None:
    initialize_run_directory(run_dir, run_id="issue-219")


def _metrics_text(run_dir: Path) -> str:
    return (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")


def _always_failing_call(
    op: str, payload: dict[str, Any], timeout: float = 600.0
) -> dict[str, Any]:
    raise RecoverableWorkerError(f"boom in {op}")


def test_restart_oserror_consumes_budget_and_trips_breaker(tmp_path: Path) -> None:
    """OSError on restart counts against the budget like Recoverable (219)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    config.voyage.max_worker_restarts = 3
    supervisor = Supervisor(run_dir, config)
    supervisor._video.call = _always_failing_call  # type: ignore[assignment]

    def _dead_spawn() -> None:
        raise OSError(2, "No such file or directory")

    supervisor._video.restart = _dead_spawn  # type: ignore[method-assign]
    with pytest.raises(FatalWorkerError, match="circuit breaker"):
        supervisor._call_with_restart(supervisor._video, "video", "000000", "generate_blocks", {})
    assert supervisor._restarts["video"] == 3
    metrics = _metrics_text(run_dir)
    assert '"event": "worker_restart_failed"' in metrics
    assert '"event": "circuit_breaker_open"' in metrics


def test_restart_oserror_then_success_recovers(tmp_path: Path) -> None:
    """One OSError restart followed by a healthy call returns normally (219)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    config.voyage.max_worker_restarts = 3
    supervisor = Supervisor(run_dir, config)
    calls: list[str] = []

    def _flaky_call(op: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
        calls.append(op)
        if len(calls) == 1:
            raise RecoverableWorkerError("first call boom")
        return {"answer": 1}

    def _flaky_restart() -> None:
        calls.append("restart")
        if calls.count("restart") == 1:
            raise OSError(12, "Cannot allocate memory")

    supervisor._video.call = _flaky_call  # type: ignore[assignment]
    supervisor._video.restart = _flaky_restart  # type: ignore[method-assign]
    assert supervisor._call_with_restart(supervisor._video, "video", "000000", "op", {}) == {
        "answer": 1
    }
    assert supervisor._restarts["video"] == 2
    assert '"event": "worker_restart_failed"' in _metrics_text(run_dir)


def test_fatal_hook_still_propagates_past_oserror_gate(tmp_path: Path) -> None:
    """FatalWorkerError from restart_hook is not swallowed by the OSError gate (219)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor._video.call = _always_failing_call  # type: ignore[assignment]
    supervisor._video.restart = lambda: None  # type: ignore[method-assign]

    def _fatal_hook(segment_id: str) -> None:
        raise FatalWorkerError("hook refuses")

    with pytest.raises(FatalWorkerError, match="hook refuses"):
        supervisor._call_with_restart(
            supervisor._video, "video", "000000", "generate_blocks", {}, _fatal_hook
        )
