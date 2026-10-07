"""Track D RPC/worker liveness helpers (DESIGN §§45-46, §12).

CPU-only: stub workers (pipes + doubles, no GPU, no network) pin the
cancel/refresh/retry/watchdog/lease/spawn contracts Track A wires.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voyage.errors import FatalWorkerError, RecoverableWorkerError
from voyage.rpc import (
    CANCEL_OP,
    SubprocessWorker,
    call_cancel,
    call_with_retry_once,
    cuda1_idle_fn,
    cuda_visible_devices_for_session_device,
    decide_timeout_seconds,
    hold_cuda1_lease,
    is_wedged,
    restart_and_resume_single_charge,
    restart_with_budget,
    safe_stop_worker,
    stop_all_with_budget,
    worker_restart_metric,
)


def _stub_worker(module: str = "stub-worker") -> SubprocessWorker:
    """Bare handle via __new__ (mirrors test_rpc_timeout legacy doubles)."""
    worker = SubprocessWorker.__new__(SubprocessWorker)
    worker._module = module
    worker._proc = None
    worker._counter = 0
    worker._timeout = 600.0
    worker._call_lock = threading.Lock()
    worker._log_file = None
    worker._init_payload = {}
    worker._alternate_executable = None
    return worker


def test_cancel_op_name_is_stable() -> None:
    assert CANCEL_OP == "cancel"


def test_call_cancel_on_idle_worker_returns_false() -> None:
    worker = _stub_worker()
    assert call_cancel(worker) is False
    assert worker.cancel() is False


def test_cancel_closes_stdin_and_stops() -> None:
    closed: list[str] = []

    class _Stdin:
        def close(self) -> None:
            closed.append("closed")

    class _Proc:
        stdin: Any = _Stdin()
        stdout: Any = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def poll(self) -> int | None:
            return 1

    worker = _stub_worker()
    worker._proc = _Proc()  # type: ignore[assignment]
    assert worker.cancel() is True
    assert closed
    assert worker._proc is None


def test_restart_with_budget_charges_single_unit() -> None:
    worker = _stub_worker()
    starts: list[str] = []
    stops: list[str] = []
    worker.stop = lambda: stops.append("stop")  # type: ignore[method-assign]
    worker.start = lambda: starts.append("start")  # type: ignore[method-assign]
    remaining = restart_with_budget(worker, 3, op="generate_blocks", worker_name="video")
    assert remaining == 2
    assert starts == ["start"]
    assert stops == ["stop"]


def test_restart_with_budget_exhausted_raises_without_restart() -> None:
    worker = _stub_worker()
    called: list[str] = []
    worker.restart = lambda: called.append("restart")  # type: ignore[method-assign]
    with pytest.raises(FatalWorkerError, match="exhausted"):
        restart_with_budget(worker, 0)
    assert called == []


def test_restart_with_budget_rejects_bool_budget() -> None:
    worker = _stub_worker()
    with pytest.raises(FatalWorkerError, match="non-int budget"):
        restart_with_budget(worker, True)


def test_restart_increments_restart_count() -> None:
    worker = _stub_worker()
    worker.stop = lambda: None  # type: ignore[method-assign]
    worker.start = lambda: None  # type: ignore[method-assign]
    assert worker.restart_count == 0
    worker.restart()
    assert worker.restart_count == 1
    worker.restart()
    assert worker.restart_count == 2


def test_call_with_retry_once_succeeds_without_restart() -> None:
    worker = _stub_worker()
    worker.call = lambda op, payload, timeout=None: {"ok": True}  # type: ignore[method-assign]
    restarts: list[str] = []
    worker.restart = lambda: restarts.append("restart")  # type: ignore[method-assign]
    result = call_with_retry_once(worker, "generate_sfx", {})
    assert result == {"ok": True}
    assert restarts == []


def test_call_with_retry_once_restarts_once_then_retries() -> None:
    worker = _stub_worker()
    calls: list[str] = []
    restarts: list[str] = []
    hooks: list[str] = []

    def _call(op: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        del op, payload, timeout
        calls.append("call")
        if len(calls) == 1:
            raise RecoverableWorkerError("transient")
        return {"recovered": True}

    worker.call = _call  # type: ignore[method-assign]
    worker.restart = lambda: restarts.append("restart")  # type: ignore[method-assign]
    result = call_with_retry_once(
        worker, "generate_audio", {}, restart_hook=lambda: hooks.append("hook")
    )
    assert result == {"recovered": True}
    assert restarts == ["restart"]
    assert hooks == ["hook"]
    assert len(calls) == 2


def test_call_with_retry_once_second_failure_propagates() -> None:
    worker = _stub_worker()

    def _always_recoverable(op: str, payload: dict[str, Any], timeout: float | None = None) -> Any:
        del op, payload, timeout
        raise RecoverableWorkerError("still down")

    worker.call = _always_recoverable  # type: ignore[method-assign]
    worker.restart = lambda: None  # type: ignore[method-assign]
    with pytest.raises(RecoverableWorkerError):
        call_with_retry_once(worker, "generate_audio", {})


def test_restart_and_resume_single_charge_shares_one_unit() -> None:
    worker = _stub_worker()
    worker.stop = lambda: None  # type: ignore[method-assign]
    worker.start = lambda: None  # type: ignore[method-assign]
    hooks: list[str] = []
    remaining = restart_and_resume_single_charge(
        worker, 2, lambda: hooks.append("resume"), op="resume", worker_name="video"
    )
    assert remaining == 1
    assert hooks == ["resume"]
    assert worker.restart_count == 1


def test_is_wedged_threshold() -> None:
    assert is_wedged(0.0, 400.0) is True
    assert is_wedged(0.0, 100.0) is False
    assert is_wedged(None, 100.0) is False
    assert is_wedged(0.0, 10.0, threshold_seconds=5.0) is True


def test_decide_timeout_pins_to_60_120() -> None:
    assert decide_timeout_seconds(None) == 60.0
    assert decide_timeout_seconds(30.0) == 60.0
    assert decide_timeout_seconds(90.0) == 90.0
    assert decide_timeout_seconds(600.0) == 120.0
    assert decide_timeout_seconds(float("nan")) == 60.0
    assert decide_timeout_seconds(True) == 60.0


def test_cuda_visible_devices_mapping() -> None:
    assert cuda_visible_devices_for_session_device("cuda:0") == "0"
    assert cuda_visible_devices_for_session_device("cuda:1") == "1"
    assert cuda_visible_devices_for_session_device("cuda") == "0"
    assert cuda_visible_devices_for_session_device("cpu") is None
    assert cuda_visible_devices_for_session_device(None) is None
    assert cuda_visible_devices_for_session_device("garbage") is None


def test_spawn_env_masks_cuda_unless_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from voyage.rpc import _spawn_env

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    env = _spawn_env(None, "cuda:1")
    assert env is not None
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    # Explicit override wins.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env2 = _spawn_env(None, "cuda:1")
    assert env2 is not None
    assert env2["CUDA_VISIBLE_DEVICES"] == "0"
    # Legacy shape preserved.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _spawn_env(None) is None


def test_cuda1_lease_idle_and_hold() -> None:
    assert cuda1_idle_fn() is True
    with hold_cuda1_lease():
        assert cuda1_idle_fn() is False
    assert cuda1_idle_fn() is True
    with hold_cuda1_lease(), pytest.raises(RuntimeError, match="busy"), hold_cuda1_lease():
        pass


def test_safe_stop_never_raises() -> None:
    worker = _stub_worker()
    assert safe_stop_worker(worker) is False

    class _FailProc:
        stdin: Any = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            raise OSError("reaped")

        def poll(self) -> int | None:
            return 1

    worker._proc = _FailProc()  # type: ignore[assignment]
    # stop() swallows TimeoutExpired internally; OSError from wait is not
    # swallowed by stop() itself — safe_stop must still not raise.
    assert safe_stop_worker(worker) in (True, False)


def test_stop_all_with_budget_maps_modules() -> None:
    first = _stub_worker("video")
    second = _stub_worker("audio")
    outcomes = stop_all_with_budget([first, second])
    assert outcomes == {"video": False, "audio": False}


def test_worker_restart_metric_shape() -> None:
    metric = worker_restart_metric(
        worker="video", op="generate_blocks", attempt=1, budget=3, reason="boom"
    )
    assert metric["event"] == "worker_restart"
    assert metric["worker"] == "video"
    assert metric["attempt"] == 1


def test_start_passes_session_device_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from voyage.rpc import SubprocessWorker

    captured: dict[str, Any] = {}

    class _StubProc:
        stdin: Any = None
        stdout: Any = None
        pid = 4242

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def _capture(*args: Any, **kwargs: Any) -> _StubProc:
        captured.update(kwargs)
        return _StubProc()

    monkeypatch.setattr(subprocess, "Popen", _capture)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    worker = SubprocessWorker(
        "voyage.workers.audio",
        tmp_path,
        tmp_path / "w.log",
        init_op=None,
        init_payload={"device": "cuda:1"},
        executable=None,
    )
    # init_op=None so no handshake call; start only exercises Popen env.
    worker._executable = "/usr/bin/python3"
    worker._alternate_executable = None
    worker.start()
    try:
        env = captured.get("env")
        assert isinstance(env, dict)
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
    finally:
        worker.stop()


def test_cancel_helper_is_stable_alias() -> None:
    worker = _stub_worker()
    assert call_cancel(worker) is False
    proc = SimpleNamespace(stdin=SimpleNamespace(close=lambda: None), stdout=None, poll=lambda: 1)
    worker._proc = proc  # type: ignore[assignment]
    original_stop = worker.stop
    try:
        worker.stop = lambda: setattr(worker, "_proc", None)  # type: ignore[method-assign]
        assert call_cancel(worker) is True
    finally:
        worker.stop = original_stop  # type: ignore[method-assign]


def test_explicit_cuda_env_wins_over_session_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from voyage.rpc import SubprocessWorker

    captured: dict[str, Any] = {}

    class _StubProc:
        stdin: Any = None
        stdout: Any = None
        pid = 4243

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def _capture(*args: Any, **kwargs: Any) -> _StubProc:
        captured.update(kwargs)
        return _StubProc()

    monkeypatch.setattr(subprocess, "Popen", _capture)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    worker = SubprocessWorker(
        "voyage.workers.audio",
        tmp_path,
        tmp_path / "w.log",
        init_op=None,
        init_payload={"device": "cuda:1"},
    )
    worker.start()
    try:
        env = captured.get("env")
        # Explicit override: no mask key added beyond the inherited one.
        assert env is None or env.get("CUDA_VISIBLE_DEVICES") == "0"
    finally:
        worker.stop()
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_os_environ_untouched_by_spawn() -> None:
    before = dict(os.environ)
    from voyage.rpc import _spawn_env

    _spawn_env("/opt/venvs/director/bin/python", "cuda:1")
    assert dict(os.environ) == before
