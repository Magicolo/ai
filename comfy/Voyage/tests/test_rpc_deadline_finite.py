"""Issue 100: non-finite timeouts fail as VoyageError, never raw select errors.

Why this file exists: `positive_seconds` rejected only `value <= 0`, and
IEEE 754 `nan <= 0` / `inf <= 0` are both False — so NaN/inf slipped
into `SubprocessWorker._timeout` and reached
`select.select(..., nan/inf)` as raw `ValueError`/`OverflowError`,
bypassing the supervisor's branch-on-class restart path. These tests pin
the two-layer fix with pipe doubles (no GPU, no subprocess): the config
validator rejects non-finite durations, and `call()` /
`_read_response_line` reject non-finite/non-positive effective timeouts
as `RecoverableWorkerError`.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from voyage.config import VoyageConfig
from voyage.errors import RecoverableWorkerError, VoyageError
from voyage.rpc import SubprocessWorker


def _pipe_worker(module: str, read_end: int) -> SubprocessWorker:
    """Worker handle reading responses from a raw pipe end (issue 001 pattern)."""
    worker = SubprocessWorker.__new__(SubprocessWorker)
    worker._module = module
    worker._proc = SimpleNamespace(  # type: ignore[assignment]
        stdin=SimpleNamespace(write=lambda data: len(data), flush=lambda: None),
        stdout=read_end,
        poll=lambda: None,
    )
    worker._counter = 0
    worker._timeout = 600.0
    worker._call_lock = threading.Lock()
    worker._log_file = None
    return worker


@pytest.mark.parametrize("hostile_value", [float("nan"), float("inf"), float("-inf")])
def test_voyage_config_rejects_non_finite_rpc_timeout(hostile_value: float) -> None:
    """NaN/inf never reach the worker timeout (the RPC-deadline leg of 100)."""
    with pytest.raises(ValidationError):
        VoyageConfig(rpc_timeout_seconds=hostile_value)


def test_voyage_config_keeps_finite_positive_contract() -> None:
    """The finite boundary is unchanged: positive passes, zero/negative fail."""
    assert VoyageConfig(rpc_timeout_seconds=600.0).rpc_timeout_seconds == 600.0
    assert VoyageConfig(rpc_timeout_seconds=0.5).rpc_timeout_seconds == 0.5
    with pytest.raises(ValidationError):
        VoyageConfig(rpc_timeout_seconds=0.0)
    with pytest.raises(ValidationError):
        VoyageConfig(rpc_timeout_seconds=-5.0)


@pytest.mark.parametrize("hostile_value", [float("nan"), float("inf"), float("-inf")])
def test_call_rejects_non_finite_per_call_timeout(hostile_value: float) -> None:
    """A caller-supplied NaN/inf raises VoyageError without touching the pipe."""
    read_end, write_end = os.pipe()
    worker = _pipe_worker("finite-timeout-probe", read_end)
    try:
        with pytest.raises(RecoverableWorkerError):
            worker.call("health", {}, timeout=hostile_value)
    finally:
        with contextlib.suppress(OSError):
            os.close(write_end)
        with contextlib.suppress(OSError):
            os.close(read_end)


@pytest.mark.parametrize("hostile_value", [float("nan"), float("inf"), float("-inf")])
def test_call_rejects_non_finite_stored_timeout(hostile_value: float) -> None:
    """A programmatically poisoned stored timeout fails in-taxonomy too."""
    read_end, write_end = os.pipe()
    worker = _pipe_worker("finite-stored-probe", read_end)
    worker._timeout = hostile_value
    try:
        with pytest.raises(VoyageError):
            worker.call("health", {})
    finally:
        with contextlib.suppress(OSError):
            os.close(write_end)
        with contextlib.suppress(OSError):
            os.close(read_end)


@pytest.mark.parametrize("hostile_value", [0.0, -1.0, float("nan"), float("inf")])
def test_call_rejects_non_positive_timeout(hostile_value: float) -> None:
    """Zero/negative join non-finite as Recoverable (never a hung or raw error)."""
    read_end, write_end = os.pipe()
    worker = _pipe_worker("positive-timeout-probe", read_end)
    try:
        with pytest.raises(RecoverableWorkerError):
            worker.call("health", {}, timeout=hostile_value)
    finally:
        with contextlib.suppress(OSError):
            os.close(write_end)
        with contextlib.suppress(OSError):
            os.close(read_end)


@pytest.mark.parametrize("hostile_value", [float("nan"), float("inf")])
def test_read_response_line_rejects_non_finite_directly(hostile_value: float) -> None:
    """The deadline reader itself refuses NaN/inf (defense past `call`)."""
    read_end, write_end = os.pipe()
    worker = _pipe_worker("finite-reader-probe", read_end)
    try:
        with pytest.raises(RecoverableWorkerError):
            worker._read_response_line(worker._proc, "health", hostile_value)  # type: ignore[arg-type]
    finally:
        with contextlib.suppress(OSError):
            os.close(write_end)
        with contextlib.suppress(OSError):
            os.close(read_end)


@pytest.mark.parametrize("hostile_value", [True, False])
def test_call_rejects_bool_timeout_282(hostile_value: bool) -> None:
    """Issue 282: `True` must not pass as a 1 s deadline (bool is int subclass)."""
    read_end, write_end = os.pipe()
    worker = _pipe_worker("bool-timeout-probe", read_end)
    try:
        with pytest.raises(RecoverableWorkerError):
            worker.call("health", {}, timeout=hostile_value)
    finally:
        with contextlib.suppress(OSError):
            os.close(write_end)
        with contextlib.suppress(OSError):
            os.close(read_end)


def test_nan_rpc_timeout_rejected_by_validator(tmp_path: Path) -> None:
    """Non-finite `rpc_timeout_seconds` is rejected by the validator."""
    with pytest.raises(ValidationError):
        VoyageConfig(rpc_timeout_seconds=float("nan"))
