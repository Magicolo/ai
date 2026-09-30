"""RPC timeout resync regression tests (issue 137).

Why this file exists: after an RPC timeout the late response bytes stay in
the pipe, so the next call reads a stale line and dies Fatal on the id
check — a recoverable slowness becomes a non-retryable Fatal outside the
restart budget. These tests pin the resync contract with stub workers
(pipes + writer threads, no GPU, no network): a slow-then-correct worker
must let the call after a timeout succeed, while a genuinely unexpected
(future/unparseable) id must still fail fast as Fatal.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from voyage.errors import FatalWorkerError, RecoverableWorkerError
from voyage.rpc import SubprocessWorker


def _pipe_worker(module: str, read_fd: int) -> SubprocessWorker:
    """Worker handle reading responses from a raw pipe fd (issue 001 pattern)."""
    worker = SubprocessWorker.__new__(SubprocessWorker)
    worker._module = module
    worker._proc = SimpleNamespace(  # type: ignore[assignment]
        stdin=SimpleNamespace(write=lambda data: len(data), flush=lambda: None),
        stdout=read_fd,
        poll=lambda: None,
    )
    worker._counter = 0
    worker._timeout = 600.0
    worker._call_lock = threading.Lock()
    worker._log_file = None
    return worker


def _result_line(request_id: str, result: dict[str, Any]) -> bytes:
    """One well-formed success response line for `request_id`."""
    return (json.dumps({"id": request_id, "ok": True, "result": result}) + "\n").encode()


def test_timeout_then_next_call_succeeds_no_fatal() -> None:
    """A late full line from a timed-out call is discarded, not Fatal (137)."""
    read_fd, write_fd = os.pipe()
    worker = _pipe_worker("slow-then-correct-worker", read_fd)

    def _writer() -> None:
        # Lands after the first call's 0.3 s deadline, during the second call.
        time.sleep(0.8)
        os.write(write_fd, _result_line("req-000001", {}))
        time.sleep(0.3)
        os.write(write_fd, _result_line("req-000002", {"answer": 1}))

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    try:
        with pytest.raises(RecoverableWorkerError, match="timed out"):
            worker.call("health", {}, timeout=0.3)
        assert worker.call("health", {}, timeout=10.0) == {"answer": 1}
    finally:
        writer.join(timeout=15.0)
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)


def test_two_stale_lines_both_discarded() -> None:
    """Two consecutive timeouts leave two late lines; the third call wins (137)."""
    read_fd, write_fd = os.pipe()
    worker = _pipe_worker("twice-slow-worker", read_fd)

    def _writer() -> None:
        time.sleep(0.8)
        os.write(write_fd, _result_line("req-000001", {}))
        time.sleep(0.3)
        os.write(write_fd, _result_line("req-000002", {}))
        time.sleep(0.3)
        os.write(write_fd, _result_line("req-000003", {"answer": 2}))

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    try:
        with pytest.raises(RecoverableWorkerError, match="timed out"):
            worker.call("health", {}, timeout=0.3)
        with pytest.raises(RecoverableWorkerError, match="timed out"):
            worker.call("health", {}, timeout=0.3)
        assert worker.call("health", {}, timeout=10.0) == {"answer": 2}
    finally:
        writer.join(timeout=15.0)
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)


def test_future_id_still_fatal() -> None:
    """A response from the future is a protocol break, not a stale line (137)."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, _result_line("req-000002", {}))
        worker = _pipe_worker("future-id-worker", read_fd)
        with pytest.raises(FatalWorkerError, match="id mismatch"):
            worker.call("health", {}, timeout=5.0)
    finally:
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)


def test_garbage_id_still_fatal() -> None:
    """An unparseable id is a protocol break, not a stale line (137)."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, _result_line("not-a-request-id", {}))
        worker = _pipe_worker("garbage-id-worker", read_fd)
        with pytest.raises(FatalWorkerError, match="id mismatch"):
            worker.call("health", {}, timeout=5.0)
    finally:
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)
