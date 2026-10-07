"""Wire recoverability + advisory embed fallback (issues 201, 225).

Why this file exists: `voyage/workers/loop.py` once mapped every
`VoyageError` to `retryable=False`, demoting an explicit worker-side
`RecoverableWorkerError` to supervisor-side `FatalWorkerError` (201) —
and `Supervisor._embed_texts` caught only `VoyageError`, so a transport
`OSError` from the director RPC killed the commit instead of degrading
to the documented `None` fallback (225). These tests pin the fixed
contract end to end (DESIGN §§45-46 for the JSONL framing, §48 for the
error classes): recoverable stays recoverable across the wire, fatal
stays fatal, and advisory embedding never fails a commit on transport
noise.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.errors import FatalWorkerError, RecoverableWorkerError, VoyageError
from voyage.persistence import read_effective_config
from voyage.rpc import SubprocessWorker
from voyage.supervisor import Supervisor
from voyage.workers.loop import serve


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


def _serve_one_line(
    monkeypatch: pytest.MonkeyPatch, handler: Any, request_id: str = "req-000001"
) -> dict[str, Any]:
    """Run one `serve` dispatch and return the parsed response dict."""
    stdin_lines = io.StringIO(
        json.dumps({"id": request_id, "op": "generate_blocks", "payload": {}}) + "\n"
    )
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin_lines)
    monkeypatch.setattr(sys, "stdout", captured)
    serve({"generate_blocks": handler})
    lines = captured.getvalue().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_recoverable_worker_error_wire_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handler raising RecoverableWorkerError keeps retryable=True (201)."""

    def _transient(payload: dict[str, Any]) -> dict[str, Any]:
        raise RecoverableWorkerError("transient gpu oom, please restart me")

    response = _serve_one_line(monkeypatch, _transient)
    assert response["ok"] is False
    assert response["error"]["code"] == "RecoverableWorkerError"
    assert response["error"]["retryable"] is True


def test_fatal_worker_error_wire_stays_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new recoverable arm must not swallow FatalWorkerError (201)."""

    def _fatal(payload: dict[str, Any]) -> dict[str, Any]:
        raise FatalWorkerError("bad geometry, retrying cannot help")

    response = _serve_one_line(monkeypatch, _fatal)
    assert response["ok"] is False
    assert response["error"]["code"] == "FatalWorkerError"
    assert response["error"]["retryable"] is False


def test_generic_voyage_error_wire_stays_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain VoyageError still maps to retryable=False (201, 007 intact)."""

    def _deterministic(payload: dict[str, Any]) -> dict[str, Any]:
        raise VoyageError("deterministic plumbing failure")

    response = _serve_one_line(monkeypatch, _deterministic)
    assert response["ok"] is False
    assert response["error"]["code"] == "VoyageError"
    assert response["error"]["retryable"] is False


def test_recoverable_wire_maps_to_supervisor_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full chain: worker Recoverable → wire retryable → supervisor Recoverable.

    Feeds the exact bytes `serve` produced into a stub supervisor-side
    worker, so the test pins `rpc.py` routing the fixed wire code to a
    restart instead of `FatalWorkerError` (201).
    """
    stdin_lines = io.StringIO(
        json.dumps({"id": "req-000001", "op": "generate_blocks", "payload": {}}) + "\n"
    )
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin_lines)
    monkeypatch.setattr(sys, "stdout", captured)

    def _transient(payload: dict[str, Any]) -> dict[str, Any]:
        raise RecoverableWorkerError("transient gpu oom, please restart me")

    serve({"generate_blocks": _transient})
    wire_line = captured.getvalue().splitlines()[0]
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, (wire_line + "\n").encode())
        worker = _pipe_worker("recoverable-wire-worker", read_fd)
        with pytest.raises(RecoverableWorkerError, match="RecoverableWorkerError"):
            worker.call("generate_blocks", {}, timeout=5.0)
    finally:
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)


def _supervisor_with_director(tmp_path: Path, behavior: Any) -> tuple[Supervisor, Path]:
    """Supervisor whose director `call` runs `behavior(op, payload, timeout)`."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="recoverable-embed")
    supervisor = Supervisor(run_dir, read_effective_config(run_dir))
    supervisor._director.call = behavior  # type: ignore[method-assign]
    return supervisor, run_dir


def _fallback_events(run_dir: Path) -> list[dict[str, Any]]:
    """Parsed `director_embed_fallback` metric events, if any."""
    metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    events = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if '"director_embed_fallback"' in line:
            events.append(json.loads(line))
    return events


def test_embed_texts_oserror_degrades_to_fallback(tmp_path: Path) -> None:
    """A transport OSError degrades to None with a visible metric (225)."""

    def _broken(op: str, payload: dict[str, Any], timeout: float | None = None) -> Any:
        raise OSError("select boom")

    supervisor, run_dir = _supervisor_with_director(tmp_path, _broken)
    assert supervisor._embed_texts(["a misty harbor"]) is None
    events = _fallback_events(run_dir)
    assert len(events) == 1
    assert events[0]["error_class"] == "OSError"


def test_embed_texts_voyage_error_still_falls_back_with_metric(tmp_path: Path) -> None:
    """The original VoyageError path keeps its fallback, now metered (225)."""

    def _sick(op: str, payload: dict[str, Any], timeout: float | None = None) -> Any:
        raise RecoverableWorkerError("director wedged")

    supervisor, run_dir = _supervisor_with_director(tmp_path, _sick)
    assert supervisor._embed_texts(["a misty harbor"]) is None
    events = _fallback_events(run_dir)
    assert len(events) == 1
    assert events[0]["error_class"] == "RecoverableWorkerError"


def test_embed_texts_memory_error_propagates(tmp_path: Path) -> None:
    """MemoryError is never an advisory fallback — it propagates (225)."""

    def _oom(op: str, payload: dict[str, Any], timeout: float | None = None) -> Any:
        raise MemoryError("out of memory")

    supervisor, _ = _supervisor_with_director(tmp_path, _oom)
    with pytest.raises(MemoryError):
        supervisor._embed_texts(["a misty harbor"])
