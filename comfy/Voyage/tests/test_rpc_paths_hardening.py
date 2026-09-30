"""Supervisor-side RPC + stored-path hardening (issues 007, 015).

Why this file exists: the worker side of 007 (`checked_request` types,
error-class wire codes, MALFORMED) is pinned in `test_commit_hardening.py`,
but the supervisor side of the wire had no pins — `SubprocessWorker.call`
accepted any `op`/`payload` and raw pydantic errors escaped the
branch-on-class taxonomy, and nothing verified the
`retryable=False -> Fatal` mapping. Likewise `resolve_stored_path`
returned `..` escapes and outside-the-run absolutes without complaint
(015). These tests pin the hardened contract with stub workers (pipes,
no GPU) and `tmp_path` runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voyage.errors import FatalWorkerError, MediaError, RecoverableWorkerError
from voyage.paths import resolve_stored_path
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


def _failure_line(request_id: str, code: str, retryable: bool) -> bytes:
    """One well-formed failure response line for `request_id`."""
    payload = {
        "id": request_id,
        "ok": False,
        "error": {"code": code, "message": "boom", "retryable": retryable},
    }
    return (json.dumps(payload) + "\n").encode()


def test_call_rejects_non_str_op_as_fatal() -> None:
    """A non-str op fails fast as Fatal, not as a raw pydantic error (007)."""
    read_fd, write_fd = os.pipe()
    worker = _pipe_worker("strict-op-worker", read_fd)
    try:
        bad_op: Any = 123
        with pytest.raises(FatalWorkerError, match="op must be str"):
            worker.call(bad_op, {})
    finally:
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)


def test_call_rejects_non_dict_payload_as_fatal() -> None:
    """A non-dict payload fails fast as Fatal, not as a raw error (007)."""
    read_fd, write_fd = os.pipe()
    worker = _pipe_worker("strict-payload-worker", read_fd)
    try:
        bad_payload: Any = ["not", "a", "dict"]
        with pytest.raises(FatalWorkerError, match="payload must be a dict"):
            worker.call("health", bad_payload)
    finally:
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)


def test_call_failure_retryable_true_is_recoverable() -> None:
    """Characterization: retryable worker failures map to Recoverable (007)."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, _failure_line("req-000001", "WORKER_ERROR", True))
        worker = _pipe_worker("retryable-worker", read_fd)
        with pytest.raises(RecoverableWorkerError, match="WORKER_ERROR"):
            worker.call("health", {}, timeout=5.0)
    finally:
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)


def test_call_failure_retryable_false_is_fatal() -> None:
    """Characterization: deterministic failures never burn restart budget (007)."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, _failure_line("req-000001", "INVALID_PAYLOAD", False))
        worker = _pipe_worker("fatal-worker", read_fd)
        with pytest.raises(FatalWorkerError, match="INVALID_PAYLOAD"):
            worker.call("health", {}, timeout=5.0)
    finally:
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)


def test_call_failure_without_error_detail_is_fatal() -> None:
    """Characterization: ok=False with no error detail fails safe as Fatal (007)."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, (json.dumps({"id": "req-000001", "ok": False}) + "\n").encode())
        worker = _pipe_worker("detail-less-worker", read_fd)
        with pytest.raises(FatalWorkerError, match="UNKNOWN"):
            worker.call("health", {}, timeout=5.0)
    finally:
        os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)


def test_resolve_stored_path_rejects_relative_dotdot_escape(tmp_path: Path) -> None:
    """A ledger `../evil.wav` resolves outside the run — fail loud (015)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(MediaError, match="escapes the run dir"):
        resolve_stored_path(run_dir, "../evil.wav")


def test_resolve_stored_path_rejects_nested_dotdot_escape(tmp_path: Path) -> None:
    """Lexical checks pass `segments/../../evil.wav` — resolve first (015)."""
    run_dir = tmp_path / "run"
    (run_dir / "segments").mkdir(parents=True)
    with pytest.raises(MediaError, match="escapes the run dir"):
        resolve_stored_path(run_dir, "segments/../../evil.wav")


def test_resolve_stored_path_rejects_existing_absolute_outside_run(
    tmp_path: Path,
) -> None:
    """A planted absolute path is untrusted even when it exists (015)."""
    planted = tmp_path / "planted.wav"
    planted.write_bytes(b"hostile")
    with pytest.raises(MediaError, match="escapes the run dir"):
        resolve_stored_path(tmp_path / "run", planted)


def test_resolve_stored_path_rejects_missing_absolute_outside_run(
    tmp_path: Path,
) -> None:
    """A missing absolute outside the run is not returned as-is (015)."""
    missing = tmp_path / "gone" / "take.wav"
    with pytest.raises(MediaError, match="escapes the run dir"):
        resolve_stored_path(tmp_path / "run", missing)


def test_resolve_stored_path_prefers_in_run_copy_over_existing_stale(
    tmp_path: Path,
) -> None:
    """Re-anchor wins over a stale absolute that still exists (015)."""
    run_dir = tmp_path / "run"
    target = run_dir / "audio" / "take_0000.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"fresh")
    stale = tmp_path / "old-home" / "audio" / "take_0000.wav"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    assert resolve_stored_path(run_dir, stale) == target


def test_resolve_stored_path_keeps_absolute_inside_run(tmp_path: Path) -> None:
    """A non-relocated absolute entry inside the run keeps working (015)."""
    run_dir = tmp_path / "run"
    inside = run_dir / "audio" / "take_0000.wav"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"data")
    assert resolve_stored_path(run_dir, inside) == inside


def test_resolve_stored_path_joins_relative_inside_run(tmp_path: Path) -> None:
    """Run-relative entries keep their lexical join (015 regression pin)."""
    run_dir = tmp_path / "run"
    assert (
        resolve_stored_path(run_dir, "audio/take_0000.wav") == run_dir / "audio" / "take_0000.wav"
    )


def test_resolve_stored_path_reanchors_moved_run(tmp_path: Path) -> None:
    """A stale absolute heals via its layout anchor when the copy exists (015)."""
    run_dir = tmp_path / "run"
    relocated = run_dir / "segments" / "000000" / "recovery.pt"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(b"tape")
    stale = Path("/old/home/x/segments/000000/recovery.pt")
    assert resolve_stored_path(run_dir, stale) == relocated
