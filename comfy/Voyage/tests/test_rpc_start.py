"""Worker start-handle hygiene tests (issue 170).

Why this file exists: `SubprocessWorker.start()` published `_proc` before
the init handshake, so a failed init left a stale handle (dead or
half-initialized child) plus an open log fd — the next attempt talked to
an uninitialized session instead of failing fast. These tests pin the
fence contract with CPU-only stub workers (nonexistent modules fail init
at process spawn, no GPU, no network): a failed start leaves `_proc`
None, the log fd closed, and `running` False, and a later start works
with no ghost of the failed attempt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.errors import RecoverableWorkerError
from voyage.rpc import SubprocessWorker


def test_failed_init_leaves_no_stale_handle(tmp_path: Path) -> None:
    """Failed init fences the handle: no proc, closed log, not running (170)."""
    worker = SubprocessWorker(
        "nonexistent_module_xyz_abc",
        tmp_path,
        tmp_path / "w.log",
        init_op="init",
        init_payload={},
        timeout=20.0,
    )
    with pytest.raises(RecoverableWorkerError, match="(closed stdout|pipe broken)"):
        worker.start()
    assert worker._proc is None
    assert worker.running is False
    assert worker._log_file is None


def test_retry_after_failed_init_starts_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start after a fenced failure mounts a fresh handle, no ghost (170)."""
    worker = SubprocessWorker(
        "nonexistent_module_xyz_abc",
        tmp_path,
        tmp_path / "w.log",
        init_op="init",
        init_payload={},
        timeout=20.0,
    )
    attempts: list[str] = []

    def _fail_once(
        self: SubprocessWorker,
        op: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        attempts.append(op)
        if len(attempts) == 1:
            raise RecoverableWorkerError("init boom")
        return {}

    monkeypatch.setattr(SubprocessWorker, "call", _fail_once)
    with pytest.raises(RecoverableWorkerError, match="init boom"):
        worker.start()
    assert worker._proc is None
    assert worker._log_file is None
    worker.start()
    try:
        assert attempts == ["init", "init"]
        assert worker._proc is not None
        assert worker._log_file is not None
        assert not worker._log_file.closed
    finally:
        worker.stop()
    assert worker._proc is None
    assert worker._log_file is None
