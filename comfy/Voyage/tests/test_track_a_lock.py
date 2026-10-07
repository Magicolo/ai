"""Track A: run-level mutual exclusion (DESIGN §73, issue 004).

Covers the public `acquire_run_lock` helper other tracks import (never
redefine): second-writer fails fast with FatalWorkerError, same-thread
nesting shares one fd, different threads contend via the real flock,
and a commit nested inside the batch lock succeeds.
"""

from __future__ import annotations

import fcntl
import os
import threading
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage.errors import FatalWorkerError
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor, acquire_run_lock, run_lock_path


def test_lock_path_names_state_json_lock(tmp_path: Path) -> None:
    """The rendezvous file is `<run>/state.json.lock` (single source)."""
    run_dir = tmp_path / "run"
    assert run_lock_path(run_dir) == run_dir / "state.json.lock"


def test_second_writer_fails_fast_with_holder(tmp_path: Path) -> None:
    """A raw flock holder makes `acquire_run_lock` raise Fatal (not block)."""
    import contextlib

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_lock_path(run_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(FatalWorkerError, match="locked by pid"), acquire_run_lock(run_dir):
            pass
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(holder_fd, fcntl.LOCK_UN)
        os.close(holder_fd)


def test_same_thread_nesting_shares_one_fd(tmp_path: Path) -> None:
    """Batch + per-commit + discard nesting in one thread succeeds."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    with acquire_run_lock(run_dir), acquire_run_lock(run_dir):
        pass
    # Released: a fresh acquire still works (no stale registry entry).
    with acquire_run_lock(run_dir):
        pass


def test_different_threads_contend(tmp_path: Path) -> None:
    """Another thread holding the lock trips Fatal (never nests across threads)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    release = threading.Event()
    held = threading.Event()

    def _holder() -> None:
        with acquire_run_lock(run_dir):
            held.set()
            release.wait(timeout=10.0)

    thread = threading.Thread(target=_holder, daemon=True)
    thread.start()
    assert held.wait(timeout=10.0)
    try:
        with pytest.raises(FatalWorkerError, match="locked by pid"), acquire_run_lock(run_dir):
            pass
    finally:
        release.set()
        thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert errors == []


def test_commit_nested_inside_batch_lock_succeeds(tmp_path: Path) -> None:
    """`commit_one_segment` inside an outer batch lock commits (re-entrant)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-lock-nested")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        with acquire_run_lock(run_dir):
            assert supervisor.commit_one_segment() == "000000"
    finally:
        supervisor.stop_workers()
