"""Lock-holder agreement (issue 081 extraction from `supervisor`).

`_read_lock_holder` is the verbatim best-effort pid read moved to
`voyage.supervisor_lock` as `read_lock_holder` so `supervisor.py`
shrinks toward the §12 split signal. Behavior contract: identical to
the pre-split `Supervisor._read_lock_holder` — the retained method
delegates (no fork), missing/empty/non-numeric/non-positive pids and
dead processes read as `"unknown"`, a live pid names itself, and the
facade re-export is the same object (single source, not a copy).
"""

from __future__ import annotations

import os
from pathlib import Path

import voyage.supervisor as supervisor
import voyage.supervisor_lock as supervisor_lock
from voyage.supervisor_lock import read_lock_holder


def test_facade_reexport_is_single_sourced() -> None:
    """The facade name is the new home object, not a copy (issue 081)."""
    assert supervisor.read_lock_holder is supervisor_lock.read_lock_holder


def test_method_agrees_with_moved_function(tmp_path: Path) -> None:
    """The retained `Supervisor` method delegates (no fork)."""
    from tests.conftest import initialize_run_directory
    from voyage import paths
    from voyage.config import load_config
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="supervisor-lock")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    instance = Supervisor(run_dir, config)
    lock_path = tmp_path / "no-such-file.lock"
    assert instance._read_lock_holder(lock_path) == "unknown"
    assert read_lock_holder(lock_path) == "unknown"


def test_missing_file_reads_unknown(tmp_path: Path) -> None:
    """A missing lock file is residue-free — unknown, never an exception."""
    assert read_lock_holder(tmp_path / "no-such-file.lock") == "unknown"


def test_empty_and_bad_text_read_unknown(tmp_path: Path) -> None:
    """Empty/non-numeric/non-positive pids never name a process."""
    for text in ("", "   ", "not-a-pid", "0", "-7"):
        lock_path = tmp_path / "holder.lock"
        lock_path.write_text(text, encoding="utf-8")
        assert read_lock_holder(lock_path) == "unknown"


def test_dead_pid_reads_unknown(tmp_path: Path) -> None:
    """A recorded pid for a dead process is residue — unknown, not the pid."""
    lock_path = tmp_path / "holder.lock"
    lock_path.write_text("42424242", encoding="utf-8")
    assert read_lock_holder(lock_path) == "unknown"


def test_live_pid_names_itself(tmp_path: Path) -> None:
    """The live holder pid is named (EPERM still names — best-effort)."""
    lock_path = tmp_path / "holder.lock"
    live_pid = str(os.getpid())
    lock_path.write_text(live_pid, encoding="utf-8")
    assert read_lock_holder(lock_path) == live_pid


def test_directory_reads_unknown(tmp_path: Path) -> None:
    """A directory at the lock path is not a pid file — unknown."""
    subdir = tmp_path / "lockdir"
    subdir.mkdir()
    assert read_lock_holder(subdir) == "unknown"
