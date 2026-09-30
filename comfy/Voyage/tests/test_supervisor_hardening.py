"""Supervisor hardening tests (issues 004, 013, 014, 016).

Covers the four owned regions in `voyage/supervisor.py`: the run-lock
acquire/holder-read path (004), the DONE-before-state crash window and
orphan adoption (013), restart-budget accounting for `restart()`/hook
failures (014), and the recovery-tape containment gate (016).

Restart/lock/tape tests stub workers or touch no workers at all (fast).
The 013 tests commit real segments through the fake backend in-container
(genuine retry-path coverage, no GPU).
"""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import load_config
from voyage.errors import FatalWorkerError, MediaError, RecoverableWorkerError
from voyage.persistence import read_state, write_state
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "supervisor-hardening") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _unstarted_supervisor(run_dir: Path) -> Supervisor:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    return Supervisor(run_dir, config)


# ---------------------------------------------------------------------------
# Issue 014 — restart()/hook failures route through the restart budget.
# ---------------------------------------------------------------------------


def _always_failing_call(
    op: str, payload: dict[str, Any], timeout: float = 600.0
) -> dict[str, Any]:
    raise RecoverableWorkerError(f"boom in {op}")


def _dead_restart() -> None:
    raise RecoverableWorkerError("init boom")


def _metrics_text(run_dir: Path) -> str:
    return (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")


def test_failed_restart_consumes_budget_and_trips_breaker(tmp_path: Path) -> None:
    """A restart whose init replay fails counts against the budget (014)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config.voyage.max_worker_restarts = 3
    supervisor = Supervisor(run_dir, config)
    supervisor._video.call = _always_failing_call  # type: ignore[method-assign]
    supervisor._video.restart = _dead_restart  # type: ignore[method-assign]
    with pytest.raises(FatalWorkerError, match="circuit breaker"):
        supervisor._call_with_restart(supervisor._video, "video", "000000", "generate_blocks", {})
    # Call failure (1) + restart failure (2) + call failure (3), then the
    # next restart failure finds the budget exhausted and trips the breaker.
    assert supervisor._restarts["video"] == 3
    metrics = _metrics_text(run_dir)
    assert metrics.count('"event": "worker_restart"') == 2
    assert '"event": "worker_restart_failed"' in metrics
    assert '"event": "circuit_breaker_open"' in metrics


def test_failed_restart_then_success_recovers(tmp_path: Path) -> None:
    """One failed restart followed by a healthy call returns normally (014)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
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
            raise RecoverableWorkerError("init boom")

    supervisor._video.call = _flaky_call  # type: ignore[method-assign]
    supervisor._video.restart = _flaky_restart  # type: ignore[method-assign]
    assert supervisor._call_with_restart(supervisor._video, "video", "000000", "op", {}) == {
        "answer": 1
    }
    assert supervisor._restarts["video"] == 2
    assert '"event": "worker_restart_failed"' in _metrics_text(run_dir)


def test_failed_restart_hook_routes_through_budget(tmp_path: Path) -> None:
    """A Recoverable failure from restart_hook consumes budget too (014)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config.voyage.max_worker_restarts = 2
    supervisor = Supervisor(run_dir, config)
    supervisor._video.call = _always_failing_call  # type: ignore[method-assign]
    supervisor._video.restart = lambda: None  # type: ignore[method-assign]

    def _dead_hook(segment_id: str) -> None:
        raise RecoverableWorkerError("hook boom")

    with pytest.raises(FatalWorkerError, match="circuit breaker"):
        supervisor._call_with_restart(
            supervisor._video, "video", "000000", "generate_blocks", {}, _dead_hook
        )
    assert supervisor._restarts["video"] == 2
    metrics = _metrics_text(run_dir)
    assert '"event": "worker_restart_failed"' in metrics
    assert '"event": "circuit_breaker_open"' in metrics


# ---------------------------------------------------------------------------
# Issue 016 — recovery-tape containment gate + discovery mirror.
# ---------------------------------------------------------------------------


def test_tape_symlink_escape_rejected(tmp_path: Path) -> None:
    """An in-run symlink pointing outside the run fails the gate (016)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not-yours")
    (run_dir / paths.SEGMENTS_DIRNAME / "evil.pt").symlink_to(outside)
    supervisor = _unstarted_supervisor(run_dir)
    with pytest.raises(MediaError, match="escapes the run dir"):
        supervisor._checked_tape_path("segments/evil.pt", "000000")


def test_tape_directory_rejected(tmp_path: Path) -> None:
    """A directory reported as the tape fails with MediaError, not IsADirectoryError (016)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    (run_dir / paths.SEGMENTS_DIRNAME / "adir").mkdir()
    supervisor = _unstarted_supervisor(run_dir)
    with pytest.raises(MediaError, match="not a regular file"):
        supervisor._checked_tape_path("segments/adir", "000000")


def test_legitimate_tape_accepted(tmp_path: Path) -> None:
    """A genuine in-run tape file still passes the hardened gate (016)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    tape = run_dir / paths.SEGMENTS_DIRNAME / "good.pt"
    tape.write_bytes(b"tape")
    supervisor = _unstarted_supervisor(run_dir)
    assert supervisor._checked_tape_path("segments/good.pt", "000000").endswith("good.pt")


def test_latest_recovery_tape_skips_planted_entries(tmp_path: Path) -> None:
    """Discovery skips symlink-escaped/dir tapes, uses the newest good one (016)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    good_segment = paths.segment_dir(run_dir, "000000")
    good_segment.mkdir(parents=True, exist_ok=True)
    (good_segment / paths.DONE_MARKER).write_bytes(b"")
    good_tape = good_segment / "recovery.pt"
    good_tape.write_bytes(b"tape")
    bad_segment = paths.segment_dir(run_dir, "000001")
    bad_segment.mkdir(parents=True, exist_ok=True)
    (bad_segment / paths.DONE_MARKER).write_bytes(b"")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not-yours")
    (bad_segment / "recovery.pt").symlink_to(outside)
    supervisor = _unstarted_supervisor(run_dir)
    assert supervisor._latest_recovery_tape() == good_tape


def test_latest_recovery_tape_skips_directory_tape(tmp_path: Path) -> None:
    """A directory at the discovery path reads as no tape, never a crash (016)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    segment = paths.segment_dir(run_dir, "000000")
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_bytes(b"")
    (segment / "recovery.pt").mkdir()
    supervisor = _unstarted_supervisor(run_dir)
    assert supervisor._latest_recovery_tape() is None


# ---------------------------------------------------------------------------
# Issue 004 — lock error taxonomy, honest holder reads, tidy exit.
# ---------------------------------------------------------------------------


def test_flock_system_error_not_misreported_as_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-contention flock OSError (EBADF/...) propagates raw, never "locked" (004)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _unstarted_supervisor(run_dir)

    def _broken_flock(_fd: int, _op: int) -> None:
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(fcntl, "flock", _broken_flock)
    with pytest.raises(OSError), supervisor._held_run_lock():
        pass


def test_stale_or_garbage_holder_reads_unknown(tmp_path: Path) -> None:
    """Dead-pid residue and garbage never name a holder that cannot hold it (004)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _unstarted_supervisor(run_dir)
    lock_path = run_dir / "state.json.lock"
    lock_path.write_text("not-a-pid", encoding="utf-8")
    assert supervisor._read_lock_holder(lock_path) == "unknown"
    lock_path.write_text("99999999", encoding="utf-8")
    assert supervisor._read_lock_holder(lock_path) == "unknown"
    lock_path.write_text("", encoding="utf-8")
    assert supervisor._read_lock_holder(lock_path) == "unknown"
    assert supervisor._read_lock_holder(run_dir / "no-such-file.lock") == "unknown"
    live_pid = str(os.getpid())
    lock_path.write_text(live_pid, encoding="utf-8")
    assert supervisor._read_lock_holder(lock_path) == live_pid


def test_lock_file_removed_on_clean_exit(tmp_path: Path) -> None:
    """The rendezvous file is tidied after a clean commit hold (004)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _unstarted_supervisor(run_dir)
    lock_path = run_dir / "state.json.lock"
    with supervisor._held_run_lock():
        assert lock_path.exists()
    assert not lock_path.exists()


def test_lock_file_kept_when_successor_waiting(tmp_path: Path) -> None:
    """Tidy-up never removes a successor's lock file (pid-guarded unlink) (004)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _unstarted_supervisor(run_dir)
    lock_path = run_dir / "state.json.lock"
    with supervisor._held_run_lock():
        pass
    # Simulate a successor that acquired after our exit: its pid owns the file.
    lock_path.write_text("424242", encoding="utf-8")
    with supervisor._held_run_lock():
        # Our own pid is recorded while we hold it; clobber it mid-hold to
        # emulate the successor-write race, then exit and check the guard.
        lock_path.write_text("424242", encoding="utf-8")
    assert lock_path.read_text(encoding="utf-8") == "424242"


# ---------------------------------------------------------------------------
# Issue 013 — DONE-before-state crash window: adopt, never overwrite.
# ---------------------------------------------------------------------------


def _commit_first_segment(run_dir: Path) -> Path:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    committed = supervisor.run_segments(1)
    assert committed == ["000000"]
    return paths.segment_dir(run_dir, "000000")


def _rewind_state_to_uncommitted(run_dir: Path) -> None:
    """Simulate the crash window: DONE durable, state.json never advanced."""
    state = read_state(run_dir)
    state.next_segment_number = 0
    state.committed_segments = 0
    state.timeline_frames = 0
    state.decision_index = 0
    write_state(run_dir, state)


def test_bare_done_dir_reclaimed_without_error(tmp_path: Path) -> None:
    """An artifact-free DONE dir (no render to protect) re-commits cleanly (013)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    segment = paths.segment_dir(run_dir, "000000")
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_bytes(b"")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    state = read_state(run_dir)
    assert (state.next_segment_number, state.committed_segments) == (1, 1)
    assert '"event": "segment_reclaimed"' in _metrics_text(run_dir)


def test_done_orphan_adopted_without_rerender(tmp_path: Path) -> None:
    """Retry adopts the DONE orphan (checksums verify) without touching media (013)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    segment = _commit_first_segment(run_dir)
    video_out = segment / "video.mp4"
    before_bytes = video_out.read_bytes()
    before_mtime = video_out.stat().st_mtime_ns
    _rewind_state_to_uncommitted(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
    finally:
        supervisor.stop_workers()
    assert video_out.read_bytes() == before_bytes
    assert video_out.stat().st_mtime_ns == before_mtime
    state = read_state(run_dir)
    assert (state.next_segment_number, state.committed_segments) == (1, 1)
    assert state.timeline_frames > 0
    assert (segment / paths.DONE_MARKER).exists()
    metrics = _metrics_text(run_dir)
    assert '"event": "segment_adopted"' in metrics


def test_corrupt_orphan_refuses_without_overwrite(tmp_path: Path) -> None:
    """An unverifiable DONE orphan fails loud; media is never re-rendered (013)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    segment = _commit_first_segment(run_dir)
    video_out = segment / "video.mp4"
    with video_out.open("ab") as handle:
        handle.write(b"\x00" * 64)
    corrupted_bytes = video_out.read_bytes()
    _rewind_state_to_uncommitted(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        with pytest.raises(MediaError, match="refusing to re-render"):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert video_out.read_bytes() == corrupted_bytes
    assert (segment / paths.DONE_MARKER).exists()
    assert read_state(run_dir).next_segment_number == 0
