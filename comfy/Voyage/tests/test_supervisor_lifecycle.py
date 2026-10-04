"""Supervisor lifecycle + control-plane tests (issues 012, 099).

012: a partial `start_workers()` failure must not orphan already-started
workers, and `run_segments` must still run `stop_workers()` when the start
itself raises (the start used to sit outside the `try/finally`).

099: a `STOP_REQUESTED` / `PAUSE_REQUESTED` written while a commit's state
advance is in flight must survive the commit — the advance is a
read-modify-write that used to write back the stale status it sampled,
clobbering the operator's request.

All run against fake backends on `tmp_path` (real media, no GPU, no
network). The 012 tests stub the worker start/stop calls (no subprocesses);
the 099 tests run a real single-segment fake commit and inject the
operator's unlocked write at the exact step-6 window by wrapping the
supervisor module's `write_state` global.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import voyage.supervisor as supervisor_module
from tests.conftest import initialize_run_directory
from voyage import paths
from voyage import persistence as persistence_module
from voyage.errors import FatalWorkerError
from voyage.models import RunState
from voyage.persistence import read_effective_config, read_state
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "supervisor-lifecycle") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _stub_worker_calls(supervisor: Supervisor) -> tuple[list[str], list[str]]:
    """Replace worker start/stop with recording doubles (no subprocesses)."""
    started: list[str] = []
    stopped: list[str] = []

    def _make_start(name: str) -> Callable[[], None]:
        def _start() -> None:
            started.append(name)

        return _start

    def _make_stop(name: str) -> Callable[[], None]:
        def _stop() -> None:
            stopped.append(name)

        return _stop

    supervisor._video.start = _make_start("video")  # type: ignore[method-assign]
    supervisor._video.stop = _make_stop("video")  # type: ignore[method-assign]
    supervisor._audio.start = _make_start("audio")  # type: ignore[method-assign]
    supervisor._audio.stop = _make_stop("audio")  # type: ignore[method-assign]
    supervisor._director.start = _make_start("director")  # type: ignore[method-assign]
    supervisor._director.stop = _make_stop("director")  # type: ignore[method-assign]
    return started, stopped


def test_start_workers_failure_stops_already_started(tmp_path: Path) -> None:
    """012: a failing second start must unwind the first (no orphan worker)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    started, stopped = _stub_worker_calls(supervisor)

    def _fail_audio_start() -> None:
        started.append("audio-attempt")
        raise FatalWorkerError("audio init boom")

    supervisor._audio.start = _fail_audio_start  # type: ignore[method-assign]
    with pytest.raises(FatalWorkerError, match="audio init boom"):
        supervisor.start_workers()
    assert started == ["video", "audio-attempt"]
    assert stopped == ["video"]
    assert supervisor._workers_running is False


def test_run_segments_partial_start_stops_workers(tmp_path: Path) -> None:
    """012: `run_segments` cleans up even when `start_workers()` raises."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    started, stopped = _stub_worker_calls(supervisor)

    def _fail_audio_start() -> None:
        started.append("audio-attempt")
        raise FatalWorkerError("audio init boom")

    supervisor._audio.start = _fail_audio_start  # type: ignore[method-assign]
    with pytest.raises(FatalWorkerError, match="audio init boom"):
        supervisor.run_segments(1)
    assert started == ["video", "audio-attempt"]
    assert "video" in stopped
    assert supervisor._workers_running is False


def _commit_with_operator_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    run_id: str,
) -> Supervisor:
    """Run one real fake commit while injecting an operator status write.

    Wraps the supervisor module's `read_state` global. A successful
    `commit_one_segment` reads state twice through that global (commit-start
    snapshot, then the step-6 `fresh` read — verified by grep: no helper on
    the success path touches it). Just after the second read returns its
    (now stale) snapshot, an unlocked read-modify-write sets `requested` —
    the exact shape of `voyage stop` / `voyage pause` via `_set_status`,
    which takes no lock — landing inside the issue-099 window between the
    commit's sample and its write-back. Returns the supervisor (workers
    left running for the caller to stop).
    """
    run_dir = tmp_path / "run"
    _init_run(run_dir, run_id=run_id)
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    reads: list[str] = []

    def _injecting_read(run: Path) -> RunState:
        snapshot = persistence_module.read_state(run)
        reads.append(snapshot.status)
        if len(reads) == 2:
            live = persistence_module.read_state(run)
            live.status = requested  # type: ignore[assignment]
            persistence_module.write_state(run, live)
        return snapshot

    monkeypatch.setattr(supervisor_module, "read_state", _injecting_read)
    assert supervisor.commit_one_segment() == "000000"
    assert reads[:2] == ["CREATED", "CREATED"]
    return supervisor


def test_commit_preserves_stop_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """099: a STOP_REQUESTED landing mid-commit survives the state advance."""
    run_dir = tmp_path / "run"
    supervisor = _commit_with_operator_write(
        tmp_path, monkeypatch, "STOP_REQUESTED", "lifecycle-099-stop"
    )
    try:
        final = read_state(run_dir)
        assert final.status == "STOP_REQUESTED"
        assert final.committed_segments == 1
        assert (paths.segment_dir(run_dir, "000000") / paths.DONE_MARKER).exists()
    finally:
        supervisor.stop_workers()


def test_commit_preserves_pause_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """099: a PAUSE_REQUESTED landing mid-commit survives the state advance."""
    run_dir = tmp_path / "run"
    supervisor = _commit_with_operator_write(
        tmp_path, monkeypatch, "PAUSE_REQUESTED", "lifecycle-099-pause"
    )
    try:
        final = read_state(run_dir)
        assert final.status == "PAUSE_REQUESTED"
        assert final.committed_segments == 1
        assert (paths.segment_dir(run_dir, "000000") / paths.DONE_MARKER).exists()
    finally:
        supervisor.stop_workers()
