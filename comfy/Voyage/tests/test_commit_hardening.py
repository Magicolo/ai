"""Commit-hardening tests (issues 001, 002, 004, 006, 007, 010, 016-S, 017, 058-S).

All against fake backends or stubbed workers (real media, no GPU) in-container:
RPC deadline reads, error-taxonomy preservation, the single-writer run lock,
worker-report bounds, audio-swap teardown, gauge timeouts, run-relative
paths, and single-step DONE.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.cli_validate import validate_run
from voyage.errors import (
    ConfigurationError,
    FatalWorkerError,
    MediaError,
    RecoverableWorkerError,
    VoyageError,
)
from voyage.persistence import read_effective_config, read_state
from voyage.rpc import SubprocessWorker
from voyage.supervisor import (
    EMBED_TIMEOUT_SECONDS,
    GAUGE_TIMEOUT_SECONDS,
    Supervisor,
)
from voyage.workers.loop import checked_request, serve


def _init_run(run_dir: Path, run_id: str = "commit-hardening") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _stub_worker(module: str) -> SubprocessWorker:
    """Bare worker handle with no process (callers stub `.call`)."""
    worker = SubprocessWorker.__new__(SubprocessWorker)
    worker._module = module
    worker._proc = None
    worker._counter = 0
    worker._timeout = 600.0
    worker._call_lock = threading.Lock()
    worker._log_file = None
    return worker


def _pipe_worker(module: str, read_fd: int) -> SubprocessWorker:
    """Worker handle reading responses from a raw pipe fd (issue 001)."""
    worker = _stub_worker(module)
    worker._proc = SimpleNamespace(  # type: ignore[assignment]
        stdin=SimpleNamespace(write=lambda data: len(data), flush=lambda: None),
        stdout=read_fd,
        poll=lambda: None,
    )
    return worker


def test_partial_line_never_passes_deadline() -> None:
    """A worker dribbling bytes without `\\n` fails at ~the deadline (001)."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"partial-response-with-no-newline")
        worker = _pipe_worker("dribble-test-worker", read_fd)
        started = time.monotonic()
        with pytest.raises(RecoverableWorkerError, match="timed out"):
            worker.call("health", {}, timeout=0.5)
        assert time.monotonic() - started < 5.0
    finally:
        os.close(write_fd)
        try:
            os.close(read_fd)
        except OSError:
            pass


def test_full_response_line_still_reads() -> None:
    """A complete response line on the same path returns its result (001)."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b'{"id": "req-000001", "ok": true, "result": {"answer": 1}}\n')
        worker = _pipe_worker("full-line-test-worker", read_fd)
        assert worker.call("health", {}, timeout=5.0) == {"answer": 1}
    finally:
        os.close(write_fd)
        try:
            os.close(read_fd)
        except OSError:
            pass


def test_oversize_response_line_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newline-less flood ends at the line cap, not at the deadline (001)."""
    import voyage.rpc as rpc_module

    monkeypatch.setattr(rpc_module, "MAX_RESPONSE_LINE_BYTES", 16)
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"x" * 64 + b"\n")
        worker = _pipe_worker("flood-test-worker", read_fd)
        started = time.monotonic()
        with pytest.raises(RecoverableWorkerError, match="exceeded"):
            worker.call("health", {}, timeout=30.0)
        assert time.monotonic() - started < 5.0
    finally:
        os.close(write_fd)
        try:
            os.close(read_fd)
        except OSError:
            pass


def test_torn_takes_ledger_rests_failed(tmp_path: Path) -> None:
    """A SIGKILL-torn takes.jsonl ends FAILED, never RUNNING (002)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    (audio_dir / "takes.jsonl").write_text("TRUNCATED\n", encoding="utf-8")
    config = read_effective_config(run_dir)
    with pytest.raises(VoyageError, match="[Ll]edger"):
        Supervisor(run_dir, config).run_segments(1)
    failed = read_state(run_dir)
    assert failed.status == "FAILED"


def test_corrupt_concept_history_rests_failed(tmp_path: Path) -> None:
    """A corrupt concepts.jsonl ends FAILED, never RUNNING (002)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    novelty_dir = run_dir / "novelty"
    novelty_dir.mkdir(exist_ok=True)
    (novelty_dir / "concepts.jsonl").write_text("NOT_JSON\n", encoding="utf-8")
    config = read_effective_config(run_dir)
    with pytest.raises(VoyageError, match="[Cc]oncept"):
        Supervisor(run_dir, config).run_segments(1)
    failed = read_state(run_dir)
    assert failed.status == "FAILED"


def test_unexpected_exception_backstops_to_failed(tmp_path: Path) -> None:
    """A non-VoyageError on the commit path still rests FAILED (002)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        import voyage.supervisor as supervisor_module

        def _zero_divide(path: Path, width: int, height: int, fps: int) -> dict[str, Any]:
            raise ZeroDivisionError("float division by zero")

        supervisor_module.validate_video = _zero_divide  # type: ignore[assignment]
        try:
            with pytest.raises(FatalWorkerError, match="ZeroDivisionError"):
                supervisor.run_segments(1)
        finally:
            import voyage.media as media_module

            supervisor_module.validate_video = media_module.validate_video
    finally:
        supervisor.stop_workers()
    failed = read_state(run_dir)
    assert failed.status == "FAILED"


def test_second_writer_fails_fast_when_locked(tmp_path: Path) -> None:
    """A second supervisor on one run dir exits loudly, never interleaves (004)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    first = Supervisor(run_dir, config)
    second = Supervisor(run_dir, config)
    second._workers_running = True
    with first._held_run_lock():
        holder = (run_dir / "state.json.lock").read_text(encoding="utf-8").strip()
        assert holder == str(os.getpid())
        started = time.monotonic()
        with pytest.raises(FatalWorkerError, match="locked"):
            second.commit_one_segment()
        assert time.monotonic() - started < 5.0


def _started_supervisor(run_dir: Path) -> Supervisor:
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    return supervisor


def test_implausible_reported_frames_rejected(tmp_path: Path) -> None:
    """`frames=10**9` fails fast instead of slicing a billion frames (006)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _started_supervisor(run_dir)
    try:
        original = supervisor._video.call

        def _lie(op: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            result = original(op, payload, timeout=timeout)
            if op == "generate_blocks":
                result = dict(result)
                result["video"] = {"frames": 10**9}
            return result

        supervisor._video.call = _lie  # type: ignore[method-assign]
        with pytest.raises(MediaError, match="implausible"):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def test_foreign_recovery_path_rejected(tmp_path: Path) -> None:
    """A tape outside the run dir fails fast instead of burning resume budget (006)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _started_supervisor(run_dir)
    try:
        original = supervisor._video.call

        def _lie(op: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            result = original(op, payload, timeout=timeout)
            if op == "generate_blocks":
                result = dict(result)
                result["video"] = {"frames": 48, "recovery_path": "/etc/passwd"}
            return result

        supervisor._video.call = _lie  # type: ignore[method-assign]
        with pytest.raises(MediaError, match="escapes the run dir"):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def test_checked_request_enforces_types() -> None:
    """Presence is not enough: mistyped fields fail before GPU work (007)."""
    with pytest.raises(KeyError, match="missing"):
        checked_request({}, frames=int)
    with pytest.raises(TypeError, match="must be int"):
        checked_request({"frames": "abc"}, frames=int)
    with pytest.raises(TypeError, match="must be int"):
        checked_request({"seed": True}, seed=int)
    checked_request({"duration": 5}, duration=float)  # ints satisfy float
    checked_request({"frames": 48}, frames=int)


def test_worker_error_class_survives_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deterministic worker error keeps its class and fatality (007)."""
    stdin_lines = io.StringIO('{"id": "req-000001", "op": "generate_blocks", "payload": {}}\n')
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin_lines)
    monkeypatch.setattr(sys, "stdout", captured)

    def _boom(payload: dict[str, Any]) -> dict[str, Any]:
        raise ConfigurationError("bad geometry")

    serve({"generate_blocks": _boom})
    lines = captured.getvalue().splitlines()
    assert len(lines) == 1
    response = json.loads(lines[0])
    assert response["id"] == "req-000001"
    assert response["ok"] is False
    assert response["error"]["code"] == "ConfigurationError"
    assert response["error"]["retryable"] is False


def test_malformed_line_answers_malformed_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """A schema-bad line with a salvageable id answers MALFORMED, no hang (007)."""
    stdin_lines = io.StringIO(
        '{"id": "req-000007", "op": 123}\n{"id": "req-000008", "op": "nope", "payload": {}}\n'
    )
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin_lines)
    monkeypatch.setattr(sys, "stdout", captured)
    serve({})
    lines = captured.getvalue().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["id"] == "req-000007"
    assert first["ok"] is False
    assert first["error"]["code"] == "MALFORMED"
    assert first["error"]["retryable"] is False
    second = json.loads(lines[1])
    assert second["id"] == "req-000008"
    assert second["error"]["code"] == "UNKNOWN_OP"


def _swap_supervisor(run_dir: Path) -> Supervisor:
    """Supervisor with the acestep+streaming swap armed, workers stubbed (010)."""
    config = read_effective_config(run_dir)
    config.audio.backend = "acestep"
    config.video.backend = "ltxv"
    return Supervisor(run_dir, config)


def test_audio_failure_propagates_not_masked(tmp_path: Path) -> None:
    """A failing take with no tape raises the audio cause, explicitly evicted (010)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _swap_supervisor(run_dir)
    calls: list[tuple[str, str]] = []

    def _script(
        worker: SubprocessWorker,
        worker_name: str,
        segment_id: str,
        op: str,
        payload: dict[str, object],
        restart_hook: Any = None,
    ) -> dict[str, object]:
        calls.append((worker_name, op))
        if op == "generate_audio":
            raise RecoverableWorkerError("ACE OOM")
        return {}

    supervisor._call_with_restart = _script  # type: ignore[method-assign]
    with pytest.raises(RecoverableWorkerError, match="ACE OOM"):
        supervisor._with_audio_gpu("000000", {"style": "drone"}, None)
    assert calls == [
        ("video", "evict_gpu"),
        ("audio", "generate_audio"),
        ("audio", "evict_gpu"),
    ]
    metrics = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "video_left_evicted"' in metrics


def test_audio_failure_with_tape_still_rebuilds(tmp_path: Path) -> None:
    """A failing take with a tape rebuilds, yet the audio cause still wins (010)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    tape = run_dir / "segments" / "000000" / "recovery.pt"
    tape.parent.mkdir(parents=True, exist_ok=True)
    tape.write_bytes(b"tape")
    supervisor = _swap_supervisor(run_dir)
    calls: list[tuple[str, str]] = []

    def _script(
        worker: SubprocessWorker,
        worker_name: str,
        segment_id: str,
        op: str,
        payload: dict[str, object],
        restart_hook: Any = None,
    ) -> dict[str, object]:
        calls.append((worker_name, op))
        if op == "generate_audio":
            raise RecoverableWorkerError("ACE OOM")
        return {}

    supervisor._call_with_restart = _script  # type: ignore[method-assign]
    with pytest.raises(RecoverableWorkerError, match="ACE OOM"):
        supervisor._with_audio_gpu("000000", {"style": "drone"}, str(tape))
    assert ("video", "rebuild") in calls
    metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    if metrics_path.exists():
        assert '"event": "video_left_evicted"' not in metrics_path.read_text(encoding="utf-8")


def test_success_path_teardown_failure_still_rebuilds(tmp_path: Path) -> None:
    """A teardown failure after a rendered take rebuilds anyway, then raises (010)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    tape = run_dir / "segments" / "000000" / "recovery.pt"
    tape.parent.mkdir(parents=True, exist_ok=True)
    tape.write_bytes(b"tape")
    supervisor = _swap_supervisor(run_dir)
    calls: list[tuple[str, str]] = []

    def _script(
        worker: SubprocessWorker,
        worker_name: str,
        segment_id: str,
        op: str,
        payload: dict[str, object],
        restart_hook: Any = None,
    ) -> dict[str, object]:
        calls.append((worker_name, op))
        if op == "evict_gpu" and worker_name == "audio":
            raise RecoverableWorkerError("audio evict boom")
        return {}

    supervisor._call_with_restart = _script  # type: ignore[method-assign]
    with pytest.raises(RecoverableWorkerError, match="audio evict boom"):
        supervisor._with_audio_gpu("000000", {"style": "drone"}, str(tape))
    assert ("video", "rebuild") in calls
    metrics = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "audio_swap_teardown_error"' in metrics


def test_gauge_probes_carry_short_timeouts(tmp_path: Path) -> None:
    """Optional gauges wait seconds per worker, never the RPC default (017)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    seen: list[float | None] = []

    def _slow(op: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        seen.append(timeout)
        time.sleep(0.2)
        raise RecoverableWorkerError("sick worker")

    supervisor._video.call = _slow  # type: ignore[method-assign]
    supervisor._audio.call = _slow  # type: ignore[method-assign]
    supervisor._director.call = _slow  # type: ignore[method-assign]
    started = time.monotonic()
    supervisor._sample_gauges("000000")
    assert time.monotonic() - started < 10.0
    assert seen == [GAUGE_TIMEOUT_SECONDS] * 3
    assert GAUGE_TIMEOUT_SECONDS == 5.0
    metrics = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "resource_gauges"' in metrics


def test_embed_call_carries_bounded_timeout(tmp_path: Path) -> None:
    """Novelty embedding degrades to fallback on a bounded wait, not 600 s (017)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    seen: list[float | None] = []

    def _embed(op: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        seen.append(timeout)
        return {"vectors": [[1.0, 0.0]]}

    supervisor._director.call = _embed  # type: ignore[method-assign]
    assert supervisor._embed_texts(["a misty harbor"]) == [[1.0, 0.0]]
    assert seen == [EMBED_TIMEOUT_SECONDS]
    assert EMBED_TIMEOUT_SECONDS == 60.0


def test_relocated_run_continues_from_relative_paths(tmp_path: Path) -> None:
    """Commit, move the run dir, commit again: relative paths survive (016-S)."""
    first_dir = tmp_path / "run-a"
    _init_run(first_dir, "relocation")
    config = read_effective_config(first_dir)
    assert Supervisor(first_dir, config).run_segments(1) == ["000000"]
    ledger_lines = (first_dir / "audio" / "takes.jsonl").read_text(encoding="utf-8").splitlines()
    assert ledger_lines
    stored_path = json.loads(ledger_lines[0])["path"]
    assert not Path(stored_path).is_absolute()
    second_dir = tmp_path / "run-b"
    first_dir.rename(second_dir)
    config = read_effective_config(second_dir)
    assert Supervisor(second_dir, config).run_segments(1) == ["000001"]
    assert validate_run(second_dir) == []


def test_done_written_without_partial_remnant(tmp_path: Path) -> None:
    """DONE lands atomically with no visible DONE.partial window (058-S)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    segment = paths.segment_dir(run_dir, "000000")
    assert (segment / paths.DONE_MARKER).exists()
    assert not (segment / "DONE.partial").exists()
    assert list((run_dir / "segments").rglob("*.partial")) == []


def test_worker_stop_closes_log_and_reaps(tmp_path: Path) -> None:
    """Restarts never leak log fds; kills never leave zombies (058-S)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        handle = supervisor._video._log_file
        assert handle is not None and not handle.closed
        supervisor.inject_worker_crash("video")
        supervisor.stop_workers()
        assert handle.closed
        assert supervisor._video._proc is None
        assert supervisor._video._log_file is None
    finally:
        supervisor.stop_workers()
