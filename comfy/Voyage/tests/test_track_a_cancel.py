"""Track A: cancellable render + stop wiring + prefetch cap (DESIGN §73).

Covers `join(timeout)` polling of the stop event, SIGINT/SIGTERM wiring
to `request_stop()`, the 12 s prefetch miss-fast with progress note, and
the PAUSED (never RUNNING) rest on KeyboardInterrupt.
"""

from __future__ import annotations

import argparse
import signal
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

import voyage.supervisor as supervisor_module
from tests.conftest import initialize_run_directory
from voyage.persistence import read_effective_config, read_state
from voyage.supervisor import Supervisor


class _RecordingProgress:
    """Minimal progress sink recording notes (never raises)."""

    def __init__(self) -> None:
        self.notes: list[str] = []
        self.verbose = False

    def note(self, message: str) -> None:
        self.notes.append(message)


class _StubDriver:
    """Background driver double: alive, no ledger side effects."""

    def is_alive(self) -> bool:
        return True


class _StubAdapter:
    """Video adapter double: fast render returning a fixed result."""

    def __init__(self, frames: int = 48) -> None:
        self.frames = frames
        self.calls = 0

    def generate_segment(self, request: Any, video_out: Path) -> Any:
        from types import SimpleNamespace

        self.calls += 1
        time.sleep(0.05)
        return SimpleNamespace(returned_frames=self.frames)


def _supervisor_with_stubs(run_dir: Path, progress: Any | None) -> Supervisor:
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config, progress=progress)
    return supervisor


def test_live_prewarm_stop_flag_still_completes_with_note(tmp_path: Path) -> None:
    """A stop set before the render still waits it out with one note."""
    from typing import cast

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-cancel-render")
    progress = _RecordingProgress()
    supervisor = _supervisor_with_stubs(run_dir, progress)
    supervisor._background = _StubDriver()
    supervisor.request_stop()
    result = supervisor._generate_segment_with_live_prewarm(
        cast(Any, _StubAdapter()), object(), run_dir / "video.mp4"
    )
    assert result.returned_frames == 48
    assert any("stop requested" in note for note in progress.notes)


def test_live_prewarm_join_uses_timeout_not_bare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final join polls with timeout (SIGINT-responsive, never bare)."""
    import threading
    from typing import cast

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-cancel-join")
    supervisor = _supervisor_with_stubs(run_dir, _RecordingProgress())
    supervisor._background = _StubDriver()
    seen_timeouts: list[float | None] = []
    real_join = threading.Thread.join

    def _recording_join(self: threading.Thread, timeout: float | None = None) -> None:
        seen_timeouts.append(timeout)
        real_join(self, timeout=timeout)

    monkeypatch.setattr(threading.Thread, "join", _recording_join)

    class _SlowAdapter(_StubAdapter):
        def generate_segment(self, request: Any, video_out: Path) -> Any:
            from types import SimpleNamespace

            time.sleep(0.6)
            return SimpleNamespace(returned_frames=self.frames)

    calls = {"drains": 0}
    real_drain = supervisor._drain_prewarm_queue

    def _boom_once(*args: Any, **kwargs: Any) -> None:
        calls["drains"] += 1
        if calls["drains"] == 1:
            raise RuntimeError("pump boom (test forces the cancellable-join path)")
        real_drain(*args, **kwargs)

    monkeypatch.setattr(supervisor, "_drain_prewarm_queue", _boom_once)
    with pytest.raises(RuntimeError, match="pump boom"):
        supervisor._generate_segment_with_live_prewarm(
            cast(Any, _SlowAdapter()), object(), run_dir / "video.mp4"
        )
    assert seen_timeouts, "expected at least one join call"
    assert all(timeout is not None for timeout in seen_timeouts), (
        f"bare join() found: {seen_timeouts}"
    )


def test_install_stop_handlers_sets_flag_and_second_raises(tmp_path: Path) -> None:
    """SIGINT handler requests stop; a second SIGINT re-raises."""
    from voyage.supervisor import install_stop_handlers

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-handlers")
    supervisor = _supervisor_with_stubs(run_dir, None)
    restore = install_stop_handlers(supervisor)
    try:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert supervisor._stop_flag is True
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    finally:
        restore()


def test_take_prefetch_cap_miss_fast_with_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A never-done prefetch misses at the cap (not the 60 s budget)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-prefetch-cap")
    progress = _RecordingProgress()
    supervisor = _supervisor_with_stubs(run_dir, progress)
    monkeypatch.setattr(supervisor_module, "PREFETCH_WAIT_CAP_SECONDS", 0.3)
    future: Future[dict[str, Any] | None] = Future()
    supervisor._prefetch_future = future
    supervisor._prefetch_target = 0
    supervisor._prefetch_submitted_at = time.monotonic()
    started = time.monotonic()
    assert supervisor._take_prefetch(0, "000000") is None
    assert time.monotonic() - started < 5.0
    assert any("timed out" in note for note in progress.notes)


def test_take_prefetch_stop_abandons_wait(tmp_path: Path) -> None:
    """A stop set before consume abandons the wait (miss, no 12 s burn)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-prefetch-stop")
    supervisor = _supervisor_with_stubs(run_dir, _RecordingProgress())
    future: Future[dict[str, Any] | None] = Future()
    supervisor._prefetch_future = future
    supervisor._prefetch_target = 0
    supervisor._prefetch_submitted_at = time.monotonic()
    supervisor.request_stop()
    started = time.monotonic()
    assert supervisor._take_prefetch(0, "000000") is None
    assert time.monotonic() - started < 5.0


def test_run_segments_keyboard_interrupt_rests_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyboardInterrupt mid-batch rests PAUSED (never strands RUNNING)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-interrupt")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers = lambda: setattr(  # type: ignore[method-assign]
        supervisor, "_workers_running", True
    )
    supervisor.stop_workers = lambda: setattr(  # type: ignore[method-assign]
        supervisor, "_workers_running", False
    )
    supervisor._workers_running = False

    def _boom() -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(supervisor, "commit_one_segment", _boom)
    committed = supervisor.run_segments(1)
    assert committed == []
    assert read_state(run_dir).status == "PAUSED"


def test_cli_main_keyboard_interrupt_maps_to_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`voyage` CLI maps KeyboardInterrupt to exit 130 (no traceback)."""
    import argparse

    import voyage.cli as cli

    def _boom(_args: argparse.Namespace) -> int:
        raise KeyboardInterrupt

    parser = cli.build_parser()
    monkeypatch.setattr(
        parser, "parse_args", lambda _argv=None: argparse.Namespace(command="generate", func=_boom)
    )
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    assert cli.main([]) == 130
    assert "PAUSED" in capsys.readouterr().err
