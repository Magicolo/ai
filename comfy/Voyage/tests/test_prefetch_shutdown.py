"""Issue 030: director prefetch is time-bounded and never gates shutdown.

Why this file exists: the speculative `decide` prefetch ran with the
600 s worker default, and `stop_workers` used `shutdown(wait=False)`,
which cannot stop a *running* future on non-daemon threads — a finished
or SIGINTed run lingered at exit up to ten minutes behind a wedged
director call. These tests pin the fix with stub directors (no GPU, no
subprocess): the prefetch carries an explicit short budget, and
`stop_workers` cancels/drains best-effort so it returns promptly even
while a prefetch is wedged. The 137 resync contract is untouched — the
prefetch still goes through `call()`, so a prefetch timeout still
leaves the pipe resyncable (pinned by `test_rpc_timeout.py`).
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.concepts import ConceptStore
from voyage.config import ProjectConfig, load_config
from voyage.models import DirectorDestination, EvolutionDecision, StyleSpec
from voyage.supervisor import PREFETCH_TIMEOUT_SECONDS, Supervisor


class _RecordingDirector:
    """Director stand-in: records `call` keyword arguments, returns a canned reply."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reply: dict[str, Any] = {"proposal": "stub"}

    def call(
        self,
        operation: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"operation": operation, "timeout": timeout})
        return dict(self.reply)

    def stop(self) -> None:
        """Stand-in for the worker shutdown (nothing to reap)."""


class _WedgedDirector:
    """Director stand-in: blocks inside `call` until released (wedged worker)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.release = threading.Event()

    def call(
        self,
        operation: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"operation": operation, "timeout": timeout})
        self.release.wait(timeout=60.0)
        return {"proposal": "late"}

    def stop(self) -> None:
        """Stand-in for the worker shutdown (nothing to reap)."""


def _stub_decision() -> EvolutionDecision:
    return EvolutionDecision(
        decision_index=0,
        destination=DirectorDestination(canonical_name="lantern valley"),
    )


def _stubbed_supervisor(
    run_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    director: _RecordingDirector | _WedgedDirector,
) -> tuple[Supervisor, ProjectConfig]:
    """Supervisor with stubbed payload builder and director (no subprocess)."""
    initialize_run_directory(run_directory)
    config, _digest = load_config(run_directory / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_directory, config)
    monkeypatch.setattr(Supervisor, "_decide_payload", lambda self, *args, **kwargs: {})
    supervisor._director = director  # type: ignore[assignment]
    supervisor._workers_running = True
    supervisor._prefetch_executor = ThreadPoolExecutor(max_workers=1)
    return supervisor, config


def _submit_prefetch(supervisor: Supervisor, config: ProjectConfig, run_directory: Path) -> None:
    store = ConceptStore(run_directory)
    style = StyleSpec(prompt="pastel neon line-art, peaceful")
    supervisor._prefetch_decide_for_next(config, 0, _stub_decision(), store, style)


def _shutdown_executor(supervisor: Supervisor) -> None:
    executor, supervisor._prefetch_executor = supervisor._prefetch_executor, None
    supervisor._prefetch_future = None
    supervisor._prefetch_target = None
    if executor is not None:
        executor.shutdown(wait=True)


def test_prefetch_call_carries_bounded_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The speculative decide carries an explicit short budget, not the 600 s default."""
    assert math.isfinite(PREFETCH_TIMEOUT_SECONDS)
    assert 0.0 < PREFETCH_TIMEOUT_SECONDS < 600.0
    run_directory = tmp_path / "run"
    director = _RecordingDirector()
    supervisor, config = _stubbed_supervisor(run_directory, monkeypatch, director)
    try:
        _submit_prefetch(supervisor, config, run_directory)
        future = supervisor._prefetch_future
        assert future is not None
        assert future.result(timeout=15.0) == {"proposal": "stub"}
        assert len(director.calls) == 1
        assert director.calls[0]["operation"] == "decide"
        assert director.calls[0]["timeout"] == PREFETCH_TIMEOUT_SECONDS
    finally:
        _shutdown_executor(supervisor)


def test_stop_workers_returns_promptly_with_wedged_prefetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stop_workers` never stalls behind a wedged prefetch thread."""
    run_directory = tmp_path / "run"
    director = _WedgedDirector()
    supervisor, config = _stubbed_supervisor(run_directory, monkeypatch, director)
    executor = supervisor._prefetch_executor
    _submit_prefetch(supervisor, config, run_directory)
    assert supervisor._prefetch_future is not None
    deadline = time.monotonic() + 10.0
    while not director.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert director.calls, "prefetch thread never entered the director call"
    started = time.monotonic()
    supervisor.stop_workers()
    elapsed = time.monotonic() - started
    assert elapsed < 5.0
    assert supervisor._prefetch_future is None
    assert supervisor._prefetch_executor is None
    assert director.calls[0]["timeout"] == PREFETCH_TIMEOUT_SECONDS
    director.release.set()
    assert executor is not None
    executor.shutdown(wait=True)
