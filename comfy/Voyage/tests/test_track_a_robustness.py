"""Track A: refresh budget, progress-sink safety, enhance budget (DESIGN §73).

Covers `_refresh_video_session` restart accounting (worker_restart /
circuit_breaker_open, FAILED mapping parity), best-effort progress sink
wrapping, and the enhancement overall budget with per-prompt notes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage.errors import FatalWorkerError, RecoverableWorkerError
from voyage.persistence import read_effective_config, read_state
from voyage.supervisor import Supervisor


class _ExplodingProgress:
    """Progress sink whose every method raises (display must never fail)."""

    verbose = False

    def note(self, _message: str) -> None:
        raise RuntimeError("sink boom")

    def bar(self, _label: str) -> Any:
        raise RuntimeError("sink boom")

    def stage(self, _label: str, _detail: str = "") -> Any:
        raise RuntimeError("sink boom")

    def segment_start(self, _number: int, _segment_id: str) -> None:
        raise RuntimeError("sink boom")

    def segment_plan(self, _plan: dict[str, Any]) -> None:
        raise RuntimeError("sink boom")

    def segment_done(self, _summary: dict[str, Any]) -> None:
        raise RuntimeError("sink boom")


class _NotesProgress:
    """Recording sink for enhance/prefetch note assertions."""

    def __init__(self) -> None:
        self.notes: list[str] = []
        self.verbose = False

    def note(self, message: str) -> None:
        self.notes.append(message)


def _metrics_text(run_dir: Path) -> str:
    from voyage import paths

    return (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")


def test_refresh_restart_failure_uses_budget_and_trips_breaker(tmp_path: Path) -> None:
    """A failing proactive restart emits worker_restart then breaker."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-refresh-budget")
    config = read_effective_config(run_dir)
    config.voyage.max_worker_restarts = 1
    supervisor = Supervisor(run_dir, config)

    def _dead_restart() -> None:
        raise RecoverableWorkerError("init boom")

    supervisor._video.restart = _dead_restart  # type: ignore[method-assign]
    supervisor._resume_video_worker = lambda segment_id: None  # type: ignore[method-assign]
    with pytest.raises(FatalWorkerError, match="circuit breaker"):
        supervisor._refresh_video_session("000000")
    metrics = _metrics_text(run_dir)
    assert '"event": "worker_restart"' in metrics
    assert '"event": "circuit_breaker_open"' in metrics


def test_progress_sink_never_fails_commit(tmp_path: Path) -> None:
    """An exploding sink still commits (best-effort display only)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-progress-safe")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config, progress=_ExplodingProgress())  # type: ignore[arg-type]
    committed = supervisor.run_segments(1)
    assert committed == ["000000"]
    assert read_state(run_dir).committed_segments == 1


def test_enhance_disabled_returns_inputs_untouched(tmp_path: Path) -> None:
    """Knob off (or non-LTX backend) never touches prompts or the network."""
    from voyage import prompt_enhancer

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-enhance-off")
    supervisor = Supervisor(run_dir, read_effective_config(run_dir), progress=None)

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("sidecar must not be called when disabled")

    import unittest.mock as mock

    prompts = ["a teapot pour", "a neon reef"]
    with mock.patch.object(prompt_enhancer, "enhance", _boom):
        out, summary = supervisor._enhance_prompts_with_budget(
            prompts, enabled=False, backend="ltx25", endpoint="http://x", segment_id="000000"
        )
    assert out == prompts
    assert summary["prompts_total"] == 2
    assert summary["prompts_changed"] == 0


def test_enhance_per_prompt_notes_and_stop_skips_rest(tmp_path: Path) -> None:
    """Per-prompt notes fire; a stop skips remaining expansions."""
    from typing import cast

    from voyage import prompt_enhancer

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-enhance-notes")
    progress = _NotesProgress()
    supervisor = Supervisor(run_dir, read_effective_config(run_dir), progress=cast(Any, progress))

    def _fake_enhance(
        text: str, endpoint: str = "", timeout: float = 0.0
    ) -> tuple[str, dict[str, int]]:
        supervisor.request_stop()
        return text + "-expanded", {"prompt_tokens": 1, "completion_tokens": 1}

    import unittest.mock as mock

    with mock.patch.object(prompt_enhancer, "enhance", _fake_enhance):
        out, summary = supervisor._enhance_prompts_with_budget(
            ["one", "two", "three"],
            enabled=True,
            backend="ltx25",
            endpoint="http://x",
            segment_id="000000",
        )
    assert out[0] == "one-expanded"
    assert out[1:] == ["two", "three"]
    assert summary["prompts_changed"] == 1
    assert any("enhancing prompt 1/3" in note for note in progress.notes)


def test_enhance_overall_budget_exhaustion_degrades(tmp_path: Path) -> None:
    """A zero overall budget degrades to inputs without calling the sidecar."""
    from typing import cast

    import voyage.supervisor as supervisor_module
    from voyage import prompt_enhancer

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-enhance-budget")
    progress = _NotesProgress()
    supervisor = Supervisor(run_dir, read_effective_config(run_dir), progress=cast(Any, progress))

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("sidecar must not be called past the budget")

    import unittest.mock as mock

    with (
        mock.patch.object(prompt_enhancer, "enhance", _boom),
        mock.patch.object(supervisor_module, "ENHANCE_OVERALL_BUDGET_SECONDS", 0.0),
    ):
        out, _summary = supervisor._enhance_prompts_with_budget(
            ["one", "two"], enabled=True, backend="ltx25", endpoint="http://x", segment_id="000000"
        )
    assert out == ["one", "two"]
    assert any("budget exhausted" in note for note in progress.notes)
