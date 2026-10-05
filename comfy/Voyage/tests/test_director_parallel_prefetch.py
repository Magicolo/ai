"""Director runs parallel to video generation (parallel-prefetch contract).

Regression cover for the sequential-director gap: the next segment's
proposal is decided in the background while the current segment renders,
then consumed (not re-decided) at the next commit. Three behaviors pin
the fix in `voyage/supervisor.py`:

1. `_take_prefetch` waits out a still-running future for this segment
   (bounded by its remaining RPC budget) instead of instantly missing
   and then queueing a second decide behind the orphan on the serial
   worker lock.
2. A hold target (`invalidated=True`) never waits — the proposal is
   discarded unread, so waiting would only stall the commit.
3. A prefetched proposal stays usable with motion-steering amendments
   (they apply post-hoc to both paths), and no prefetch is submitted
   for drift-hold targets (it would be consumed as `invalidated`
   while contending with prompt enhancement on the single-slot
   llama sidecar).

Real `Future` objects, CPU-only, no workers started except where noted.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.concepts import ConceptStore
from voyage.config import resolve_config
from voyage.models import (
    DirectorDestination,
    DirectorVideoPlan,
    EvolutionDecision,
    StyleSpec,
)
from voyage.persistence import read_effective_config, read_state
from voyage.supervisor import Supervisor


def _metric_events(run_dir: Path, event: str) -> list[dict[str, Any]]:
    metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    return [
        entry
        for entry in (
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if entry.get("event") == event
    ]


def _unstarted_supervisor(run_dir: Path) -> Supervisor:
    return Supervisor(run_dir, read_effective_config(run_dir))


def _valid_raw(index: int = 0) -> dict[str, Any]:
    """Accept-loop-passing raw: schema-valid, staged, style-clean, novel."""
    decision = EvolutionDecision(
        decision_index=index,
        destination=DirectorDestination(canonical_name=f"parallel valley {index}"),
        video=DirectorVideoPlan(stages=[f"a calm neon valley holds, stage {index}"]),
    )
    return dict(decision.model_dump())


def test_take_waits_for_inflight_prefetch(tmp_path: Path) -> None:
    """A still-running future for this segment is waited out, then hits."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    supervisor = _unstarted_supervisor(run_dir)
    future: Future[dict[str, Any] | None] = Future()
    supervisor._prefetch_future = future
    supervisor._prefetch_target = 3
    supervisor._prefetch_submitted_at = time.monotonic()
    raw = {"decision_index": 0}

    def _complete_late() -> None:
        time.sleep(0.5)
        future.set_result(raw)

    worker = threading.Thread(target=_complete_late, daemon=True)
    worker.start()
    started = time.monotonic()
    try:
        assert supervisor._take_prefetch(3, "000003") == raw
    finally:
        worker.join()
    assert time.monotonic() - started >= 0.4
    assert len(_metric_events(run_dir, "director_prefetch_hit")) == 1
    assert _metric_events(run_dir, "director_prefetch_miss") == []


def test_invalidated_take_never_waits(tmp_path: Path) -> None:
    """A hold discards the proposal unread — no waiting on a running future."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    supervisor = _unstarted_supervisor(run_dir)
    supervisor._prefetch_future = Future()
    supervisor._prefetch_target = 3
    supervisor._prefetch_submitted_at = time.monotonic()
    started = time.monotonic()
    assert (
        supervisor._take_prefetch(3, "000003", invalidated=True, invalidation_reason="drift_hold")
        is None
    )
    assert time.monotonic() - started < 10.0
    assert len(_metric_events(run_dir, "director_prefetch_miss")) == 1
    assert _metric_events(run_dir, "director_prefetch_hit") == []
    assert _metric_events(run_dir, "director_prefetch_invalidated") == []


def test_prefetch_serves_first_candidate_with_amendments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Motion steering no longer forces a redundant synchronous decide."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    supervisor = _unstarted_supervisor(run_dir)

    def _never_called(
        self: Supervisor,
        worker: Any,
        worker_name: str,
        segment_id: str,
        op: str,
        payload: dict[str, object],
        restart_hook: Any = None,
    ) -> dict[str, object]:
        raise AssertionError("sync decide must not run when the prefetch serves attempt 0")

    monkeypatch.setattr(Supervisor, "_call_with_restart", _never_called)
    monkeypatch.setattr(Supervisor, "_embed_texts", lambda self, texts: None)
    config = supervisor._config
    state = read_state(run_dir)
    store = ConceptStore(run_dir / "novelty")
    style = StyleSpec(prompt="pastel neon line-art, peaceful")
    decision, _tokens = supervisor._accept_director_decision(
        config,
        state,
        store,
        style,
        "000000",
        amendments=["gentle continuous motion throughout the shot"],
        prefetched_raw=_valid_raw(),
    )
    assert decision.destination_concept == "parallel valley 0"
    assert "gentle continuous motion throughout the shot" in decision.video.stages[0]


def test_prefetch_submit_skipped_for_drift_hold(tmp_path: Path) -> None:
    """No background decide is spent on a segment the hold will discard."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    config = resolve_config(read_effective_config(run_dir), drift_every_n_segments=2)
    supervisor = Supervisor(run_dir, config)
    supervisor._workers_running = True
    supervisor._prefetch_executor = ThreadPoolExecutor(max_workers=1)
    try:
        decision = EvolutionDecision(
            decision_index=0,
            destination=DirectorDestination(canonical_name="parallel valley 0"),
            video=DirectorVideoPlan(stages=["a calm neon valley holds, stage 0"]),
        )
        store = ConceptStore(run_dir / "novelty")
        style = StyleSpec(prompt="pastel neon line-art, peaceful")
        # Target 1 is a hold (1 % 2 != 0) — the submit must be skipped.
        supervisor._prefetch_decide_for_next(config, 0, decision, store, style)
        assert supervisor._prefetch_target is None
        assert supervisor._prefetch_future is None
    finally:
        supervisor._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        supervisor._workers_running = False
