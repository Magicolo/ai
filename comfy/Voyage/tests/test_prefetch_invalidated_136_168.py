"""Prefetch `hit` logged before the discard (issue 168; 136 retired).

`_take_prefetch` emitted `director_prefetch_hit` at consumption time, but
the proposal was then discarded by the drift-cadence hold — so the soak
hit-rate counted proposals that never entered the accept loop. Fix: an
`invalidated` third outcome (`director_prefetch_invalidated`); the
hit-rate stays `hit / (hit + miss)` by construction. (Issue 136's
amendment-discard site is gone with the piggyback inspector; the
`invalidated` unit path below keeps covering the logging shape.) Real
fake-backend commits (real ffmpeg media), CPU-only.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _unstarted_supervisor(run_dir: Path) -> Supervisor:
    config = read_effective_config(run_dir)
    return Supervisor(run_dir, config)


def _ready_supervisor(run_dir: Path, number: int, raw: dict[str, Any]) -> Supervisor:
    supervisor = _unstarted_supervisor(run_dir)
    future: Future[dict[str, Any] | None] = Future()
    future.set_result(raw)
    supervisor._prefetch_future = future
    supervisor._prefetch_target = number
    supervisor._prefetch_submitted_at = time.monotonic()
    return supervisor


def _prefetch_events(run_dir: Path) -> list[str]:
    return [
        line
        for line in (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if "prefetch" in line
    ]


def test_usable_prefetch_still_logs_hit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    supervisor = _ready_supervisor(run_dir, 3, {"decision_index": 0})
    assert supervisor._take_prefetch(3, "000003") == {"decision_index": 0}
    events = _prefetch_events(run_dir)
    assert any("director_prefetch_hit" in line for line in events)
    assert not any("director_prefetch_invalidated" in line for line in events)


def test_invalidated_prefetch_logs_invalidated_not_hit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    supervisor = _ready_supervisor(run_dir, 3, {"decision_index": 0})
    assert (
        supervisor._take_prefetch(3, "000003", invalidated=True, invalidation_reason="amendments")
        is None
    )
    events = _prefetch_events(run_dir)
    assert not any("director_prefetch_hit" in line for line in events)
    invalidated = [line for line in events if "director_prefetch_invalidated" in line]
    assert len(invalidated) == 1
    assert "amendments" in invalidated[0]


def test_drift_hold_skips_prefetch_submit_logs_miss(tmp_path: Path) -> None:
    """168 e2e: held segments no longer spend a background decide at all.

    The submit for a drift-hold target is skipped (it would be consumed
    as `invalidated` while contending with prompt enhancement on the
    single-slot sidecar), so the held segment logs a plain `miss` —
    not a `hit`, and no `invalidated` either.
    """
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="hold168")
    config = read_effective_config(run_dir)
    from voyage.config import resolve_config

    config = resolve_config(config, drift_every_n_segments=2)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
        assert supervisor.commit_one_segment() == "000001"
    finally:
        supervisor.stop_workers()
    events = _prefetch_events(run_dir)
    seg1 = [line for line in events if '"segment_id": "000001"' in line]
    assert not any("director_prefetch_hit" in line for line in seg1)
    assert not any("director_prefetch_invalidated" in line for line in seg1)
    assert any("director_prefetch_miss" in line for line in seg1)
