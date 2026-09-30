"""Issues 136 + 168: prefetch `hit` logged before the discard.

`_take_prefetch` emitted `director_prefetch_hit` at consumption time, but
the proposal was then discarded — by fresh inspect amendments (136) or by
the drift-cadence hold (168, one call deeper) — so the soak hit-rate
counted proposals that never entered the accept loop. Shared fix: an
`invalidated` third outcome (`director_prefetch_invalidated`), covering
both discard sites; the hit-rate stays `hit / (hit + miss)` by
construction. Real fake-backend commits (real ffmpeg media), CPU-only.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import load_config
from voyage.segment_manifest import load_metrics
from voyage.supervisor import Supervisor


def _unstarted_supervisor(run_dir: Path) -> Supervisor:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
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


def test_drift_hold_logs_invalidated_not_hit(tmp_path: Path) -> None:
    """168 e2e: every held segment with a ready prefetch used to count a hit."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="hold168")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    from voyage.config import apply_draft_overrides

    config = apply_draft_overrides(config, drift_every_n_segments=2)
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
    invalidated = [line for line in seg1 if "director_prefetch_invalidated" in line]
    assert len(invalidated) == 1
    assert "drift_hold" in invalidated[0]


def test_amended_prefetch_logs_invalidated_not_hit(tmp_path: Path) -> None:
    """136 e2e: fake testsrc deterministically yields amendments, so the
    segment-1 prefetch is discarded — it must not count as a hit."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="amend136", seed=7, visual_inspector=True)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
        assert supervisor.commit_one_segment() == "000001"
    finally:
        supervisor.stop_workers()
    amendments = load_metrics(paths.segment_dir(run_dir, "000000"))["visual"]["amendments"]
    assert amendments != []
    events = _prefetch_events(run_dir)
    seg1 = [line for line in events if '"segment_id": "000001"' in line]
    assert not any("director_prefetch_hit" in line for line in seg1)
    invalidated = [line for line in seg1 if "director_prefetch_invalidated" in line]
    assert len(invalidated) == 1
    assert "amendments" in invalidated[0]
