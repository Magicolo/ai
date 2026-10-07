"""Benchmark scratch routing under run `tmp/` (issue 287).

CPU-only: the harness runs on fake clocks, the worker helpers read
in-memory `_INIT_PARAMS` — no GPU, no renders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voyage.workers import video_causvid, video_ltx23, video_ltx25, video_ltxv
from voyage.workers.video_common import run_benchmark_harness

_WORKERS: tuple[Any, ...] = (video_ltxv, video_causvid, video_ltx25, video_ltx23)


def test_harness_stages_inside_given_parent(tmp_path: Path) -> None:
    """`staging_parent` routes the scratch dir instead of bare /tmp."""
    seen: list[Path] = []
    ticks = iter([0.0, 1.0, 2.0, 3.0])

    def probe(output_path: Path, measured: bool) -> None:
        del measured
        seen.append(output_path)
        output_path.write_bytes(b"probe")

    staging = tmp_path / "run" / "tmp"
    staging.mkdir(parents=True)
    outcome = run_benchmark_harness(
        0, 1, "voyage-test-bench-", probe, clock=lambda: next(ticks), staging_parent=staging
    )
    assert len(outcome.wall_seconds) == 1
    assert seen and all(path.parent.parent == staging for path in seen)


def test_worker_staging_helper_returns_none_without_init() -> None:
    """Legacy/test callers without a recorded scratch keep TMPDIR behavior."""
    for worker in _WORKERS:
        original = dict(worker._INIT_PARAMS)
        worker._INIT_PARAMS.clear()
        try:
            assert worker._benchmark_staging_parent() is None
        finally:
            worker._INIT_PARAMS.update(original)


def test_worker_staging_helper_returns_recorded_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "run" / "tmp"
    scratch.mkdir(parents=True)
    for worker in _WORKERS:
        original = dict(worker._INIT_PARAMS)
        worker._INIT_PARAMS.update({"scratch_dir": str(scratch)})
        try:
            assert worker._benchmark_staging_parent() == scratch
        finally:
            worker._INIT_PARAMS.clear()
            worker._INIT_PARAMS.update(original)


def test_worker_staging_helper_ignores_blank_scratch() -> None:
    for worker in _WORKERS:
        original = dict(worker._INIT_PARAMS)
        worker._INIT_PARAMS.update({"scratch_dir": "   "})
        try:
            assert worker._benchmark_staging_parent() is None
        finally:
            worker._INIT_PARAMS.clear()
            worker._INIT_PARAMS.update(original)
