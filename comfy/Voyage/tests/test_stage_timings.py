"""Per-stage wall-time breakdown in segment_committed metrics (slice 2).

Each committed segment records how long its stages took (inspect,
director, video, audio, validate, commit) so experiment runs reveal the
real bottleneck instead of one opaque elapsed number.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import load_config
from voyage.persistence import read_state
from voyage.supervisor import Supervisor


def _committed_events(run_dir: Path) -> list[dict[str, object]]:
    lines = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in lines.splitlines() if line.strip()]
    return [e for e in events if e.get("event") == "segment_committed"]


def test_segment_committed_carries_stage_breakdown(tmp_path: Path) -> None:
    """segment_committed includes per-stage seconds that add up to elapsed."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="timings")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
    finally:
        supervisor.stop_workers()
    assert read_state(run_dir).committed_segments == 1

    events = _committed_events(run_dir)
    assert len(events) == 1
    stages = events[0].get("stages")
    assert isinstance(stages, dict)
    assert set(stages) == {"inspect", "director", "video", "audio", "validate", "commit"}
    elapsed = events[0]["elapsed_seconds"]
    assert isinstance(elapsed, (int, float))
    total = 0.0
    for name, seconds in stages.items():
        assert isinstance(seconds, (int, float)), name
        assert seconds >= 0.0, name
        assert seconds <= elapsed, name
        total += float(seconds)
    assert total <= elapsed + 0.01  # +10ms: six round(x, 3) stages can sum 3ms over
