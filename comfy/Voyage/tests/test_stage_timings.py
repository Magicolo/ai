"""Per-stage wall-time breakdown in segment_committed metrics (slice 2).

Each committed segment records how long its stages took (inspect,
director, video, audio, validate, commit) so experiment runs reveal the
real bottleneck instead of one opaque elapsed number.
"""

from __future__ import annotations

import json
from pathlib import Path

from voyage import paths
from voyage.config import default_config_toml, load_config
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_state,
    write_manifest,
    write_state,
)
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "timings") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.CONFIG_FILENAME).write_text(
        default_config_toml(run_id, "pastel neon line-art, peaceful", 11),
        encoding="utf-8",
    )
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


def _committed_events(run_dir: Path) -> list[dict[str, object]]:
    lines = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in lines.splitlines() if line.strip()]
    return [e for e in events if e.get("event") == "segment_committed"]


def test_segment_committed_carries_stage_breakdown(tmp_path: Path) -> None:
    """segment_committed includes per-stage seconds that add up to elapsed."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
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
