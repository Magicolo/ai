"""Inspector wiring tests: amendment safety + no supervisor piggyback.

The supervisor-side visual-inspector piggyback was removed: commits never
gain a `visual` metrics section, so `inspect scoreboard` rows carry
`metrics=None`. The worker `inspect` op, the registry weight pins, and the
prompt amendment helpers stay (the scoreboard standalone tooling reads
whatever `visual.metrics` a segment carries). The enabled-path merge test
is gone with the piggyback; the amendment-markers property survives.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.errors import ProposalRejected
from voyage.models import StyleSpec
from voyage.persistence import read_effective_config, read_state
from voyage.prompts import (
    apply_feedback_amendments,
    check_prompt_against_style,
    feedback_amendments,
)
from voyage.supervisor import Supervisor

VISUAL_METRIC_KEYS = (
    "motion_energy",
    "visual_complexity",
    "semantic_change_rate",
    "palette_distance",
    "style_similarity",
    "scene_boundary_strength",
)


def _init_run(run_dir: Path) -> None:
    initialize_run_directory(run_dir, run_id="iwire", seed=7)


def _read_metrics(run_dir: Path, segment_id: str) -> dict[str, object]:
    from voyage.segment_manifest import load_metrics

    raw = load_metrics(paths.segment_dir(run_dir, segment_id))
    assert isinstance(raw, dict)
    return raw


def test_committed_run_writes_no_visual_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
    finally:
        supervisor.stop_workers()
    assert "visual" not in _read_metrics(run_dir, "000000")
    assert read_state(run_dir).committed_segments == 1


def test_amendments_never_trip_style_markers() -> None:
    style = StyleSpec(prompt="pastel neon line-art, peaceful")
    extremes = [
        {"motion_energy": 0.0, "visual_complexity": 0.0, "semantic_change_rate": 1.0},
        {"motion_energy": 1.0, "visual_complexity": 1.0, "semantic_change_rate": 0.0},
        {"palette_distance": 1.0, "style_similarity": 0.0, "scene_boundary_strength": 1.0},
    ]
    for measured in extremes:
        amended = apply_feedback_amendments(
            "a calm neon landscape", feedback_amendments(measured, style)
        )
        try:
            check_prompt_against_style(amended, style)
        except ProposalRejected as exc:
            raise AssertionError(f"amendment tripped style markers: {measured}") from exc
