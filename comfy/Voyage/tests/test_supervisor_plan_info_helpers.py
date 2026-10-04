"""Plan-info agreement (issue 081 extraction from `supervisor`).

`_segment_plan_info` is the verbatim console-plan dict moved to
`voyage.supervisor_plan_info` as `segment_plan_info` so `supervisor.py`
shrinks toward the §12 split signal. Behavior contract: identical to
the pre-split `Supervisor._segment_plan_info` — the retained method
delegates (no fork), planned-frames fall back to the configured segment
size on missing/non-int/non-positive reports, and seeds/cuts normalize
to lists — and the facade re-export is the same object (single source,
not a copy).
"""

from __future__ import annotations

from pathlib import Path

import voyage.supervisor as supervisor
import voyage.supervisor_plan_info as supervisor_plan_info
from tests.conftest import initialize_run_directory
from voyage.config import ProjectConfig
from voyage.director import PHASE_ORDER, DeterministicDirector
from voyage.models import EvolutionDecision
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor
from voyage.supervisor_plan_info import segment_plan_info


def _harness(tmp_path: Path) -> tuple[Supervisor, ProjectConfig, EvolutionDecision]:
    """Initialized run + config + deterministic decision (fake, no GPU)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="supervisor-plan-info")
    config = read_effective_config(run_dir)
    instance = Supervisor(run_dir, config)
    decision = DeterministicDirector("test style").propose(
        decision_index=0,
        current_concept="plain",
        destination_concept="meadow",
        phase=PHASE_ORDER[0],
    )
    return instance, config, decision


def test_facade_reexport_is_single_sourced() -> None:
    """The facade name is the new home object, not a copy (issue 081)."""
    assert supervisor.segment_plan_info is supervisor_plan_info.segment_plan_info


def test_method_agrees_with_moved_function(tmp_path: Path) -> None:
    """The retained `Supervisor` method delegates (no fork)."""
    instance, config, decision = _harness(tmp_path)
    payload: dict[str, object] = {"frames": 12, "seeds": [1, 2], "scene_cuts": []}
    via_method = instance._segment_plan_info(
        config, 1, "000001", decision, ["prompt"], payload, 1, False, False
    )
    via_function = segment_plan_info(
        config, 1, "000001", decision, ["prompt"], payload, 1, False, False
    )
    assert via_method == via_function
    assert via_method["planned_frames"] == 12
    assert via_method["segment_id"] == "000001"


def test_missing_frames_fall_back_to_config(tmp_path: Path) -> None:
    """A payload without frames uses the configured segment size."""
    instance, config, decision = _harness(tmp_path)
    info = instance._segment_plan_info(
        config, 2, "000002", decision, ["prompt"], {}, 1, True, False
    )
    assert info["planned_frames"] == config.video.segment_frames
    assert info["planned_duration"] == config.video.segment_frames / config.video.fps


def test_non_positive_frames_fall_back_to_config(tmp_path: Path) -> None:
    """Non-int/non-positive frame reports never poison the plan."""
    instance, config, decision = _harness(tmp_path)
    for bad in (0, -5, "twelve"):
        info = segment_plan_info(
            config, 3, "000003", decision, ["p"], {"frames": bad}, 1, False, True
        )
        assert info["planned_frames"] == config.video.segment_frames


def test_seeds_and_cuts_normalize_to_lists(tmp_path: Path) -> None:
    """Scalar seeds and cuts normalize; geometry/fps mirror the config."""
    instance, config, decision = _harness(tmp_path)
    info = segment_plan_info(
        config,
        4,
        "000004",
        decision,
        ["p1", "p2"],
        {"frames": 8, "seed": 7, "scene_cuts": "cut"},
        2,
        False,
        False,
    )
    assert info["video_seeds"] == [7]
    assert info["scene_cuts"] == []
    assert info["geometry"] == f"{config.video.width}x{config.video.height}"
    assert info["fps"] == config.video.fps
    assert info["blocks"] == 2
