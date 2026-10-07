"""Periodic scene cut: every 3rd segment renders fresh (scene_cut=True)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.conftest import initialize_run_directory
from voyage.persistence import read_effective_config
from voyage.supervisor import SCENE_CUT_EVERY_N_SEGMENTS, Supervisor


def _streaming_supervisor(run_dir: Path) -> Supervisor:
    config = read_effective_config(run_dir)
    config.video.backend = "causvid"
    config.video.blocks_per_segment = 1
    return Supervisor(run_dir, config)


def _canned_proposal() -> Any:
    decision = SimpleNamespace(
        destination_concept="Resonant Vortex",
        phase="drift",
        novelty_accepted=True,
        transition=SimpleNamespace(mechanism="morph", intermediate_stages=["a", "b"]),
        audio=SimpleNamespace(
            music_caption="soft pulse", energy=0.5, texture="haze", environment=["dock"]
        ),
        notes="canned",
    )
    prompt_plan = SimpleNamespace(stages=[SimpleNamespace(prompt="stage zero")])
    from voyage.supervisor import ProposedSegment

    return ProposedSegment(
        decision=decision,
        prompt_plan=prompt_plan,
        block_prompts=["vortex evolving"],
        num_blocks=1,
        prefetch_hit=False,
        drift_hold=False,
        director_tokens={"prompt_tokens": 0, "completion_tokens": 0},
    )


def _render_scene_cuts(run_dir: Path, numbers: list[int]) -> list[list[bool]]:
    from voyage.supervisor import ProposedSegment  # noqa: F401  (re-export check)

    supervisor = _streaming_supervisor(run_dir)
    seen: list[dict[str, Any]] = []
    original = supervisor._call_with_restart

    def _stub(
        worker: Any,
        worker_name: str,
        segment_id: str,
        operation: str,
        payload: dict[str, object],
        restart_hook: Any = None,
    ) -> dict[str, object]:
        seen.append({"op": operation, "payload": dict(payload)})
        return {"video": {"frames": 96, "novel_frames": 96, "conditioning_frames": 25, "fps": 24}}

    supervisor._call_with_restart = _stub  # type: ignore[assignment]
    try:
        config = read_effective_config(run_dir)
        config.video.backend = "causvid"
        config.video.blocks_per_segment = 1
        state = SimpleNamespace(
            destination_concept="Resonant Vortex",
            current_concept="Luminous Fracture",
            timeline_frames=121,
        )
        cuts: list[list[bool]] = []
        for number in numbers:
            segment_id = f"{number:06d}"
            segment = run_dir / "segments" / segment_id
            segment.mkdir(parents=True, exist_ok=True)
            supervisor._render_video(
                config,
                state,  # type: ignore[arg-type]
                number,
                segment_id,
                segment,
                _canned_proposal(),
                {},
            )
            payload = seen[-1]["payload"]
            assert isinstance(payload["scene_cuts"], list)
            cuts.append(list(payload["scene_cuts"]))
        return cuts
    finally:
        supervisor._call_with_restart = original  # type: ignore[method-assign]


def test_cadence_constant_is_three() -> None:
    assert SCENE_CUT_EVERY_N_SEGMENTS == 3


def test_every_third_segment_cuts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="scene-cut-cadence")
    cuts = _render_scene_cuts(run_dir, [1, 2, 4, 5])
    assert cuts[0] == [False]
    assert cuts[1] == [True]
    assert cuts[2] == [False]
    assert cuts[3] == [True]


def test_first_segment_never_cuts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="scene-cut-first")
    (cuts,) = _render_scene_cuts(run_dir, [0])
    assert cuts == [False]
