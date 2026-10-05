"""LTX always-continue: drift never forces a fresh segment (issue ltx25-compare).

TDD red test for Track A: supervisor._render_video must send
scene_cut=False / scene_cuts=[False, ...] even when
destination_concept != current_concept. Fresh happens only when the
worker has no tail (first segment / missing tail), never on drift.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.conftest import initialize_run_directory
from voyage.persistence import read_effective_config
from voyage.supervisor import ProposedSegment, Supervisor


def _streaming_supervisor(run_dir: Path) -> Supervisor:
    config = read_effective_config(run_dir)
    config.video.backend = "causvid"
    config.video.blocks_per_segment = 1
    return Supervisor(run_dir, config)


def _canned_proposal() -> ProposedSegment:
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
    return ProposedSegment(
        decision=decision,  # type: ignore[arg-type]
        prompt_plan=prompt_plan,  # type: ignore[arg-type]
        block_prompts=["vortex evolving"],
        num_blocks=1,
        prefetch_hit=False,
        drift_hold=False,
        director_tokens={"prompt_tokens": 0, "completion_tokens": 0},
    )


def _stub_video_call(
    supervisor: Supervisor,
    result: dict[str, object],
    seen: list[dict[str, Any]],
) -> Any:
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
        return dict(result)

    supervisor._call_with_restart = _stub  # type: ignore[assignment]
    return original


def test_render_video_never_cuts_on_drift(tmp_path: Path) -> None:
    """Destination != current must still continue (scene_cuts all False)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="ltx-always")
    supervisor = _streaming_supervisor(run_dir)
    seen: list[dict[str, Any]] = []
    original = _stub_video_call(
        supervisor,
        {"video": {"frames": 96, "novel_frames": 96, "conditioning_frames": 25, "fps": 24}},
        seen,
    )
    try:
        config = read_effective_config(run_dir)
        config.video.backend = "causvid"
        config.video.blocks_per_segment = 1
        state = SimpleNamespace(
            destination_concept="Resonant Vortex",
            current_concept="Luminous Fracture",
            timeline_frames=121,
        )
        segment = run_dir / "segments" / "000001"
        segment.mkdir(parents=True, exist_ok=True)
        supervisor._render_video(
            config,
            state,  # type: ignore[arg-type]
            1,
            "000001",
            segment,
            _canned_proposal(),
            {},
        )
        assert len(seen) == 1
        payload = seen[0]["payload"]
        assert payload["scene_cuts"] == [False]
    finally:
        supervisor._call_with_restart = original  # type: ignore[method-assign]


def test_render_video_first_segment_still_fresh_capable(tmp_path: Path) -> None:
    """Single-block payload shape stays [False] (worker decides fresh by tail)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="ltx-always-fresh")
    supervisor = _streaming_supervisor(run_dir)
    seen: list[dict[str, Any]] = []
    original = _stub_video_call(
        supervisor,
        {"video": {"frames": 121, "novel_frames": 121, "conditioning_frames": 0, "fps": 24}},
        seen,
    )
    try:
        config = read_effective_config(run_dir)
        config.video.backend = "causvid"
        config.video.blocks_per_segment = 1
        state = SimpleNamespace(
            destination_concept="Luminous Fracture",
            current_concept="Luminous Fracture",
            timeline_frames=0,
        )
        segment = run_dir / "segments" / "000000"
        segment.mkdir(parents=True, exist_ok=True)
        supervisor._render_video(
            config,
            state,  # type: ignore[arg-type]
            0,
            "000000",
            segment,
            _canned_proposal(),
            {},
        )
        assert seen[0]["payload"]["scene_cuts"] == [False]
    finally:
        supervisor._call_with_restart = original  # type: ignore[method-assign]
