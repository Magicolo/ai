"""Issue 119: `models.py` calibration/plan knobs need range validation.

`bands_ordered` rejects `min > max` but accepts ordered-yet-out-of-domain
bands (`5.0..6.0` flags every segment BELOW forever) and unbounded scalars
(`style_similarity_min=99`, `energy=999`, negative durations/sizes). The
model layer is the single source everything else trusts — fail one bad
value loudly at parse time. CPU-only.
"""

from __future__ import annotations

import pytest

from voyage.models import (
    ArtifactRef,
    DirectorAudioPlan,
    PromptStage,
    StyleSpec,
    TransitionPlan,
)


def test_stylespec_rejects_out_of_unit_bands() -> None:
    with pytest.raises(ValueError):
        StyleSpec(prompt="x", motion_energy_min=5.0, motion_energy_max=6.0)
    with pytest.raises(ValueError):
        StyleSpec(prompt="x", style_similarity_min=99.0)
    with pytest.raises(ValueError):
        StyleSpec(prompt="x", surrealism=5.0)
    with pytest.raises(ValueError):
        StyleSpec(prompt="x", transition_smoothness=-2.0)


def test_stylespec_accepts_unit_boundaries() -> None:
    style = StyleSpec(
        prompt="x",
        motion_energy_min=0.0,
        motion_energy_max=1.0,
        visual_complexity_min=0.0,
        visual_complexity_max=1.0,
        semantic_drift_min=0.0,
        semantic_drift_max=1.0,
        style_similarity_min=0.0,
        surrealism=0.0,
        transition_smoothness=1.0,
    )
    assert style.motion_energy_max == 1.0


def test_audio_plan_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        DirectorAudioPlan(energy=999.0)
    with pytest.raises(ValueError):
        DirectorAudioPlan(energy=-0.5)
    with pytest.raises(ValueError):
        DirectorAudioPlan(tempo_bpm=-120)
    with pytest.raises(ValueError):
        DirectorAudioPlan(tempo_bpm=0)


def test_transition_plan_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        TransitionPlan(transition_strength=42.0)
    with pytest.raises(ValueError):
        TransitionPlan(transition_strength=-1.0)
    with pytest.raises(ValueError):
        TransitionPlan(estimated_duration_seconds=-1.0)


def test_artifact_ref_rejects_empty_path_and_negative_bytes() -> None:
    with pytest.raises(ValueError):
        ArtifactRef(path="")
    with pytest.raises(ValueError):
        ArtifactRef(path="seg/video.mp4", bytes=-10)


def test_prompt_stage_rejects_negative_block_start() -> None:
    with pytest.raises(ValueError):
        PromptStage(stage=0, block_start=-3, block_end=5, prompt="x")


def test_valid_models_still_construct() -> None:
    assert DirectorAudioPlan().tempo_bpm == 74
    assert TransitionPlan().estimated_duration_seconds == 64.0
    assert ArtifactRef(path="seg/video.mp4").bytes is None
    assert PromptStage(stage=0, block_start=2, block_end=5, prompt="x").block_start == 2
