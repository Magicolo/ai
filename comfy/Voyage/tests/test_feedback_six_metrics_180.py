"""Issue 180: `feedback_amendments` must steer on all six §43 metrics.

`palette_distance` + `scene_boundary_strength` were measured, labeled, and
then ignored (zero amendments at 0.99 vs the 0.30 ceiling the director
context itself prints). Charter decision: both steer, with `StyleSpec`
bands defaulting to the director-context ceilings. CPU-only.
"""

from __future__ import annotations

import pytest

from voyage.models import StyleSpec
from voyage.prompts import feedback_amendments


def _style() -> StyleSpec:
    return StyleSpec(prompt="line art")


def test_palette_distance_above_ceiling_amends() -> None:
    amendments = feedback_amendments({"palette_distance": 0.99}, _style())
    assert len(amendments) == 1


def test_scene_boundary_above_ceiling_amends() -> None:
    amendments = feedback_amendments({"scene_boundary_strength": 0.99}, _style())
    assert len(amendments) == 1


def test_all_six_metrics_steer_in_isolation() -> None:
    style = _style()
    assert feedback_amendments({"motion_energy": 0.99}, style) == [
        "calm static composition, minimal motion"
    ]
    assert feedback_amendments({"visual_complexity": 0.99}, style) == [
        "sparse composition, few simple shapes, large empty areas"
    ]
    assert feedback_amendments({"semantic_change_rate": 0.0}, style) == [
        "gradual visible transformation unfolding across the shot"
    ]
    assert feedback_amendments({"style_similarity": 0.0}, style) == [
        "strictly in the charter style, signature palette and linework"
    ]
    assert feedback_amendments({"palette_distance": 0.99}, style) == [
        "muted restrained palette, no blown highlights"
    ]
    assert feedback_amendments({"scene_boundary_strength": 0.99}, style) == [
        "single continuous shot, no cuts"
    ]


def test_in_band_palette_and_boundary_steer_nothing() -> None:
    assert feedback_amendments({"palette_distance": 0.10}, _style()) == []
    assert feedback_amendments({"scene_boundary_strength": 0.10}, _style()) == []


def test_nonfinite_palette_and_boundary_steer_nothing() -> None:
    assert feedback_amendments({"palette_distance": float("nan")}, _style()) == []
    assert feedback_amendments({"scene_boundary_strength": float("inf")}, _style()) == []


def test_charter_bands_are_configurable() -> None:
    style = StyleSpec(prompt="x", palette_distance_max=1.0, scene_boundary_strength_max=1.0)
    assert (
        feedback_amendments({"palette_distance": 0.99, "scene_boundary_strength": 0.99}, style)
        == []
    )
    with pytest.raises(ValueError):
        StyleSpec(prompt="x", palette_distance_max=2.0)
    with pytest.raises(ValueError):
        StyleSpec(prompt="x", scene_boundary_strength_max=-0.1)
