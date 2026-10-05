"""Presentation-fps knob for slow-motion finalize (Phase 0).

When `presentation_fps` is set and the model pass interpolates by
`interpolate`, the timeline stretches by
`source_fps * interpolate / presentation_fps` (e.g. 24fps x2 presented
at 32fps = 1.5x slow motion) instead of lifting fps with minterpolate.
Unset (`None`, the default) ships `round(source_fps * interpolate)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from pydantic import ValidationError

from voyage.config import AugmentConfig, preset_config, resolve_config
from voyage.media import plan_augmentation, slowmo_factor
from voyage.persistence import read_effective_config


def test_presentation_fps_defaults_to_unset() -> None:
    assert AugmentConfig().presentation_fps is None


def test_presentation_fps_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        AugmentConfig(presentation_fps=0)
    with pytest.raises(ValidationError):
        AugmentConfig(presentation_fps=-1)


def test_presentation_fps_accepts_positive() -> None:
    assert AugmentConfig(presentation_fps=32).presentation_fps == 32


def test_plan_unset_presentation_pins_interpolated_rate() -> None:
    plan = plan_augmentation(768, 512, 24.0, interpolate=2)
    assert plan.out_fps == 48
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True


def test_plan_with_presentation_suppresses_lift_for_slowmo() -> None:
    plan = plan_augmentation(
        768, 512, 24.0, interpolate=2, presentation_fps=32, model_interpolate=2
    )
    assert plan.out_fps == 32
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_plan_model_lift_covers_exact_output() -> None:
    plan = plan_augmentation(
        768, 512, 24.0, interpolate=2, presentation_fps=48, model_interpolate=2
    )
    assert plan.out_fps == 48
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_plan_presentation_without_model_lift_keeps_minterpolate() -> None:
    plan = plan_augmentation(768, 512, 24.0, presentation_fps=32)
    assert plan.out_fps == 32
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True


def test_slowmo_factor_values() -> None:
    assert slowmo_factor(24.0, 2, 32) == pytest.approx(1.5)
    assert slowmo_factor(24.0, 4, 24) == pytest.approx(4.0)
    assert slowmo_factor(24.0, 1, 24) == pytest.approx(1.0)


def test_resolve_config_applies_presentation_fps(tmp_path: Path) -> None:
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="pres", style="s")
    config = read_effective_config(run_dir)
    assert config.augment.presentation_fps is None
    resolved = resolve_config(config, presentation_fps=32)
    assert resolved.augment.presentation_fps == 32


def test_augment_overrides_passes_presentation_fps() -> None:
    from voyage.cli_core import _augment_overrides

    args = argparse.Namespace(
        upscale=None,
        interpolate=None,
        presentation_fps=32,
    )
    assert _augment_overrides(args)["presentation_fps"] == 32
    absent = argparse.Namespace(
        upscale=None,
        interpolate=None,
        presentation_fps=None,
    )
    assert "presentation_fps" not in _augment_overrides(absent)


def test_preset_parses_without_presentation_key(tmp_path: Path) -> None:

    config = preset_config("pres", "s", 0)
    assert config.augment.presentation_fps is None
