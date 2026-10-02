"""Presentation-fps knob for slow-motion finalize (Phase 0).

When `presentation_fps` is set and the model pass interpolates by
`interp_multiplier`, the timeline stretches by
`source_fps * multiplier / presentation_fps` (e.g. 24fps x2 presented
at 32fps = 1.5x slow motion) instead of lifting fps with minterpolate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from pydantic import ValidationError

from voyage.config import AugmentConfig, default_config_toml, resolve_config
from voyage.media import plan_augmentation, slowmo_factor


def test_presentation_fps_defaults_to_none() -> None:
    assert AugmentConfig().presentation_fps is None


def test_presentation_fps_rejects_negative_and_maps_zero_to_none() -> None:
    with pytest.raises(ValidationError):
        AugmentConfig(presentation_fps=-1)
    # Zero from TOML means "unset", matching the min_fps 0-disables idiom.
    assert AugmentConfig(presentation_fps=0).presentation_fps is None


def test_presentation_fps_accepts_positive() -> None:
    assert AugmentConfig(presentation_fps=32).presentation_fps == 32


def test_plan_without_multiplier_keeps_minterpolate_lift() -> None:
    plan = plan_augmentation(768, 512, 24.0, 768, 512, 32, 24, 1216, 704)
    assert plan.out_fps == 32
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True


def test_plan_with_multiplier_suppresses_lift_for_slowmo() -> None:
    plan = plan_augmentation(768, 512, 24.0, 768, 512, 32, 24, 1216, 704, interp_multiplier=2)
    assert plan.out_fps == 32
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_plan_multiplier_covers_exact_output() -> None:
    plan = plan_augmentation(768, 512, 24.0, 768, 512, 48, 24, 1216, 704, interp_multiplier=2)
    assert plan.out_fps == 48
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_slowmo_factor_values() -> None:
    assert slowmo_factor(24.0, 2, 32) == pytest.approx(1.5)
    assert slowmo_factor(24.0, 4, 24) == pytest.approx(4.0)
    assert slowmo_factor(24.0, 1, 24) == pytest.approx(1.0)


def test_resolve_config_applies_presentation_fps(tmp_path: Path) -> None:
    from tests.conftest import initialize_run_directory
    from voyage import paths
    from voyage.config import load_config

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="pres", style="s")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert config.augment.presentation_fps is None
    resolved = resolve_config(config, presentation_fps=32)
    assert resolved.augment.presentation_fps == 32


def test_augment_overrides_passes_presentation_fps() -> None:
    from voyage.cli_core import _augment_overrides

    args = argparse.Namespace(
        no_augment=False,
        min_fps=None,
        min_resolution=None,
        use_model_pass=None,
        interp_multiplier=None,
        presentation_fps=32,
    )
    assert _augment_overrides(args)["presentation_fps"] == 32
    absent = argparse.Namespace(
        no_augment=False,
        min_fps=None,
        min_resolution=None,
        use_model_pass=None,
        interp_multiplier=None,
        presentation_fps=None,
    )
    assert "presentation_fps" not in _augment_overrides(absent)


def test_default_toml_parses_without_presentation_key(tmp_path: Path) -> None:
    import tomllib

    from voyage.config import ProjectConfig

    toml_text = default_config_toml("pres", "s", 0)
    assert "presentation_fps" not in toml_text
    data = tomllib.loads(toml_text)
    config = ProjectConfig(**data)
    assert config.augment.presentation_fps is None
