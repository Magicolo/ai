"""Tests for the family catalog."""

from __future__ import annotations

import pytest

from zoomy.errors import ZoomyError
from zoomy.family_catalog import FAMILY_CATALOG, find_family


def test_family_keys_are_unique() -> None:
    """Every family key is unique so dropdown values never collide."""
    keys = [family.key for family in FAMILY_CATALOG]
    assert len(keys) == len(set(keys))


def test_z_families_share_one_sequence() -> None:
    """The fast and quality Z variants continue the same zoom sequence."""
    fast = find_family(FAMILY_CATALOG, "z_fast")
    quality = find_family(FAMILY_CATALOG, "z_quality")
    assert fast.sequence_key == quality.sequence_key == "z_image"


def test_ernie_family_settings() -> None:
    """Ernie uses its verified turbo recipe and zeroed negative conditioning."""
    ernie = find_family(FAMILY_CATALOG, "ernie_turbo")
    assert ernie.base_model_file == "ernie-image-turbo.safetensors"
    assert ernie.text_encoder_type == "flux2"
    assert ernie.model_shift is None
    assert ernie.sampler_name == "euler"
    assert ernie.scheduler_name == "simple"
    assert ernie.sampler_steps == 8
    assert ernie.classifier_free_guidance == 1.0
    assert ernie.negative_prompt is None


def test_z_fast_and_quality_sampler_recipes() -> None:
    """Each Z variant carries its verified sampler recipe."""
    fast = find_family(FAMILY_CATALOG, "z_fast")
    quality = find_family(FAMILY_CATALOG, "z_quality")
    assert (fast.sampler_name, fast.scheduler_name, fast.sampler_steps) == (
        "ddim",
        "normal",
        6,
    )
    assert fast.classifier_free_guidance == 1.0
    assert (quality.sampler_name, quality.scheduler_name, quality.sampler_steps) == (
        "res_multistep",
        "beta",
        22,
    )
    assert quality.classifier_free_guidance == 4.0
    assert fast.model_shift == 3.0
    assert quality.model_shift == 3.0


def test_ernie_lora_is_selected_by_default_and_z_loras_are_not() -> None:
    """Defaults mirror the source workflows: Ernie styled, Z photorealistic."""
    ernie = find_family(FAMILY_CATALOG, "ernie_turbo")
    assert [lora.selected_by_default for lora in ernie.loras] == [True]
    for family_key in ("z_fast", "z_quality"):
        family = find_family(FAMILY_CATALOG, family_key)
        assert [lora.selected_by_default for lora in family.loras] == [False, False]


def test_every_family_declares_complete_parameters() -> None:
    """No family ships with blank prompts or missing model files."""
    for family in FAMILY_CATALOG:
        assert family.default_prompt
        assert family.music_prompt
        assert family.sound_effect_prompt
        assert family.sound_effect_negative_prompt
        assert family.cold_start_image.endswith(".png")
        assert family.base_model_file.endswith(".safetensors")
        assert family.text_encoder_file.endswith(".safetensors")
        assert family.autoencoder_file.endswith(".safetensors")
        assert family.loras


def test_find_family_rejects_unknown_key() -> None:
    """An unknown key lists the available families in the error."""
    with pytest.raises(ZoomyError, match="available families"):
        find_family(FAMILY_CATALOG, "does_not_exist")
