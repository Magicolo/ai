"""Tests for the family catalog."""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from zoomy.engine_protocol import FRAME_SIZE_ALIGNMENT_PIXELS
from zoomy.errors import ZoomyError
from zoomy.family_catalog import FAMILY_CATALOG, find_family

KNOWN_FAMILY_KEYS = tuple(family.key for family in FAMILY_CATALOG)


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


@given(key=st.sampled_from(KNOWN_FAMILY_KEYS))
def test_find_family_roundtrip(key: str) -> None:
    """Every catalog key resolves to the family carrying it."""
    assert find_family(FAMILY_CATALOG, key).key == key


@given(key=st.text(max_size=30))
def test_find_family_rejects_any_unknown_key(key: str) -> None:
    """Anything outside the catalog raises, never returns a wrong family."""
    assume(key not in KNOWN_FAMILY_KEYS)
    with pytest.raises(ZoomyError, match="available families"):
        find_family(FAMILY_CATALOG, key)


def test_lora_strengths_are_positive() -> None:
    """Default strengths are usable slider values, never zero or negative."""
    for family in FAMILY_CATALOG:
        for lora in family.loras:
            assert lora.default_strength > 0


def test_ernie_presets_match_official_sizes() -> None:
    """Ernie offers exactly the seven official Baidu frame sizes."""
    ernie = find_family(FAMILY_CATALOG, "ernie_turbo")
    assert [(preset.width, preset.height) for preset in ernie.resolution_presets] == [
        (1024, 1024),
        (1264, 848),
        (848, 1264),
        (1376, 768),
        (768, 1376),
        (1200, 896),
        (896, 1200),
    ]


def test_z_presets_offer_draft_and_hd_sizes() -> None:
    """Z offers a fast 512 draft plus HD sizes for quick experiments."""
    for family_key in ("z_fast", "z_quality"):
        family = find_family(FAMILY_CATALOG, family_key)
        sizes = [(preset.width, preset.height) for preset in family.resolution_presets]
        assert (512, 512) in sizes
        assert (1376, 768) in sizes
        assert (768, 1376) in sizes


def test_every_preset_is_engine_aligned() -> None:
    """Every preset is a positive multiple of 16, so the VAE accepts it."""
    for family in FAMILY_CATALOG:
        assert family.resolution_presets
        for preset in family.resolution_presets:
            assert preset.display_name
            assert preset.width > 0
            assert preset.height > 0
            assert preset.width % FRAME_SIZE_ALIGNMENT_PIXELS == 0
            assert preset.height % FRAME_SIZE_ALIGNMENT_PIXELS == 0


def test_preset_display_names_are_unique_per_family() -> None:
    """Duplicate preset labels would make the dropdown ambiguous."""
    for family in FAMILY_CATALOG:
        names = [preset.display_name for preset in family.resolution_presets]
        assert len(names) == len(set(names))
