"""Tests for the frame workflow builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zoomy.family_catalog import FAMILY_CATALOG, LoraDefinition, find_family
from zoomy.frame_workflow import (
    CROP_BORDER_PIXELS,
    FRAME_HEIGHT_PIXELS,
    FRAME_WIDTH_PIXELS,
    FrameRenderRequest,
    build_frame_workflow,
)

if TYPE_CHECKING:
    from typing import Any

    from zoomy.family_catalog import FamilyDefinition

CHALKBOARD_LORA = LoraDefinition(
    file_name="Chalkboard01-1_CE_ZIMG_AIT4k.safetensors",
    display_name="Chalkboard",
    default_strength=0.85,
    selected_by_default=False,
)
CLAY_ART_LORA = LoraDefinition(
    file_name="ClayArt01a_CE_ZIMG_AIT3k.safetensors",
    display_name="Clay Art",
    default_strength=0.85,
    selected_by_default=False,
)


def _build_document(
    family: FamilyDefinition,
    *,
    frame_count: int,
    lora_selections: tuple[tuple[LoraDefinition, float], ...] = (),
) -> dict[str, dict[str, Any]]:
    request = FrameRenderRequest(
        family=family,
        prompt="a prompt",
        negative_prompt="a negative",
        frame_count=frame_count,
        lora_selections=lora_selections,
        seed=12345,
        output_directory="/comfy/output",
    )
    return build_frame_workflow(request)


def test_cold_start_uses_the_seed_image() -> None:
    """Frame zero loads the family's cold-start image from ComfyUI input."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=0)
    assert "load_input_image" in document
    assert document["load_input_image"]["inputs"]["image"] == family.cold_start_image
    assert "load_previous_frame" not in document


def test_warm_start_loads_only_the_latest_frame() -> None:
    """Later frames read exactly one file: the newest of the sequence."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=5)
    loader_inputs = document["load_previous_frame"]["inputs"]
    assert loader_inputs["directory"] == "/comfy/output/Zoomy/z_image"
    assert loader_inputs["image_load_cap"] == 1
    assert loader_inputs["skip_first_images"] == 4
    assert loader_inputs["select_every_nth"] == 1
    assert "load_input_image" not in document


def test_loras_chain_in_selection_order() -> None:
    """Selected LoRAs apply in order, chaining model and clip through each."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    selections = ((CHALKBOARD_LORA, 0.5), (CLAY_ART_LORA, 0.9))
    document = _build_document(family, frame_count=1, lora_selections=selections)
    first_inputs = document["apply_lora_0"]["inputs"]
    second_inputs = document["apply_lora_1"]["inputs"]
    assert first_inputs["model"] == ["load_base_model", 0]
    assert first_inputs["clip"] == ["load_text_encoder", 0]
    assert first_inputs["strength_model"] == 0.5
    assert second_inputs["model"] == ["apply_lora_0", 0]
    assert second_inputs["clip"] == ["apply_lora_0", 1]
    assert second_inputs["strength_model"] == 0.9
    assert document["sample_next_frame"]["inputs"]["model"] == ["apply_model_shift", 0]


def test_without_loras_the_base_model_reaches_the_sampler() -> None:
    """With no LoRA selected the sampler sees the (shifted) base model."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=1)
    assert "apply_lora_0" not in document
    assert document["sample_next_frame"]["inputs"]["model"] == ["apply_model_shift", 0]


def test_model_shift_applies_only_when_the_family_defines_one() -> None:
    """Ernie has no shift node; Z families shift by 3.0."""
    ernie = find_family(FAMILY_CATALOG, "ernie_turbo")
    ernie_document = _build_document(ernie, frame_count=1)
    assert "apply_model_shift" not in ernie_document
    assert ernie_document["sample_next_frame"]["inputs"]["model"] == ["load_base_model", 0]
    z_fast = find_family(FAMILY_CATALOG, "z_fast")
    z_document = _build_document(z_fast, frame_count=1)
    assert z_document["apply_model_shift"]["inputs"]["shift"] == 3.0
    assert z_document["sample_next_frame"]["inputs"]["model"] == ["apply_model_shift", 0]


def test_negative_conditioning_follows_the_family() -> None:
    """Ernie zeroes its negative; Z families encode a real negative prompt."""
    ernie = find_family(FAMILY_CATALOG, "ernie_turbo")
    ernie_document = _build_document(ernie, frame_count=1)
    assert "encode_negative_prompt" not in ernie_document
    assert ernie_document["zero_negative_prompt"]["inputs"]["conditioning"] == ["encode_prompt", 0]
    assert ernie_document["sample_next_frame"]["inputs"]["negative"] == [
        "zero_negative_prompt",
        0,
    ]
    z_fast = find_family(FAMILY_CATALOG, "z_fast")
    z_document = _build_document(z_fast, frame_count=1)
    assert "zero_negative_prompt" not in z_document
    assert z_document["encode_negative_prompt"]["inputs"]["text"] == "a negative"
    assert z_document["sample_next_frame"]["inputs"]["negative"] == [
        "encode_negative_prompt",
        0,
    ]


def test_crop_and_rescale_implement_the_zoom_geometry() -> None:
    """The crop shaves the border; the rescale returns to full frame size."""
    family = find_family(FAMILY_CATALOG, "ernie_turbo")
    document = _build_document(family, frame_count=1)
    crop_inputs = document["crop_previous_frame"]["inputs"]
    assert crop_inputs["width"] == FRAME_WIDTH_PIXELS - 2 * CROP_BORDER_PIXELS
    assert crop_inputs["height"] == FRAME_HEIGHT_PIXELS - 2 * CROP_BORDER_PIXELS
    assert crop_inputs["x"] == CROP_BORDER_PIXELS
    assert crop_inputs["y"] == CROP_BORDER_PIXELS
    rescale_inputs = document["rescale_cropped_frame"]["inputs"]
    assert rescale_inputs["upscale_method"] == "bicubic"
    assert rescale_inputs["width"] == FRAME_WIDTH_PIXELS
    assert rescale_inputs["height"] == FRAME_HEIGHT_PIXELS
    assert rescale_inputs["crop"] == "disabled"


def test_sampler_carries_family_settings_and_request_seed() -> None:
    """The KSampler mixes family recipe values with the request's seed."""
    family = find_family(FAMILY_CATALOG, "z_quality")
    document = _build_document(family, frame_count=1)
    sampler_inputs = document["sample_next_frame"]["inputs"]
    assert sampler_inputs["seed"] == 12345
    assert sampler_inputs["steps"] == family.sampler_steps
    assert sampler_inputs["cfg"] == family.classifier_free_guidance
    assert sampler_inputs["sampler_name"] == family.sampler_name
    assert sampler_inputs["scheduler"] == family.scheduler_name
    assert sampler_inputs["denoise"] == family.denoise_strength
    assert sampler_inputs["latent_image"] == ["encode_frame_pixels", 0]


def test_save_prefix_targets_the_sequence_directory() -> None:
    """Frames save under Zoomy/<sequence_key>/ so the repository finds them."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=1)
    assert document["save_next_frame"]["inputs"]["filename_prefix"] == "Zoomy/z_image/frame"
    assert document["save_next_frame"]["inputs"]["images"] == ["decode_next_frame", 0]


def test_every_link_references_an_existing_node() -> None:
    """No input points at a node key the workflow does not define."""
    for family in FAMILY_CATALOG:
        document = _build_document(family, frame_count=2)
        assert_all_links_resolve(document)


def assert_all_links_resolve(document: dict[str, dict[str, Any]]) -> None:
    """Fail when any [key, slot] link targets a missing node key."""
    for node in document.values():
        for value in node["inputs"].values():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
                and not isinstance(value, str)
            ):
                assert value[0] in document, f"link {value} targets a missing node"
