"""Tests for the finalize workflow builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zoomy.family_catalog import FAMILY_CATALOG, find_family
from zoomy.finalize_workflow import (
    FinalizeRequest,
    build_finalize_workflow,
    compute_audio_seconds,
    compute_interpolated_frame_count,
)

if TYPE_CHECKING:
    from typing import Any

    from zoomy.family_catalog import FamilyDefinition


def _build_document(family: FamilyDefinition, *, frame_count: int) -> dict[str, dict[str, Any]]:
    request = FinalizeRequest(
        family=family,
        frame_count=frame_count,
        output_directory="/comfy/output",
    )
    return build_finalize_workflow(request)


def test_interpolated_frame_count_formula() -> None:
    """Interpolation yields (n - 1) * 4 + 1 frames."""
    assert compute_interpolated_frame_count(1) == 1
    assert compute_interpolated_frame_count(4) == 13
    assert compute_interpolated_frame_count(20) == 77


def test_audio_seconds_floors_at_the_sync_minimum() -> None:
    """Short sequences clamp to 1.0s; longer ones scale at 32 fps."""
    assert compute_audio_seconds(13) == 1.0
    assert compute_audio_seconds(32) == 1.0
    assert compute_audio_seconds(77) == pytest.approx(77 / 32)


def test_duration_flows_to_every_audio_node() -> None:
    """The computed seconds reach the music encoder, noise, and sampler."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=4)
    assert document["encode_music_prompt"]["inputs"]["duration"] == 1.0
    assert document["create_music_noise"]["inputs"]["seconds"] == 1.0
    assert document["sample_sound_effects"]["inputs"]["duration"] == 1.0


def test_frame_sequence_loader_reads_every_frame() -> None:
    """The loader streams the whole directory without caps or skips."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=4)
    loader_inputs = document["load_frame_sequence"]["inputs"]
    assert loader_inputs["directory"] == "/comfy/output/Zoomy/z_image"
    assert loader_inputs["image_load_cap"] == 0
    assert loader_inputs["skip_first_images"] == 0
    assert loader_inputs["select_every_nth"] == 1


def test_interpolation_and_padding_settings() -> None:
    """FILM multiplies by four and the pad node guarantees 17 frames."""
    family = find_family(FAMILY_CATALOG, "ernie_turbo")
    document = _build_document(family, frame_count=4)
    interpolate_inputs = document["interpolate_frames"]["inputs"]
    assert interpolate_inputs["multiplier"] == 4
    assert interpolate_inputs["images"] == ["load_frame_sequence", 0]
    assert document["pad_interpolated_frames"]["inputs"] == {
        "images": ["interpolate_frames", 0],
        "min_frames": 17,
    }


def test_video_combines_unpadded_frames_with_merged_audio() -> None:
    """The video uses the raw interpolation; only MMAudio sees the padded copy."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=4)
    video_inputs = document["render_video"]["inputs"]
    assert video_inputs["images"] == ["interpolate_frames", 0]
    assert video_inputs["audio"] == ["merge_audio_tracks", 0]
    assert video_inputs["frame_rate"] == 32
    assert video_inputs["loop_count"] == 0
    assert video_inputs["pingpong"] is False
    assert video_inputs["save_output"] is True
    assert video_inputs["format"] == "video/h264-mp4"
    assert video_inputs["filename_prefix"] == "Zoomy_z_image"


def test_music_chain_carries_the_proven_recipe() -> None:
    """ACE-Step runs with the settings verified against the source workflow."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=4)
    encoder_inputs = document["encode_music_prompt"]["inputs"]
    assert encoder_inputs["tags"] == family.music_prompt
    assert encoder_inputs["lyrics"] == ""
    assert encoder_inputs["seed"] == 31
    assert encoder_inputs["bpm"] == 100
    assert encoder_inputs["keyscale"] == "E minor"
    sampler_inputs = document["sample_music"]["inputs"]
    assert sampler_inputs["steps"] == 8
    assert sampler_inputs["cfg"] == 1.0
    assert sampler_inputs["sampler_name"] == "euler"
    assert sampler_inputs["scheduler"] == "simple"
    assert sampler_inputs["denoise"] == 1.0
    assert document["decode_music"]["class_type"] == "VAEDecodeAudio"


def test_sound_effect_chain_carries_the_proven_recipe() -> None:
    """MMAudio runs at the verified steps, guidance, and attenuation."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=4)
    sampler_inputs = document["sample_sound_effects"]["inputs"]
    assert sampler_inputs["prompt"] == family.sound_effect_prompt
    assert sampler_inputs["negative_prompt"] == family.sound_effect_negative_prompt
    assert sampler_inputs["steps"] == 25
    assert sampler_inputs["cfg"] == 4.5
    assert sampler_inputs["seed"] == 7
    assert sampler_inputs["mask_away_clip"] is True
    assert sampler_inputs["force_offload"] is True
    assert sampler_inputs["images"] == ["pad_interpolated_frames", 0]
    soften_inputs = document["soften_sound_effects"]["inputs"]
    assert soften_inputs["volume"] == -6
    merge_inputs = document["merge_audio_tracks"]["inputs"]
    assert merge_inputs["audio1"] == ["decode_music", 0]
    assert merge_inputs["audio2"] == ["soften_sound_effects", 0]
    assert merge_inputs["merge_method"] == "add"


def test_builder_rejects_empty_sequences() -> None:
    """Zero frames is a programming error the rendering layer translates."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    with pytest.raises(ValueError, match="frames"):
        _build_document(family, frame_count=0)


def test_every_link_references_an_existing_node() -> None:
    """No input points at a node key the workflow does not define."""
    for family in FAMILY_CATALOG:
        document = _build_document(family, frame_count=5)
        for node in document.values():
            for value in node["inputs"].values():
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and isinstance(value[0], str)
                    and isinstance(value[1], int)
                ):
                    assert value[0] in document, f"link {value} targets a missing node"
