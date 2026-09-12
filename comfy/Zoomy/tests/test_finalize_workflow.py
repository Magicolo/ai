"""Tests for the finalize workflow builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from zoomy.family_catalog import FAMILY_CATALOG, find_family
from zoomy.final_assembly import MUSIC_CROSSFADE_SECONDS, SOUND_EFFECT_CROSSFADE_SECONDS
from zoomy.finalize_workflow import (
    MUSIC_SEED,
    SEGMENT_MUSIC_EXTENSION_SECONDS,
    SEGMENT_SOUND_EXTENSION_SECONDS,
    SEGMENT_SOURCE_FRAMES,
    FinalizeRequest,
    SegmentWindow,
    build_finalize_segment_workflow,
    build_finalize_workflow,
    compute_audio_seconds,
    compute_interpolated_frame_count,
    compute_segment_music_seconds,
    compute_segment_sound_seconds,
    compute_segment_video_seconds,
    compute_segment_windows,
    needs_segmentation,
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


@given(frame_count=st.integers(min_value=1, max_value=500))
def test_audio_formula_holds_across_sequence_lengths(frame_count: int) -> None:
    """Interpolation and the floored duration follow the proven formulas."""
    interpolated = (frame_count - 1) * 4 + 1
    assert compute_interpolated_frame_count(frame_count) == interpolated
    assert compute_audio_seconds(interpolated) == pytest.approx(max(interpolated / 32, 1.0))
    assert compute_audio_seconds(interpolated) >= 1.0


@given(
    first=st.integers(min_value=1, max_value=300),
    second=st.integers(min_value=1, max_value=300),
)
def test_audio_seconds_grows_monotonically(first: int, second: int) -> None:
    """Longer sequences never produce shorter audio tracks."""
    assume(first <= second)
    first_seconds = compute_audio_seconds(compute_interpolated_frame_count(first))
    second_seconds = compute_audio_seconds(compute_interpolated_frame_count(second))
    assert first_seconds <= second_seconds


@given(frame_count=st.integers(max_value=0))
def test_builder_rejects_non_positive_frame_counts(frame_count: int) -> None:
    """Zero or negative counts fail fast instead of building nonsense."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    with pytest.raises(ValueError, match="frames"):
        _build_document(family, frame_count=frame_count)


def test_duration_flows_to_every_audio_node() -> None:
    """Music scores video plus its overlap; effects score video plus theirs."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    document = _build_document(family, frame_count=4)
    assert document["encode_music_prompt"]["inputs"]["duration"] == 2.0
    assert document["create_music_noise"]["inputs"]["seconds"] == 2.0
    assert document["sample_sound_effects"]["inputs"]["duration"] == 1.25


def test_segment_durations_carry_the_blend_overlap() -> None:
    """A 48-frame window scores ~6.9 s of music and ~6.2 s of effects."""
    assert compute_segment_video_seconds(48) == pytest.approx(189 / 32)
    assert compute_segment_music_seconds(48) == pytest.approx(189 / 32 + 1.0)
    assert compute_segment_sound_seconds(48) == pytest.approx(189 / 32 + 0.25)


def test_extension_matches_assembly_overlaps() -> None:
    """Over-generated tails must equal the overlaps the assembly blends."""
    assert SEGMENT_MUSIC_EXTENSION_SECONDS == MUSIC_CROSSFADE_SECONDS
    assert SEGMENT_SOUND_EXTENSION_SECONDS == SOUND_EFFECT_CROSSFADE_SECONDS


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


def test_short_sequences_need_no_segmentation() -> None:
    """Sequences that fit one segment finalize in a single pass."""
    assert needs_segmentation(1) is False
    assert needs_segmentation(SEGMENT_SOURCE_FRAMES) is False
    assert needs_segmentation(300) is True


def test_single_segment_covers_short_sequences() -> None:
    """A short sequence yields one window starting at frame zero."""
    windows = compute_segment_windows(4)
    assert windows == (SegmentWindow(index=0, skip_first_images=0, frame_count=4),)


def test_segment_boundary_splits_windows() -> None:
    """One frame past the segment size opens a second window for the tail."""
    windows = compute_segment_windows(SEGMENT_SOURCE_FRAMES + 1)
    assert windows == (
        SegmentWindow(index=0, skip_first_images=0, frame_count=SEGMENT_SOURCE_FRAMES),
        SegmentWindow(index=1, skip_first_images=SEGMENT_SOURCE_FRAMES, frame_count=1),
    )


@given(frame_count=st.integers(min_value=1, max_value=5000))
def test_segment_windows_tile_the_sequence_without_gaps(frame_count: int) -> None:
    """Windows are contiguous, bounded, and cover exactly the whole sequence."""
    windows = compute_segment_windows(frame_count)
    assert windows
    assert windows[0].skip_first_images == 0
    covered_frames = 0
    for position, window in enumerate(windows):
        assert window.index == position
        assert 1 <= window.frame_count <= SEGMENT_SOURCE_FRAMES
        assert window.skip_first_images == covered_frames
        covered_frames += window.frame_count
    assert covered_frames == frame_count


@given(frame_count=st.integers(max_value=0))
def test_segment_windows_reject_non_positive_frame_counts(frame_count: int) -> None:
    """Zero or negative counts fail fast instead of yielding empty windows."""
    with pytest.raises(ValueError, match="frames"):
        compute_segment_windows(frame_count)


def _build_segment_document(
    family: FamilyDefinition, *, frame_count: int, window: SegmentWindow
) -> dict[str, dict[str, Any]]:
    request = FinalizeRequest(
        family=family,
        frame_count=frame_count,
        output_directory="/comfy/output",
    )
    return build_finalize_segment_workflow(request, window)


def test_segment_loader_reads_only_its_window() -> None:
    """The segment loader skips to its window and caps the frame count."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    window = SegmentWindow(index=2, skip_first_images=96, frame_count=48)
    document = _build_segment_document(family, frame_count=295, window=window)
    loader_inputs = document["load_frame_sequence"]["inputs"]
    assert loader_inputs["directory"] == "/comfy/output/Zoomy/z_image"
    assert loader_inputs["image_load_cap"] == 48
    assert loader_inputs["skip_first_images"] == 96
    assert loader_inputs["select_every_nth"] == 1


def test_segment_renders_music_and_effects_twins() -> None:
    """Each segment renders two indexed videos: one per soundtrack stem.

    Separate twins let the Python assembly join music with a long crossfade
    and effects with a short one, instead of blending both on one cut.
    """
    family = find_family(FAMILY_CATALOG, "z_fast")
    window = SegmentWindow(index=2, skip_first_images=96, frame_count=48)
    document = _build_segment_document(family, frame_count=295, window=window)
    music_inputs = document["render_video_music"]["inputs"]
    effects_inputs = document["render_video_effects"]["inputs"]
    assert music_inputs["filename_prefix"] == "Zoomy_z_image_seg002_music"
    assert effects_inputs["filename_prefix"] == "Zoomy_z_image_seg002_sfx"
    assert music_inputs["images"] == ["interpolate_frames", 0]
    assert effects_inputs["images"] == ["interpolate_frames", 0]
    assert music_inputs["audio"] == ["decode_music", 0]
    assert effects_inputs["audio"] == ["soften_sound_effects", 0]
    assert "merge_audio_tracks" not in document


def test_segment_music_seed_evolves_with_the_window_index() -> None:
    """Music varies per segment while tags, tempo, and key stay shared."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    first = _build_segment_document(
        family,
        frame_count=295,
        window=SegmentWindow(index=0, skip_first_images=0, frame_count=48),
    )
    third = _build_segment_document(
        family,
        frame_count=295,
        window=SegmentWindow(index=2, skip_first_images=96, frame_count=48),
    )
    first_encoder = first["encode_music_prompt"]["inputs"]
    third_encoder = third["encode_music_prompt"]["inputs"]
    assert first_encoder["seed"] == MUSIC_SEED
    assert third_encoder["seed"] == MUSIC_SEED + 2
    assert third_encoder["tags"] == first_encoder["tags"]
    assert third_encoder["bpm"] == first_encoder["bpm"]
    assert third_encoder["keyscale"] == first_encoder["keyscale"]
    assert first["sample_music"]["inputs"]["seed"] == MUSIC_SEED
    assert third["sample_music"]["inputs"]["seed"] == MUSIC_SEED + 2


def test_segment_audio_duration_matches_the_window() -> None:
    """A 48-frame window scores overlap tails, not the whole sequence."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    window = SegmentWindow(index=0, skip_first_images=0, frame_count=48)
    document = _build_segment_document(family, frame_count=295, window=window)
    expected_music = pytest.approx(((48 - 1) * 4 + 1) / 32 + 1.0)
    expected_effects = pytest.approx(((48 - 1) * 4 + 1) / 32 + 0.25)
    assert document["encode_music_prompt"]["inputs"]["duration"] == expected_music
    assert document["create_music_noise"]["inputs"]["seconds"] == expected_music
    assert document["sample_sound_effects"]["inputs"]["duration"] == expected_effects


def test_segment_saves_full_length_stems() -> None:
    """Stems keep the overlap tails the video twins trim away."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    window = SegmentWindow(index=2, skip_first_images=96, frame_count=48)
    document = _build_segment_document(family, frame_count=295, window=window)
    music_stem = document["save_music_stem"]
    sound_stem = document["save_sound_stem"]
    assert music_stem["class_type"] == "SaveAudioAdvanced"
    assert sound_stem["class_type"] == "SaveAudioAdvanced"
    assert music_stem["inputs"]["audio"] == ["decode_music", 0]
    assert sound_stem["inputs"]["audio"] == ["soften_sound_effects", 0]
    assert music_stem["inputs"]["filename_prefix"] == "Zoomy_z_image_seg002_music_stem"
    assert sound_stem["inputs"]["filename_prefix"] == "Zoomy_z_image_seg002_sfx_stem"
    assert music_stem["inputs"]["format"] == "flac"
    assert sound_stem["inputs"]["format"] == "flac"


def test_segment_builder_rejects_empty_windows() -> None:
    """A window with no frames is a programming error, like an empty sequence."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    request = FinalizeRequest(family=family, frame_count=295, output_directory="/comfy/output")
    with pytest.raises(ValueError, match="frames"):
        build_finalize_segment_workflow(
            request, SegmentWindow(index=7, skip_first_images=295, frame_count=0)
        )


def test_segment_documents_link_cleanly() -> None:
    """Every segment window builds a graph with no dangling references."""
    for family in FAMILY_CATALOG:
        for window in compute_segment_windows(100):
            document = _build_segment_document(family, frame_count=100, window=window)
            for node in document.values():
                for value in node["inputs"].values():
                    if (
                        isinstance(value, list)
                        and len(value) == 2
                        and isinstance(value[0], str)
                        and isinstance(value[1], int)
                    ):
                        assert value[0] in document, f"link {value} targets a missing node"
