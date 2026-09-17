"""Tests for the backend-neutral generation contracts."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zoomy.engine_protocol import (
    CROP_BORDER_PIXELS,
    CROP_HEIGHT_PIXELS,
    CROP_WIDTH_PIXELS,
    FRAME_HEIGHT_PIXELS,
    FRAME_WIDTH_PIXELS,
    INTERPOLATION_MULTIPLIER,
    MINIMUM_AUDIO_SECONDS,
    SEGMENT_MUSIC_EXTENSION_SECONDS,
    SEGMENT_SOUND_EXTENSION_SECONDS,
    SEGMENT_SOURCE_FRAMES,
    VIDEO_FRAMES_PER_SECOND,
    ZOOM_FACTOR_PER_FRAME,
    FinalizeRequest,
    FrameRenderRequest,
    ProgressUpdate,
    compute_audio_seconds,
    compute_frames_for_seconds,
    compute_interpolated_frame_count,
    compute_segment_music_seconds,
    compute_segment_sound_seconds,
    compute_segment_video_seconds,
    compute_segment_windows,
    needs_segmentation,
)
from zoomy.family_catalog import FAMILY_CATALOG, find_family


def test_geometry_matches_the_zoom_dive() -> None:
    """The crop rule centers the dive: (1376 - 1356) / 2 = 10 px per side."""
    assert CROP_WIDTH_PIXELS == FRAME_WIDTH_PIXELS - 2 * CROP_BORDER_PIXELS
    assert CROP_HEIGHT_PIXELS == FRAME_HEIGHT_PIXELS - 2 * CROP_BORDER_PIXELS
    assert pytest.approx(FRAME_WIDTH_PIXELS / CROP_WIDTH_PIXELS) == ZOOM_FACTOR_PER_FRAME
    assert pytest.approx(1.0147, rel=1e-3) == ZOOM_FACTOR_PER_FRAME


def test_interpolation_count_matches_film_math() -> None:
    """Four frames interpolate to thirteen: (4 - 1) * 4 + 1."""
    assert compute_interpolated_frame_count(1) == 1
    assert compute_interpolated_frame_count(4) == 13
    assert compute_interpolated_frame_count(48) == 189


def test_audio_seconds_floors_at_the_latent_minimum() -> None:
    """Short sequences still clear the 1.0 s ACE-Step latent floor."""
    assert compute_audio_seconds(1) == MINIMUM_AUDIO_SECONDS
    assert compute_audio_seconds(13) == MINIMUM_AUDIO_SECONDS
    assert compute_audio_seconds(189) == pytest.approx(189 / VIDEO_FRAMES_PER_SECOND)


def test_segment_budget_matches_verified_vram_envelope() -> None:
    """48 source frames stay single-pass; 49 need segmentation."""
    assert not needs_segmentation(SEGMENT_SOURCE_FRAMES)
    assert needs_segmentation(SEGMENT_SOURCE_FRAMES + 1)


def test_segment_audio_extends_video_by_the_assembly_overlaps() -> None:
    """Segment stems over-generate exactly the join crossfades."""
    assert compute_segment_music_seconds(48) == pytest.approx(
        compute_segment_video_seconds(48) + SEGMENT_MUSIC_EXTENSION_SECONDS
    )
    assert compute_segment_sound_seconds(48) == pytest.approx(
        compute_segment_video_seconds(48) + SEGMENT_SOUND_EXTENSION_SECONDS
    )


def test_segment_windows_reject_empty_sequences() -> None:
    """Zero frames cannot tile into windows."""
    with pytest.raises(ValueError, match="0 frames"):
        compute_segment_windows(0)


def test_segment_windows_reject_empty_size() -> None:
    """A zero segment size would loop forever, so it is refused."""
    with pytest.raises(ValueError, match="Segment size"):
        compute_segment_windows(10, segment_size=0)


def test_segment_windows_tile_without_gaps() -> None:
    """Windows cover exactly the source frames, capped at the segment size."""
    windows = compute_segment_windows(100)
    assert [window.skip_first_images for window in windows] == [0, 48, 96]
    assert [window.frame_count for window in windows] == [48, 48, 4]
    assert [window.index for window in windows] == [0, 1, 2]
    assert sum(window.frame_count for window in windows) == 100


@given(
    frame_count=st.integers(min_value=1, max_value=2000),
    segment_size=st.integers(min_value=1, max_value=200),
)
def test_segment_windows_always_tile(frame_count: int, segment_size: int) -> None:
    """Arbitrary counts tile contiguously within the size cap."""
    windows = compute_segment_windows(frame_count, segment_size=segment_size)
    assert sum(window.frame_count for window in windows) == frame_count
    assert all(1 <= window.frame_count <= segment_size for window in windows)
    assert [window.skip_first_images for window in windows] == [
        index * segment_size for index in range(len(windows))
    ]


def test_request_dataclasses_carry_no_backend_fields() -> None:
    """Requests hold generation inputs only — no directories, no timeouts."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    frame_request = FrameRenderRequest(
        family=family,
        prompt="a prompt",
        negative_prompt="a negative",
        frame_count=3,
        lora_selections=(),
        seed=7,
    )
    assert frame_request.frame_count == 3
    assert frame_request.frame_width == FRAME_WIDTH_PIXELS
    assert frame_request.frame_height == FRAME_HEIGHT_PIXELS
    finalize_request = FinalizeRequest(family=family, frame_count=3)
    assert finalize_request.frame_count == 3
    update = ProgressUpdate(message="done")
    assert update.frame_path is None
    assert update.video_path is None
    assert update.elapsed_seconds is None
    assert INTERPOLATION_MULTIPLIER == 4


def test_frames_for_seconds_hits_the_duration() -> None:
    """81 source frames interpolate to 321 frames: 10.03 s at 32 fps."""
    assert compute_frames_for_seconds(10.0) == 81
    assert compute_frames_for_seconds(1.0) == 9
    assert compute_frames_for_seconds(0.5) == 5


def test_frames_for_seconds_rejects_non_positive_durations() -> None:
    """Zero or negative durations cannot size a sequence."""
    with pytest.raises(ValueError, match="positive"):
        compute_frames_for_seconds(0.0)
    with pytest.raises(ValueError, match="positive"):
        compute_frames_for_seconds(-2.5)


@given(target_seconds=st.floats(min_value=0.001, max_value=3600, allow_nan=False))
def test_frames_for_seconds_is_minimal(target_seconds: float) -> None:
    """The count is the smallest whose interpolated video covers the target."""
    frame_count = compute_frames_for_seconds(target_seconds)
    assert compute_interpolated_frame_count(frame_count) / VIDEO_FRAMES_PER_SECOND >= target_seconds
    if frame_count > 1:
        assert (
            compute_interpolated_frame_count(frame_count - 1) / VIDEO_FRAMES_PER_SECOND
            < target_seconds
        )
