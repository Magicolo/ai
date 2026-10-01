"""Agreement tests for the LTXV validators split (issue 036).

The spatial/frame validators move verbatim into
`voyage.workers.video_ltxv_validators`; `video_ltxv` keeps explicit-`as`
facade re-exports so every existing `video_ltxv.<name>` importer holds.
"""

from __future__ import annotations


def test_validators_are_single_sourced_through_facade() -> None:
    from voyage.workers import video_ltxv, video_ltxv_validators

    assert video_ltxv.padded_size is video_ltxv_validators.padded_size
    assert video_ltxv.validate_spatial_size is video_ltxv_validators.validate_spatial_size
    assert video_ltxv.validate_frame_count is video_ltxv_validators.validate_frame_count
    assert video_ltxv.validate_fps is video_ltxv_validators.validate_fps
    assert (
        video_ltxv.validate_conditioning_start is video_ltxv_validators.validate_conditioning_start
    )
    assert video_ltxv.SPATIAL_GRANULARITY is video_ltxv_validators.SPATIAL_GRANULARITY


def test_validators_keep_behavior() -> None:
    import pytest

    from voyage.workers import video_ltxv_validators

    assert video_ltxv_validators.padded_size(500) == 512
    video_ltxv_validators.validate_spatial_size(768, 512)
    video_ltxv_validators.validate_frame_count(121)
    video_ltxv_validators.validate_fps(24)
    video_ltxv_validators.validate_conditioning_start(8, 121)
    with pytest.raises(ValueError, match="not divisible by 32"):
        video_ltxv_validators.validate_spatial_size(768, 432)
    with pytest.raises(ValueError, match="8n\\+1"):
        video_ltxv_validators.validate_frame_count(100)
    with pytest.raises(ValueError, match="positive"):
        video_ltxv_validators.validate_fps(0)


def test_spatial_granularity_stays_32() -> None:
    from voyage.workers import video_ltxv_validators

    assert video_ltxv_validators.SPATIAL_GRANULARITY == 32
