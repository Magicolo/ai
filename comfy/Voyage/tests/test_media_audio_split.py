"""Agreement tests for the media audio-join split (issue 036).

The probe/slice/blend/join audio group moves verbatim into
`voyage.media_audio`; `media` keeps explicit-`as` facade re-exports so
every existing `voyage.media.<name>` importer holds. Six test modules
patch the `run_capture`/`probe`/`slice_take`/`_cached_slice_take` seams
live on the defining module (`voyage.media_audio`, re-homed in the same
pass); `test_commit_side_integrity_095_101_104.py` keeps its
`voyage.media` patch target (dirty-foreign, skipped — recorded in 036).
"""

from __future__ import annotations


def test_audio_group_is_single_sourced_through_facade() -> None:
    from voyage import media, media_audio

    assert media.AV_ALIGNMENT_TOLERANCE_SECONDS is media_audio.AV_ALIGNMENT_TOLERANCE_SECONDS
    assert media.FPS_MATCH_TOLERANCE is media_audio.FPS_MATCH_TOLERANCE
    assert media.MIN_FADE_GRAPH_SECONDS is media_audio.MIN_FADE_GRAPH_SECONDS
    assert media.MIN_OVERLAP_BLEND_SECONDS is media_audio.MIN_OVERLAP_BLEND_SECONDS
    assert media.ABSORPTION_EPSILON_SECONDS is media_audio.ABSORPTION_EPSILON_SECONDS
    assert media.MIN_SLICE_PIECE_SECONDS is media_audio.MIN_SLICE_PIECE_SECONDS
    assert media.MAX_SLICES_PER_WINDOW is media_audio.MAX_SLICES_PER_WINDOW
    assert (
        media.DURATION_FRAME_ESTIMATE_SLACK_FRAMES
        is media_audio.DURATION_FRAME_ESTIMATE_SLACK_FRAMES
    )
    assert media.PAIR_BLEND_INPUT_COUNT is media_audio.PAIR_BLEND_INPUT_COUNT
    assert media.FFMPEG_TIMEOUT_SECONDS is media_audio.FFMPEG_TIMEOUT_SECONDS
    assert media.av_drift_seconds is media_audio.av_drift_seconds
    assert media.check_av_alignment is media_audio.check_av_alignment
    assert media.check_free_space is media_audio.check_free_space
    assert media.run_capture is media_audio.run_capture
    assert media.probe is media_audio.probe
    assert media.validate_video is media_audio.validate_video
    assert media._probe_video_fps is media_audio._probe_video_fps
    assert media.validate_audio is media_audio.validate_audio
    assert media.slice_take is media_audio.slice_take
    assert media._take_joint_fade is media_audio._take_joint_fade
    assert media.probed_take_seconds is media_audio.probed_take_seconds
    assert media.assemble_segment_audio is media_audio.assemble_segment_audio
    assert media._sha256_file is media_audio._sha256_file
    assert media._verify_segment is media_audio._verify_segment
    assert media._check_segment_committed is media_audio._check_segment_committed
    assert media._segment_timeline is media_audio._segment_timeline
    assert media._slice_cache_key is media_audio._slice_cache_key
    assert media._cached_slice_take is media_audio._cached_slice_take
    assert media._concat_fallback_audio is media_audio._concat_fallback_audio
    assert media._audio_duration_seconds is media_audio._audio_duration_seconds
    assert media._blend_fade_seconds is media_audio._blend_fade_seconds
    assert media._blend_pair is media_audio._blend_pair
    assert media._join_audio_single_graph is media_audio._join_audio_single_graph
    assert media.build_final_audio is media_audio.build_final_audio


def test_drift_math_stays_exact() -> None:
    from voyage import media_audio

    assert media_audio.av_drift_seconds(10.0, 9.5) == 0.5
    assert media_audio.av_drift_seconds(9.5, 10.0) == 0.5
    assert media_audio.check_av_alignment(10.0, 9.5, "seg") == 0.5


def test_alignment_budget_still_rejects() -> None:
    import pytest

    from voyage import media_audio
    from voyage.errors import MediaError

    assert media_audio.AV_ALIGNMENT_TOLERANCE_SECONDS == 0.6
    with pytest.raises(MediaError, match="drift"):
        media_audio.check_av_alignment(10.0, 9.0, "seg")


def test_blend_fade_formula_stays_clamped() -> None:
    import pytest

    from voyage import media_audio
    from voyage.errors import MediaError

    assert media_audio._blend_fade_seconds(10.0, 10.0, 0.4) == 0.4
    assert media_audio._blend_fade_seconds(0.5, 10.0, 0.4) == 0.25
    assert media_audio.PAIR_BLEND_INPUT_COUNT == 2
    with pytest.raises(MediaError, match="non-positive overlap"):
        media_audio._blend_fade_seconds(10.0, 10.0, 0.0)
