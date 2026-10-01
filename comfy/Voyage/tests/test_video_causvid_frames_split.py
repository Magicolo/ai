"""Agreement tests for the CausVid frame-math split (issue 036).

The rollout accounting helpers move verbatim into
`voyage.workers.video_causvid_frames`; `video_causvid` keeps explicit-`as`
facade re-exports so every existing `video_causvid.<name>` importer holds.
"""

from __future__ import annotations


def test_frame_helpers_are_single_sourced_through_facade() -> None:
    from voyage.workers import video_causvid, video_causvid_frames

    assert video_causvid.dropped_tail_frames is video_causvid_frames.dropped_tail_frames
    assert video_causvid.novel_frames_per_rollout is video_causvid_frames.novel_frames_per_rollout
    assert video_causvid.split_tail_novel is video_causvid_frames.split_tail_novel
    assert video_causvid.reencode_window_frames is video_causvid_frames.reencode_window_frames


def test_frame_helpers_keep_behavior() -> None:
    import pytest

    from voyage.workers import video_causvid_frames

    assert video_causvid_frames.dropped_tail_frames(3) == 9
    assert video_causvid_frames.novel_frames_per_rollout(81, 3) == 72
    assert video_causvid_frames.split_tail_novel(81, 3) == (9, 72)
    assert video_causvid_frames.reencode_window_frames(3) == 9
    with pytest.raises(ValueError, match="nothing would be committed"):
        video_causvid_frames.novel_frames_per_rollout(9, 3)


def test_overlap_three_stays_nine_and_seventy_two() -> None:
    from voyage.workers import video_causvid_frames

    dropped = video_causvid_frames.dropped_tail_frames(3)
    assert dropped == 9
    assert video_causvid_frames.reencode_window_frames(3) == dropped
