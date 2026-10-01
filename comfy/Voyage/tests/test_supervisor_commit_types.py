"""Supervisor commit-type split surface tests (issue 081).

Behavior-preservation contract for the `voyage/supervisor_commit_types.py`
extraction: the commit-pipeline NamedTuples live once in the new module,
and `voyage.supervisor` re-exports the identical objects.
"""

from __future__ import annotations

import voyage.supervisor as supervisor
import voyage.supervisor_commit_types as commit_types


def test_commit_types_are_single_sourced() -> None:
    """Commit NamedTuples live once, in supervisor_commit_types (issue 081)."""
    assert supervisor.ProposedSegment is commit_types.ProposedSegment
    assert supervisor.RenderedVideo is commit_types.RenderedVideo
    assert supervisor.CoveredAudio is commit_types.CoveredAudio


def test_commit_types_field_names_unchanged() -> None:
    """Field names are byte-stable across the move (issue 081)."""
    assert list(commit_types.ProposedSegment._fields) == [
        "decision",
        "prompt_plan",
        "block_prompts",
        "num_blocks",
        "prefetch_hit",
        "drift_hold",
        "director_tokens",
    ]
    assert list(commit_types.RenderedVideo._fields) == [
        "frames",
        "duration",
        "video_time",
        "recovery_tape",
        "video_stage_ms",
        "joint_audio_path",
    ]
    assert list(commit_types.CoveredAudio._fields) == [
        "audio_plan",
        "audio_ahead",
        "take_action",
        "take_reason",
    ]
