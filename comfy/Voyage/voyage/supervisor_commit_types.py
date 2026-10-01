"""Commit-pipeline NamedTuples for the supervisor (DESIGN §73).

Split from `voyage.supervisor` (issue 081): the transactional commit's
pure proposal/render/cover types as importable NamedTuples with no
supervisor state. `voyage.supervisor` re-exports every name below so
existing importers keep working; new code imports from here directly.

- `ProposedSegment`: director proposal + staged prompt plan (issue 020).
- `RenderedVideo`: video outcome with the issue-006 ceiling gate.
- `CoveredAudio`: audio outcome driving audio_buffer_seconds.
"""

from __future__ import annotations

from typing import NamedTuple

from voyage.models import AudioPlan, EvolutionDecision, PromptPlan


class ProposedSegment(NamedTuple):
    """Director proposal + staged prompt plan for one commit (issue 020).

    Pure proposal: no media rendered, no state advanced. Built by
    `_propose_segment`, consumed by `_render_video` / `_commit_segment`.
    """

    decision: EvolutionDecision
    prompt_plan: PromptPlan
    block_prompts: list[str]
    num_blocks: int
    prefetch_hit: bool
    drift_hold: bool
    director_tokens: dict[str, int]


class RenderedVideo(NamedTuple):
    """Video outcome of one commit (issue 020).

    `frames` is the worker-reported count after the issue-006 ceiling
    gate (never the raw report); `video_time` is the timeline offset the
    audio coverage starts from; `recovery_tape` is the validated absolute
    wire path (None when the worker reported none); `joint_audio_path`
    carries a joint-audio backend's worker-side soundtrack file for the
    supervisor to commit as the segment audio.wav (None for take-based
    backends — DESIGN §140 ltx plan).
    """

    frames: int
    duration: float
    video_time: float
    recovery_tape: str | None
    video_stage_ms: dict[str, float]
    joint_audio_path: str | None = None


class CoveredAudio(NamedTuple):
    """Audio outcome of one commit (issue 020).

    The segment's AudioPlan plus the seconds of music coverage remaining
    ahead of the new segment end (drives audio_buffer_seconds) and the
    planner action/reason (surfaced in console summaries).
    """

    audio_plan: AudioPlan
    audio_ahead: float
    take_action: str
    take_reason: str
