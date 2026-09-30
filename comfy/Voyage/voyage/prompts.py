"""Prompt staging (DESIGN §§18, 75-76).

Every video prompt is constructed by code from three layers:

    STYLE_PREFIX + WORLD_STATE / TRANSITION DESCRIPTION + MOTION / CAMERA

The director only supplies the middle layer (world/transition/stage
texts); the supervisor enforces the timeline and the immutable style
prefix is injected at every block so style can never drift out of the
prompt chain.
"""

from __future__ import annotations

import math

from voyage.models import PromptPlan, PromptStage, StyleSpec

# Motion-pace bands derived from the charter's motion_energy_max (§18.1
# step 4): at/below calm the camera barely drifts, at/below slow it
# glides, above slow it may travel at a moderate pace. Thresholds are
# charter-relative, not perceptual absolutes — tune against StyleSpec.
_MOTION_CALM_MAX = 0.35
_MOTION_SLOW_MAX = 0.6

# Fragments that attempt to override the human-owned style charter (§18.1
# step 5). Matched case-insensitively as substrings. The ladder is:
# (1) this blocklist rejects the known phrasings via ProposalRejected,
# (2) `enforce_style` re-injects the immutable prefix every block so a
# paraphrase that slips past still cannot remove the charter, (3) the
# supervisor accept loop re-checks every stage against the charter.
# Tier 1 stays narrow enough that legitimate scene language never trips
# it; the seed corpus in the 026 tests pins the documented paraphrases.
STYLE_OVERRIDE_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "ignore prior instructions",
    "disregard the style",
    "override the style",
    "forget the style",
    "new style:",
    "change the style to",
    "system prompt",
    "jailbreak",
)


def motion_constraints(style: StyleSpec) -> str:
    """Camera/motion tail derived from the charter (§18.1 step 4)."""
    if style.motion_energy_max <= _MOTION_CALM_MAX:
        pace = "very slow"
    elif style.motion_energy_max <= _MOTION_SLOW_MAX:
        pace = "slow"
    else:
        pace = "moderate"
    return (
        f"{pace} continuous camera movement, gentle organic motion, "
        "no abrupt cuts, no scene change within the shot"
    )


def enforce_style(prompt: str, style: StyleSpec) -> str:
    """Code-level style injection (§18.1 steps 1-4).

    Trims whitespace, prepends the immutable style prefix, appends motion
    constraints derived from the charter. Never trusts the director to
    carry style on its own.
    """
    middle = " ".join(prompt.split())
    parts = [style.prompt.strip(), middle, motion_constraints(style)]
    return ", ".join(part for part in parts if part)


def detect_style_override(prompt: str) -> str | None:
    """Return the offending marker if the text tries to override the charter."""
    lowered = prompt.lower()
    for marker in STYLE_OVERRIDE_MARKERS:
        if marker in lowered:
            return marker
    return None


def check_prompt_against_style(prompt: str, style: StyleSpec) -> None:
    """Reject director text that attempts a style override (§18.1 step 5)."""
    marker = detect_style_override(prompt)
    if marker is not None:
        from voyage.errors import ProposalRejected

        raise ProposalRejected(f"director prompt attempts style override ({marker!r})")


def feedback_amendments(measured: dict[str, float], style: StyleSpec) -> list[str]:
    """Translate §43 metric deviations into prompt amendments (Phase 5).

    Pure function over the deterministic MEASURED block: each amendment
    nudges the next segment's middle-layer text back toward the charter
    band. Empty input → no amendments (inspector disabled/skipped).
    """
    if not measured:
        return []
    amendments = []
    motion = measured.get("motion_energy")
    # Non-finite readings mean "unknown", never a deviation (issue 144):
    # NaN > max and NaN < min are both False, so unguarded code steers
    # nothing while the label path claims WITHIN.
    if motion is not None and math.isfinite(motion):
        if motion > style.motion_energy_max:
            amendments.append("calm static composition, minimal motion")
        elif motion < style.motion_energy_min:
            amendments.append("gentle continuous motion throughout the shot")
    complexity = measured.get("visual_complexity")
    if (
        complexity is not None
        and math.isfinite(complexity)
        and complexity > style.visual_complexity_max
    ):
        amendments.append("sparse composition, few simple shapes, large empty areas")
    drift = measured.get("semantic_change_rate")
    if drift is not None and math.isfinite(drift) and drift < style.semantic_drift_min:
        amendments.append("gradual visible transformation unfolding across the shot")
    similarity = measured.get("style_similarity")
    if (
        similarity is not None
        and math.isfinite(similarity)
        and similarity < style.style_similarity_min
    ):
        amendments.append("strictly in the charter style, signature palette and linework")
    return amendments


def apply_feedback_amendments(prompt: str, amendments: list[str]) -> str:
    """Append feedback amendments to a middle-layer prompt (§43)."""
    parts = [prompt.strip(), *(item.strip() for item in amendments if item.strip())]
    return ", ".join(part for part in parts if part)


def compose_prompt(style: str, concept: str, phase: str, seed_hint: str = "") -> str:
    parts = [style.strip(), concept.strip(), f"phase: {phase.strip()}"]
    if seed_hint:
        parts.append(seed_hint)
    return ", ".join(part for part in parts if part)


def compose_block_prompt(
    style: StyleSpec,
    world_text: str,
    transition_text: str = "",
) -> str:
    """Three-layer renderer prompt: STYLE + WORLD/TRANSITION + MOTION."""
    middle = world_text.strip()
    if transition_text.strip():
        middle = f"{middle}. {transition_text.strip()}"
    check_prompt_against_style(middle, style)
    return enforce_style(middle, style)


def build_prompt_plan(
    segment_id: str,
    style: str,
    concept: str,
    phase: str,
    block_starts: list[int],
    block_ends: list[int],
) -> PromptPlan:
    if len(block_starts) != len(block_ends):
        raise ValueError("block_starts and block_ends must have equal length")
    stages = [
        PromptStage(
            stage=index,
            block_start=start,
            block_end=end,
            prompt=compose_prompt(style, concept, phase),
        )
        for index, (start, end) in enumerate(zip(block_starts, block_ends, strict=True))
    ]
    return PromptPlan(segment_id=segment_id, stages=stages)


def build_staged_prompt_plan(
    segment_id: str,
    style: StyleSpec,
    stage_texts: list[str],
    transition_texts: list[str],
    num_blocks: int,
    blocks_per_stage: int,
    *,
    strict: bool = False,
    repeat_transitions: bool = True,
) -> PromptPlan:
    """Map semantic stages onto block ranges (§18.2).

    Each stage covers `blocks_per_stage` blocks; the final stage absorbs
    any remainder. Stage texts beyond the block range are dropped —
    silently by default (documented long-standing behavior), or loudly
    with `strict=True`, which raises ValueError naming the dropped
    count instead (issue 026: callers should know 4 of 5 stages
    vanished). Short transition lists repeat their last entry by
    default; `repeat_transitions=False` holds them empty past their
    range so "repeat last" and "hold" stay distinguishable.
    """
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if blocks_per_stage <= 0:
        raise ValueError("blocks_per_prompt_stage must be positive")
    if not stage_texts:
        raise ValueError("stage_texts must not be empty")
    stages: list[PromptStage] = []
    # Balanced staging (~blocks_per_stage each): round to the nearest stage
    # count so the tail never becomes a 1-block stub; the final stage
    # absorbs whatever remains.
    num_stages = max(1, (num_blocks + blocks_per_stage // 2) // blocks_per_stage)
    if strict and len(stage_texts) > num_stages:
        dropped = len(stage_texts) - num_stages
        raise ValueError(
            f"{dropped} stage texts exceed the {num_stages} planned stages "
            f"({num_blocks} blocks at {blocks_per_stage} per stage)"
        )
    for index in range(num_stages):
        start = index * blocks_per_stage
        if index < num_stages - 1:
            end = start + blocks_per_stage - 1
        else:
            end = num_blocks - 1  # final stage absorbs the remainder
        text = stage_texts[min(index, len(stage_texts) - 1)]
        transition = ""
        if transition_texts:
            if repeat_transitions:
                transition = transition_texts[min(index, len(transition_texts) - 1)]
            elif index < len(transition_texts):
                transition = transition_texts[index]
        stages.append(
            PromptStage(
                stage=index,
                block_start=start,
                block_end=end,
                prompt=compose_block_prompt(style, text, transition),
            )
        )
    return PromptPlan(segment_id=segment_id, stages=stages)
