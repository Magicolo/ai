"""Prompt staging (DESIGN §§18, 75-76).

Every video prompt is constructed by code from three layers:

    STYLE_PREFIX + WORLD_STATE / TRANSITION DESCRIPTION + MOTION / CAMERA

The director only supplies the middle layer (world/transition/stage
texts); the supervisor enforces the timeline and the immutable style
prefix is injected at every block so style can never drift out of the
prompt chain.
"""

from __future__ import annotations

from voyage.models import PromptPlan, PromptStage, StyleSpec

# Fragments that attempt to override the human-owned style charter (§18.1
# step 5). Matched case-insensitively as substrings; the check is
# deliberately narrow so legitimate scene language never trips it.
STYLE_OVERRIDE_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "disregard the style",
    "override the style",
    "forget the style",
    "new style:",
    "change the style to",
)


def motion_constraints(style: StyleSpec) -> str:
    """Camera/motion tail derived from the charter (§18.1 step 4)."""
    if style.motion_energy_max <= 0.35:
        pace = "very slow"
    elif style.motion_energy_max <= 0.6:
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
) -> PromptPlan:
    """Map semantic stages onto block ranges (§18.2).

    Each stage covers `blocks_per_stage` blocks; the final stage absorbs
    any remainder. Stage texts beyond the block range are dropped.
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
    for index in range(num_stages):
        start = index * blocks_per_stage
        if index < num_stages - 1:
            end = start + blocks_per_stage - 1
        else:
            end = num_blocks - 1  # final stage absorbs the remainder
        text = stage_texts[min(index, len(stage_texts) - 1)]
        transition = ""
        if transition_texts:
            transition = transition_texts[min(index, len(transition_texts) - 1)]
        stages.append(
            PromptStage(
                stage=index,
                block_start=start,
                block_end=end,
                prompt=compose_block_prompt(style, text, transition),
            )
        )
    return PromptPlan(segment_id=segment_id, stages=stages)
