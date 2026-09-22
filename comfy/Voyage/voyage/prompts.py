"""Prompt staging (DESIGN §§18, 75-76).

The LLM expresses transitions in words; the supervisor enforces the
timeline. The immutable style prefix is injected at every block so
style can never drift out of the prompt chain.
"""

from __future__ import annotations

from voyage.models import PromptPlan, PromptStage


def compose_prompt(style: str, concept: str, phase: str, seed_hint: str = "") -> str:
    parts = [style.strip(), concept.strip(), f"phase: {phase.strip()}"]
    if seed_hint:
        parts.append(seed_hint)
    return ", ".join(part for part in parts if part)


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
