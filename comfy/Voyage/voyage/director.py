"""World director (DESIGN task group F, §§43-44, 51, 74).

Backends share one contract: given the bounded director input (§20),
return a validated `EvolutionDecision`. The deterministic fallback holds
the transition grammar and never corrupts state; the Qwen worker lives
behind the same schema so the supervisor path (validate → novelty →
style check → accept) stays identical.
"""

from __future__ import annotations

from typing import Any, Protocol

from voyage.models import (
    DirectorAudioPlan,
    DirectorDestination,
    DirectorNovelty,
    DirectorVideoPlan,
    EvolutionDecision,
    TransitionPhase,
    TransitionPlan,
)

PHASE_ORDER: list[TransitionPhase] = [
    "ESTABLISH",
    "DRIFT",
    "TRANSFORM",
    "DESTABILIZE",
    "EMERGE",
    "STABILIZE",
]

DIRECTOR_SYSTEM_PROMPT = """\
You are an autonomous audiovisual art director for an infinite voyage.
The human owns the STYLE CHARTER: you must never alter, dilute, or
override it. You own subject matter: invent new worlds, design gradual
transitions, plan music that may evolve more strongly than visuals.
Rules: the voyage evolves continuously; transitions are gradual and
describe change mechanisms, never abrupt substitution; old canonical
concepts are forbidden unless revisits are allowed; output must be
machine-readable JSON only, no prose, no markdown fences.\
"""


def build_director_user_message(
    style_charter: str,
    current_world: str,
    current_transition: str,
    recent_summary: str,
    forbidden_summary: str,
    audio_state: str,
    controller_metrics: str,
    retry_feedback: str = "",
) -> str:
    """Bounded director input (§20). Never a raw transcript."""
    sections = [
        f"STYLE CHARTER\n{style_charter}",
        f"CURRENT WORLD\n{current_world}",
        f"CURRENT TRANSITION\n{current_transition}",
        f"RECENT WORLD SUMMARY\n{recent_summary}",
        f"FORBIDDEN CONCEPT SUMMARY\n{forbidden_summary}",
        f"CURRENT AUDIO STATE\n{audio_state}",
        f"TARGET CONTROLLER METRICS\n{controller_metrics}",
    ]
    if retry_feedback:
        sections.append(
            f"PREVIOUS PROPOSAL REJECTED\n{retry_feedback}\nPropose a different concept."
        )
    sections.append(
        "Respond with a single JSON object with keys: destination "
        "{canonical_name, summary}, transition {mechanism, "
        "transition_strength, estimated_duration_seconds, "
        "intermediate_stages, major_transition}, video {stages: "
        "[3-5 short scene descriptions, ordered from current world to "
        "destination]}, audio {music_caption, energy 0-1, tempo_bpm, "
        "texture, environment}, novelty {why_new, distinguishes_from}. "
        "Strict types: transition.mechanism MUST be exactly one of "
        "'material_metamorphosis', 'environmental_transformation', "
        "'scale_shift', 'geometric_transformation', 'physical_rule_change', "
        "'lighting_transformation', 'perceptual_transformation', 'hybrid'. "
        "audio.environment MUST be a JSON array of short strings. novelty "
        "MUST be an object with why_new and distinguishes_from strings."
    )
    return "\n\n".join(sections)


class DirectorBackend(Protocol):
    def propose(
        self,
        decision_index: int,
        current_concept: str,
        destination_concept: str,
        phase: TransitionPhase,
    ) -> EvolutionDecision: ...


def deterministic_decision(
    decision_index: int,
    current_concept: str,
    destination_concept: str,
    phase: TransitionPhase,
    style: str,
) -> EvolutionDecision:
    """Fallback content: continue the current stage, preserve style (§51)."""
    position = PHASE_ORDER.index(phase)
    if decision_index > 0 and decision_index % 2 == 0 and position < len(PHASE_ORDER) - 1:
        phase = PHASE_ORDER[position + 1]
    concept = destination_concept or current_concept or style
    return EvolutionDecision(
        decision_index=decision_index,
        destination=DirectorDestination(
            canonical_name=concept, summary="deterministic continuation"
        ),
        transition=TransitionPlan(
            source_concept=current_concept,
            destination_concept=concept,
            mechanism="hybrid",
            intermediate_stages=[f"the {concept} holds and deepens"],
        ),
        video=DirectorVideoPlan(stages=[concept]),
        audio=DirectorAudioPlan(),
        novelty=DirectorNovelty(why_new="fallback holds the current concept"),
        phase=phase,
        novelty_accepted=True,
        notes="deterministic-fallback: continue current stage, preserve style",
    )


class DeterministicDirector:
    """Fallback director: hold the transition grammar, never corrupt state."""

    def __init__(self, style: str) -> None:
        self._style = style

    def propose(
        self,
        decision_index: int,
        current_concept: str,
        destination_concept: str,
        phase: TransitionPhase,
    ) -> EvolutionDecision:
        return deterministic_decision(
            decision_index, current_concept, destination_concept, phase, self._style
        )


def director_input_from_state(
    state: Any,
    style_charter: str,
    recent_summary: str,
    forbidden_summary: str,
    audio_state: str,
) -> dict[str, Any]:
    """Assemble the bounded §20 input from supervisor state (no transcript)."""
    return {
        "style_charter": style_charter,
        "current_world": state.current_concept or "(voyage start: no world yet)",
        "current_transition": (
            f"{state.current_concept} -> {state.destination_concept} [{state.phase}]"
        ),
        "recent_summary": recent_summary,
        "forbidden_summary": forbidden_summary,
        "audio_state": audio_state,
        "controller_metrics": (
            f"decision_index={state.decision_index} "
            f"committed_segments={state.committed_segments} "
            f"timeline_frames={state.timeline_frames}"
        ),
    }
