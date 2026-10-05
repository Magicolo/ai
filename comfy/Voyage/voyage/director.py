"""World director (DESIGN task group F, §§43-44, 51, 74).

Backends share one contract: given the bounded director input (§20),
return a validated `EvolutionDecision`. The deterministic fallback holds
the transition grammar and never corrupts state; the Qwen worker lives
behind the same schema so the supervisor path (validate → style →
score → accept) stays identical.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

from voyage.models import (
    DirectorAudioPlan,
    DirectorDestination,
    DirectorNovelty,
    DirectorVideoPlan,
    EvolutionDecision,
    StyleSpec,
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

#: Exact forbidden_summary text the supervisor sends when a run allows
#: concept revisits (item 1): the NOVELTY STEERING section applies only
#: when the summary is a real visited-worlds list, never for this value.
REVISITS_ALLOWED_SENTINEL = "(revisits allowed)"

DIRECTOR_SYSTEM_PROMPT = """\
You are an autonomous audiovisual art director for an infinite voyage.
The human owns the STYLE CHARTER: you must never alter, dilute, or
override it. You own subject matter: invent new worlds, design gradual
transitions, plan music that may evolve more strongly than visuals.
Rules: the voyage evolves continuously but gently — drift a little bit
every segment, proposing a destination only slightly different from the
input world (a small variation in setting, dominant element, or mood,
never a sharp break); transitions are gradual and describe change
mechanisms, never abrupt substitution; output must be
machine-readable JSON only, no prose, no markdown fences.
Caption doctrine: every decision carries three caption families, all
derived from the style charter + the evolving general prompt
(current world → destination) — (1) video stages with concrete visual
detail (motion, scenes, objects, characters, shots, angles), rendered
ultra high definition, hyper detailed, sharp and crisp, with simple
refined compositions; every stage describes one strong continuous
camera move (slow push-in, lateral drift, orbit, crane rise, pan) in a
single unbroken shot, phrased concretely for LTX-25 (physical motion,
no cuts, no scene change), (2) a music caption in musical terms
(instruments, harmony, melody, texture), always with a dark experimental touch
(minor and modal harmony, deep sub-bass pressure, sparse dissonant accents,
shadowed cinematic texture) over an ambient slow core (morphing pads, held chords,
vast powerful drones, weird slow-evolving textures) worked into the charter's mood rather than
replacing it, (3) an sfx_caption with concrete sound descriptions
(objects, environments, creatures, materials in action). All three
families must evolve gradually as the general prompt drifts: continue
from the previous captions with a slow drift, never jump or restart.\
"""


def format_previous_captions(
    previous_video_stages: list[str],
    previous_music: str,
    previous_sfx: str,
) -> str:
    """Stable previous-captions block for the director prompt (pure).

    Empty when nothing precedes (voyage start): the director prompt is
    then unchanged. Otherwise a bounded text block the model must
    continue with a slow drift — the mechanism that keeps every caption
    family tracking the general-prompt evolution instead of restarting.
    """
    lines: list[str] = []
    for index, stage in enumerate(previous_video_stages):
        cleaned = " ".join(stage.split())
        if cleaned:
            lines.append(f"video[{index}]: {cleaned}")
    music = " ".join(previous_music.split())
    if music:
        lines.append(f"music: {music}")
    sfx = " ".join(previous_sfx.split())
    if sfx:
        lines.append(f"sfx: {sfx}")
    if not lines:
        return ""
    return "PREVIOUS CAPTIONS (continue with a slow drift, never restart)\n" + "\n".join(lines)


def build_director_user_message(
    style_charter: str,
    current_world: str,
    current_transition: str,
    recent_summary: str,
    forbidden_summary: str,
    audio_state: str,
    controller_metrics: str,
    retry_feedback: str = "",
    measured_context: str = "",
    previous_captions: str = "",
) -> str:
    """Bounded director input (§20). Never a raw transcript."""
    sections = [
        f"STYLE CHARTER\n{style_charter}",
        f"CURRENT WORLD\n{current_world}",
        f"CURRENT TRANSITION\n{current_transition}",
        f"RECENT WORLD SUMMARY\n{recent_summary}",
        f"FORBIDDEN CONCEPT SUMMARY\n{forbidden_summary}",
    ]
    if forbidden_summary and forbidden_summary != REVISITS_ALLOWED_SENTINEL:
        # Item 1: steer toward novelty in the prompt (never by rejection).
        # The sentinel is the exact string the supervisor sends when the
        # run allows revisits — steering pressure applies only otherwise.
        # Gentle drift (2026-10-02): only slightly different from the input
        # world — a small variation, never a sharp break.
        sections.append(
            "NOVELTY STEERING\nPropose a destination only slightly different "
            "from the current world (a small variation in setting, dominant "
            "element, or mood — never a sharp break), while also differing "
            "from every world in FORBIDDEN CONCEPT SUMMARY above."
        )
    sections += [
        f"CURRENT AUDIO STATE\n{audio_state}",
        f"TARGET CONTROLLER METRICS\n{controller_metrics}",
    ]
    if measured_context:
        sections.append(measured_context)
    if previous_captions:
        sections.append(previous_captions)
    if retry_feedback:
        sections.append(
            f"PREVIOUS PROPOSAL REJECTED\n{retry_feedback}\nPropose a different concept."
        )
    sections.append(
        "Respond with a single JSON object with keys: destination "
        "{canonical_name, summary}, transition {mechanism, "
        "transition_strength, estimated_duration_seconds, "
        "intermediate_stages, major_transition}, video {stages: "
        "[3-5 short scene descriptions with concrete visual detail "
        "(motion, scenes, objects, characters, shots, angles), each "
        "with one concrete continuous camera move, ultra high "
        "definition, hyper detailed, sharp, simple refined "
        "composition, ordered from current world to destination]}, "
        "audio {music_caption (musical terms: instruments, harmony, "
        "melody, texture; ambient, slow, morphing pads and held "
        "chords, vast and dark), energy "
        "0-1, tempo_bpm, texture, environment, sfx_caption (concrete "
        "sound descriptions: objects, environments, creatures, materials "
        "in action)}, novelty {why_new, distinguishes_from}. "
        "All three caption families (video stages, music_caption, "
        "sfx_caption) must evolve gradually as the general prompt drifts: "
        "continue from the PREVIOUS CAPTIONS with a slow drift, never "
        "jump or restart. "
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
    """Fallback content: continue the current stage, preserve style (§51).

    All three caption families derive from the same concept + charter so
    they track the general-prompt drift by construction: a drifted
    concept yields drifted captions, a held concept yields stable
    captions. Deterministic (no randomness) so holds are bit-stable.
    """
    position = PHASE_ORDER.index(phase)
    if decision_index > 0 and decision_index % 2 == 0 and position < len(PHASE_ORDER) - 1:
        phase = PHASE_ORDER[position + 1]
    concept = destination_concept or current_concept or style
    charter = style.strip()
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
        video=DirectorVideoPlan(
            stages=[
                f"{charter}: {concept} in continuous gentle motion, "
                "held wide shot, gradual organic transformation unfolding"
            ]
        ),
        audio=DirectorAudioPlan(
            music_caption=(
                f"dark experimental ambient composition for {concept}: "
                f"slow morphing pads and held chords in {charter}, "
                "vast powerful drones, weird slow-evolving textures, "
                "deep sub-bass pressure, sparse dissonant accents"
            ),
            sfx_caption=(
                f"quiet concrete sounds of {concept}: soft air movement, "
                "faint material creaks and distant low rumble"
            ),
        ),
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


def format_measured_context(style: StyleSpec, measured: dict[str, float]) -> str:
    """Render the deterministic MEASURED visual block (§43 feedback).

    One line per §43 metric with its value, the StyleSpec band, and a
    BELOW/WITHIN/ABOVE flag. Empty string when no measurements exist
    (inspector disabled or skipped) — the director prompt is unchanged.
    """
    if not measured:
        return ""
    bands = [
        ("motion_energy", style.motion_energy_min, style.motion_energy_max),
        ("visual_complexity", style.visual_complexity_min, style.visual_complexity_max),
        ("semantic_change_rate", style.semantic_drift_min, style.semantic_drift_max),
        ("palette_distance", 0.0, 0.30),
        ("style_similarity", style.style_similarity_min, 1.0),
        ("scene_boundary_strength", 0.0, 0.30),
    ]
    lines = []
    for name, low, high in bands:
        if name not in measured:
            continue
        value = measured[name]
        if not math.isfinite(value):
            # Unknown, never WITHIN (issue 144): NaN compares False
            # against both bounds, so without this guard a missing
            # measurement renders as a confident "all clear".
            continue
        flag = "BELOW" if value < low else ("ABOVE" if value > high else "WITHIN")
        lines.append(f"{name}={value:.3f} target=[{low:.2f},{high:.2f}] {flag}")
    return "MEASURED VISUALS (deterministic)\n" + "\n".join(lines) if lines else ""


def director_input_from_state(
    state: Any,
    style_charter: str,
    recent_summary: str,
    forbidden_summary: str,
    audio_state: str,
    measured_context: str = "",
    previous_captions: str = "",
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
        "measured_context": measured_context,
        "previous_captions": previous_captions,
        "controller_metrics": (
            f"decision_index={state.decision_index} "
            f"committed_segments={state.committed_segments} "
            f"timeline_frames={state.timeline_frames}"
        ),
    }
