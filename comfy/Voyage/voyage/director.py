"""World director (DESIGN task group F, §§43-44, 51, 74).

V1 implements the deterministic fallback as the default backend: continue
the current stage, reduce semantic mutation, preserve style. A Qwen3-8B
worker plugs in behind `DirectorBackend` later; the supervisor path
(validate → novelty → style check → accept) stays identical.
"""

from __future__ import annotations

from typing import Protocol

from voyage.models import EvolutionDecision, TransitionPhase

PHASE_ORDER: list[TransitionPhase] = [
    "ESTABLISH",
    "DRIFT",
    "TRANSFORM",
    "DESTABILIZE",
    "EMERGE",
    "STABILIZE",
]


class DirectorBackend(Protocol):
    def propose(
        self,
        decision_index: int,
        current_concept: str,
        destination_concept: str,
        phase: TransitionPhase,
    ) -> EvolutionDecision: ...


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
        # Deterministic continuation: keep the destination, advance one phase
        # every two decisions so ESTABLISH/STABILIZE are perceivable.
        position = PHASE_ORDER.index(phase)
        if decision_index > 0 and decision_index % 2 == 0 and position < len(PHASE_ORDER) - 1:
            phase = PHASE_ORDER[position + 1]
        concept = destination_concept or current_concept or self._style
        return EvolutionDecision(
            decision_index=decision_index,
            destination_concept=concept,
            phase=phase,
            novelty_accepted=True,
            notes="deterministic-fallback: continue current stage, preserve style",
        )
