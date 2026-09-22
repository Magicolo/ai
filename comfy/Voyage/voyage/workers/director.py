"""Director worker: `python -m voyage.workers.director`."""

from __future__ import annotations

from typing import Any, cast

from voyage.director import DeterministicDirector
from voyage.models import TransitionPhase
from voyage.workers.loop import checked_request, serve


def handle_decide(payload: dict[str, Any]) -> dict[str, Any]:
    checked_request(
        payload,
        decision_index=int,
        current_concept=str,
        destination_concept=str,
        phase=str,
        style=str,
    )
    backend = DeterministicDirector(style=str(payload["style"]))
    phase = cast("TransitionPhase", str(payload["phase"]))
    decision = backend.propose(
        decision_index=int(payload["decision_index"]),
        current_concept=str(payload["current_concept"]),
        destination_concept=str(payload["destination_concept"]),
        phase=phase,
    )
    return decision.model_dump()


def main() -> None:
    serve(
        {
            "init": lambda _payload: {"status": "READY", "backend": "deterministic"},
            "health": lambda _payload: {"status": "READY", "backend": "deterministic"},
            "decide": handle_decide,
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()
