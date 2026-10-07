"""Issue 260: director_input_from_state takes the real RunState.

No cycle (director already imports voyage.models); the six touched
attributes typecheck under strict mypy.
"""

from __future__ import annotations

import typing

from voyage.director import director_input_from_state
from voyage.models import RunState


def test_state_param_is_runstate() -> None:
    """The boundary constructor is typed, not Any (260)."""
    hints = typing.get_type_hints(director_input_from_state)
    assert hints["state"] is RunState


def test_assembler_reads_six_attributes() -> None:
    """All six attributes flow into the bounded input (260)."""
    state = RunState(
        name="probe",
        current_concept="harbor",
        destination_concept="reef",
        phase="DRIFT",
        decision_index=3,
        committed_segments=5,
        timeline_frames=240,
    )
    payload = director_input_from_state(
        state,
        style_charter="charter",
        recent_summary="recent",
        forbidden_summary="forbidden",
        audio_state="audio",
    )
    assert payload["current_world"] == "harbor"
    assert "harbor -> reef [DRIFT]" in payload["current_transition"]
    assert "decision_index=3" in payload["controller_metrics"]
    assert "committed_segments=5" in payload["controller_metrics"]
    assert "timeline_frames=240" in payload["controller_metrics"]
