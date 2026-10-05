"""Post-commit cheap motion sense: never fails, steers the next prompt.

Why: frozen segments used to commit silently — the six-metric inspector
was never wired on the live path. The cheap tier (three 160px thumbnails
+ pixel-delta energy) runs inline after the state advance and steers the
next proposal via measured_context/amendments; it never gates a commit.
"""

from __future__ import annotations

import io
from pathlib import Path

from voyage.console import VoyageConsole
from voyage.director import format_measured_context
from voyage.models import StyleSpec
from voyage.motion_sense import MotionReading, sense_motion
from voyage.prompts import feedback_amendments


def _style() -> StyleSpec:
    return StyleSpec(prompt="charter")


def test_missing_clip_reads_unknown_never_raises() -> None:
    reading = sense_motion(Path("/nonexistent/clip.mp4"))
    assert reading.energy is None
    assert reading.seconds >= 0.0
    assert isinstance(reading, MotionReading)


def test_unknown_reading_steers_nothing() -> None:
    assert feedback_amendments({}, _style()) == []
    assert format_measured_context(_style(), {}) == ""


def test_low_motion_steers_next_prompt() -> None:
    measured = {"motion_energy": 0.01}
    assert feedback_amendments(measured, _style()) == [
        "gentle continuous motion throughout the shot"
    ]
    rendered = format_measured_context(_style(), measured)
    assert "motion_energy=0.010" in rendered
    assert "BELOW" in rendered


def test_healthy_motion_steers_nothing() -> None:
    measured = {"motion_energy": 0.25}
    assert feedback_amendments(measured, _style()) == []
    assert "WITHIN" in format_measured_context(_style(), measured)


def test_console_reports_low_motion_and_sense_latency() -> None:
    stream = io.StringIO()
    VoyageConsole(stream=stream).segment_done(
        {
            "number": 3,
            "segment_id": "000003",
            "frames": 48,
            "duration": 2.0,
            "take_ids": [],
            "take_action": "keep",
            "take_reason": "",
            "beats": 4,
            "bpm": 120.0,
            "video_backend": "fake",
            "overlap_fraction": 0.1,
            "overlap_cap_seconds": 0.5,
            "stage_seconds": {},
            "elapsed": 2.1,
            "motion_energy": 0.042,
            "motion_seconds": 0.21,
        }
    )
    text = stream.getvalue()
    assert "motion energy 0.042" in text
    assert "0.21s sense" in text
    assert "LOW-MOTION" in text


def test_console_tolerates_missing_motion_keys() -> None:
    stream = io.StringIO()
    VoyageConsole(stream=stream).segment_done(
        {
            "number": 3,
            "segment_id": "000003",
            "frames": 48,
            "duration": 2.0,
            "take_ids": [],
            "take_action": "keep",
            "take_reason": "",
            "beats": 4,
            "bpm": 120.0,
            "video_backend": "fake",
            "overlap_fraction": 0.1,
            "overlap_cap_seconds": 0.5,
            "stage_seconds": {},
            "elapsed": 2.1,
        }
    )
    assert "SEGMENT 000003 committed" in stream.getvalue()
