"""Feedback loop tests (Phase 5, DESIGN §43): MEASURED context + amendments."""

from voyage.director import build_director_user_message, format_measured_context
from voyage.models import StyleSpec
from voyage.prompts import apply_feedback_amendments, feedback_amendments


def _style() -> StyleSpec:
    return StyleSpec(prompt="line art")


def test_measured_context_empty_when_no_measurements() -> None:
    assert format_measured_context(_style(), {}) == ""


def test_measured_context_flags_bands() -> None:
    style = _style()
    text = format_measured_context(
        style,
        {
            "motion_energy": 0.90,
            "visual_complexity": 0.40,
            "style_similarity": 0.10,
        },
    )
    assert "motion_energy=0.900" in text
    assert "ABOVE" in text
    assert "visual_complexity=0.400" in text
    assert "WITHIN" in text
    assert "style_similarity=0.100" in text
    assert "BELOW" in text


def test_measured_context_skips_unknown_metrics() -> None:
    text = format_measured_context(_style(), {"motion_energy": 0.25})
    assert "motion_energy" in text
    assert "scene_boundary_strength" not in text


def test_user_message_carries_measured_block() -> None:
    base = build_director_user_message(
        "charter", "world", "trans", "recent", "forbidden", "audio", "metrics"
    )
    assert "MEASURED" not in base
    with_block = build_director_user_message(
        "charter",
        "world",
        "trans",
        "recent",
        "forbidden",
        "audio",
        "metrics",
        measured_context="MEASURED VISUALS\nmotion_energy=0.900",
    )
    assert "MEASURED VISUALS" in with_block


def test_amendments_empty_when_no_measurements() -> None:
    assert feedback_amendments({}, _style()) == []


def test_amendments_calm_when_motion_high() -> None:
    amendments = feedback_amendments({"motion_energy": 0.90}, _style())
    assert any("minimal motion" in item for item in amendments)


def test_amendments_gentle_when_motion_low() -> None:
    amendments = feedback_amendments({"motion_energy": 0.01}, _style())
    assert any("gentle continuous motion" in item for item in amendments)


def test_amendments_sparse_when_complexity_high() -> None:
    amendments = feedback_amendments({"visual_complexity": 0.95}, _style())
    assert any("sparse composition" in item for item in amendments)


def test_amendments_transform_when_drift_low() -> None:
    amendments = feedback_amendments({"semantic_change_rate": 0.01}, _style())
    assert any("gradual visible transformation" in item for item in amendments)


def test_amendments_charter_when_similarity_low() -> None:
    amendments = feedback_amendments({"style_similarity": 0.10}, _style())
    assert any("charter style" in item for item in amendments)


def test_amendments_quiet_within_bands() -> None:
    assert (
        feedback_amendments(
            {
                "motion_energy": 0.25,
                "visual_complexity": 0.40,
                "semantic_change_rate": 0.20,
                "style_similarity": 0.90,
            },
            _style(),
        )
        == []
    )


def test_apply_amendments_appends() -> None:
    assert (
        apply_feedback_amendments("a valley", ["calm static composition"])
        == "a valley, calm static composition"
    )
    assert apply_feedback_amendments("a valley", []) == "a valley"
