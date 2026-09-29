"""Three caption families with drift evolution (slice 1, TDD).

The director generates video / music / SFX captions from the style
charter + evolving general prompt. Every caption family must track the
slow per-segment concept drift: a new concept yields new captions, a
held concept yields stable captions. SFX evolution must never trigger
a music-take repaint (the slow-loop planner keys on music only).
"""

from __future__ import annotations

import json
from pathlib import Path

from voyage.audio.planner import AudioPlanner
from voyage.director import (
    build_director_user_message,
    deterministic_decision,
    director_input_from_state,
    format_previous_captions,
)
from voyage.models import DirectorAudioPlan, EvolutionDecision


def test_audio_plan_carries_sfx_caption_with_backward_compat_default() -> None:
    plan = DirectorAudioPlan()
    assert plan.sfx_caption == ""
    legacy = {
        "music_caption": "slow ambient electronic composition",
        "energy": 0.48,
        "tempo_bpm": 74,
        "texture": "",
        "environment": [],
    }
    revived = DirectorAudioPlan.model_validate(legacy)
    assert revived.sfx_caption == ""
    assert revived.music_caption == "slow ambient electronic composition"


def test_director_shape_text_requires_three_caption_families() -> None:
    message = build_director_user_message(
        style_charter="pastel neon line-art",
        current_world="a reef",
        current_transition="none",
        recent_summary="reef",
        forbidden_summary="none",
        audio_state="quiet",
        controller_metrics="ok",
    )
    assert "sfx_caption" in message
    assert "music_caption" in message
    assert "instruments" in message.lower()
    assert "concrete" in message.lower() or "sound" in message.lower()


def test_director_shape_text_requires_gradual_caption_evolution() -> None:
    message = build_director_user_message(
        style_charter="pastel neon line-art",
        current_world="a reef",
        current_transition="none",
        recent_summary="reef",
        forbidden_summary="none",
        audio_state="quiet",
        controller_metrics="ok",
        previous_captions="PREVIOUS CAPTIONS\nvideo: reef dawn\nmusic: soft pads\nsfx: water",
    )
    assert "PREVIOUS CAPTIONS" in message
    lowered = message.lower()
    assert "gradual" in lowered or "evolve" in lowered or "drift" in lowered


def test_deterministic_captions_track_concept_drift() -> None:
    first = deterministic_decision(0, "reef", "crystal reef", "DRIFT", "pastel neon")
    second = deterministic_decision(1, "crystal reef", "glass desert", "DRIFT", "pastel neon")
    assert first.video.stages != second.video.stages
    assert first.audio.music_caption != second.audio.music_caption
    assert first.audio.sfx_caption != second.audio.sfx_caption
    for caption in (
        *first.video.stages,
        first.audio.music_caption,
        first.audio.sfx_caption,
    ):
        assert "crystal reef" in caption or "pastel neon" in caption


def test_deterministic_captions_hold_when_concept_holds() -> None:
    first = deterministic_decision(0, "reef", "reef", "ESTABLISH", "pastel neon")
    second = deterministic_decision(1, "reef", "reef", "ESTABLISH", "pastel neon")
    assert first.video.stages == second.video.stages
    assert first.audio.music_caption == second.audio.music_caption
    assert first.audio.sfx_caption == second.audio.sfx_caption


def test_format_previous_captions_is_stable_text() -> None:
    text = format_previous_captions(
        previous_video_stages=["reef dawn", "reef noon"],
        previous_music="soft pads",
        previous_sfx="lapping water",
    )
    assert "reef dawn" in text
    assert "soft pads" in text
    assert "lapping water" in text
    assert format_previous_captions([], "", "") == ""


def test_director_input_carries_previous_captions() -> None:
    import tempfile

    from voyage.config import default_config_toml, load_config
    from voyage.persistence import initial_state

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "voyage.toml"
        config_path.write_text(
            default_config_toml("demo", "pastel neon line-art, peaceful", 1),
            encoding="utf-8",
        )
        config, _ = load_config(config_path)
        state = initial_state(config)
        payload = director_input_from_state(
            state,
            style_charter=config.style,
            recent_summary="(no concepts yet)",
            forbidden_summary="(revisits allowed)",
            audio_state="style=ambient energy=0.5",
            previous_captions="PREVIOUS CAPTIONS\nvideo: reef",
        )
        assert payload["previous_captions"] == "PREVIOUS CAPTIONS\nvideo: reef"


def test_previous_transition_captions_load_best_effort(tmp_path: Path) -> None:
    from voyage.supervisor import previous_transition_captions

    assert previous_transition_captions(tmp_path, 0) == ""
    segment = tmp_path / "segments" / "000000"
    segment.mkdir(parents=True)
    decision = deterministic_decision(0, "reef", "crystal reef", "DRIFT", "pastel neon")
    (segment / "transition.json").write_text(json.dumps(decision.model_dump()), encoding="utf-8")
    text = previous_transition_captions(tmp_path, 1)
    assert "crystal reef" in text
    (segment / "transition.json").write_text("{torn", encoding="utf-8")
    assert previous_transition_captions(tmp_path, 1) == ""


def test_planner_ignores_sfx_drift_for_music_repaint() -> None:
    planner = AudioPlanner(take_seconds=45.0, ahead_seconds=20.0, takes=[])
    first = planner.plan(0.0, "soft pads", 1, 0)
    assert first.action == "render" and first.take is not None
    planner.record(first.take)
    evolved = EvolutionDecision.model_validate(
        {
            "decision_index": 1,
            "destination": {"canonical_name": "reef"},
            "audio": {
                "music_caption": "soft pads",
                "sfx_caption": "crackling arcs, lapping water",
                "energy": 0.5,
                "tempo_bpm": 74,
                "texture": "",
                "environment": [],
            },
        }
    )
    assert evolved.audio.sfx_caption != ""
    keep = planner.plan(1.0, evolved.audio.music_caption, 2, 1)
    assert keep.action == "keep"
