"""Audio RPC/library parameter validation (issue 063).

CPU-only: every rejection happens before the ACE stack loads or ffmpeg
runs, so no GPU, weights, or audio rendering is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from voyage.audio.acestep import (
    validate_bpm,
    validate_duration_seconds,
    validate_reference_audio,
    validate_task_type,
)
from voyage.config import AudioConfig, default_config_toml, load_config, resolve_config
from voyage.workers import audio_acestep


def test_bpm_rejects_zero_and_out_of_range() -> None:
    for bad_bpm in (0, -12, 301, 1000000):
        with pytest.raises(ValueError, match="bpm"):
            validate_bpm(bad_bpm)


def test_bpm_accepts_none_and_musical_range() -> None:
    validate_bpm(None)
    for good_bpm in (1, 60, 110, 140, 300):
        validate_bpm(good_bpm)


def test_duration_rejects_non_positive_and_non_finite() -> None:
    for bad_duration in (0.0, -3.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="duration_seconds"):
            validate_duration_seconds(bad_duration)


def test_duration_accepts_positive_values() -> None:
    validate_duration_seconds(0.5)
    validate_duration_seconds(15.0)


def test_task_type_rejects_unknown_modes() -> None:
    for bad_task in ("not-a-task", "", "variation"):
        with pytest.raises(ValueError, match="task_type"):
            validate_task_type(bad_task)


def test_task_type_accepts_known_modes() -> None:
    validate_task_type("text2music")
    validate_task_type("repaint")


def test_reference_audio_rejects_non_paths_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reference_audio"):
        validate_reference_audio(12345)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reference_audio"):
        validate_reference_audio(str(tmp_path / "missing.wav"))


def test_reference_audio_accepts_none_and_existing_file(tmp_path: Path) -> None:
    validate_reference_audio(None)
    existing = tmp_path / "take.wav"
    existing.write_bytes(b"fake-wav")
    validate_reference_audio(existing)
    validate_reference_audio(str(existing))


def test_sample_rate_and_channels_bounds() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        audio_acestep.validate_sample_rate(0)
    with pytest.raises(ValueError, match="channels"):
        audio_acestep.validate_channels(0)
    with pytest.raises(ValueError, match="channels"):
        audio_acestep.validate_channels(3)
    audio_acestep.validate_sample_rate(48000)
    audio_acestep.validate_channels(1)
    audio_acestep.validate_channels(2)


def _generate_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "segment_id": "000001",
        "style": "pastel neon line-art, peaceful",
        "energy": 0.5,
        "seed": 7,
        "output_path": "/tmp/voyage-audio-probe/take.wav",
        "sample_rate": 48000,
        "channels": 2,
        "duration_seconds": 15.0,
    }
    payload.update(overrides)
    return payload


def test_handle_generate_audio_rejects_bad_bpm_before_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> Any:
        raise AssertionError("stack must not load for invalid params")

    monkeypatch.setattr(audio_acestep, "_require_stack", _boom)
    with pytest.raises(ValueError, match="bpm"):
        audio_acestep.handle_generate_audio(_generate_payload(bpm=0))


def test_handle_generate_audio_rejects_bad_audio_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> Any:
        raise AssertionError("stack must not load for invalid params")

    monkeypatch.setattr(audio_acestep, "_require_stack", _boom)
    with pytest.raises(ValueError, match="sample_rate"):
        audio_acestep.handle_generate_audio(_generate_payload(sample_rate=0))
    with pytest.raises(ValueError, match="channels"):
        audio_acestep.handle_generate_audio(_generate_payload(channels=0))
    with pytest.raises(ValueError, match="duration_seconds"):
        audio_acestep.handle_generate_audio(_generate_payload(duration_seconds=-3.0))
    with pytest.raises(ValueError, match="task_type"):
        audio_acestep.handle_generate_audio(_generate_payload(task_type="not-a-task"))
    with pytest.raises(ValueError, match="reference_audio"):
        audio_acestep.handle_generate_audio(_generate_payload(reference_audio=12345))


# --- 088 fold: tests/test_audio_take_ahead_guard.py (7 tests) ---
# """Issue 013 guard: take_seconds must exceed ahead_seconds (config validation).
#
# The pathological per-segment video↔audio GPU swap hides behind the
# take_seconds >> ahead_seconds invariant (today only an invariant test, no
# validator). This promotes the invariant into an AudioConfig validator so
# bad values fail at config load, not after GPU hours.
# """


def test_audio_config_defaults_satisfy_take_ahead_rule() -> None:
    audio = AudioConfig()
    assert audio.take_seconds > audio.ahead_seconds


def test_audio_config_rejects_take_equal_to_ahead() -> None:
    with pytest.raises(ValidationError, match="must exceed ahead_seconds"):
        AudioConfig(take_seconds=20.0, ahead_seconds=20.0)


def test_audio_config_rejects_take_shorter_than_ahead() -> None:
    with pytest.raises(ValidationError, match="must exceed ahead_seconds"):
        AudioConfig(take_seconds=10.0, ahead_seconds=20.0)


def test_audio_config_rejection_names_the_swap_cost() -> None:
    with pytest.raises(ValidationError, match="GPU swap"):
        AudioConfig(take_seconds=5.0, ahead_seconds=20.0)


def test_audio_config_accepts_take_just_above_ahead() -> None:
    audio = AudioConfig(take_seconds=20.5, ahead_seconds=20.0)
    assert audio.take_seconds == pytest.approx(20.5)


def test_resolve_config_take_override_below_ahead_fails() -> None:
    from voyage.config import ProjectConfig

    base = ProjectConfig(style="take-ahead probe", audio=AudioConfig())
    with pytest.raises(ValidationError, match="must exceed ahead_seconds"):
        resolve_config(base, take_seconds=10.0)


def test_default_toml_still_loads_under_the_rule(tmp_path: Path) -> None:
    path = tmp_path / "voyage.toml"
    path.write_text(
        default_config_toml("take-ahead", "pastel neon line-art, peaceful", 7),
        encoding="utf-8",
    )
    config, _digest = load_config(path)
    assert config.audio.take_seconds > config.audio.ahead_seconds
