"""Issue 013 guard: take_seconds must exceed ahead_seconds (config validation).

The pathological per-segment video↔audio GPU swap hides behind the
take_seconds >> ahead_seconds invariant (today only an invariant test, no
validator). This promotes the invariant into an AudioConfig validator so
bad values fail at config load, not after GPU hours.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from voyage.config import AudioConfig, default_config_toml, load_config, resolve_config


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
