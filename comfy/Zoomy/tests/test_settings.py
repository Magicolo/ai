"""Tests for environment-driven settings."""

from __future__ import annotations

from typing import Literal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from zoomy.errors import ZoomyError
from zoomy.settings import Settings

SETTING_NAMES = (
    "ZOOMY_MODELS_DIRECTORY",
    "ZOOMY_SEED_DIRECTORY",
    "ZOOMY_MUSIC_PROJECT_DIRECTORY",
    "ZOOMY_OUTPUT_DIRECTORY",
    "ZOOMY_INTERFACE_ADDRESS",
    "ZOOMY_INTERFACE_PORT",
    "ZOOMY_CUDA_DEVICE",
)

# Operating systems forbid NUL bytes in environment variables, and lone
# surrogates cannot round-trip through this container's encoder, so the
# generators below exclude both; anything else is fair game.
NON_ROUND_TRIPPABLE_CATEGORIES: tuple[Literal["Cs"], ...] = ("Cs",)
environment_text = st.text(
    alphabet=st.characters(
        blacklist_characters="\x00", blacklist_categories=NON_ROUND_TRIPPABLE_CATEGORIES
    )
)

# Blank environment values (empty or whitespace-only) fall back to defaults,
# generated directly since filtering arbitrary text for blankness starves.
blank_text = st.text(
    alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c", chr(0xA0)]),
    max_size=10,
)


def test_from_environment_uses_defaults_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ZOOMY_* variable unset yields the documented defaults."""
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    parsed = Settings.from_environment()
    assert parsed.models_directory == "/models"
    assert parsed.seed_directory == "/seed"
    assert parsed.music_project_directory == "/music-project"
    assert parsed.output_directory == "/output"
    assert parsed.interface_address == "0.0.0.0"
    assert parsed.interface_port == 7861
    assert parsed.cuda_device == "cuda:0"


def test_from_environment_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each ZOOMY_* variable overrides its setting."""
    monkeypatch.setenv("ZOOMY_MODELS_DIRECTORY", "/other/models")
    monkeypatch.setenv("ZOOMY_SEED_DIRECTORY", "/other/seed")
    monkeypatch.setenv("ZOOMY_MUSIC_PROJECT_DIRECTORY", "/other/music")
    monkeypatch.setenv("ZOOMY_OUTPUT_DIRECTORY", "/other/output")
    monkeypatch.setenv("ZOOMY_INTERFACE_ADDRESS", "127.0.0.1")
    monkeypatch.setenv("ZOOMY_INTERFACE_PORT", "7000")
    monkeypatch.setenv("ZOOMY_CUDA_DEVICE", "cuda:1")
    parsed = Settings.from_environment()
    assert parsed.models_directory == "/other/models"
    assert parsed.seed_directory == "/other/seed"
    assert parsed.music_project_directory == "/other/music"
    assert parsed.output_directory == "/other/output"
    assert parsed.interface_address == "127.0.0.1"
    assert parsed.interface_port == 7000
    assert parsed.cuda_device == "cuda:1"


def test_from_environment_rejects_non_integer_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-integer ZOOMY_INTERFACE_PORT raises a friendly error."""
    monkeypatch.setenv("ZOOMY_INTERFACE_PORT", "not-a-number")
    with pytest.raises(ZoomyError, match="ZOOMY_INTERFACE_PORT"):
        Settings.from_environment()


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw_value=blank_text)
def test_blank_port_falls_back_to_default(monkeypatch: pytest.MonkeyPatch, raw_value: str) -> None:
    """Blank port strings behave as unset (matching text settings)."""
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOOMY_INTERFACE_PORT", raw_value)
    parsed = Settings.from_environment()
    assert parsed.interface_port == 7861


def test_empty_text_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty directory strings behave as unset."""
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOOMY_MODELS_DIRECTORY", "")
    monkeypatch.setenv("ZOOMY_CUDA_DEVICE", "")
    parsed = Settings.from_environment()
    assert parsed.models_directory == "/models"
    assert parsed.cuda_device == "cuda:0"


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw_value=blank_text)
def test_blank_text_values_fall_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch, raw_value: str
) -> None:
    """Whitespace-only text values behave as unset (matching the port)."""
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOOMY_MODELS_DIRECTORY", raw_value)
    monkeypatch.setenv("ZOOMY_OUTPUT_DIRECTORY", raw_value)
    monkeypatch.setenv("ZOOMY_CUDA_DEVICE", raw_value)
    parsed = Settings.from_environment()
    assert parsed.models_directory == "/models"
    assert parsed.output_directory == "/output"
    assert parsed.cuda_device == "cuda:0"


def test_padded_text_values_are_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surrounding padding is not part of a directory or device name."""
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOOMY_MODELS_DIRECTORY", "  /other/models\t")
    parsed = Settings.from_environment()
    assert parsed.models_directory == "/other/models"
