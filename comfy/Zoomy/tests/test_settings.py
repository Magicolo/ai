"""Tests for environment-driven settings."""

from __future__ import annotations

import math
from typing import Literal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from zoomy.errors import ZoomyError
from zoomy.settings import Settings

SETTING_NAMES = (
    "ZOOMY_COMFY_ADDRESS",
    "ZOOMY_OUTPUT_DIRECTORY",
    "ZOOMY_INTERFACE_ADDRESS",
    "ZOOMY_INTERFACE_PORT",
    "ZOOMY_OPERATION_TIMEOUT_SECONDS",
    "ZOOMY_POLL_INTERVAL_SECONDS",
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
    settings = Settings.from_environment()
    assert settings.comfy_address == "http://comfy:8188"
    assert settings.output_directory == "/comfy/output"
    assert settings.interface_address == "0.0.0.0"
    assert settings.interface_port == 7861
    assert settings.operation_timeout_seconds == 1800.0
    assert settings.poll_interval_seconds == 2.0


def test_from_environment_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each ZOOMY_* variable overrides its setting."""
    monkeypatch.setenv("ZOOMY_COMFY_ADDRESS", "http://comfy-service:8188")
    monkeypatch.setenv("ZOOMY_OUTPUT_DIRECTORY", "/other/output")
    monkeypatch.setenv("ZOOMY_INTERFACE_ADDRESS", "127.0.0.1")
    monkeypatch.setenv("ZOOMY_INTERFACE_PORT", "7000")
    monkeypatch.setenv("ZOOMY_OPERATION_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("ZOOMY_POLL_INTERVAL_SECONDS", "0.5")
    settings = Settings.from_environment()
    assert settings.comfy_address == "http://comfy-service:8188"
    assert settings.output_directory == "/other/output"
    assert settings.interface_address == "127.0.0.1"
    assert settings.interface_port == 7000
    assert settings.operation_timeout_seconds == 90.0
    assert settings.poll_interval_seconds == 0.5


def test_from_environment_rejects_non_integer_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-integer ZOOMY_INTERFACE_PORT raises a friendly error."""
    monkeypatch.setenv("ZOOMY_INTERFACE_PORT", "not-a-number")
    with pytest.raises(ZoomyError, match="ZOOMY_INTERFACE_PORT"):
        Settings.from_environment()


def test_from_environment_rejects_non_numeric_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric ZOOMY_OPERATION_TIMEOUT_SECONDS raises a friendly error."""
    monkeypatch.setenv("ZOOMY_OPERATION_TIMEOUT_SECONDS", "soon")
    with pytest.raises(ZoomyError, match="ZOOMY_OPERATION_TIMEOUT_SECONDS"):
        Settings.from_environment()


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw_value=environment_text)
def test_operation_timeout_is_finite_or_rejected(
    monkeypatch: pytest.MonkeyPatch, raw_value: str
) -> None:
    """Any timeout string yields a finite setting or a friendly error."""
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOOMY_OPERATION_TIMEOUT_SECONDS", raw_value)
    try:
        parsed = Settings.from_environment()
    except ZoomyError:
        return
    assert math.isfinite(parsed.operation_timeout_seconds)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw_value=environment_text)
def test_poll_interval_is_finite_or_rejected(
    monkeypatch: pytest.MonkeyPatch, raw_value: str
) -> None:
    """Any interval string yields a finite setting or a friendly error."""
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOOMY_POLL_INTERVAL_SECONDS", raw_value)
    try:
        parsed = Settings.from_environment()
    except ZoomyError:
        return
    assert math.isfinite(parsed.poll_interval_seconds)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw_value=blank_text)
def test_blank_numeric_values_fall_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch, raw_value: str
) -> None:
    """Blank numeric strings behave as unset (matching text settings)."""
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZOOMY_INTERFACE_PORT", raw_value)
    monkeypatch.setenv("ZOOMY_OPERATION_TIMEOUT_SECONDS", raw_value)
    monkeypatch.setenv("ZOOMY_POLL_INTERVAL_SECONDS", raw_value)
    parsed = Settings.from_environment()
    assert parsed.interface_port == 7861
    assert parsed.operation_timeout_seconds == 1800.0
    assert parsed.poll_interval_seconds == 2.0
