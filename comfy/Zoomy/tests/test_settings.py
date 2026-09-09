"""Tests for environment-driven settings."""

from __future__ import annotations

import pytest

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
