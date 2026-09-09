"""Application settings read from ``ZOOMY_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from zoomy.errors import ZoomyError

DEFAULT_COMFY_ADDRESS = "http://comfy:8188"
DEFAULT_OUTPUT_DIRECTORY = "/comfy/output"
DEFAULT_INTERFACE_ADDRESS = "0.0.0.0"
DEFAULT_INTERFACE_PORT = 7861
DEFAULT_OPERATION_TIMEOUT_SECONDS = 1800.0
DEFAULT_POLL_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for one zoomy process."""

    comfy_address: str
    output_directory: str
    interface_address: str
    interface_port: int
    operation_timeout_seconds: float
    poll_interval_seconds: float

    @classmethod
    def from_environment(cls) -> Settings:
        """Build settings from ``ZOOMY_*`` environment variables."""
        return cls(
            comfy_address=_read_text("ZOOMY_COMFY_ADDRESS", DEFAULT_COMFY_ADDRESS),
            output_directory=_read_text("ZOOMY_OUTPUT_DIRECTORY", DEFAULT_OUTPUT_DIRECTORY),
            interface_address=_read_text("ZOOMY_INTERFACE_ADDRESS", DEFAULT_INTERFACE_ADDRESS),
            interface_port=_read_integer("ZOOMY_INTERFACE_PORT", DEFAULT_INTERFACE_PORT),
            operation_timeout_seconds=_read_float(
                "ZOOMY_OPERATION_TIMEOUT_SECONDS", DEFAULT_OPERATION_TIMEOUT_SECONDS
            ),
            poll_interval_seconds=_read_float(
                "ZOOMY_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS
            ),
        )


def _read_text(name: str, default: str) -> str:
    """Return the environment value or the default when unset or blank."""
    return os.environ.get(name, default) or default


def _read_integer(name: str, default: int) -> int:
    """Parse an integer environment value, raising a friendly error on garbage."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError as failure:
        message = f"Environment variable {name} must contain an integer, received {raw!r}"
        raise ZoomyError(message) from failure
    return parsed


def _read_float(name: str, default: float) -> float:
    """Parse a floating-point environment value, raising a friendly error."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError as failure:
        message = f"Environment variable {name} must contain a number, received {raw!r}"
        raise ZoomyError(message) from failure
    return parsed
