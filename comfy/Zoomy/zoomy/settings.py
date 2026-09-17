"""Application settings read from ``ZOOMY_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from zoomy.errors import ZoomyError

DEFAULT_MODELS_DIRECTORY = "/models"
DEFAULT_SEED_DIRECTORY = "/seed"
DEFAULT_MUSIC_PROJECT_DIRECTORY = "/music-project"
DEFAULT_OUTPUT_DIRECTORY = "/output"
DEFAULT_INTERFACE_ADDRESS = "0.0.0.0"
DEFAULT_INTERFACE_PORT = 7861
DEFAULT_CUDA_DEVICE = "cuda:0"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for one zoomy process."""

    models_directory: str
    seed_directory: str
    music_project_directory: str
    output_directory: str
    interface_address: str
    interface_port: int
    cuda_device: str

    @classmethod
    def from_environment(cls) -> Settings:
        """Build settings from ``ZOOMY_*`` environment variables."""
        return cls(
            models_directory=_read_text("ZOOMY_MODELS_DIRECTORY", DEFAULT_MODELS_DIRECTORY),
            seed_directory=_read_text("ZOOMY_SEED_DIRECTORY", DEFAULT_SEED_DIRECTORY),
            music_project_directory=_read_text(
                "ZOOMY_MUSIC_PROJECT_DIRECTORY", DEFAULT_MUSIC_PROJECT_DIRECTORY
            ),
            output_directory=_read_text("ZOOMY_OUTPUT_DIRECTORY", DEFAULT_OUTPUT_DIRECTORY),
            interface_address=_read_text("ZOOMY_INTERFACE_ADDRESS", DEFAULT_INTERFACE_ADDRESS),
            interface_port=_read_integer("ZOOMY_INTERFACE_PORT", DEFAULT_INTERFACE_PORT),
            cuda_device=_read_text("ZOOMY_CUDA_DEVICE", DEFAULT_CUDA_DEVICE),
        )


def _read_text(name: str, default: str) -> str:
    """Return the stripped environment value, or the default when unset or blank.

    Directories, addresses, and device names never want surrounding padding,
    so the returned value is stripped; a whitespace-only value means
    "not configured", matching :func:`_read_integer`.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _read_integer(name: str, default: int) -> int:
    """Parse an integer environment value, raising a friendly error on garbage.

    A missing or blank value falls back to the default, matching the text
    settings (an explicitly blank variable means "not configured").
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw)
    except ValueError as failure:
        message = f"Environment variable {name} must contain an integer, received {raw!r}"
        raise ZoomyError(message) from failure
    return parsed
