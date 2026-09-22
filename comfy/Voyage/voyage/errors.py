"""Explicit exception taxonomy (DESIGN §48). Supervisor restart decisions
branch on error class, never on message string matching."""

from __future__ import annotations


class VoyageError(Exception):
    """Base class for all voyage errors."""


class ConfigurationError(VoyageError):
    """Invalid config, TOML, or CLI arguments."""


class WorkerError(VoyageError):
    """Base class for worker failures."""


class RecoverableWorkerError(WorkerError):
    """Worker failed but the run can continue after restart/retry."""


class FatalWorkerError(WorkerError):
    """Worker failed in a way that ends the run."""


class MediaError(VoyageError):
    """ffmpeg/ffprobe failure or media validation failure."""


class StateError(VoyageError):
    """Corrupt or inconsistent persistent state."""


class ModelCompatibilityError(VoyageError):
    """Model/checkpoint incompatible with the selected profile."""


class DiskSpaceError(VoyageError):
    """Free-space reserve would be crossed by the next commit."""
