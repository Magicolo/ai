"""Shared worker validators (issue 084, DESIGN §§45-46).

Single home for the triplicated worker validators: audio/SFX
`validate_sample_rate` + `validate_channels` were identical in three
modules, `validate_output_path` in three more, and the video geometry /
fps / frame-countValidators lived only in the fake video worker while the
GPU workers re-checked the same shapes inline. Every worker imports from
here; the old per-module names stay as thin re-exports so existing
imports (`audio_acestep.validate_sample_rate`, ...) keep working.

Pure stdlib (math only) — no torch, no ffmpeg, no filesystem side
effects — so slim-image and CPU tests import this freely. The fake SFX
worker imports from here instead of the GPU SFX worker, fixing the
reverse fake→GPU dependency (slim import no longer pulls GPU modules).

Duration: this module owns the base `validate_duration_seconds`
(finite and > 0, shared by the audio workers). MMAudio's bounded window
(`(0, MAX_WINDOW_SECONDS]`) stays in `voyage.audio.mmaudio_sfx`, which
adds the ceiling on top — one base, one specialization, not three copies.
"""

from __future__ import annotations

import math

MIN_ENERGY = 0.0
MAX_ENERGY = 1.0
"""Director energy knob bounds: the fake backend maps it to tone frequency."""


def validate_sample_rate(sample_rate: int) -> None:
    """Reject non-positive output sample rates."""
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not math.isfinite(sample_rate)
        or sample_rate <= 0
    ):
        raise ValueError(f"sample_rate must be positive (got {sample_rate})")


def validate_channels(channels: int) -> None:
    """Reject non-mono/stereo channel counts."""
    if isinstance(channels, bool) or channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2 (got {channels})")


def validate_output_path(output_path: str) -> None:
    """Reject empty output paths before creating parent directories."""
    if not output_path.strip():
        raise ValueError("output_path must be non-empty")


def validate_geometry(width: int, height: int) -> None:
    """Reject non-positive frame dimensions before any ffmpeg side effect."""
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, float))
        or not math.isfinite(width)
        or width <= 0
    ):
        raise ValueError(f"width must be positive (got {width})")
    if (
        isinstance(height, bool)
        or not isinstance(height, (int, float))
        or not math.isfinite(height)
        or height <= 0
    ):
        raise ValueError(f"height must be positive (got {height})")


def validate_fps(fps: int) -> None:
    """Reject non-positive frame rates before rendering."""
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(f"fps must be positive (got {fps})")


def validate_frame_count(frames: int) -> None:
    """Reject non-positive block lengths before rendering."""
    if (
        isinstance(frames, bool)
        or not isinstance(frames, (int, float))
        or not math.isfinite(frames)
        or frames <= 0
    ):
        raise ValueError(f"frames must be positive (got {frames})")


def validate_energy(energy: float) -> None:
    """Reject non-finite/out-of-range director energy (issue 063 class).

    Runs before any ffmpeg side effect so a bad knob fails as
    INVALID_PAYLOAD (fatal), never as a confusing ffmpeg error.
    """
    if not math.isfinite(energy) or not MIN_ENERGY <= energy <= MAX_ENERGY:
        raise ValueError(f"energy must be finite and {MIN_ENERGY}..{MAX_ENERGY} (got {energy})")


def validate_window_id(window_id: str) -> None:
    """Reject empty window ids before rendering."""
    if not window_id.strip():
        raise ValueError("window_id must be non-empty")


def validate_duration_seconds(duration_seconds: float) -> None:
    """Reject non-positive/non-finite take lengths before rendering."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(f"duration_seconds must be finite and > 0 (got {duration_seconds})")
