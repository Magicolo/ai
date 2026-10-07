"""Lightweight post-commit motion sense (DESIGN §43 cheap tier).

Why this module exists: frozen segments used to commit silently — the
six-metric inspector was never wired on the live path. This is the
cheapest useful detector: three 160px frames via the existing
`sample_frames` select-filter path plus pixel-delta `motion_energy`.
Typical cost is 100-300ms of ffmpeg thumbnail decode, no GPU, no model
load — small enough to run inline after the state advance without
stalling the next segment's video render.

Contract: `sense_motion` never raises. A missing/corrupt clip returns
`energy=None` (unknown, never a deviation) so the commit still succeeds
and the next prompt simply gets no steering.

Timing vocab (Track E, DESIGN §59): the sense wall time feeds the
generation `motion_sense` key via `motion_stage_fragment` (and
`logrotate.build_stage_seconds(..., sense_ms=...)`); the prompt-enhance
split lives in `logrotate.build_stage_seconds(..., enhance_ms=...)`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionReading:
    """One cheap motion sample: energy 0..1 or None when unknown."""

    energy: float | None
    seconds: float


#: Energy below which a segment counts as low-motion for display/steering.
#: Mirrors the `StyleSpec.motion_energy_min` default (0.20) so the console
#: flag agrees with the amendment floor without the console needing the
#: full style spec.
LOW_MOTION_FLOOR = 0.20

#: Generation timing key for the post-commit sense wall time (Track E).
SENSE_STAGE_KEY = "motion_sense"

#: Generation timing key for the prompt-enhance split (Track E companion).
#: The value lives in `logrotate.build_stage_seconds`, not here — this
#: constant keeps the two halves spelling the key identically.
ENHANCE_STAGE_KEY = "prompt_enhance"


def sense_motion(video_path: Path, *, count: int = 3, width: int = 160) -> MotionReading:
    """Sample `video_path` and return pixel-delta motion energy + wall time."""
    started = time.monotonic()
    try:
        from voyage.vision.metrics import motion_energy, sample_frames

        frames = sample_frames(video_path, count=count, width=width)
        energy: float | None = float(motion_energy(frames))
    except Exception:  # noqa: BLE001 - best-effort sensor, never fails a commit
        energy = None
    return MotionReading(energy=energy, seconds=round(time.monotonic() - started, 3))


def motion_stage_fragment(reading: MotionReading) -> dict[str, float]:
    """One-key generation timing fragment for a sense reading (Track E).

    Pure helper so Track A fills the `motion_sense` split without spelling
    the key: `{SENSE_STAGE_KEY: reading.seconds}` rounded to milliseconds.
    Unknown readings (sense skipped) still report their wall time — the
    steering input is None but the timing is real.
    """
    return {SENSE_STAGE_KEY: round(float(reading.seconds), 3)}
