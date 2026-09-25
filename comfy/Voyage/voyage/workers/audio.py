"""Audio worker: `python -m voyage.workers.audio`.

Fake backend behind the same `generate_audio` contract as the real ACE-Step
worker (`audio_acestep`): renders a deterministic sine tone with ffmpeg so
the commit/validate/finalize path is genuine with no GPU, no weights, no
network (DESIGN §§45-46 transport, §37 audio). `output_path` arrives as an
absolute wire path from the supervisor (codebase invariant: absolute on the
wire, run-relative refs in store); this worker only ensures its parent
directory exists. No `torch` import at module scope or lazily — the hard GPU
ban (§12) holds trivially here.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

from voyage.fake_backends import FakeAudioBackend
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts

_backend = FakeAudioBackend()

BENCHMARK_WARMUP_TAKES = 1
BENCHMARK_MEASURED_TAKES = 3
BENCHMARK_DURATION_SECONDS = 2.0
BENCHMARK_SAMPLE_RATE = 48000
BENCHMARK_CHANNELS = 2
BENCHMARK_ENERGY = 0.5
"""Probe shape for `handle_benchmark` (§104): startup excluded, sine renders only."""

MIN_ENERGY = 0.0
MAX_ENERGY = 1.0
"""Director energy knob bounds: the fake backend maps it to tone frequency."""


def validate_energy(energy: float) -> None:
    """Reject non-finite/out-of-range director energy (issue 063 class).

    Runs before any ffmpeg side effect so a bad knob fails as
    INVALID_PAYLOAD (fatal), never as a confusing ffmpeg error.
    """
    if not math.isfinite(energy) or not MIN_ENERGY <= energy <= MAX_ENERGY:
        raise ValueError(f"energy must be finite and {MIN_ENERGY}..{MAX_ENERGY} (got {energy})")


def validate_duration_seconds(duration_seconds: float) -> None:
    """Reject non-positive/non-finite take lengths before rendering."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(f"duration_seconds must be finite and > 0 (got {duration_seconds})")


def validate_sample_rate(sample_rate: int) -> None:
    """Reject non-positive output sample rates."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive (got {sample_rate})")


def validate_channels(channels: int) -> None:
    """Reject non-mono/stereo channel counts."""
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2 (got {channels})")


def validate_output_path(output_path: str) -> None:
    """Reject empty output paths before creating parent directories."""
    if not output_path.strip():
        raise ValueError("output_path must be non-empty")


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    """Liveness probe: no payload fields required, never touches the backend."""
    del payload
    return {"status": "READY", "backend": _backend.name}


def handle_generate_audio(payload: dict[str, Any]) -> dict[str, Any]:
    """Render one sine-tone segment; all numeric fields validated first."""
    checked_request(
        payload,
        segment_id=str,
        style=str,
        energy=float,
        seed=int,
        output_path=str,
        sample_rate=int,
        channels=int,
        duration_seconds=float,
    )
    energy = float(payload["energy"])
    validate_energy(energy)
    duration = float(payload["duration_seconds"])
    validate_duration_seconds(duration)
    sample_rate = int(payload["sample_rate"])
    validate_sample_rate(sample_rate)
    channels = int(payload["channels"])
    validate_channels(channels)
    output_raw = str(payload["output_path"])
    validate_output_path(output_raw)
    output = Path(output_raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _backend.generate_segment(
        output,
        style=str(payload["style"]),
        energy=energy,
        seed=int(payload["seed"]),
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=duration,
    )
    return {"artifacts": [str(output)], "audio": result}


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured sine renders (startup excluded, §104).

    Counts are validated before any render (issue 060): without the check
    `measured=0` divides by zero after doing all the work.
    """
    warmup = int(payload.get("warmup", BENCHMARK_WARMUP_TAKES))
    measured = int(payload.get("measured", BENCHMARK_MEASURED_TAKES))
    validate_benchmark_counts(warmup, measured)
    duration = float(payload.get("duration_seconds", BENCHMARK_DURATION_SECONDS))
    validate_duration_seconds(duration)
    sample_rate = int(payload.get("sample_rate", BENCHMARK_SAMPLE_RATE))
    validate_sample_rate(sample_rate)
    channels = int(payload.get("channels", BENCHMARK_CHANNELS))
    validate_channels(channels)
    walls: list[float] = []
    with tempfile.TemporaryDirectory(prefix="voyage-bench-") as staging_directory:
        for take_index in range(warmup + measured):
            started = time.monotonic()
            _backend.generate_segment(
                Path(staging_directory) / f"bench_{take_index}.wav",
                style="benchmark",
                energy=BENCHMARK_ENERGY,
                seed=take_index,
                sample_rate=sample_rate,
                channels=channels,
                duration_seconds=duration,
            )
            elapsed = time.monotonic() - started
            if take_index >= warmup:
                walls.append(elapsed)
    mean = sum(walls) / len(walls)
    return {
        "backend": _backend.name,
        "warmup_takes": warmup,
        "measured_takes": measured,
        "take_wall_seconds": [round(wall, 3) for wall in walls],
        "takes_per_second": round(1.0 / mean, 3),
        "audio_seconds_per_wall_second": round(duration / mean, 3),
    }


def handle_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Fake checkpoint: stateless backend, so the id is derived, not stored."""
    return {"checkpoint_id": f"audio-{payload.get('segment_id', 'none')}"}


def handle_evict_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    """No GPU state held; kept so the supervisor's evict path is uniform."""
    del payload
    return {"evicted": True}


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    """Fake resume: echoes the checkpoint id without loading anything."""
    return {"resumed": True, "checkpoint_id": payload.get("checkpoint_id")}


def handle_shutdown(payload: dict[str, Any]) -> dict[str, Any]:
    """Stop the worker loop; nothing resident to release."""
    del payload
    return {"stopped": True}


def main() -> None:
    """Serve the fake-audio op map over the shared JSONL loop."""
    serve(
        {
            "init": handle_health,
            "health": handle_health,
            "generate_audio": handle_generate_audio,
            "benchmark": handle_benchmark,
            "checkpoint": handle_checkpoint,
            "evict_gpu": handle_evict_gpu,
            "resume": handle_resume,
            "shutdown": handle_shutdown,
        }
    )


if __name__ == "__main__":
    main()
