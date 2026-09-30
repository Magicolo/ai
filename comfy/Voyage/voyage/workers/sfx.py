"""SFX worker: `python -m voyage.workers.sfx`.

Fake backend behind the same `generate_sfx` contract as the real
MMAudio worker (`sfx_mmaudio`): renders deterministic seeded noise so
the finalize-time windowing/sharding/mix path is genuine with no GPU,
no weights, no network. No `torch` import at module scope or lazily —
the hard GPU ban (§12) holds trivially here.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from voyage.audio.mmaudio_sfx import validate_duration_seconds
from voyage.fake_backends import FakeSfxBackend
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts
from voyage.workers.sfx_mmaudio import validate_channels, validate_sample_rate

_backend = FakeSfxBackend()

BENCHMARK_DURATION_SECONDS = 2.0
"""Probe window for `handle_benchmark` (§104): startup excluded, noise only."""


def validate_window_id(window_id: str) -> None:
    """Reject empty window ids before rendering."""
    if not window_id.strip():
        raise ValueError("window_id must be non-empty")


def validate_output_path(output_path: str) -> None:
    """Reject empty output paths before creating parent directories."""
    if not output_path.strip():
        raise ValueError("output_path must be non-empty")


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    """Liveness probe: no payload fields required, never touches the backend."""
    del payload
    return {"status": "READY", "backend": _backend.name}


def handle_generate_sfx(payload: dict[str, Any]) -> dict[str, Any]:
    """Render one seeded-noise window; all fields validated first."""
    checked_request(
        payload,
        window_id=str,
        caption=str,
        video_path=str,
        start_seconds=float,
        duration_seconds=float,
        seed=int,
        output_path=str,
        sample_rate=int,
        channels=int,
    )
    validate_window_id(str(payload["window_id"]))
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
    result = _backend.generate_window(
        output,
        caption=str(payload["caption"]),
        seed=int(payload["seed"]),
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=duration,
    )
    return {
        "artifacts": [str(output)],
        "sfx": {"path": str(output), "backend": _backend.name, **result},
    }


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured noise windows (startup excluded, §104)."""
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    validate_benchmark_counts(warmup, measured)
    duration = float(payload.get("duration_seconds", BENCHMARK_DURATION_SECONDS))
    validate_duration_seconds(duration)
    walls: list[float] = []
    with tempfile.TemporaryDirectory(prefix="voyage-sfx-bench-") as staging_directory:
        for window_index in range(warmup + measured):
            started = time.monotonic()
            _backend.generate_window(
                Path(staging_directory) / f"bench_{window_index}.wav",
                caption="benchmark",
                seed=window_index,
                sample_rate=48000,
                channels=2,
                duration_seconds=duration,
            )
            elapsed = time.monotonic() - started
            if window_index >= warmup:
                walls.append(elapsed)
    mean = sum(walls) / len(walls)
    return {
        "backend": _backend.name,
        "warmup_windows": warmup,
        "measured_windows": measured,
        "window_wall_seconds": [round(wall, 3) for wall in walls],
        "windows_per_second": round(1.0 / mean, 3),
        "audio_seconds_per_wall_second": round(duration / mean, 3),
    }


def handle_evict_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    """No GPU state held; kept so the supervisor's evict path is uniform."""
    del payload
    return {"evicted": True}


def handle_shutdown(payload: dict[str, Any]) -> dict[str, Any]:
    """Stop the worker loop; nothing resident to release."""
    del payload
    return {"stopped": True}


def main() -> None:
    """Serve the fake-SFX op map over the shared JSONL loop."""
    serve(
        {
            "init": handle_health,
            "health": handle_health,
            "generate_sfx": handle_generate_sfx,
            "benchmark": handle_benchmark,
            "evict_gpu": handle_evict_gpu,
            "shutdown": handle_shutdown,
        }
    )


if __name__ == "__main__":
    main()
