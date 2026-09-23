"""Audio worker: `python -m voyage.workers.audio`."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from voyage.fake_backends import FakeAudioBackend
from voyage.workers.loop import checked_request, serve

_backend = FakeAudioBackend()


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "READY", "backend": _backend.name}


def handle_generate_audio(payload: dict[str, Any]) -> dict[str, Any]:
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
    output = Path(str(payload["output_path"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _backend.generate_segment(
        output,
        style=str(payload["style"]),
        energy=float(payload["energy"]),
        seed=int(payload["seed"]),
        sample_rate=int(payload["sample_rate"]),
        channels=int(payload["channels"]),
        duration_seconds=float(payload["duration_seconds"]),
    )
    return {"artifacts": [str(output)], "audio": result}


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured sine renders (startup excluded, §104)."""
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    duration = float(payload.get("duration_seconds", 2.0))
    sample_rate = int(payload.get("sample_rate", 48000))
    channels = int(payload.get("channels", 2))
    walls: list[float] = []
    with tempfile.TemporaryDirectory(prefix="voyage-bench-") as tmp:
        for index in range(warmup + measured):
            started = time.monotonic()
            _backend.generate_segment(
                Path(tmp) / f"bench_{index}.wav",
                style="benchmark",
                energy=0.5,
                seed=index,
                sample_rate=sample_rate,
                channels=channels,
                duration_seconds=duration,
            )
            elapsed = time.monotonic() - started
            if index >= warmup:
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


def main() -> None:
    serve(
        {
            "init": handle_health,
            "health": handle_health,
            "generate_audio": handle_generate_audio,
            "benchmark": handle_benchmark,
            "checkpoint": lambda payload: {
                "checkpoint_id": f"audio-{payload.get('segment_id', 'none')}"
            },
            "evict_gpu": lambda _payload: {"evicted": True},
            "resume": lambda payload: {
                "resumed": True,
                "checkpoint_id": payload.get("checkpoint_id"),
            },
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()
