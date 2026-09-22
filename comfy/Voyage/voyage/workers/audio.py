"""Audio worker: `python -m voyage.workers.audio`."""

from __future__ import annotations

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


def main() -> None:
    serve(
        {
            "init": handle_health,
            "health": handle_health,
            "generate_audio": handle_generate_audio,
            "checkpoint": lambda payload: {
                "checkpoint_id": f"audio-{payload.get('segment_id', 'none')}"
            },
            "resume": lambda payload: {
                "resumed": True,
                "checkpoint_id": payload.get("checkpoint_id"),
            },
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()
