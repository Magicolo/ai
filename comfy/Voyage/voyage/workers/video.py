"""Video worker: `python -m voyage.workers.video`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voyage.fake_backends import FakeVideoBackend
from voyage.workers.loop import checked_request, serve

_backend = FakeVideoBackend()


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "READY", "backend": _backend.name}


def handle_generate_blocks(payload: dict[str, Any]) -> dict[str, Any]:
    checked_request(
        payload,
        segment_id=str,
        prompt=str,
        seed=int,
        output_path=str,
        width=int,
        height=int,
        fps=int,
        frames=int,
    )
    output = Path(str(payload["output_path"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _backend.generate_segment(
        output,
        prompt=str(payload["prompt"]),
        seed=int(payload["seed"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        fps=int(payload["fps"]),
        frames=int(payload["frames"]),
    )
    return {
        "blocks_generated": 1,
        "artifacts": [str(output)],
        "video": result,
    }


def handle_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    return {"checkpoint_id": f"video-{payload.get('segment_id', 'none')}"}


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    return {"resumed": True, "checkpoint_id": payload.get("checkpoint_id")}


def main() -> None:
    serve(
        {
            "init": handle_health,
            "health": handle_health,
            "generate_blocks": handle_generate_blocks,
            "checkpoint": handle_checkpoint,
            "resume": handle_resume,
            "evict_gpu": lambda _payload: {"evicted": True},
            "rebuild": lambda _payload: {"rebuilt": True},
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()
