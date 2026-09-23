"""Video worker: `python -m voyage.workers.video`."""

from __future__ import annotations

import tempfile
import time
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


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured testsrc renders (startup excluded, §104)."""
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    width = int(payload.get("width", 320))
    height = int(payload.get("height", 180))
    fps = int(payload.get("fps", 24))
    frames = int(payload.get("frames", 24))
    walls: list[float] = []
    with tempfile.TemporaryDirectory(prefix="voyage-bench-") as tmp:
        for index in range(warmup + measured):
            started = time.monotonic()
            _backend.generate_segment(
                Path(tmp) / f"bench_{index}.mp4",
                prompt="benchmark",
                seed=index,
                width=width,
                height=height,
                fps=fps,
                frames=frames,
            )
            elapsed = time.monotonic() - started
            if index >= warmup:
                walls.append(elapsed)
    mean = sum(walls) / len(walls)
    return {
        "backend": _backend.name,
        "warmup_blocks": warmup,
        "measured_blocks": measured,
        "frames_per_block": frames,
        "resolution": f"{width}x{height}",
        "block_wall_seconds": [round(wall, 3) for wall in walls],
        "blocks_per_second": round(1.0 / mean, 3),
        "fps_equivalent": round(frames / mean, 3),
        "vram_peak_gib": "unknown",
        "vram_avg_gib": "unknown",
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
            "benchmark": handle_benchmark,
            "checkpoint": handle_checkpoint,
            "resume": handle_resume,
            "evict_gpu": lambda _payload: {"evicted": True},
            "rebuild": lambda _payload: {"rebuilt": True},
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()
