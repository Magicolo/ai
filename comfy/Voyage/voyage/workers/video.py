"""Video worker: `python -m voyage.workers.video`.

Fake backend behind the single-prompt `generate_blocks` wire form: renders a
deterministic `testsrc` clip with ffmpeg so commit/validate/finalize run
genuinely with no GPU, no weights, no network (DESIGN §§45-46 transport;
the streaming multi-block form lives in the GPU workers owned by other
tracks and is mapped by `VideoBackendAdapter.build_payload`). `output_path`
arrives as an absolute wire path from the supervisor (codebase invariant:
absolute on the wire, run-relative refs in store); this worker only ensures
its parent directory exists. No `torch` import at module scope or lazily —
the hard GPU ban (§12) holds trivially here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voyage.fake_backends import FakeVideoBackend
from voyage.workers._validators import (
    validate_fps,
    validate_frame_count,
    validate_geometry,
    validate_output_path,
)
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts
from voyage.workers.video_common import run_benchmark_harness, standard_serve_map_with_cancel

__all__ = [
    "validate_geometry",
    "validate_fps",
    "validate_frame_count",
    "validate_output_path",
]

_backend = FakeVideoBackend()

BENCHMARK_WARMUP_BLOCKS = 1
BENCHMARK_MEASURED_BLOCKS = 3
BENCHMARK_WIDTH = 320
BENCHMARK_HEIGHT = 180
BENCHMARK_FPS = 24
BENCHMARK_FRAMES = 24
"""Probe shape for `handle_benchmark` (§104): startup excluded, testsrc renders only."""


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    """Liveness probe: no payload fields required, never touches the backend."""
    del payload
    return {"status": "READY", "backend": _backend.name}


def handle_generate_blocks(payload: dict[str, Any]) -> dict[str, Any]:
    """Render one testsrc block; geometry validated before any ffmpeg work."""
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
    width = int(payload["width"])
    height = int(payload["height"])
    validate_geometry(width, height)
    fps = int(payload["fps"])
    validate_fps(fps)
    frames = int(payload["frames"])
    validate_frame_count(frames)
    output_raw = str(payload["output_path"])
    validate_output_path(output_raw)
    output = Path(output_raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _backend.generate_segment(
        output,
        prompt=str(payload["prompt"]),
        seed=int(payload["seed"]),
        width=width,
        height=height,
        fps=fps,
        frames=frames,
    )
    return {
        "blocks_generated": 1,
        "artifacts": [str(output)],
        "video": result,
    }


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured testsrc renders (startup excluded, §104).

    Counts are validated before any render (issue 060): without the check
    `measured=0` divides by zero after doing all the work. The warmup /
    measured loop lives in `video_common.run_benchmark_harness` (single
    home, issue 084); this handler only validates the probe shape and
    formats the backend-specific report, so the response keys are unchanged.
    """
    warmup = int(payload.get("warmup", BENCHMARK_WARMUP_BLOCKS))
    measured = int(payload.get("measured", BENCHMARK_MEASURED_BLOCKS))
    validate_benchmark_counts(warmup, measured)
    width = int(payload.get("width", BENCHMARK_WIDTH))
    height = int(payload.get("height", BENCHMARK_HEIGHT))
    validate_geometry(width, height)
    fps = int(payload.get("fps", BENCHMARK_FPS))
    validate_fps(fps)
    frames = int(payload.get("frames", BENCHMARK_FRAMES))
    validate_frame_count(frames)

    counter = 0

    def _probe(path: Path, _measured: bool) -> None:
        # Varying seed per probe (the original loop used seed=block_index):
        # the hue angle differs per block while timing stays comparable.
        nonlocal counter
        _backend.generate_segment(
            path,
            prompt="benchmark",
            seed=counter,
            width=width,
            height=height,
            fps=fps,
            frames=frames,
        )
        counter += 1

    outcome = run_benchmark_harness(warmup, measured, "voyage-bench-", _probe)
    walls = outcome.wall_seconds
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


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    """Fake resume: echoes the checkpoint id without loading anything."""
    return {"resumed": True, "checkpoint_id": payload.get("checkpoint_id")}


def handle_evict_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    """No GPU state held; kept so the supervisor's evict path is uniform."""
    del payload
    return {"evicted": True}


def handle_rebuild(payload: dict[str, Any]) -> dict[str, Any]:
    """No resident session to rebuild; kept so the supervisor path is uniform."""
    del payload
    return {"rebuilt": True}


def main() -> None:
    """Serve the fake-video op map over the shared JSONL loop.

    The nine-op map lives in `video_common.standard_serve_map` (single
    home, issue 084); `backend_name="video"` preserves the historical
    `"video-<segment_id>"` checkpoint prefix (the backend itself is named
    `"fake"`, which would relabel checkpoints to `"fake-..."`). The Track D
    `cancel` wire name rides via `standard_serve_map_with_cancel` (idle
    handler — mid-render cancel stays client-side abandon-and-restart).
    """
    serve(
        standard_serve_map_with_cancel(
            "video",
            handle_init=handle_health,
            handle_health=handle_health,
            handle_generate_blocks=handle_generate_blocks,
            handle_benchmark=handle_benchmark,
            handle_evict_gpu=handle_evict_gpu,
            handle_rebuild=handle_rebuild,
            handle_resume=handle_resume,
        )
    )


if __name__ == "__main__":
    main()
