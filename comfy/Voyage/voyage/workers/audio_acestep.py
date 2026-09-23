"""ACE-Step audio worker: `python -m voyage.workers.audio_acestep`.

Same `generate_audio` contract as the fake audio worker, but renders real
music takes with the resident ACE-Step stack (DESIGN §37). ACE renders
FLAC at its native 48kHz; the worker converts to the requested WAV shape
so downstream validation and assembly never branch on backend.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from voyage.audio.acestep import AceStepStack, evict, initialize, render_take
from voyage.workers.loop import checked_request, serve

_stack: AceStepStack | None = None
_models_dir = "/models"
_device = "cuda:0"


def _require_stack() -> AceStepStack:
    global _stack
    if _stack is None:
        # Lazy load: init only records where/how, so the process can start
        # while video still owns the GPU; the stack loads on first render.
        _stack = initialize(_models_dir, _device)
    return _stack


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    global _models_dir, _device
    _models_dir = str(payload.get("models_dir", "/models"))
    _device = str(payload.get("device", "cuda:0"))
    return {"status": "READY", "backend": "acestep", "device": _device, "loaded": False}


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    del payload
    return {"status": "READY", "backend": "acestep", "loaded": _stack is not None}


def _convert(rendered_flac: Path, output: Path, sample_rate: int, channels: int) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(rendered_flac),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg take convert failed: {completed.stderr.strip()}")


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
    with tempfile.TemporaryDirectory(prefix="voyage-take-") as staging:
        rendered = render_take(
            _require_stack(),
            caption=str(payload["style"]),
            duration_seconds=float(payload["duration_seconds"]),
            seed=int(payload["seed"]),
            save_path=Path(staging) / "take.flac",
            energy=float(payload["energy"]),
            task_type=str(payload.get("task_type", "text2music")),
            src_audio=payload.get("reference_audio"),
            repaint_start=float(payload.get("repaint_start", 0.0)),
            repaint_end=float(payload.get("repaint_end", -1.0)),
        )
        _convert(
            rendered,
            output,
            sample_rate=int(payload["sample_rate"]),
            channels=int(payload["channels"]),
        )
    return {
        "artifacts": [str(output)],
        "audio": {"path": str(output), "backend": "acestep"},
    }


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured take renders with VRAM peaks (§104).

    Requires the resident ACE stack (`init` first); without it this is an
    error, not a silent fake measurement.
    """
    import time

    import torch

    _require_stack()
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    probe: dict[str, Any] = {
        "segment_id": str(payload.get("segment_id", "benchmark")),
        "style": str(payload.get("style", "pastel neon line-art, peaceful")),
        "energy": float(payload.get("energy", 0.5)),
        "seed": int(payload.get("seed", 0)),
        "sample_rate": int(payload.get("sample_rate", 48000)),
        "channels": int(payload.get("channels", 2)),
        "duration_seconds": float(payload.get("duration_seconds", 15.0)),
    }
    walls: list[float] = []
    peaks: list[float] = []
    with tempfile.TemporaryDirectory(prefix="voyage-bench-") as tmp:
        for index in range(warmup + measured):
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            handle_generate_audio({**probe, "output_path": str(Path(tmp) / f"t{index}.wav")})
            elapsed = time.monotonic() - started
            peak_gib = torch.cuda.max_memory_allocated() / 1024**3
            if index >= warmup:
                walls.append(elapsed)
                peaks.append(peak_gib)
    mean = sum(walls) / len(walls)
    duration = probe["duration_seconds"]
    return {
        "backend": "acestep",
        "warmup_takes": warmup,
        "measured_takes": measured,
        "take_wall_seconds": [round(wall, 3) for wall in walls],
        "takes_per_second": round(1.0 / mean, 3),
        "audio_seconds_per_wall_second": round(duration / mean, 3),
        "vram_peak_gib": round(max(peaks), 2),
        "vram_avg_gib": round(sum(peaks) / len(peaks), 2),
    }


def handle_evict_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    """Unload the ACE stack so video can reclaim the GPU (§40)."""
    del payload
    global _stack
    if _stack is not None:
        evict(_stack)
        _stack = None
    return {"evicted": True}


def handle_shutdown(payload: dict[str, Any]) -> dict[str, Any]:
    del payload
    global _stack
    if _stack is not None:
        evict(_stack)
        _stack = None
    return {"stopped": True}


def main() -> None:
    serve(
        {
            "init": handle_init,
            "health": handle_health,
            "generate_audio": handle_generate_audio,
            "benchmark": handle_benchmark,
            "evict_gpu": handle_evict_gpu,
            "checkpoint": lambda payload: {
                "checkpoint_id": f"audio-{payload.get('segment_id', 'none')}"
            },
            "resume": lambda payload: {
                "resumed": True,
                "checkpoint_id": payload.get("checkpoint_id"),
            },
            "shutdown": handle_shutdown,
        }
    )


if __name__ == "__main__":
    main()
