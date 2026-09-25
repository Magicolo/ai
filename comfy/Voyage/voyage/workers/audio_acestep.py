"""ACE-Step audio worker: `python -m voyage.workers.audio_acestep`.

Same `generate_audio` contract as the fake audio worker, but renders real
music takes with the resident ACE-Step stack (DESIGN §37). ACE renders
FLAC at its native 48kHz; the worker converts to the requested WAV shape
so downstream validation and assembly never branch on backend.

GPU ban (§12): the only top-level ACE import is `voyage.audio.acestep`,
which itself keeps `torch`/`acestep` behind function-local imports — this
module never imports `torch` at top level. `torch` appears only inside
`handle_benchmark` for peak-memory accounting, behind a `find_spec` guard.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from voyage.audio.acestep import (
    AceStepStack,
    evict,
    initialize,
    render_take,
    validate_bpm,
    validate_duration_seconds,
    validate_reference_audio,
    validate_task_type,
)
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts

_stack: AceStepStack | None = None
_models_dir = "/models"
_device = "cuda:0"

BYTES_PER_GIB = 1024**3
"""Byte-to-GiB divisor for VRAM peak reporting (benchmark only)."""


def _require_torch() -> None:
    """Fail fast with ImportError when `torch` is absent (slim image).

    The benchmark's peak-memory accounting needs `torch.cuda`; without the
    guard the bare import raises ImportError anyway, but naming the needing
    op keeps the failure attributable. Same class as the historical
    failure, so slim-image behavior is unchanged.
    """
    if importlib.util.find_spec("torch") is None:
        raise ImportError(
            "audio_acestep benchmark needs optional dependency 'torch' "
            "(slim image carries the fake audio worker only)"
        )


def validate_sample_rate(sample_rate: int) -> None:
    """Reject non-positive output sample rates (issue 063)."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive (got {sample_rate})")


def validate_channels(channels: int) -> None:
    """Reject non-mono/stereo channel counts (issue 063)."""
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2 (got {channels})")


def _require_stack() -> AceStepStack:
    global _stack
    if _stack is None:
        # Lazy load: init only records where/how, so the process can start
        # while video still owns the GPU; the stack loads on first render.
        _stack = initialize(_models_dir, _device)
    return _stack


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    """Record where/how the ACE stack will load (no GPU touched here).

    Init only records: the stack loads lazily on first render so the
    process can start while video still owns the GPU (sequential residency,
    DESIGN §40). Optional fields are type-checked when present so a
    mistyped `init` fails as INVALID_PAYLOAD instead of misdirecting the
    later load.
    """
    global _models_dir, _device
    if "models_dir" in payload and not isinstance(payload["models_dir"], str):
        raise TypeError(
            f"init field 'models_dir' must be str, got {type(payload['models_dir']).__name__}"
        )
    if "device" in payload and not isinstance(payload["device"], str):
        raise TypeError(f"init field 'device' must be str, got {type(payload['device']).__name__}")
    _models_dir = str(payload.get("models_dir", "/models"))
    _device = str(payload.get("device", "cuda:0"))
    return {"status": "READY", "backend": "acestep", "device": _device, "loaded": False}


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    del payload
    return {"status": "READY", "backend": "acestep", "loaded": _stack is not None}


def _convert(rendered_flac: Path, output: Path, sample_rate: int, channels: int) -> None:
    """Convert the native ACE FLAC take to the requested WAV shape (ffmpeg arg-list).

    No shell: the arg-list form keeps spaces in paths safe. A non-zero
    ffmpeg exit raises RuntimeError (not a VoyageError), so the worker loop
    maps it to retryable WORKER_ERROR — a convert failure after a good
    render smells transient (disk/memory/ffmpeg), worth one supervisor
    restart rather than an instant Fatal.
    """
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
    # Issue 063: validate every numeric/enum field before any filesystem
    # or GPU side effect — invalid takes fail here, not deep in ACE/ffmpeg.
    bpm_raw = payload.get("bpm")
    parsed_bpm = int(bpm_raw) if bpm_raw is not None else None
    validate_bpm(parsed_bpm)
    duration = float(payload["duration_seconds"])
    validate_duration_seconds(duration)
    sample_rate = int(payload["sample_rate"])
    validate_sample_rate(sample_rate)
    channels = int(payload["channels"])
    validate_channels(channels)
    raw_task_type = payload.get("task_type", "text2music")
    if not isinstance(raw_task_type, str):
        raise ValueError(f"task_type must be a string (got {raw_task_type!r})")
    validate_task_type(raw_task_type)
    reference_audio = payload.get("reference_audio")
    validate_reference_audio(reference_audio)
    output_raw = str(payload["output_path"])
    if not output_raw.strip():
        raise ValueError("output_path must be non-empty")
    output = Path(output_raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voyage-take-") as staging:
        rendered = render_take(
            _require_stack(),
            caption=str(payload["style"]),
            duration_seconds=duration,
            seed=int(payload["seed"]),
            save_path=Path(staging) / "take.flac",
            energy=float(payload["energy"]),
            task_type=raw_task_type,
            src_audio=reference_audio,
            repaint_start=float(payload.get("repaint_start", 0.0)),
            repaint_end=float(payload.get("repaint_end", -1.0)),
            bpm=parsed_bpm,
        )
        _convert(
            rendered,
            output,
            sample_rate=sample_rate,
            channels=channels,
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
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    validate_benchmark_counts(warmup, measured)
    _require_torch()
    import torch

    _require_stack()
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
    with tempfile.TemporaryDirectory(prefix="voyage-bench-") as staging_directory:
        for take_index in range(warmup + measured):
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            handle_generate_audio(
                {**probe, "output_path": str(Path(staging_directory) / f"t{take_index}.wav")}
            )
            elapsed = time.monotonic() - started
            peak_gib = torch.cuda.max_memory_allocated() / BYTES_PER_GIB
            if take_index >= warmup:
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
    """Release the ACE stack and stop the worker loop."""
    del payload
    global _stack
    if _stack is not None:
        evict(_stack)
        _stack = None
    return {"stopped": True}


def handle_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Derive a checkpoint id (the stack itself is not serializable)."""
    return {"checkpoint_id": f"audio-{payload.get('segment_id', 'none')}"}


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    """Acknowledge a checkpoint id; the stack rebuilds lazily on next render."""
    return {"resumed": True, "checkpoint_id": payload.get("checkpoint_id")}


def main() -> None:
    """Serve the ACE-Step audio op map over the shared JSONL loop."""
    serve(
        {
            "init": handle_init,
            "health": handle_health,
            "generate_audio": handle_generate_audio,
            "benchmark": handle_benchmark,
            "evict_gpu": handle_evict_gpu,
            "checkpoint": handle_checkpoint,
            "resume": handle_resume,
            "shutdown": handle_shutdown,
        }
    )


if __name__ == "__main__":
    main()
