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

import os
import subprocess as subprocess
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
from voyage.workers._resident import BYTES_PER_GIB, require_torch
from voyage.workers._validators import (
    validate_channels,
    validate_sample_rate,
)
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts

# Compat re-exports (issue 084): `validate_sample_rate` / `validate_channels`
# are the shared `_validators` implementations (single home); the names stay
# bound here so existing `audio_acestep.validate_*` imports keep working.

__all__ = ["BYTES_PER_GIB", "validate_sample_rate", "validate_channels"]

_stack: AceStepStack | None = None
_models_dir = "/models"
_device = "cuda:0"
_upstream_cache_dir: Path | None = None
"""Dedicated CWD for upstream ACE-Step relative writes (DESIGN §37, below)."""

_INIT_STR_KEYS = ("models_dir", "device")
"""`init` fields this worker records (issue 127).

Anything else is a caller typo — reject it before any side effect
(including the CWD redirect below) instead of booting defaults.
"""


def _redirect_upstream_writes() -> Path:
    """Point CWD at a dedicated tmp dir so upstream relative writes miss the run dir.

    Upstream ACE-Step writes `.cache/acestep/progress_estimates.json`
    relative to the worker process CWD, and workers spawn with CWD=run_dir —
    every music render littered the run directory. Redirecting CWD to a fresh
    tmp dir (created once per process, reused across re-inits) moves those
    writes out of the run. All voyage paths are absolute (payload
    `output_path` invariant, `TemporaryDirectory` staging, absolute
    `models_dir`), so the chdir is side-effect free — the `video_causvid`
    `_enter_causvid_tree` precedent. Runs before any ACE-Step
    library call; idempotent.
    """
    global _upstream_cache_dir
    if _upstream_cache_dir is None:
        _upstream_cache_dir = Path(tempfile.mkdtemp(prefix="voyage-acestep-cwd-"))
    os.chdir(_upstream_cache_dir)
    return _upstream_cache_dir


def _require_torch() -> None:
    """Fail fast with ImportError when `torch` is absent (slim image).

    Shared guard mechanics live in `voyage.workers._resident.require_torch`;
    this thin wrapper names the needing stack so the failure stays
    attributable (same class as the historical failure, so slim-image
    behavior is unchanged).
    """
    require_torch("audio_acestep benchmark")


def _session_device_index() -> int:
    """CUDA index of the session `_device` ("cuda:N" → N, else 0, never raises)."""
    try:
        prefix, _, index = _device.partition(":")
        if prefix == "cuda" and index.strip().isdigit():
            return int(index)
    except (AttributeError, ValueError):
        pass
    return 0


def _cuda_available() -> bool:
    """Whether a CUDA context exists (guard for every peak-memory call, 156)."""
    import torch

    return bool(torch.cuda.is_available())


def _reset_peak_stats() -> None:
    """Reset the session device's peak counter, or no-op off-GPU (156)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(_session_device_index())


def _peak_gib() -> float | None:
    """Session-device peak GiB, or None off-GPU (156, honest unknowns)."""
    import torch

    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(_session_device_index())) / BYTES_PER_GIB


def _benchmark_report(
    *,
    warmup: int,
    measured: int,
    duration: float,
    walls: list[float],
    peaks: list[float],
    cuda_available: bool,
    device: str,
) -> dict[str, Any]:
    """Shape the benchmark report (156, pure): device + availability always,
    VRAM peaks null off-GPU (extends `_benchmark_env`'s honest-unknown doctrine)."""
    mean = sum(walls) / len(walls)
    return {
        "backend": "acestep",
        "device": device,
        "cuda_available": cuda_available,
        "warmup_takes": warmup,
        "measured_takes": measured,
        "take_wall_seconds": [round(wall, 3) for wall in walls],
        "takes_per_second": round(1.0 / mean, 3),
        "audio_seconds_per_wall_second": round(duration / mean, 3),
        "vram_peak_gib": round(max(peaks), 2) if peaks else None,
        "vram_avg_gib": round(sum(peaks) / len(peaks), 2) if peaks else None,
    }


def _require_stack() -> AceStepStack:
    global _stack
    if _stack is None:
        # Lazy load: init only records where/how, so the process can start
        # while video still owns the GPU; the stack loads on first render.
        _redirect_upstream_writes()
        _stack = initialize(_models_dir, _device)
    return _stack


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    """Record where/how the ACE stack will load (no GPU touched here).

    Init only records: the stack loads lazily on first render so the
    process can start while video still owns the GPU (sequential residency,
    DESIGN §40). Optional fields are type-checked when present so a
    mistyped `init` fails as INVALID_PAYLOAD instead of misdirecting the
    later load. Redirects CWD out of the run dir first (before any ACE-Step
    library call) so upstream relative writes land in a tmp dir, never in
    the run (see `_redirect_upstream_writes`).
    """
    unknown = sorted(set(payload) - set(_INIT_STR_KEYS))
    if unknown:
        raise TypeError(f"init got unknown field(s) {unknown} (known: {sorted(_INIT_STR_KEYS)})")
    _redirect_upstream_writes()
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


def _cuda_mem_info_gib() -> tuple[float, float] | None:
    """Device memory in GiB, or None without CUDA (Stage C gauge half).

    Never raises: health is best-effort, and the slim image has no torch
    at all — a missing stack or a CPU box simply yields no vram fields
    instead of a failed probe. Units mirror the video workers' health.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return round(free_bytes / 1024**3, 1), round(total_bytes / 1024**3, 1)


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    del payload
    info: dict[str, Any] = {
        "status": "READY",
        "backend": "acestep",
        "loaded": _stack is not None,
    }
    memory = _cuda_mem_info_gib()
    if memory is not None:
        info["vram_free_gib"], info["vram_total_gib"] = memory
    return info


def _convert(rendered_flac: Path, output: Path, sample_rate: int, channels: int) -> None:
    """Convert the native ACE FLAC take to the requested WAV shape (ffmpeg arg-list).

    No shell: the arg-list form keeps spaces in paths safe. A non-zero
    ffmpeg exit raises RuntimeError (not a VoyageError), so the worker loop
    maps it to retryable WORKER_ERROR — a convert failure after a good
    render smells transient (disk/memory/ffmpeg), worth one supervisor
    restart rather than an instant Fatal. Carries the tree-standard
    `-hide_banner -nostdin` daemon hygiene (issues 053): without `-nostdin`
    an ffmpeg reading stdin in a pipelined supervisor can steal RPC bytes
    or block, and the full banner pollutes worker logs.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
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
    cuda_available = _cuda_available()
    with tempfile.TemporaryDirectory(prefix="voyage-bench-") as staging_directory:
        for take_index in range(warmup + measured):
            _reset_peak_stats()
            started = time.monotonic()
            handle_generate_audio(
                {**probe, "output_path": str(Path(staging_directory) / f"t{take_index}.wav")}
            )
            elapsed = time.monotonic() - started
            peak_gib = _peak_gib()
            if take_index >= warmup:
                walls.append(elapsed)
                if peak_gib is not None:
                    peaks.append(peak_gib)
    duration = probe["duration_seconds"]
    return _benchmark_report(
        warmup=warmup,
        measured=measured,
        duration=duration,
        walls=walls,
        peaks=peaks,
        cuda_available=cuda_available,
        device=_device,
    )


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
