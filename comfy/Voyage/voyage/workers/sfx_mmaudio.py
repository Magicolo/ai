"""MMAudio SFX worker: `python -m voyage.workers.sfx_mmaudio`.

Renders video-synced effects windows behind the `generate_sfx` op for
the finalize-time SFX pass (three-caption doctrine: the caption is the
director's SFX family). Same resident-stack discipline as the ACE-Step
worker — lazy init (process starts while video owns the GPU), `evict_gpu`
before the music stack loads, benchmark with VRAM peaks for the 2060
ladder (§104).

GPU ban (§12): `torch`/`mmaudio` only load inside functions via
`voyage.audio.mmaudio_sfx`, never at module scope.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from voyage.audio.mmaudio_sfx import (
    CLIP_FPS,
    CLIP_SIZE,
    SYNC_FPS,
    SYNC_SIZE,
    SfxStack,
    evict,
    initialize,
    render_window,
    validate_duration_seconds,
    validate_model_size,
)
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts

_stack: SfxStack | None = None
_models_dir = "/models"
_device = "cuda:0"
_model_size = "large_44k_v2"

BYTES_PER_GIB = 1024**3
"""Byte-to-GiB divisor for VRAM peak reporting (benchmark only)."""


def _require_torch() -> None:
    """Fail fast with ImportError when `torch` is absent (slim image)."""
    if importlib.util.find_spec("torch") is None:
        raise ImportError(
            "sfx_mmaudio benchmark needs optional dependency 'torch' "
            "(slim image carries the fake SFX worker only)"
        )


def validate_sample_rate(sample_rate: int) -> None:
    """Reject non-positive output sample rates (issue 063 class)."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive (got {sample_rate})")


def validate_channels(channels: int) -> None:
    """Reject non-mono/stereo channel counts."""
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2 (got {channels})")


def _require_stack() -> SfxStack:
    global _stack
    if _stack is None:
        # Lazy load: init only records where/how, so the process can
        # start while video still owns the GPU; the stack loads on the
        # first window (sequential residency, DESIGN §40).
        _stack = initialize(_models_dir, _device, _model_size)
    return _stack


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    """Record where/how the MMAudio stack will load (no GPU touched here)."""
    global _models_dir, _device, _model_size
    if "models_dir" in payload and not isinstance(payload["models_dir"], str):
        raise TypeError(
            f"init field 'models_dir' must be str, got {type(payload['models_dir']).__name__}"
        )
    if "device" in payload and not isinstance(payload["device"], str):
        raise TypeError(f"init field 'device' must be str, got {type(payload['device']).__name__}")
    if "model_size" in payload:
        if not isinstance(payload["model_size"], str):
            raise TypeError(
                f"init field 'model_size' must be str, got {type(payload['model_size']).__name__}"
            )
        validate_model_size(payload["model_size"])
    _models_dir = str(payload.get("models_dir", "/models"))
    _device = str(payload.get("device", "cuda:0"))
    _model_size = str(payload.get("model_size", "large_44k_v2"))
    return {
        "status": "READY",
        "backend": "mmaudio",
        "device": _device,
        "model_size": _model_size,
        "loaded": False,
    }


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    """Liveness probe reporting whether the stack is resident."""
    del payload
    return {"status": "READY", "backend": "mmaudio", "loaded": _stack is not None}


def _extract_frames(
    video_path: str, start_seconds: float, duration_seconds: float
) -> tuple[Any, Any, float]:
    """ffmpeg two-pass conditioning extract: 8 fps @ 384 px (CLIP) + 25 fps @ 224 px (sync).

    Returns `(clip_batch, sync_batch, resolved_seconds)` — CPU float32
    tensors (T,3,H,W), sync normalized to [-1, 1]. Counts are exact
    (int(rate × resolved)); when the source yields fewer frames than
    requested (short tail), the duration truncates to reality like the
    canonical loader and the caller fails loud past 0.05 s of drift.
    Arg-lists only (never shell); a non-zero exit raises RuntimeError →
    retryable WORKER_ERROR (transient disk/ffmpeg smell, worth one
    supervisor restart).
    """
    import numpy as np
    import torch

    def _grab(rate: float, size: int, count: int) -> Any:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            video_path,
            "-t",
            f"{duration_seconds:.6f}",
            "-vf",
            f"fps={rate},scale={size}:{size}",
            "-frames:v",
            str(count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"sfx frame extract failed: {completed.stderr.decode().strip()}")
        pixels = len(completed.stdout) // (size * size * 3)
        if pixels == 0:
            raise RuntimeError(f"sfx frame extract yielded no frames ({video_path})")
        array = np.frombuffer(completed.stdout, dtype=np.uint8)
        frames = array[: pixels * size * size * 3].reshape(pixels, size, size, 3)
        tensor = torch.from_numpy(frames.copy()).float().div_(255.0).permute(0, 3, 1, 2)
        return tensor

    wanted_clip = int(CLIP_FPS * duration_seconds)
    wanted_sync = int(SYNC_FPS * duration_seconds)
    clip = _grab(CLIP_FPS, CLIP_SIZE, wanted_clip)
    sync = _grab(SYNC_FPS, SYNC_SIZE, wanted_sync)
    resolved = min(len(clip) / CLIP_FPS, len(sync) / SYNC_FPS)
    sync = sync.mul_(2.0).sub_(1.0)
    return clip, sync, resolved


def _convert(rendered_flac: Path, output: Path, sample_rate: int, channels: int) -> None:
    """Convert the native 44.1 kHz FLAC window to the requested WAV shape."""
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
        raise RuntimeError(f"ffmpeg sfx convert failed: {completed.stderr.strip()}")


def handle_generate_sfx(payload: dict[str, Any]) -> dict[str, Any]:
    """Render one caption-conditioned effects window for [start, start+duration)."""
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
    duration = float(payload["duration_seconds"])
    validate_duration_seconds(duration)
    start = float(payload["start_seconds"])
    if start < 0.0:
        raise ValueError(f"start_seconds must be >= 0 (got {start})")
    sample_rate = int(payload["sample_rate"])
    validate_sample_rate(sample_rate)
    channels = int(payload["channels"])
    validate_channels(channels)
    video_raw = str(payload["video_path"])
    if not video_raw.strip():
        raise ValueError("video_path must be non-empty")
    output_raw = str(payload["output_path"])
    if not output_raw.strip():
        raise ValueError("output_path must be non-empty")
    output = Path(output_raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    clip_frames, sync_frames, resolved = _extract_frames(video_raw, start, duration)
    if resolved <= 0.0:
        raise RuntimeError(f"sfx window {payload['window_id']}: source yielded no frames")
    if duration - resolved > 0.6:
        raise RuntimeError(
            f"sfx window {payload['window_id']}: source yielded {resolved:.2f}s "
            f"for {duration:.2f}s requested — timeline/video mismatch, failing loud"
        )
    with tempfile.TemporaryDirectory(prefix="voyage-sfx-") as staging:
        rendered = render_window(
            _require_stack(),
            caption=str(payload["caption"]),
            negative_caption=str(payload.get("negative_caption", "")),
            clip_frames=clip_frames,
            sync_frames=sync_frames,
            duration_seconds=resolved,
            seed=int(payload["seed"]),
            save_path=Path(staging) / "window.flac",
        )
        _convert(rendered, output, sample_rate=sample_rate, channels=channels)
    return {
        "artifacts": [str(output)],
        "sfx": {
            "path": str(output),
            "backend": "mmaudio",
            "model_size": _model_size,
            "duration_seconds": resolved,
        },
    }


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured 8 s windows with VRAM peaks (§104, ladder input).

    Requires the resident stack (`init` first); without it this is an
    error, not a silent fake measurement. The probe video is synthetic
    (ffmpeg testsrc) — the ladder measures stack residency + forward
    cost, not content.
    """
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    validate_benchmark_counts(warmup, measured)
    _require_torch()
    import torch

    _require_stack()
    duration = float(payload.get("duration_seconds", 8.0))
    validate_duration_seconds(duration)
    walls: list[float] = []
    peaks: list[float] = []
    with tempfile.TemporaryDirectory(prefix="voyage-sfx-bench-") as staging_directory:
        staging = Path(staging_directory)
        probe_video = staging / "probe.mp4"
        probe = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size=768x512:rate=24:duration={duration}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(probe_video),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeError(f"sfx benchmark probe render failed: {probe.stderr.strip()}")
        for window_index in range(warmup + measured):
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            handle_generate_sfx(
                {
                    "window_id": f"bench_{window_index}",
                    "caption": "benchmark effects: wind, distant rumble",
                    "video_path": str(probe_video),
                    "start_seconds": 0.0,
                    "duration_seconds": duration,
                    "seed": window_index,
                    "output_path": str(staging / f"w{window_index}.wav"),
                    "sample_rate": 48000,
                    "channels": 2,
                }
            )
            elapsed = time.monotonic() - started
            peak_gib = torch.cuda.max_memory_allocated() / BYTES_PER_GIB
            if window_index >= warmup:
                walls.append(elapsed)
                peaks.append(peak_gib)
    mean = sum(walls) / len(walls)
    return {
        "backend": "mmaudio",
        "model_size": _model_size,
        "warmup_windows": warmup,
        "measured_windows": measured,
        "window_wall_seconds": [round(wall, 3) for wall in walls],
        "windows_per_second": round(1.0 / mean, 3),
        "audio_seconds_per_wall_second": round(duration / mean, 3),
        "vram_peak_gib": round(max(peaks), 2),
        "vram_avg_gib": round(sum(peaks) / len(peaks), 2),
    }


def handle_evict_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    """Unload the MMAudio stack so music/video can reclaim the GPU (§40)."""
    del payload
    global _stack
    if _stack is not None:
        evict(_stack)
        _stack = None
    return {"evicted": True}


def handle_shutdown(payload: dict[str, Any]) -> dict[str, Any]:
    """Release the MMAudio stack and stop the worker loop."""
    del payload
    global _stack
    if _stack is not None:
        evict(_stack)
        _stack = None
    return {"stopped": True}


def handle_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Derive a checkpoint id (the stack itself is not serializable)."""
    return {"checkpoint_id": f"sfx-{payload.get('window_id', 'none')}"}


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    """Acknowledge a checkpoint id; the stack rebuilds lazily on next render."""
    return {"resumed": True, "checkpoint_id": payload.get("checkpoint_id")}


def main() -> None:
    """Serve the MMAudio SFX op map over the shared JSONL loop."""
    serve(
        {
            "init": handle_init,
            "health": handle_health,
            "generate_sfx": handle_generate_sfx,
            "benchmark": handle_benchmark,
            "evict_gpu": handle_evict_gpu,
            "checkpoint": handle_checkpoint,
            "resume": handle_resume,
            "shutdown": handle_shutdown,
        }
    )


if __name__ == "__main__":
    main()
