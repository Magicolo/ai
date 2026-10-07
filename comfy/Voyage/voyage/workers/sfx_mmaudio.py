"""MMAudio SFX worker: `python -m voyage.workers.sfx_mmaudio`.

Renders video-synced effects windows behind the `generate_sfx` op for
the finalize-time SFX pass (three-caption doctrine: the caption is the
director's SFX family). Same resident-stack discipline as the ACE-Step
worker — lazy init (process starts while video owns the GPU), `evict_gpu`
before the music stack loads, benchmark with VRAM peaks for the 2060
ladder (DESIGN §104).

GPU ban (DESIGN §12): `torch`/`mmaudio` only load inside functions via
`voyage.audio.mmaudio_sfx`, never at module scope.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.audio.mmaudio_sfx import (
    CLIP_FPS,
    CLIP_SIZE,
    SYNC_FPS,
    SYNC_SIZE,
    SfxStack,
    check_stacked_bytes,
    evict,
    initialize,
    render_window,
    validate_duration_seconds,
    validate_model_size,
)
from voyage.workers._resident import BYTES_PER_GIB, require_torch
from voyage.workers._validators import (
    validate_channels,
    validate_sample_rate,
)
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts

# Compat re-exports (issue 084): `validate_sample_rate` / `validate_channels`
# are the shared `_validators` implementations (single home); the names stay
# bound here so existing imports keep working.

__all__ = ["BYTES_PER_GIB", "validate_sample_rate", "validate_channels"]

_stack: SfxStack | None = None
_models_dir = "/models"
_device = "cuda:0"
_model_size = "large_44k_v2"
_scratch_dir: str | None = None
"""Run scratch root for window/bench staging (`init` payload, boba /tmp-quota incident)."""

_INIT_STR_KEYS = ("models_dir", "device", "model_size", "scratch_dir")
"""`init` fields this worker records (issue 127).

Anything else is a caller typo (`model_is`, `backemd`, `model_dir`) —
reject it at startup instead of booting defaults with the pinned
weights silently unused.
"""


def _require_torch() -> None:
    """Fail fast with ImportError when `torch` is absent (slim image).

    Shared guard mechanics live in `voyage.workers._resident.require_torch`;
    this thin wrapper names the needing stack so the failure stays
    attributable (same class as the historical failure, so slim-image
    behavior is unchanged).
    """
    require_torch("sfx_mmaudio benchmark")


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
        "backend": "mmaudio",
        "model_size": _model_size,
        "device": device,
        "cuda_available": cuda_available,
        "warmup_windows": warmup,
        "measured_windows": measured,
        "window_wall_seconds": [round(wall, 3) for wall in walls],
        "windows_per_second": round(1.0 / mean, 3),
        "audio_seconds_per_wall_second": round(duration / mean, 3),
        "vram_peak_gib": round(max(peaks), 2) if peaks else None,
        "vram_avg_gib": round(sum(peaks) / len(peaks), 2) if peaks else None,
    }


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
    global _models_dir, _device, _model_size, _scratch_dir
    unknown = sorted(set(payload) - set(_INIT_STR_KEYS))
    if unknown:
        raise TypeError(f"init got unknown field(s) {unknown} (known: {sorted(_INIT_STR_KEYS)})")
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
    if "scratch_dir" in payload and not isinstance(payload["scratch_dir"], str):
        raise TypeError(
            f"init field 'scratch_dir' must be str, got {type(payload['scratch_dir']).__name__}"
        )
    _models_dir = str(payload.get("models_dir", "/models"))
    _device = str(payload.get("device", "cuda:0"))
    _model_size = str(payload.get("model_size", "large_44k_v2"))
    if "scratch_dir" in payload:
        _scratch_dir = str(payload["scratch_dir"])
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


#: Single-pass conditioning geometry (issue 045): decode once at the top
#: of each branch (sync rate 25 fps, clip size 384 px), then derive both
#: branches in-process — temporal subsample for CLIP, CPU downscale for
#: sync. One ffmpeg spawn per window instead of two.
SINGLE_PASS_FPS = SYNC_FPS
SINGLE_PASS_SIZE = CLIP_SIZE


def sfx_single_pass_argv(
    video_path: str, start_seconds: float, duration_seconds: float, frame_count: int
) -> list[str]:
    """ffmpeg argv for the single conditioning decode (issue 045, pure).

    Arg-lists only (never shell). Emits `frame_count` rawvideo rgb24
    frames at the single-pass geometry; the caller derives the CLIP
    (temporal subsample) and sync (CPU downscale) branches from them.
    """
    if frame_count < 1:
        raise ValueError(f"frame_count must be >= 1 (got {frame_count})")
    return [
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
        f"fps={SINGLE_PASS_FPS:g},scale={SINGLE_PASS_SIZE}:{SINGLE_PASS_SIZE}",
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def derive_clip_indices(frame_total: int, wanted_clip: int) -> list[int]:
    """Evenly spaced single-pass indices for the CLIP branch (issue 045).

    The single pass runs at the sync rate, so CLIP keeps its native 8 fps
    cadence by subsampling (`wanted == 1` takes the middle frame — the
    same choice the vision sampler documents).
    """
    if frame_total < 1:
        raise ValueError(f"frame_total must be >= 1 (got {frame_total})")
    if wanted_clip < 1:
        raise ValueError(f"wanted_clip must be >= 1 (got {wanted_clip})")
    if wanted_clip == 1:
        return [frame_total // 2]
    return [round(index * (frame_total - 1) / (wanted_clip - 1)) for index in range(wanted_clip)]


def _drain_in_background(stream: Any, sink: list[bytes]) -> Any:
    """Read `stream` to EOF on a daemon thread, appending one chunk (issue 286).

    The single-pass extract streams ffmpeg stdout frame-by-frame while
    stderr would otherwise sit undrained — past the ~64 KiB OS pipe
    buffer the child blocks on stderr while the parent blocks on stdout
    (textbook pipe deadlock, biting hardest on the failure path that most
    needs a loud error). A daemon drain keeps the pipe empty; the
    trailing `communicate()` still reaps the process and collects any
    tail the drainer did not see. The thread never raises (a dead stream
    reads as EOF) and is daemon, so a wedged child cannot pin worker
    teardown.
    """
    import threading

    def _drain() -> None:
        try:
            data = stream.read()
        except Exception:  # noqa: BLE001 — drain is best-effort; the returncode check reports failures
            return
        if data:
            sink.append(data)

    drainer = threading.Thread(target=_drain, name="sfx-ffmpeg-stderr-drain", daemon=True)
    drainer.start()
    return drainer


_SFX_EXTRACT_WAIT_SECONDS = 30.0
"""Bound for the trailing extract reap (issue 286).

The stdout read loop ends at EOF/truncation; `communicate()` after that
only reaps an already-dead ffmpeg, so 30 s is generous — a wedged child
is killed and reaped instead of hanging the worker until the 600 s
supervisor RPC deadline.
"""


def _read_frame_bytes(stdout: Any, stride: int) -> bytes | None:
    """Read exactly one frame; None on clean EOF, loud on a short tail."""
    chunks: list[bytes] = []
    remaining = stride
    while remaining > 0:
        piece: bytes = stdout.read(remaining)
        if not piece:
            if chunks:
                raise RuntimeError(
                    f"sfx frame extract hit a truncated stream "
                    f"({stride - remaining} of {stride} bytes)"
                )
            return None
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def _extract_frames(
    video_path: str, start_seconds: float, duration_seconds: float
) -> tuple[Any, Any, float]:
    """ffmpeg single-pass conditioning extract (issue 045).

    One rawvideo decode at 25 fps @ 384 px, streamed frame-by-frame
    into a single `torch.stack` — no `capture_output` byte hold, no
    per-branch spawns, no numpy-copy fan-out. The CLIP branch
    subsamples the stream temporally (already 384 px, no resize); the
    sync branch downscales 384 → 224 CPU-side and normalizes to
    [-1, 1]. Counts are exact (int(rate × resolved)); when the source
    yields fewer frames than requested (short tail), the duration
    truncates to reality like the canonical loader and the caller
    fails loud past 0.05 s of drift (padded to the sync floor inside
    `render_window` instead — see `pad_to_sync_floor`).
    Arg-lists only (never shell); a non-zero exit raises RuntimeError →
    retryable WORKER_ERROR (transient disk/ffmpeg smell, worth one
    supervisor restart).
    """
    import numpy as np
    import torch

    wanted_clip = int(CLIP_FPS * duration_seconds)
    wanted_sync = int(SYNC_FPS * duration_seconds)
    argv = sfx_single_pass_argv(video_path, start_seconds, duration_seconds, wanted_sync)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None or proc.stderr is None:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"sfx frame extract could not capture ffmpeg pipes ({video_path})")
    stride = SINGLE_PASS_SIZE * SINGLE_PASS_SIZE * 3
    raw_frames: list[Any] = []
    stderr_chunks: list[bytes] = []
    drainer = _drain_in_background(proc.stderr, stderr_chunks)
    try:
        while len(raw_frames) < wanted_sync:
            chunk = _read_frame_bytes(proc.stdout, stride)
            if chunk is None:
                break
            raw_frames.append(
                torch.from_numpy(np.frombuffer(chunk, dtype=np.uint8)).reshape(
                    SINGLE_PASS_SIZE, SINGLE_PASS_SIZE, 3
                )
            )
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            _, tail = proc.communicate(timeout=_SFX_EXTRACT_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, tail = proc.communicate()
        drainer.join(timeout=_SFX_EXTRACT_WAIT_SECONDS)
        stderr = b"".join(stderr_chunks) + (tail or b"")
    if not raw_frames:
        raise RuntimeError(f"sfx frame extract yielded no frames ({video_path})")
    if proc.returncode != 0 and len(raw_frames) < wanted_sync:
        raise RuntimeError(f"sfx frame extract failed: {stderr.decode().strip()}")
    stream = torch.stack(raw_frames).permute(0, 3, 1, 2).float().div_(255.0)
    indices = derive_clip_indices(len(raw_frames), min(wanted_clip, len(raw_frames)))
    clip = stream[indices]
    sync = torch.nn.functional.interpolate(
        stream, size=(SYNC_SIZE, SYNC_SIZE), mode="bilinear", align_corners=False
    ).mul_(2.0)
    sync = sync.sub_(1.0)
    resolved = min(len(clip) / CLIP_FPS, len(sync) / SYNC_FPS)
    return clip, sync, resolved


def _check_convert_result(completed: Any) -> None:
    """Fail loud on a non-zero ffmpeg SFX convert (TRY301 inner home)."""
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg sfx convert failed: {completed.stderr.strip()}")


def _convert(rendered_flac: Path, output: Path, sample_rate: int, channels: int) -> None:
    """Convert the native 44.1 kHz FLAC window to the requested WAV shape.

    H1 atomic twin of the ACE worker: converts to a sibling
    `*.partial.wav` then `os.replace` publishes, so a killed convert
    never leaves a half-written stem the ledger could adopt.
    """
    staged = output.parent / f"{output.stem}.partial.wav"
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
        str(staged),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        _check_convert_result(completed)
        os.replace(staged, output)
    except BaseException:
        with contextlib.suppress(OSError):
            staged.unlink()
        raise


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
    check_stacked_bytes(duration)
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
    with tempfile.TemporaryDirectory(
        prefix="voyage-sfx-window-", dir=paths.staging_parent(_scratch_dir)
    ) as staging:
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

    _require_stack()
    duration = float(payload.get("duration_seconds", 8.0))
    validate_duration_seconds(duration)
    walls: list[float] = []
    peaks: list[float] = []
    cuda_available = _cuda_available()
    with tempfile.TemporaryDirectory(
        prefix="voyage-sfx-bench-", dir=paths.staging_parent(_scratch_dir)
    ) as staging_directory:
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
            _reset_peak_stats()
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
            peak_gib = _peak_gib()
            if window_index >= warmup:
                walls.append(elapsed)
                if peak_gib is not None:
                    peaks.append(peak_gib)
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
