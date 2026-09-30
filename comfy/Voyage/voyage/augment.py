"""GPU augment runner orchestration: chunked Real-ESRGAN upscale + FILM interpolate (Track D spike).

Pure orchestration — stdlib only, never torch (supervisor section 12 GPU
ban): chunk math, ffmpeg chunk decode/encode (arg-lists, verified outputs),
device selection, and chunk-level parallelism.

Reuse check (2026-09-29): `voyage.media` has `plan_augmentation`, but it
computes presentation geometry/fps floors (Track B: out box, minterpolate
vs plain fps) — a different concern from chunk windows and `(n-1)*m+1`
frame counts, so there is nothing to import here. TODO (unify): if
media.py ever gains chunk/frame-count plan math, move
`interpolated_frame_count` there and import it instead of duplicating.

Parallelism contract (SFX pairing): augment chunks run on cuda:0 while the
MMAudio SFX stack (when present) renders on cuda:1 — the two stages share
nothing but chunk boundaries, so a 2-GPU box runs them side by side and a
1-GPU box runs chunks serially on cuda:0. `augment_plan` stamps each chunk
with its device round-robin so the pairing is visible in the plan;
`run_augment_chunks` owns the serial-vs-ThreadPoolExecutor(2) switch.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from voyage.errors import MediaError

DEFAULT_CHUNK_FRAMES = 32
"""Source frames per chunk: mirrors VHS_BatchManager `frames_per_batch` (video_export node 10)."""

DEFAULT_INTERP_MULTIPLIER = 4
"""FILM interpolation multiplier: matches the 4x video-export recipe."""

DEFAULT_UPSCALE_FACTOR = 2
"""Presentation upscale factor (4x model + Lanczos 0.5 downscale = the 2x recipe)."""

AUGMENT_DEVICE_PRIMARY = "cuda:0"
"""Video-augment device: always used, and the only device on a 1-GPU box."""

AUGMENT_DEVICE_SECONDARY = "cuda:1"
"""Second device for chunk parallelism; the SFX stack's device when paired."""

MAX_PARALLEL_DEVICES = 2
"""Chunk fan-out cap: one worker per device (the SFX pairing needs no more)."""

CRF_MINIMUM = 0
"""Best-quality h264 CRF bound for chunk encodes."""

CRF_MAXIMUM = 51
"""Worst-quality h264 CRF bound for chunk encodes."""

T = TypeVar("T")
"""Outcome type of the per-chunk worker passed to `run_augment_chunks`."""


def _require_count(name: str, value: int, minimum: int) -> int:
    """Validate an integer count: ints only (bools rejected), at least `minimum`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int (got {type(value).__name__})")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {value})")
    return value


def interpolated_frame_count(source_frames: int, multiplier: int) -> int:
    """Frames after FILM interpolation: `(n-1)*m+1` (a single frame passes through)."""
    sources = _require_count("source_frames", source_frames, 1)
    factor = _require_count("multiplier", multiplier, 1)
    return (sources - 1) * factor + 1


def chunk_windows(
    total_frames: int, chunk: int = DEFAULT_CHUNK_FRAMES
) -> Iterator[tuple[int, int]]:
    """Yield `(start, count)` source-frame windows tiling `[0, total_frames)` contiguously."""
    total = _require_count("total_frames", total_frames, 0)
    size = _require_count("chunk", chunk, 1)
    for start in range(0, total, size):
        yield (start, min(size, total - start))


@dataclass(frozen=True)
class AugmentChunk:
    """One augment unit: source window + expected output frames + assigned device."""

    index: int
    start_frame: int
    source_frames: int
    expected_frames: int
    device: str


def augment_plan(
    total_frames: int,
    *,
    chunk: int = DEFAULT_CHUNK_FRAMES,
    multiplier: int = DEFAULT_INTERP_MULTIPLIER,
    devices: tuple[str, ...] | None = None,
) -> list[AugmentChunk]:
    """Plan chunked augmentation: windows, per-chunk output counts, round-robin devices.

    Chunk outputs do NOT sum to the unchunked `(n-1)*m+1` total: each chunk
    interpolates independently, so one boundary pair per chunk joint is
    skipped (the same trade-off VHS_BatchManager documents). Callers that
    need exact end-to-end counts must use the unchunked formula.
    """
    _require_count("multiplier", multiplier, 1)
    resolved = augment_devices() if devices is None else devices
    if not resolved:
        raise ValueError("augment_plan needs at least one device (got none)")
    plan: list[AugmentChunk] = []
    for index, (start, count) in enumerate(chunk_windows(total_frames, chunk)):
        plan.append(
            AugmentChunk(
                index=index,
                start_frame=start,
                source_frames=count,
                expected_frames=interpolated_frame_count(count, multiplier),
                device=resolved[index % len(resolved)],
            )
        )
    return plan


def run_capture(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg-style argv (arg-list, never shell); mirrors `media.run_capture`."""
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def ffmpeg_decode_chunk(
    source_video: Path,
    dest_dir: Path,
    start_frame: int,
    frame_count: int,
) -> list[Path]:
    """Decode one source window to PNG frames via a frame-accurate select filter.

    `dest_dir` must be a fresh per-chunk directory (stale `frame_*.png`
    files would be picked up by the glob below). Seeking is exact but
    linear from the file start — a `-ss` fast-seek follow-up can skip the
    already-decoded prefix once chunk offsets grow large.
    """
    start = _require_count("start_frame", start_frame, 0)
    count = _require_count("frame_count", frame_count, 1)
    dest_dir.mkdir(parents=True, exist_ok=True)
    end = start + count - 1
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(source_video),
        "-vf",
        f"select='between(n\\,{start}\\,{end})',setpts=N/FRAME_RATE/TB",
        "-vsync",
        "0",
        str(dest_dir / "frame_%06d.png"),
    ]
    proc = run_capture(argv)
    if proc.returncode != 0:
        raise MediaError(f"chunk decode failed for frames {start}-{end}: {proc.stderr[-2000:]}")
    frames = sorted(dest_dir.glob("frame_*.png"))
    if not frames:
        raise MediaError(f"chunk decode produced no frames for {start}-{end}")
    for frame in frames:
        if frame.stat().st_size == 0:
            raise MediaError(f"chunk decode produced empty frame {frame}")
    return frames


def ffmpeg_encode_chunk(
    frames_pattern: Path,
    dest: Path,
    fps: int,
    *,
    crf: int = 15,
) -> Path:
    """Encode a chunk's PNG sequence (`frame_%06d.png` pattern) to h264."""
    rate = _require_count("fps", fps, 1)
    quality = _require_count("crf", crf, CRF_MINIMUM)
    if quality > CRF_MAXIMUM:
        raise ValueError(f"crf must be <= {CRF_MAXIMUM} (got {quality})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-framerate",
        str(rate),
        "-i",
        str(frames_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(quality),
        str(dest),
    ]
    proc = run_capture(argv)
    if proc.returncode != 0:
        raise MediaError(f"chunk encode failed for {dest}: {proc.stderr[-2000:]}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise MediaError(f"chunk encode produced empty output {dest}")
    return dest


def cuda_visible_device_count(env: Mapping[str, str] | None = None) -> int | None:
    """Entries in CUDA_VISIBLE_DEVICES, or None when unset (caller falls back to nvidia-smi)."""
    source = os.environ if env is None else env
    raw = source.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return sum(1 for entry in raw.split(",") if entry.strip() != "")


def nvidia_smi_device_count() -> int:
    """GPUs listed by `nvidia-smi -L`; 0 when the binary is missing or fails."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if proc.returncode != 0:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("GPU "))


def augment_devices(
    *,
    env: Mapping[str, str] | None = None,
    smi_count: int | None = None,
) -> tuple[str, ...]:
    """Devices for chunk work: `(cuda:0,)` by default, plus `cuda:1` when visible.

    Visibility prefers CUDA_VISIBLE_DEVICES, then the `smi_count` test
    seam, then a live `nvidia-smi -L` probe. Unknown/zero visibility still
    yields `(cuda:0,)` — the caller gates on real CUDA — except an
    explicitly emptied CUDA_VISIBLE_DEVICES, which honors the admin's
    hide-GPU intent by yielding `()` (caller skips augment). Capped at two
    devices: the SFX pairing never needs more.
    """
    explicit = cuda_visible_device_count(env)
    if explicit is not None:
        if explicit == 0:
            return ()
        count = explicit
    elif smi_count is not None:
        count = smi_count
    else:
        count = nvidia_smi_device_count()
    if count >= MAX_PARALLEL_DEVICES:
        return (AUGMENT_DEVICE_PRIMARY, AUGMENT_DEVICE_SECONDARY)
    return (AUGMENT_DEVICE_PRIMARY,)


def run_augment_chunks(
    chunks: list[AugmentChunk],
    worker: Callable[[AugmentChunk, str], T],
) -> list[T]:
    """Run chunks on their planned devices, preserving chunk order in the outcomes.

    One unique device → serial loop; two → ThreadPoolExecutor(2), one
    thread per device (the SFX pairing's cuda:0/cuda:1 split). Executor.map
    keeps outcome order identical to chunk order either way, so callers can
    concatenate chunk outputs directly.
    """
    devices = list(dict.fromkeys(chunk.device for chunk in chunks))
    if len(devices) <= 1:
        return [worker(chunk, chunk.device) for chunk in chunks]
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_DEVICES, len(devices))) as pool:
        return list(pool.map(lambda chunk: worker(chunk, chunk.device), chunks))
