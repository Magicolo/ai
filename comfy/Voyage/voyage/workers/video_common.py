"""Shared video-worker scaffolding (issue 019).

Extracted from `video_longlive.py` / `video_ltxv.py` / `video_causvid.py`,
whose operational skeleton (mp4 writes, atomic JSON tape writes, benchmark
warmup+measured loops, `serve()` dispatch maps) was copy-pasted three ways
and had already diverged on tape keys and profile fields. Workers keep
their generate/denoise logic and session classes; everything here is
plumbing that must change once, not three times.

Deliberate non-shares (documented, not migrated): LongLive's recovery tape
is a torch `.pt` tensor bundle (not JSON — see its `_load_recovery_tape`),
and its segment mp4 write uses `imageio.get_writer` inside `generate_blocks`
— both stay in `video_longlive.py`.

Top level is numpy + stdlib only (both in every image); imageio stays
function-level — the slim gates image has no imageio, so CPU tests stub
`sys.modules["imageio.v2"]`.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from voyage.workers.loop import Handler, validate_benchmark_counts

TAIL_FILENAME = "video_tail.mp4"
"""Crash-recovery anchor beside the segment video (ltxv + causvid)."""

TAPE_FILENAME = "recovery.pt"
"""Recovery-tape filename (kept for supervisor discovery; JSON content)."""


def write_tape_atomic(tape_path: Path, tape: dict[str, Any]) -> Path:
    """Atomically write a JSON recovery tape (sorted keys, trailing newline).

    Writes to a `.tmp` sibling then renames, so a crash mid-write never
    leaves a half-written tape behind. Returns `tape_path`.
    """
    tape_text = json.dumps(tape, indent=2, sort_keys=True) + "\n"
    tape_tmp = tape_path.with_suffix(".tmp")
    tape_tmp.write_text(tape_text, encoding="utf-8")
    tape_tmp.replace(tape_path)
    return tape_path


def clip_array_to_uint8(scaled_frames: NDArray[Any]) -> NDArray[np.uint8]:
    """Clip float frames in ~[0, 1] to uint8 (issue 065).

    VAE overshoot outside [0, 1] wraps modulo 256 on a bare `astype`
    (1.01 becomes near-black) — clip first, then convert.
    """
    return np.clip(scaled_frames * 255.0, 0, 255).astype("uint8")


def save_mp4(frames: list[NDArray[np.uint8]], output_path: Path, frames_per_second: int) -> None:
    """Write (H, W, C) uint8 frames as h264 mp4 (ltxv + causvid writer)."""
    if frames_per_second <= 0:
        raise ValueError(f"mp4 frame rate must be positive (got {frames_per_second})")
    import imageio.v2 as imageio  # type: ignore[import-not-found]

    imageio.mimsave(str(output_path), frames, fps=frames_per_second, codec="libx264")


class BenchmarkHarnessOutcome(NamedTuple):
    """Measured-only results from :func:`run_benchmark_harness`."""

    wall_seconds: list[float]
    peak_gib: list[float]


def run_benchmark_harness(
    warmup_count: int,
    measured_count: int,
    temporary_prefix: str,
    probe: Callable[[Path, bool], None],
    clock: Callable[[], float] = time.monotonic,
    reset_peak_memory: Callable[[], None] | None = None,
    read_peak_gib: Callable[[], float] | None = None,
) -> BenchmarkHarnessOutcome:
    """Run warmup + measured probes in a scratch dir, timing each.

    Owns the `TemporaryDirectory`, the warmup/measured loop, wall timing and
    VRAM-peak sampling; callers pass a `probe` that renders one artifact to
    the given path (`measured` is true for measured iterations — collect
    results then — and false for warmup) and format their own
    backend-specific response from the returned measured-only walls/peaks.
    Counts are validated first so bad values fail fast on CPU without
    touching GPU state.
    """
    validate_benchmark_counts(warmup_count, measured_count)
    wall_seconds: list[float] = []
    peak_gib: list[float] = []
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as tmp:
        for index in range(warmup_count + measured_count):
            if reset_peak_memory is not None:
                reset_peak_memory()
            started = clock()
            probe(Path(tmp) / f"b{index}.mp4", index >= warmup_count)
            elapsed = clock() - started
            peak = read_peak_gib() if read_peak_gib is not None else 0.0
            if index >= warmup_count:
                wall_seconds.append(elapsed)
                peak_gib.append(peak)
    return BenchmarkHarnessOutcome(wall_seconds=wall_seconds, peak_gib=peak_gib)


def standard_serve_map(
    backend_name: str,
    *,
    handle_init: Handler,
    handle_health: Handler,
    handle_generate_blocks: Handler,
    handle_benchmark: Handler,
    handle_evict_gpu: Handler,
    handle_rebuild: Handler,
    handle_resume: Handler,
) -> dict[str, Handler]:
    """Build the standard nine-op worker `serve()` dispatch map.

    Every video worker serves the same ops; only the `checkpoint` id prefix
    varies, which `backend_name` supplies (e.g. `"ltxv"` →
    `"ltxv-<segment_id>"`).
    """
    return {
        "init": handle_init,
        "health": handle_health,
        "generate_blocks": handle_generate_blocks,
        "benchmark": handle_benchmark,
        "evict_gpu": handle_evict_gpu,
        "rebuild": handle_rebuild,
        "checkpoint": lambda payload: {
            "checkpoint_id": f"{backend_name}-{payload.get('segment_id', 'none')}"
        },
        "resume": handle_resume,
        "shutdown": lambda _payload: {"stopped": True},
    }
