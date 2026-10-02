"""Finalize drain over ledgered interp chunks (issue: independent augment workers).

DESIGN §57: the drain is the finalize-side consumer — it never renders
pixels, it only verifies ledgered `interpolated_<NN>` PNG dirs, encodes each
to a chunk mp4, and concats them in plan order. All GPU work stays in the
pollers (`augment_upscale_poller`, `augment_interp_poller`); the drain is
stdlib-only with `encode_fn` / `concat_fn` seams so tests never need ffmpeg.

Ledger-truth resume: a non-empty `chunk_<NN>.mp4` beside a ledgered
`interpolated` record is complete work and is never re-encoded; anything
else (missing/empty mp4, `*.partial` leftovers pruned at entry) is redone.
A chunk the upscale poller started but the interp poller never finished
fails loud — finalize must run the pollers to completion first.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voyage import augment_sidecar as sidecar
from voyage.atomic import fsync_dir
from voyage.errors import MediaError

CHUNK_MP4_TEMPLATE = "chunk_{index:02d}.mp4"
"""Per-chunk intermediate filename (mirrors the finalize model pass)."""

INTERMEDIATE_FILENAME = "model_intermediate.mp4"
"""Concat output the presentation encode consumes downstream."""

CONCAT_LIST_FILENAME = "chunks.txt"
"""Concat-demuxer list beside the chunk mp4s."""

EncodeFn = Callable[[Path, Path, float], Path]
"""(png_dir, dest_mp4, out_fps) -> dest_mp4; production uses chunk ffmpeg."""

ConcatFn = Callable[[list[Path], Path], Path]
"""(chunk_mp4s_in_order, dest) -> dest; production uses stream-copy concat."""


@dataclass(frozen=True)
class DrainResult:
    """Outcome of draining one interp plan directory."""

    plan_dir: Path
    chunk_mp4s: tuple[Path, ...]
    intermediate_mp4: Path
    chunks_drained: int


def _require_out_fps(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"out_fps must be a number (got {type(value).__name__})")
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"out_fps must be finite and > 0 (got {value!r})")
    return rate


def _default_encode(png_dir: Path, dest: Path, out_fps: float) -> Path:
    from voyage import augment

    return augment.ffmpeg_encode_chunk(png_dir / "frame_%06d.png", dest, int(round(out_fps)))


def concat_chunk_mp4s(chunks: list[Path], dest: Path) -> Path:
    """Stream-copy concat of chunk mp4s in order (shared finalize default).

    Used by the per-plan drain above and by the multi-segment finalize
    wiring (`augment_finalize`), so every concat in the durable path shares
    one ffmpeg shape.
    """
    from voyage import augment
    from voyage.media_audio import run_capture

    concat_list = augment._write_chunk_concat_list(chunks, dest.parent / CONCAT_LIST_FILENAME)
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(dest),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"sidecar chunk concat failed: {proc.stderr[-2000:]}")
    return dest


def _default_concat(chunks: list[Path], dest: Path) -> Path:
    return concat_chunk_mp4s(chunks, dest)


def _interpolated_records(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Last-wins ledger record per chunk index for the interpolated stage."""
    by_index: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("stage") != sidecar.STAGE_INTERPOLATED:
            continue
        index = record.get("chunk_index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        by_index[index] = record
    return by_index


def drain_interpolated_plan(
    plan_dir: Path,
    *,
    out_fps: float,
    encode_fn: EncodeFn | None = None,
    concat_fn: ConcatFn | None = None,
) -> DrainResult:
    """Encode every ledgered interpolated chunk dir and concat them in order."""
    if not isinstance(plan_dir, Path):
        raise TypeError(f"plan_dir must be a Path (got {type(plan_dir).__name__})")
    rate = _require_out_fps(out_fps)
    render = encode_fn if encode_fn is not None else _default_encode
    join = concat_fn if concat_fn is not None else _default_concat
    if not plan_dir.is_dir():
        raise MediaError(f"augment plan dir missing: {plan_dir}")
    sidecar.prune_stale_partials(plan_dir)
    ledger = plan_dir / sidecar.CHUNKS_LEDGER_FILENAME
    records = sidecar.load_chunk_ledger(ledger)
    started = sorted(sidecar.completed_stages(records).keys())
    if not started:
        raise MediaError(f"no chunks recorded in {ledger} (nothing to drain)")
    rendered = _interpolated_records(records)
    waiting = [index for index in started if index not in rendered]
    if waiting:
        raise MediaError(
            f"chunks still waiting for interp in {plan_dir.name}: {waiting} "
            "(run the interp poller to completion, then drain again)"
        )
    chunk_mp4s: list[Path] = []
    for index in started:
        expected = rendered[index].get("expected_frames")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
            raise MediaError(
                f"chunk {index} has a bad expected_frames {expected!r} (not drainable)"
            )
        png_dir = plan_dir / f"interpolated_{index:02d}"
        if not png_dir.is_dir():
            raise MediaError(f"interpolated dir missing for chunk {index}: {png_dir}")
        found = sorted(png_dir.glob("frame_*.png"))
        if len(found) != expected:
            raise MediaError(
                f"{png_dir.name} holds {len(found)} PNGs "
                f"(ledger expects {expected}; rerun the interp poller)"
            )
        dest = plan_dir / CHUNK_MP4_TEMPLATE.format(index=index)
        if dest.exists() and dest.stat().st_size > 0:
            chunk_mp4s.append(dest)
            continue
        # SFX convention (`<name>.partial.wav`): the temp keeps the real
        # suffix so ffmpeg infers the format; `<name>.mp4.partial` breaks
        # format inference ("Unable to find a suitable output format").
        partial = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
        render(png_dir, partial, rate)
        if not partial.exists() or partial.stat().st_size == 0:
            raise MediaError(f"chunk encode produced empty output {partial}")
        os.replace(partial, dest)
        fsync_dir(plan_dir)
        chunk_mp4s.append(dest)
    intermediate = plan_dir / INTERMEDIATE_FILENAME
    join(chunk_mp4s, intermediate)
    if not intermediate.exists() or intermediate.stat().st_size == 0:
        raise MediaError(f"chunk concat produced empty output {intermediate}")
    return DrainResult(
        plan_dir=plan_dir,
        chunk_mp4s=tuple(chunk_mp4s),
        intermediate_mp4=intermediate,
        chunks_drained=len(chunk_mp4s),
    )
