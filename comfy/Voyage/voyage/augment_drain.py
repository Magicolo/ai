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
import shutil
import time
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


_PLAN_HASH_LENGTH = 16
"""Expected plan-dir name length (mirrors `plan_hash_for`, 16 hex chars)."""

_PLAN_HASH_DIGITS = frozenset("0123456789abcdef")
"""Lowercase hexadecimal alphabet for plan-dir name recognition."""

_SECONDS_PER_DAY = 86400.0
"""Grace-period unit (parameter is days, comparison is seconds)."""


def _is_plan_hash_name(name: str) -> bool:
    """Whether a dir name looks like a sidecar plan hash (16 lowercase hex)."""
    return len(name) == _PLAN_HASH_LENGTH and all(
        character in _PLAN_HASH_DIGITS for character in name
    )


def _newest_modification_time(plan_dir: Path) -> float | None:
    """Newest modification time under a plan dir (None when unreadable/empty).

    Why recursive: creating PNGs inside `interpolated_00/` bumps that
    subdir, not the parent plan dir — the parent mtime alone would call
    an actively-written plan old. Best-effort: any unreadable entry is
    skipped, never fail-loud (GC must never break finalize).
    """
    newest: float | None = None
    try:
        candidates = [plan_dir, *plan_dir.rglob("*")]
    except OSError:
        return None
    for candidate in candidates:
        try:
            stamp = candidate.stat().st_mtime
        except OSError:
            continue
        if newest is None or stamp > newest:
            newest = stamp
    return newest


def prune_orphan_plan_dirs(
    run_dir: Path,
    *,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    crf: int,
    preset: str,
    grace_days: float = 7.0,
    now_seconds: float | None = None,
) -> int:
    """Delete stale unreferenced sidecar plan dirs (DESIGN §§56-57, orphan GC).

    Why this exists: any settings/weights/segment re-render forks
    `<plan-hash>/` via `plan_dir_for_segment` (old hash never reused),
    so `run/augment/` accumulates orphans with no eviction. Live dirs are
    recomputed from the current committed segments (`plan_dir_for_segment`
    per source plus `seam_plan_dir` per adjacent pair, same derivation as
    the pollers/drain) and always kept; `morph_joints` is kept wholesale
    (record.json ledger, not sidecar — different GC). Only 16-hex plan
    dirs older than `grace_days` (default 7, by newest modification time
    under the dir) are deleted; anything live, young, non-hash-named, or
    unreadable is kept. Returns the pruned count. NOT wired into finalize
    (explicit later decision) — callers invoke it deliberately.

    Stdlib-only like the rest of this module (supervisor §12 GPU ban):
    segment sources and plan derivations import locally to avoid cycles.
    """
    from voyage.augment_seam import seam_plan_dir
    from voyage.augment_sidecar import AUGMENT_DIRNAME, plan_dir_for_segment
    from voyage.augment_upscale_poller import committed_segment_sources

    if not isinstance(run_dir, Path):
        raise TypeError(f"run_dir must be a Path (got {type(run_dir).__name__})")
    if isinstance(grace_days, bool) or not isinstance(grace_days, (int, float)):
        raise TypeError(f"grace_days must be a number (got {type(grace_days).__name__})")
    if not math.isfinite(float(grace_days)) or float(grace_days) < 0:
        raise ValueError(f"grace_days must be finite and >= 0 (got {grace_days!r})")
    if now_seconds is not None and (
        isinstance(now_seconds, bool)
        or not isinstance(now_seconds, (int, float))
        or not math.isfinite(float(now_seconds))
    ):
        raise ValueError(f"now_seconds must be finite or None (got {now_seconds!r})")
    current_time = float(now_seconds) if now_seconds is not None else time.time()
    cutoff_time = current_time - float(grace_days) * _SECONDS_PER_DAY
    augment_root = run_dir / AUGMENT_DIRNAME
    if not augment_root.is_dir():
        return 0
    sources, _skipped = committed_segment_sources(run_dir)
    live_dirs: set[Path] = set()
    for source in sources:
        live_dirs.add(
            plan_dir_for_segment(
                run_dir,
                source_key=source.source_key,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=out_fps,
                upscale_factor=upscale_factor,
                crf=crf,
                preset=preset,
            )
        )
    for first, second in zip(sources, sources[1:], strict=False):
        live_dirs.add(
            seam_plan_dir(
                run_dir,
                key_a=first.source_key,
                key_b=second.source_key,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=out_fps,
                upscale_factor=upscale_factor,
                crf=crf,
                preset=preset,
            )
        )
    pruned = 0
    for child in sorted(augment_root.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        if child.name == "morph_joints":
            continue
        if child in live_dirs:
            continue
        if not _is_plan_hash_name(child.name):
            continue
        newest = _newest_modification_time(child)
        if newest is None or newest > cutoff_time:
            continue
        try:
            shutil.rmtree(child)
        except OSError:
            continue
        pruned += 1
    return pruned
