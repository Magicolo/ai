"""Finalize drain over ledgered interp chunks (issue: independent augment workers).

DESIGN §57: the drain is the finalize-side consumer — it never renders
pixels, it only verifies ledgered `interpolated_<NN>` PNG dirs, encodes each
to a chunk mp4, and concats them in plan order. All GPU work stays in the
pollers (`augment_upscale_poller`, `augment_interp_poller`); the drain is
stdlib-only with `encode_fn` / `concat_fn` seams so tests never need ffmpeg.

Ledger-truth resume: an exact-key `chunk_mp4` record plus a non-empty
`chunk_<NN>.mp4` is complete work and is never re-encoded; anything
else (missing/empty mp4, `*.partial` leftovers pruned at entry) is
redone through `ensure_chunk_mp4`. A chunk the upscale poller started
but the interp poller never finished fails loud — finalize must run
the pollers to completion first.
"""

from __future__ import annotations

import contextlib
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


def _chunk_key_from_record(record: dict[str, Any]) -> sidecar.ChunkKey:
    """Rebuild the interp `ChunkKey` a ledger record was recorded with (fail-loud).

    The drain never invents keys: the exact recorded key drives both the
    mp4-record match (any settings change misses and re-encodes) and the
    PNG-count expectation. A record missing fields, or carrying a
    non-positive `expected_frames`, is not drainable.
    """
    try:
        index = record["chunk_index"]
    except KeyError as exc:
        raise MediaError(f"chunk record carries a bad key ({exc}; not drainable)") from exc
    if isinstance(index, bool) or not isinstance(index, int):
        raise MediaError(f"chunk record carries a bad chunk_index {index!r} (not drainable)")
    try:
        key = sidecar.ChunkKey(
            chunk_index=index,
            start_frame=int(record["start_frame"]),
            source_frames=int(record["source_frames"]),
            expected_frames=int(record["expected_frames"]),
            upscale_factor=int(record["upscale_factor"]),
            multiplier=int(record["multiplier"]),
            crf=int(record["crf"]),
            preset=str(record["preset"]),
            source_key=str(record["source_key"]),
            weights_key=str(record["weights_key"]),
            out_width=int(record["out_width"]),
            out_height=int(record["out_height"]),
            out_fps=int(record["out_fps"]),
            chunk_frames=int(record.get("chunk_frames", 32)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError(f"chunk record carries a bad key ({exc}; not drainable)") from exc
    if key.expected_frames <= 0:
        raise MediaError(
            f"chunk {key.chunk_index} has a bad expected_frames "
            f"{key.expected_frames} (not drainable)"
        )
    return key


def _prune_chunk_png_dirs(plan_dir: Path, chunk_index: int) -> None:
    """Best-effort removal of a chunk's superseded PNG dirs (the mp4 is truth).

    Runs only after the durable `chunk_NN.mp4` plus its ledger record are
    in place, so a crash before this point re-encodes from intact PNGs
    and a crash inside it converges on the next `ensure_chunk_mp4` call
    (which prunes again on its fast path).
    """
    for prefix in (f"upscaled_{chunk_index:02d}", f"interpolated_{chunk_index:02d}"):
        child = plan_dir / prefix
        with contextlib.suppress(OSError):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)


def ensure_chunk_mp4(
    plan_dir: Path,
    key: sidecar.ChunkKey,
    *,
    out_fps: float,
    run_dir: Path | None = None,
    encode_fn: EncodeFn | None = None,
) -> Path:
    """Encode `chunk_NN.mp4` from `interpolated_NN/` PNGs when incomplete.

    Idempotent resume: an exact-key `chunk_mp4` record plus a non-empty
    mp4 returns immediately (never re-encoded — same fast path the drain
    has always had, now ledger-backed); otherwise the interp PNGs must be
    complete (fail-loud like the drain) and are encoded via `encode_fn`
    (default chunk ffmpeg) through a `.partial.mp4` + atomic replace,
    recorded under `run_dir` (default: the `run/augment/<hash>` layout
    parent, fail-loud elsewhere), and then both PNG dirs are pruned —
    the mp4 supersedes ~20x its PNG weight. Pruning runs on the fast
    path too, so every call converges PNGs to deleted. Called per chunk
    by the interp poller right after rendering (cleanup during
    generation) and by the drain (cleanup at finalize time).
    """
    if not isinstance(plan_dir, Path):
        raise TypeError(f"plan_dir must be a Path (got {type(plan_dir).__name__})")
    rate = _require_out_fps(out_fps)
    render = encode_fn if encode_fn is not None else _default_encode
    if run_dir is None:
        if plan_dir.parent.name != sidecar.AUGMENT_DIRNAME:
            raise MediaError(
                f"chunk mp4 needs run_dir outside {sidecar.AUGMENT_DIRNAME}/<hash>: {plan_dir}"
            )
        run_dir = plan_dir.parent.parent
    ledger = plan_dir / sidecar.CHUNKS_LEDGER_FILENAME
    records = sidecar.load_chunk_ledger(ledger)
    if not sidecar.chunk_mp4_complete(plan_dir, records, key):
        interp_dir = plan_dir / f"interpolated_{key.chunk_index:02d}"
        if not sidecar.chunk_output_complete(interp_dir, key.expected_frames):
            raise MediaError(
                f"interpolated dir missing or short for chunk {key.chunk_index}: {interp_dir} "
                f"(expected {key.expected_frames} PNGs; rerun the interp poller)"
            )
        dest = sidecar.chunk_mp4_path(plan_dir, key.chunk_index)
        # SFX convention (`<name>.partial.wav`): the temp keeps the real
        # suffix so ffmpeg infers the format; `<name>.mp4.partial` breaks
        # format inference ("Unable to find a suitable output format").
        partial = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
        render(interp_dir, partial, rate)
        if not partial.exists() or partial.stat().st_size == 0:
            raise MediaError(f"chunk encode produced empty output {partial}")
        os.replace(partial, dest)
        fsync_dir(plan_dir)
        relative = os.path.relpath(dest, run_dir).replace(os.sep, "/")
        sidecar.append_chunk_record(ledger, key, stage=sidecar.STAGE_CHUNK_MP4, path=relative)
    _prune_chunk_png_dirs(plan_dir, key.chunk_index)
    return sidecar.chunk_mp4_path(plan_dir, key.chunk_index)


def sweep_chunk_mp4s(
    run_dir: Path,
    *,
    encode_fn: EncodeFn | None = None,
) -> tuple[int, int]:
    """Retroactive chunk-mp4 pass over every plan dir (finalize-start sweep).

    Encodes (+ records + prunes PNGs, via `ensure_chunk_mp4`) each
    fully-interpolated chunk still missing its durable mp4 — one-time
    catch-up for runs rendered before per-chunk mp4s existed, so their
    gigabytes of PNGs collapse to megabytes before polling starts.
    Chunks with incomplete interp PNGs are skipped, never fail-loud
    (the pollers below heal them); returns `(ensured, skipped)`.
    """
    if not isinstance(run_dir, Path):
        raise TypeError(f"run_dir must be a Path (got {type(run_dir).__name__})")
    try:
        plan_dirs = sorted(
            child for child in (run_dir / sidecar.AUGMENT_DIRNAME).iterdir() if child.is_dir()
        )
    except OSError:
        return (0, 0)
    ensured = 0
    skipped = 0
    for plan_dir in plan_dirs:
        if plan_dir.is_symlink() or plan_dir.name == "morph_joints":
            continue
        ledger = plan_dir / sidecar.CHUNKS_LEDGER_FILENAME
        records = sidecar.load_chunk_ledger(ledger)
        for _index, record in sorted(_interpolated_records(records).items()):
            try:
                key = _chunk_key_from_record(record)
            except MediaError:
                skipped += 1
                continue
            if sidecar.chunk_mp4_complete(plan_dir, records, key):
                continue
            interp_dir = plan_dir / f"interpolated_{key.chunk_index:02d}"
            if not sidecar.chunk_output_complete(interp_dir, key.expected_frames):
                skipped += 1
                continue
            ensure_chunk_mp4(
                plan_dir, key, out_fps=float(key.out_fps), run_dir=run_dir, encode_fn=encode_fn
            )
            records = sidecar.load_chunk_ledger(ledger)
            ensured += 1
    return (ensured, skipped)


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
        # Ledger-backed mp4 fast path: a complete durable mp4 is never
        # re-encoded, and encoding one prunes its ~20x PNG weight — the
        # drain is the finalize-time half of the per-segment cleanup
        # (the interp poller ensures each chunk the same way during
        # generation, so a fully pre-warmed plan drains PNG-free).
        key = _chunk_key_from_record(rendered[index])
        chunk_mp4s.append(ensure_chunk_mp4(plan_dir, key, out_fps=rate, encode_fn=render))
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
    per source plus one per fix-stage joint video already on disk — joints
    render no new work here, existing videos only) and always kept;
    `morph_joints` is kept wholesale (record.json ledger, not sidecar —
    different GC; the media native path still uses it). Old `seam_*`
    plan dirs are intentionally NOT live: they age out via `grace_days`.
    Only 16-hex plan dirs older than `grace_days` (default 7, by newest
    modification time under the dir) are deleted; anything live, young,
    non-hash-named, or unreadable is kept. Returns the pruned count. NOT
    wired into finalize (explicit later decision) — callers invoke it
    deliberately.

    Stdlib-only like the rest of this module (supervisor §12 GPU ban):
    segment sources and plan derivations import locally to avoid cycles.
    """
    from voyage.augment_joints import existing_joint_videos
    from voyage.augment_sidecar import AUGMENT_DIRNAME, plan_dir_for_segment
    from voyage.augment_upscale_poller import committed_segment_sources
    from voyage.hashing import sha256_file

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
    for joint_video in existing_joint_videos(run_dir):
        try:
            joint_sha = sha256_file(joint_video)
        except (OSError, ValueError):
            continue
        live_dirs.add(
            plan_dir_for_segment(
                run_dir,
                source_key=joint_sha,
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
