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
        ],
        # Unbounded: the chunk list scales with total video length.
        timeout=None,
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
        # Track C probe-before-prune: the fresh encode must probe to
        # `expected_frames` BEFORE the PNG donors are pruned — a short
        # encode (kill mid-write, torn mp4) re-raises here while its
        # PNGs are still intact for the retry. Unprobable (no ffprobe)
        # skips the count check but keeps the non-empty gate above.
        probed = sidecar._probe_mp4_frames(partial)
        if probed is not None and probed != int(key.expected_frames):
            with contextlib.suppress(OSError):
                partial.unlink()
            raise MediaError(
                f"chunk encode probed {probed}f for chunk {key.chunk_index} "
                f"(expected {key.expected_frames}f) — PNGs kept for retry"
            )
        os.replace(partial, dest)
        fsync_dir(plan_dir)
        relative = os.path.relpath(dest, run_dir).replace(os.sep, "/")
        sidecar.append_chunk_record(ledger, key, stage=sidecar.STAGE_CHUNK_MP4, path=relative)
    else:
        # Track C probe-before-prune on the fast path too: a ledgered
        # mp4 that probes short (truncated since record) is treated as
        # missing — fall through and re-encode from intact PNGs when
        # present, else fail loud (the heal sweep strips the record).
        if not sidecar.chunk_mp4_frames_match(plan_dir, key):
            raise MediaError(
                f"chunk mp4 probes short for chunk {key.chunk_index} "
                f"(expected {key.expected_frames}f; re-render the chunk)"
            )
    _prune_chunk_png_dirs(plan_dir, key.chunk_index)
    return sidecar.chunk_mp4_path(plan_dir, key.chunk_index)


def settle_uses_output_truth(
    plan_dir: Path,
    records: list[dict[str, Any]],
    key: sidecar.ChunkKey,
) -> bool:
    """Whether a chunk is settled (Track C output-truth, not ledger-only).

    Settled means the exact-key `chunk_mp4` record exists AND the mp4
    probes to `expected_frames` (the durable artifact is truth — its PNG
    donors were pruned by design, so surviving PNG dirs never veto it),
    OR — when no durable mp4 exists yet — the interp PNGs are complete
    with matching geometry. Ledger-only completion once settled short
    chunks (torn encodes with intact records); this is the gate poll/drain
    settle checks must use.
    """
    if sidecar.chunk_mp4_complete(plan_dir, records, key):
        return True
    from voyage.augment import chunk_frames_match_size as _size_match

    interp_dir = plan_dir / f"interpolated_{key.chunk_index:02d}"
    if not sidecar.chunk_output_complete(interp_dir, int(key.expected_frames)):
        return False
    try:
        size = (int(key.out_width), int(key.out_height))
    except (TypeError, ValueError):
        return True
    return bool(_size_match(interp_dir, size))


def sweep_chunk_mp4s(
    run_dir: Path,
    *,
    encode_fn: EncodeFn | None = None,
    live_dirs: set[Path] | None = None,
) -> tuple[int, int]:
    """Retroactive chunk-mp4 pass over every plan dir (finalize-start sweep).

    Encodes (+ records + prunes PNGs, via `ensure_chunk_mp4`) each
    fully-interpolated chunk still missing its durable mp4 — one-time
    catch-up for runs rendered before per-chunk mp4s existed, so their
    gigabytes of PNGs collapse to megabytes before polling starts.
    Chunks with incomplete interp PNGs are skipped, never fail-loud
    (the pollers below heal them); returns `(ensured, skipped)`.

    Track C liveness: `live_dirs` (the `plan_dir_for_segment` set for
    the current usable segments + joints) restricts the sweep to live
    dirs — orphaned plan dirs are GC's concern (prune-then-sweep order),
    never re-encoded here. None (default) keeps the legacy all-dirs
    sweep for direct callers/tests.
    """
    if not isinstance(run_dir, Path):
        raise TypeError(f"run_dir must be a Path (got {type(run_dir).__name__})")
    ensured = 0
    skipped = 0
    for plan_dir in sidecar.list_plan_dirs(run_dir):
        if plan_dir.is_symlink() or plan_dir.name == "morph_joints":
            continue
        if live_dirs is not None and plan_dir not in live_dirs:
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


_LAST_GC_FINGERPRINT: tuple[Any, ...] | None = None
"""Inputs of the last completed orphan-GC pass (issue 252).

Repeat finalize starts in one process re-ran the whole GC (joint
re-hashes + recursive mtime stats) with nothing changed. When the
cheap fingerprint below still matches, there is nothing new to
collect and the pass returns 0. Skips defer collection only — they
never delete wrongly (a changed tree changes the fingerprint and
runs the full pass). Wall time (`now_seconds`) is deliberately NOT
part of the fingerprint: an orphan aging past grace waits for the
next fingerprint change instead of forcing a full pass per call.
"""


def _gc_fingerprint(
    *,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    crf: int,
    preset: str,
    grace_days: float,
    run_dir: Path,
    joint_videos: list[Path],
) -> tuple[Any, ...]:
    """Cheap GC inputs identity (no hashes, no recursive stats).

    Segment manifests (id + source key + file identity) catch
    commits/re-renders; the augment top-level listing (names + dir
    mtimes) catches new/removed plan dirs; joint video identities
    catch re-rendered bridges. All shallow: one `iterdir` + a few
    stats, versus the full pass's recursive stats + gigabyte hashes.
    """
    from voyage import paths
    from voyage.augment_upscale_poller import committed_segment_sources

    segment_bits: list[tuple[str, str, int, int]] = []
    try:
        sources, _skipped = committed_segment_sources(run_dir)
    except (OSError, ValueError):
        sources = []
    for source in sorted(sources, key=lambda item: item.segment_id):
        manifest = source.segment_dir / paths.SEGMENT_MANIFEST_FILENAME
        try:
            stat = manifest.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = (-1, -1)
        segment_bits.append((source.segment_id, source.source_key, stamp[0], stamp[1]))
    top_bits: list[tuple[str, int]] = []
    for child in sidecar.list_plan_dirs(run_dir):
        try:
            stamp_ns = child.stat().st_mtime_ns
        except OSError:
            stamp_ns = -1
        top_bits.append((child.name, stamp_ns))
    joint_bits: list[tuple[str, int, int]] = []
    for joint_video in sorted(str(video) for video in joint_videos):
        try:
            stat = Path(joint_video).stat()
            joint_bits.append((joint_video, stat.st_mtime_ns, stat.st_size))
        except OSError:
            joint_bits.append((joint_video, -1, -1))
    return (
        str(run_dir),
        weights_key,
        out_width,
        out_height,
        out_fps,
        upscale_factor,
        crf,
        preset,
        float(grace_days),
        tuple(segment_bits),
        tuple(top_bits),
        tuple(joint_bits),
    )


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

    Track C joint/morph GC: liveness derives from the CURRENT adjacent
    usable pairs only (the committed segment order at call time — a
    re-trimmed run's stale pairs are not live), and `joint_sources/`,
    `joint_timeline/`, `morph_native/` artifacts age out via the same
    `grace_days` (see `prune_stale_joint_artifacts`).

    Stdlib-only like the rest of this module (supervisor §12 GPU ban):
    segment sources and plan derivations import locally to avoid cycles.
    """
    from voyage.augment_joints import existing_joint_videos, joint_video_sha
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
    joint_list = existing_joint_videos(run_dir)
    global _LAST_GC_FINGERPRINT
    fingerprint = _gc_fingerprint(
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
        grace_days=float(grace_days),
        run_dir=run_dir,
        joint_videos=joint_list,
    )
    if fingerprint == _LAST_GC_FINGERPRINT:
        # Same tree, same settings, same segments: the last completed
        # pass already collected everything collectible (issue 252).
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
    for joint_video in joint_list:
        try:
            joint_sha = joint_video_sha(joint_video)
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
    for child in sidecar.list_plan_dirs(run_dir):
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
    # Track C joint/morph grace-aging: stale fix-stage artifacts (joint
    # units for non-adjacent pairs, joint_timeline/morph_native outputs)
    # age out under the same grace — liveness is the current adjacent
    # usable pairs only (see helper). Best-effort: never fails the GC.
    with contextlib.suppress(OSError, ValueError, TypeError, ImportError):
        pruned += prune_stale_joint_artifacts(
            run_dir, grace_days=float(grace_days), now_seconds=current_time
        )
    _LAST_GC_FINGERPRINT = fingerprint
    return pruned


def _live_adjacent_pair_keys(run_dir: Path) -> set[tuple[str, str]]:
    """Current adjacent usable pair keys (segment ids), empty on failure.

    Liveness single source for joint/morph GC: only pairs adjacent in
    the CURRENT committed order are live — a re-trimmed or extended
    run's stale pairs age out instead of pinning artifacts forever.
    """
    try:
        from voyage.augment_upscale_poller import committed_segment_sources
    except ImportError:
        return set()
    try:
        sources, _ = committed_segment_sources(run_dir)
    except (OSError, ValueError, TypeError):
        return set()
    ordered = sorted(sources, key=lambda source: source.segment_id)
    return {(ordered[i].segment_id, ordered[i + 1].segment_id) for i in range(len(ordered) - 1)}


def prune_stale_joint_artifacts(
    run_dir: Path, *, grace_days: float = 7.0, now_seconds: float | None = None
) -> int:
    """Age out stale joint/morph artifacts under grace (Track C, best-effort).

    Covers `augment/joint_sources/` units whose pair is no longer adjacent
    in the current committed order, plus `augment/joint_timeline/` and
    `augment/morph_native/` outputs older than `grace_days` (by newest
    mtime under each root). Live pairs (current adjacency) and young
    artifacts are always kept; missing roots prune nothing. Returns the
    pruned count. Never raises for I/O (GC must never break finalize).
    """
    if isinstance(grace_days, bool) or not isinstance(grace_days, (int, float)):
        raise TypeError(f"grace_days must be a number (got {type(grace_days).__name__})")
    import math as _math

    if not _math.isfinite(float(grace_days)) or float(grace_days) < 0:
        raise ValueError(f"grace_days must be finite and >= 0 (got {grace_days!r})")
    current_time = float(now_seconds) if now_seconds is not None else time.time()
    cutoff = current_time - float(grace_days) * _SECONDS_PER_DAY
    pruned = 0
    live_pairs = _live_adjacent_pair_keys(run_dir)
    # Joint-source units: dirname encodes the pair (leftID_rightID or
    # joint_II_JJ) — keep only current-adjacent pairs, age out the rest.
    try:
        from voyage.augment_joints import joint_sources_root
    except ImportError:
        joint_sources_root = None  # type: ignore[assignment]
    if joint_sources_root is not None:
        try:
            units_root = joint_sources_root(run_dir)
        except (OSError, ValueError, TypeError):
            units_root = None
        if units_root is not None and units_root.is_dir():
            try:
                children = sorted(units_root.iterdir())
            except OSError:
                children = []
            for child in children:
                try:
                    if not child.is_dir() or child.is_symlink():
                        continue
                except OSError:
                    continue
                # Pair liveness: keep when the dirname names a live pair.
                name = child.name
                live = any(left in name and right in name for left, right in live_pairs)
                if live:
                    continue
                newest = _newest_modification_time(child)
                if newest is None or newest > cutoff:
                    continue
                try:
                    shutil.rmtree(child)
                    pruned += 1
                except OSError:
                    continue
    # Timeline/morph outputs: whole-root aging (their contents rebuild
    # from the current pairs each finalize — stale roots are safe to
    # drop past grace).
    for dirname in ("joint_timeline", "morph_native"):
        root = run_dir / "augment" / dirname
        try:
            if not root.is_dir() or root.is_symlink():
                continue
        except OSError:
            continue
        newest = _newest_modification_time(root)
        if newest is None or newest > cutoff:
            continue
        try:
            shutil.rmtree(root)
            pruned += 1
        except OSError:
            continue
    return pruned
