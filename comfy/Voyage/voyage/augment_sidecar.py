"""Durable augment sidecar ledger (DESIGN §§56-57, §140; issues 153/166).

Pure stdlib orchestration (supervisor §12 GPU ban): the independent
upscale/interp polling workers share one `chunks.jsonl` ledger under
`run_dir/augment/<plan-hash>/` so either worker (or a finalize drain)
can resume after a crash without re-rendering completed chunks.
Mirrors the SFX `sfx.jsonl` contract (serial append, flush+fsync,
last-wins dedupe, partial pruning) — never torch, never ffmpeg.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from voyage.atomic import fsync_dir
from voyage.hashing import sha256_file, sha256_text

AUGMENT_DIRNAME = "augment"
"""Run-relative dir holding one subdir per finalize plan hash."""

CHUNKS_LEDGER_FILENAME = "chunks.jsonl"
"""Ledger filename inside each plan dir (SFX `sfx.jsonl` twin)."""

PLAN_HASH_LENGTH = 16
"""Hex chars kept from the plan sha (enough to isolate plans, short for `ls`)."""

STAGE_UPSCALED = "upscaled"
"""Ledger stage: the upscale poller finished this chunk."""

STAGE_INTERPOLATED = "interpolated"
"""Ledger stage: the interp poller finished this chunk."""

_KNOWN_STAGES = frozenset({STAGE_UPSCALED, STAGE_INTERPOLATED})


@dataclass(frozen=True)
class ChunkKey:
    """Identity of one sidecar chunk (all must match for a ledger hit).

    `chunk_frames` is the configured window size: the same plan dir with
    a different chunking would otherwise tile different `(start, count)`
    windows onto the same indexes, and the index-only skip would reuse
    wrong outputs. Records written before the field existed carry the
    default and are additionally guarded by the pollers' exact-window
    check (`stage_indexes_matching`), so they re-render at most once.
    """

    chunk_index: int
    start_frame: int
    source_frames: int
    expected_frames: int
    upscale_factor: int
    multiplier: int
    crf: int
    preset: str
    source_key: str
    weights_key: str
    out_width: int
    out_height: int
    out_fps: int
    chunk_frames: int = 32


def plan_hash_for(
    *,
    source_key: str,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    multiplier: int,
    crf: int,
    preset: str,
) -> str:
    """Short hash isolating one finalize plan (a re-finalize at new settings misses)."""
    canonical = "|".join(
        [
            source_key,
            weights_key,
            str(out_width),
            str(out_height),
            str(out_fps),
            str(upscale_factor),
            str(multiplier),
            str(crf),
            preset,
        ]
    )
    return sha256_text(canonical)[:PLAN_HASH_LENGTH]


def sidecar_dir(run_dir: Path, plan_hash: str) -> Path:
    """Durable plan dir: `run_dir/augment/<plan-hash>/` (survives crashes, unlike tmp)."""
    if not isinstance(run_dir, Path):
        raise TypeError(f"run_dir must be a Path (got {type(run_dir).__name__})")
    if not plan_hash:
        raise ValueError("plan_hash must be non-empty")
    return run_dir / AUGMENT_DIRNAME / plan_hash


def plan_dir_for_segment(
    run_dir: Path,
    *,
    source_key: str,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    crf: int,
    preset: str,
) -> Path:
    """Durable plan dir for one segment (shared upscale-plan contract).

    Both pollers and the finalize drain derive through here: `out_fps` is
    the committed-segment (source) fps and the stored plan always carries
    `multiplier=1`, so upscale outputs are multiplier-independent and a
    settings change (geometry, recipe, either weight leg) misses old
    outputs instead of reusing them. The interp poller records its own
    keys with the real multiplier inside the same dir.
    """
    return sidecar_dir(
        run_dir,
        plan_hash_for(
            source_key=source_key,
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            out_fps=out_fps,
            upscale_factor=upscale_factor,
            multiplier=1,
            crf=crf,
            preset=preset,
        ),
    )


def _require_stage(stage: str) -> str:
    if stage not in _KNOWN_STAGES:
        raise ValueError(f"stage must be one of {sorted(_KNOWN_STAGES)} (got {stage!r})")
    return stage


def append_chunk_record(ledger: Path, key: ChunkKey, *, stage: str, path: str) -> None:
    """Durably append one finished chunk (flush + fsync + fsync_dir, serial appends)."""
    if not isinstance(key, ChunkKey):
        raise TypeError(f"key must be a ChunkKey (got {type(key).__name__})")
    resolved_stage = _require_stage(stage)
    if not path:
        raise ValueError("path must be a non-empty run-relative string")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {**asdict(key), "stage": resolved_stage, "path": path}
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_dir(ledger.parent)


def load_chunk_ledger(ledger: Path) -> list[dict[str, Any]]:
    """Read the sidecar ledger (missing file → empty; blank lines tolerated).

    Torn trailing lines (a crash mid-append between `write` and the
    newline+fsync) are skipped, not fatal: the interrupted chunk simply
    has no record and is re-rendered on the next poll.
    """
    if not ledger.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def chunk_cache_hit(records: list[dict[str, Any]], key: ChunkKey, *, stage: str) -> bool:
    """Whether any ledger record satisfies this chunk+stage (exact key match, pure)."""
    resolved_stage = _require_stage(stage)
    wanted = asdict(key)
    for record in records:
        if record.get("stage") != resolved_stage:
            continue
        if all(record.get(field) == value for field, value in wanted.items()):
            return True
    return False


def completed_stages(records: list[dict[str, Any]]) -> dict[int, set[str]]:
    """Chunk index → finished stages (union over records; malformed rows skipped)."""
    completed: dict[int, set[str]] = {}
    for record in records:
        stage = record.get("stage")
        index = record.get("chunk_index")
        if stage not in _KNOWN_STAGES or isinstance(index, bool) or not isinstance(index, int):
            continue
        completed.setdefault(index, set()).add(str(stage))
    return completed


def missing_chunk_indexes(
    records: list[dict[str, Any]], indexes: list[int], *, stage: str
) -> list[int]:
    """Ordered indexes in `indexes` with no finished record for `stage`."""
    resolved_stage = _require_stage(stage)
    done = completed_stages(records)
    return [index for index in indexes if resolved_stage not in done.get(index, set())]


def stage_indexes_matching(
    records: list[dict[str, Any]],
    windows: list[tuple[int, int]],
    *,
    stage: str,
    chunk_frames: int,
) -> list[int]:
    """Ordered chunk indexes whose ledger record exactly matches its window.

    Index-only matching (`missing_chunk_indexes` over `completed_stages`)
    trusts that the recorded chunk covers the same `(start, count)` window
    the current `chunk_frames` setting tiles. A settings change that
    re-tiles the same plan dir (or a pre-`chunk_frames` record) would
    otherwise skip wrong outputs. An index counts as done only when a
    record for `stage` carries the same `start_frame`, `source_frames`,
    and `chunk_frames` (records predating the field match on the window
    alone, grandfathering the pinned default-32 ledgers).
    """
    resolved_stage = _require_stage(stage)
    if isinstance(chunk_frames, bool) or not isinstance(chunk_frames, int) or chunk_frames < 1:
        raise ValueError(f"chunk_frames must be an int >= 1 (got {chunk_frames!r})")
    matching: list[int] = []
    for index, (start, count) in enumerate(windows):
        for record in records:
            if record.get("stage") != resolved_stage:
                continue
            if record.get("chunk_index") != index:
                continue
            if record.get("start_frame") != start:
                continue
            if record.get("source_frames") != count:
                continue
            recorded_frames = record.get("chunk_frames")
            if recorded_frames is not None and recorded_frames != chunk_frames:
                continue
            matching.append(index)
            break
    return matching


def chunk_output_complete(output_dir: Path, expected: int) -> bool:
    """Whether a ledgered chunk's output dir holds all expected frames.

    Output-truth companion to the ledger: the ledger alone never skips —
    only a present dir with exactly `expected` non-empty `frame_*.png`
    files counts as complete. A crash, cleanup, or disk corruption can
    remove output after its record was appended; anything incomplete is
    re-rendered (or waited on) instead of deadlocking the skip.
    """
    if not output_dir.is_dir() or output_dir.is_symlink():
        return False
    try:
        frames = sorted(output_dir.glob("frame_*.png"))
    except OSError:
        return False
    if len(frames) != expected:
        return False
    for frame in frames:
        try:
            if not frame.is_file() or frame.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def prune_stale_partials(target: Path) -> int:
    """Remove crashed-render partial leftovers (best-effort, returns pruned count).

    Covers `*.partial` files/dirs (upscale/interp PNG renders) and
    `*.partial.*` files (chunk mp4 renders, which keep their real suffix
    per the SFX `<name>.partial.wav` convention so ffmpeg infers the
    format): anything partial is, by the ledger-truth rule, incomplete
    work — a finished chunk always has a ledger record, so unledgered
    partials are safe to drop on every poll.
    """
    pruned = 0
    if target.is_dir():
        seen: set[Path] = set()
        for pattern in ("*.partial", "*.partial.*"):
            for partial in sorted(target.glob(pattern)):
                if partial in seen:
                    continue
                seen.add(partial)
                with contextlib.suppress(OSError):
                    if partial.is_dir() and not partial.is_symlink():
                        shutil.rmtree(partial)
                    else:
                        partial.unlink()
                    pruned += 1
    return pruned


def _interp_sha_from_weights_key(weights_key: object) -> str | None:
    """Interp-leg sha from a ledger weights_key (both wild shapes; else None).

    Legacy records carry `sha(interp)|sha(esrgan)`; newer records carry a
    `backend|` prefix (`film|sha|sha` / `rife|sha|sha`). Anything else is
    unknown provenance and must never drive deletion.
    """
    if not isinstance(weights_key, str):
        return None
    parts = weights_key.split("|")
    if len(parts) == 2 and all(parts):
        return parts[0]
    if len(parts) == 3 and parts[0] in ("film", "rife") and all(parts):
        return parts[1]
    return None


def _esrgan_sha_from_weights_key(weights_key: object) -> str | None:
    """Esrgan-leg sha from a ledger weights_key (both wild shapes; else None).

    Legacy records carry `sha(interp)|sha(esrgan)`; newer records carry a
    `backend|` prefix (`film|sha|sha` / `rife|sha|sha`). Anything else is
    unknown provenance and must never drive adoption.
    """
    if not isinstance(weights_key, str):
        return None
    parts = weights_key.split("|")
    if len(parts) == 2 and all(parts):
        return parts[1]
    if len(parts) == 3 and parts[0] in ("film", "rife") and all(parts):
        return parts[2]
    return None


def find_upscaled_donor(
    run_dir: Path,
    *,
    exclude_dir: Path,
    source_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    chunk_frames: int,
    chunk_index: int,
    start_frame: int,
    source_frames: int,
    esrgan_sha: str,
) -> dict[str, Any] | None:
    """First adoptable upscaled record from another plan dir, else None.

    A donor is an `upscaled`-stage record in a DIFFERENT plan dir whose
    window tuple + source + geometry + recipe match this chunk and whose
    record weights_key carries the same esrgan sha (backend switches
    change only the interp leg, so cross-backend upscaled PNGs are
    reusable). Donor outputs must be complete on disk. Records with
    unparseable keys skip silently (never adopt unknown provenance).
    """
    if not isinstance(run_dir, Path):
        raise TypeError(f"run_dir must be a Path (got {type(run_dir).__name__})")
    if not esrgan_sha:
        return None
    try:
        candidates = sorted((run_dir / AUGMENT_DIRNAME).iterdir())
    except OSError:
        return None
    for candidate in candidates:
        if candidate == exclude_dir:
            continue
        try:
            records = load_chunk_ledger(candidate / CHUNKS_LEDGER_FILENAME)
        except OSError:
            continue
        for record in records:
            if record.get("stage") != STAGE_UPSCALED:
                continue
            if record.get("chunk_index") != chunk_index:
                continue
            if record.get("start_frame") != start_frame:
                continue
            if record.get("source_frames") != source_frames:
                continue
            if record.get("chunk_frames", chunk_frames) != chunk_frames:
                continue
            if record.get("source_key") != source_key:
                continue
            if record.get("out_width") != out_width:
                continue
            if record.get("out_height") != out_height:
                continue
            if record.get("out_fps") != out_fps:
                continue
            if record.get("upscale_factor") != upscale_factor:
                continue
            if record.get("expected_frames") != source_frames:
                continue
            if _esrgan_sha_from_weights_key(record.get("weights_key")) != esrgan_sha:
                continue
            rel = record.get("path")
            if not isinstance(rel, str) or not rel:
                continue
            if rel.startswith("/") or ".." in rel.split("/"):
                continue
            if not chunk_output_complete(run_dir / rel, source_frames):
                continue
            return record
    return None


def prune_stale_interp_plans(
    run_dir: Path, *, interp_backend: str, weights: Any, whole_dir: bool = False
) -> int:
    """Prune stale interp outputs from known-other-backend plan dirs.

    Selector: the dir holds at least one `interpolated`-stage
    ledger record and every interpolated record keys to the
    non-active backend's current weights-file sha
    (`_interp_sha_from_weights_key` handles both wild shapes;
    unparseable or unknown shas never match). Upscale-only dirs
    never match (upscale pixels are backend-independent), and
    dirs with current-backend records never match either.

    Two modes: with `whole_dir=False` (default) only the
    `interpolated_*` output subdirs are deleted (the ledger,
    the upscaled pixels and the dir itself stay — the dir
    remains a valid upscale donor for cross-key adoption and
    the interp leg re-renders into it); with `whole_dir=True`
    the whole plan dir goes. Use False at finalize start
    (frees the bulk under disk pressure while keeping donors)
    and True pre-publish once every poll and drain on every
    path has finished.

    Skips silently (returns 0) when the non-active leg file is
    missing or unreadable — never deletes blind. Best-effort
    per dir, mirroring `prune_stale_partials`.
    """
    if not isinstance(run_dir, Path):
        raise TypeError(f"run_dir must be a Path (got {type(run_dir).__name__})")
    if interp_backend not in ("film", "rife"):
        raise ValueError(f"interp_backend must be 'film' or 'rife' (got {interp_backend!r})")
    if not isinstance(whole_dir, bool):
        raise TypeError(f"whole_dir must be a bool (got {type(whole_dir).__name__})")
    other = "film" if interp_backend == "rife" else "rife"
    leg_path = getattr(weights, other, None)
    try:
        other_sha: str | None = sha256_file(leg_path) if leg_path is not None else None
    except (AttributeError, OSError, TypeError, ValueError):
        other_sha = None
    if not other_sha:
        return 0
    pruned = 0
    try:
        candidates = sorted((run_dir / AUGMENT_DIRNAME).iterdir())
    except OSError:
        return 0
    for plan_dir in candidates:
        if not plan_dir.is_dir() or plan_dir.is_symlink():
            continue
        records = load_chunk_ledger(plan_dir / CHUNKS_LEDGER_FILENAME)
        shas = [
            _interp_sha_from_weights_key(record.get("weights_key"))
            for record in records
            if record.get("stage") == STAGE_INTERPOLATED
        ]
        if not shas or any(sha is None or sha != other_sha for sha in shas):
            continue
        if whole_dir:
            with contextlib.suppress(OSError):
                shutil.rmtree(plan_dir)
                pruned += 1
        else:
            # Free the bulk now, keep the dir: the ledger stays (so the
            # dir remains a valid upscale donor for cross-key adoption)
            # and the upscaled pixels stay (backend-independent).
            removed_any = False
            for child in sorted(plan_dir.iterdir()):
                if (
                    child.is_dir()
                    and not child.is_symlink()
                    and child.name.startswith(f"{STAGE_INTERPOLATED}_")
                ):
                    with contextlib.suppress(OSError):
                        shutil.rmtree(child)
                        removed_any = True
            if removed_any:
                pruned += 1
    return pruned
