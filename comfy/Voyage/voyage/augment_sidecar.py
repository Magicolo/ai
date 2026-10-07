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
import fcntl
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from voyage.atomic import atomic_write_bytes, fsync_dir
from voyage.augment import chunk_frames_match_size
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

STAGE_CHUNK_MP4 = "chunk_mp4"
"""Ledger stage: the durable per-chunk mp4 is encoded (PNG-independent).

Appended by `ensure_chunk_mp4` (drain) with the SAME `ChunkKey` as the
chunk's interp record, right before both PNG dirs are pruned: the mp4
supersedes ~20x its PNG weight, and an exact-key record plus a
non-empty `chunk_NN.mp4` lets every poller skip the chunk before any
PNG-count gate (pruned PNGs never rejoin the missing set).
"""

_KNOWN_STAGES = frozenset({STAGE_UPSCALED, STAGE_INTERPOLATED, STAGE_CHUNK_MP4})


def list_plan_dirs(run_dir: Path) -> list[Path]:
    """Sorted `augment/` subdirs, [] when absent/unreadable (issue 252).

    Sweep, heal, and GC share this enumeration instead of each
    re-walking the tree. Per-caller filters (symlink skips,
    `morph_joints`, hash-name checks, ledger presence) stay at the
    call sites — only the walk is shared, so behavior is unchanged.
    """
    try:
        return sorted(child for child in (run_dir / AUGMENT_DIRNAME).iterdir() if child.is_dir())
    except OSError:
        return []


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
        # Exclusive lock: parallel finalize workers may append to the
        # same ledger concurrently (flock releases on close regardless).
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
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


def chunk_mp4_path(plan_dir: Path, chunk_index: int) -> Path:
    """Durable chunk-mp4 path (the per-chunk encoded artifact)."""
    return plan_dir / f"chunk_{chunk_index:02d}.mp4"


def _probe_mp4_frames(mp4: Path) -> int | None:
    """Probed presented frames for one chunk mp4, None when unprobable.

    Header `nb_frames` first (no decode), decode-count fallback — the
    same shape as `media.presented_frames` but stdlib-local (this module
    stays torch/ffmpeg-wrapper free; failures read as None, never raise).
    """
    try:
        import subprocess as _subprocess

        header = _subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=nw=1:nk=1",
                str(mp4),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if header.returncode == 0:
        try:
            count = int(header.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            count = 0
        if count > 0:
            return count
    try:
        import subprocess as _subprocess

        proc = _subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nw=1:nk=1",
                str(mp4),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def chunk_mp4_file_complete(plan_dir: Path, chunk_index: int) -> bool:
    """Whether the durable chunk mp4 exists and is non-empty (pure file truth)."""
    try:
        mp4 = chunk_mp4_path(plan_dir, chunk_index)
        return mp4.is_file() and mp4.stat().st_size > 0
    except OSError:
        return False


def chunk_mp4_frames_match(plan_dir: Path, key: ChunkKey) -> bool:
    """Whether the durable chunk mp4 probes to `key.expected_frames` (Track C).

    File-exists + non-empty is not enough: a truncated encode (kill
    mid-concat, torn chunk) probes short and must re-render. Unprobable
    (no ffprobe in the test image) reads as True when the file is
    non-empty — the count gate only hardens environments with ffprobe,
    never unit tests/fakes. Mismatch reads as missing (caller treats it
    as incomplete, never as complete).
    """
    if not chunk_mp4_file_complete(plan_dir, key.chunk_index):
        return False
    probed = _probe_mp4_frames(chunk_mp4_path(plan_dir, key.chunk_index))
    if probed is None:
        return True
    return probed == int(key.expected_frames)


def chunk_mp4_complete(plan_dir: Path, records: list[dict[str, Any]], key: ChunkKey) -> bool:
    """Whether chunk `key` needs no further PNG work (record + file truth).

    Complete means an exact-key `chunk_mp4`-stage record exists (the same
    `ChunkKey` the interp record carries, so any settings change still
    re-renders) plus a non-empty `chunk_NN.mp4` beside it that probes to
    `key.expected_frames` (Track C: a truncated encode probes short and
    re-renders — file-exists alone once shipped short chunks silently).
    Pollers check this BEFORE any PNG-count gate: pruned PNG dirs must
    skip, never rejoin the missing set. The drain encodes through
    `ensure_chunk_mp4` instead of checking this directly (it must also
    produce the mp4).
    """
    if not chunk_cache_hit(records, key, stage=STAGE_CHUNK_MP4):
        return False
    if not chunk_mp4_file_complete(plan_dir, key.chunk_index):
        return False
    return chunk_mp4_frames_match(plan_dir, key)


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
            donor_dir = run_dir / rel
            if not chunk_output_complete(donor_dir, source_frames):
                continue
            if not chunk_frames_match_size(donor_dir, (out_width, out_height)):
                continue
            return record
    return None


def _strip_stale_interpolated_records(ledger: Path, *, other_sha: str) -> bool:
    """Strip stale interpolated-stage records from a pruned ledger (else False).

    The `whole_dir=False` prune deletes `interpolated_*` outputs but leaves
    the ledger — without this strip the ledger keeps records for pixels
    that no longer exist, and the next restart's sidecar validator reports
    `ledgered but output missing` and aborts before the pollers (which
    would otherwise heal the gap by re-rendering) ever run. Upscaled
    records stay untouched: they remain valid cross-backend donors. Only
    records whose interp leg keys to `other_sha` are stripped; anything
    unparseable or foreign stays (never delete unknown provenance).
    `chunk_mp4`-stage records keyed to the stale leg go with the
    interpolated ones (their pixels are the stale backend's); the prune
    deletes the matching `chunk_*.mp4` files alongside the
    `interpolated_*` dirs.
    Atomic rewrite (temp + fsync + replace + fsync_dir); OSError →
    False (best-effort, retried on the next prune pass).
    """
    try:
        raw_lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    kept: list[str] = []
    stripped_any = False
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            kept.append(line)
            continue
        if (
            isinstance(record, dict)
            and record.get("stage") in (STAGE_INTERPOLATED, STAGE_CHUNK_MP4)
            and _interp_sha_from_weights_key(record.get("weights_key")) == other_sha
        ):
            stripped_any = True
            continue
        kept.append(line)
    if not stripped_any:
        return False
    payload = ("".join(line + "\n" for line in kept)).encode("utf-8")
    try:
        atomic_write_bytes(ledger, payload)
    except OSError:
        return False
    return True


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
    `interpolated_*` output subdirs and the `chunk_*.mp4` files are
    deleted (the upscaled pixels and the dir itself stay — the dir
    remains a valid upscale donor for cross-key adoption and the interp
    leg re-renders into it) and the orphaned `interpolated`- and
    `chunk_mp4`-stage ledger records are stripped
    (same atomic-rewrite durability as every ledger append; upscaled
    records stay, so donors survive). Without the strip the ledger
    would keep pointing at deleted pixels and the next restart's
    sidecar validator would abort before the pollers heal it). With
    `whole_dir=True` the whole plan dir goes.
    Use False at finalize start (frees the bulk under disk pressure
    while keeping donors) and True pre-publish once every poll and
    drain on every path has finished.

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
            # Free the bulk now, keep the dir and its donors: the
            # upscaled pixels stay (backend-independent) and the
            # upscaled ledger records stay with them, while the
            # orphaned interpolated records are stripped (a ledger
            # pointing at deleted pixels would abort the next
            # restart's validation before the pollers heal it).
            # The strip is NOT gated on removed_any: a previous
            # prune (or any external deletion) may already have
            # taken the outputs, leaving a dangling ledger behind.
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
                elif (
                    child.is_file()
                    and not child.is_symlink()
                    and child.name.startswith("chunk_")
                    and child.name.endswith(".mp4")
                ):
                    # Stale-leg chunk mp4s (and their `.partial.mp4`
                    # leftovers): encoded from the deleted interp pixels,
                    # so they must go with them — otherwise the drain
                    # would concat the wrong backend's frames.
                    with contextlib.suppress(OSError):
                        child.unlink()
                        removed_any = True
            stripped = _strip_stale_interpolated_records(
                plan_dir / CHUNKS_LEDGER_FILENAME, other_sha=other_sha
            )
            if removed_any or stripped:
                pruned += 1
    return pruned


def _strip_dangling_chunk_records(ledger: Path) -> int:
    """Drop chunk records whose outputs are missing or incomplete (DESIGN §56).

    Mirrors `_check_sidecar_plan_consistency` in `voyage/cli_validate.py`
    exactly: groups records by latest-per-`(chunk_index, stage)` among the
    known stages, resolves the output dir as
    `ledger.parent / f"{stage}_{index:02d}"`, and requires
    `chunk_output_complete` for the latest record's `expected_frames`.
    Two `chunk_mp4`-stage exceptions: those groups resolve to
    `chunk_NN.mp4` file truth instead of a PNG dir, and PNG-stage groups
    of a chunk with a complete durable mp4 are exempt (their PNG dirs
    were pruned by design after the mp4 superseded them).
    Records whose frames additionally fail `chunk_frames_match_size`
    against the record's own `out_width`/`out_height` (valid ints only)
    are stripped too — donor adoption once propagated a mixed-geometry
    chunk whose counts were intact, and no validator sees that. The whole
    group goes (all records of a group share one output dir), while
    unparseable/foreign lines and records the validator skips are kept.
    Rewrite is atomic via `atomic_write_bytes`; returns removed count.
    `chunk_frames_match_size` fails open without PIL, so slim-image runs
    strip on count-truth only (unit tests and fakes unaffected).
    """
    try:
        text = ledger.read_text(encoding="utf-8")
    except OSError:
        return 0
    raw_lines = text.splitlines()
    lines: list[str] = []
    parsed: list[dict[str, Any] | None] = []
    for raw in raw_lines:
        if not raw.strip():
            continue
        lines.append(raw)
        try:
            record = json.loads(raw)
        except ValueError:
            parsed.append(None)
            continue
        parsed.append(record if isinstance(record, dict) else None)

    def _group(record: dict[str, Any]) -> tuple[int, str] | None:
        try:
            index = int(record["chunk_index"])
            expected = int(record["expected_frames"])
        except (KeyError, TypeError, ValueError):
            return None
        stage = record.get("stage")
        if stage not in _KNOWN_STAGES or expected <= 0:
            return None
        return (index, stage)

    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for record in parsed:
        if record is None:
            continue
        group = _group(record)
        if group is not None:
            latest[group] = record
    dangling: set[tuple[int, str]] = set()
    # Chunks with a complete durable mp4 keep their PNG-stage records:
    # the PNG dirs were pruned BY DESIGN after the mp4 superseded them,
    # so PNG-missing must not read as dangling (the pollers skip
    # mp4-complete chunks before any PNG gate for the same reason).
    mp4_exempt = {
        index
        for (index, stage) in latest
        if stage == STAGE_CHUNK_MP4 and chunk_mp4_file_complete(ledger.parent, index)
    }
    for group, record in latest.items():
        index, stage = group
        if stage == STAGE_CHUNK_MP4:
            if not chunk_mp4_file_complete(ledger.parent, index):
                dangling.add(group)
            continue
        if index in mp4_exempt:
            continue
        expected = int(record["expected_frames"])
        chunk_dir = ledger.parent / f"{stage}_{index:02d}"
        if not chunk_output_complete(chunk_dir, expected):
            dangling.add(group)
            continue
        try:
            size = (int(record["out_width"]), int(record["out_height"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not chunk_frames_match_size(chunk_dir, size):
            dangling.add(group)
    if not dangling:
        return 0
    kept = [
        line
        for line, record in zip(lines, parsed, strict=True)
        if record is None or _group(record) not in dangling
    ]
    removed = len(lines) - len(kept)
    if not removed:
        return 0
    payload = ("".join(line + "\n" for line in kept)).encode("utf-8")
    try:
        atomic_write_bytes(ledger, payload)
    except OSError:
        return 0
    return removed


def heal_augment_ledgers(run_dir: Path) -> int:
    """Strip dangling chunk records in every plan ledger; return removed count.

    Self-healing hook for generate/finalize start (the kaolin case: healed
    outputs were deleted while their ledger records stayed, and the
    pre-finalize validator aborts on ledgered-but-missing instead of
    re-rendering). Stripped records fail the pollers' output-truth gates,
    so the next pass re-renders them. Never raises — per-plan-dir
    `OSError` is skipped so one unreadable plan never blocks the sweep.
    """
    stripped = 0
    for plan_dir in list_plan_dirs(run_dir):
        ledger = plan_dir / CHUNKS_LEDGER_FILENAME
        if not ledger.is_file():
            continue
        try:
            stripped += _strip_dangling_chunk_records(ledger)
        except OSError:
            continue
    return stripped
