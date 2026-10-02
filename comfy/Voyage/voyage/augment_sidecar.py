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
from voyage.hashing import sha256_text

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
    """Identity of one sidecar chunk (all must match for a ledger hit)."""

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
    """Read the sidecar ledger (missing file → empty; blank lines tolerated)."""
    if not ledger.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            parsed = json.loads(stripped)
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
