"""Independent interp poller (DESIGN §§56-57, augment plan §140).

Polls the plan dirs published by the upscale poller and interpolates each
upscaled chunk with FILM on the augment device (`cuda:1` on the 2-GPU box,
leaving `cuda:0` to video generation) without synchronizing with the
generation loop. Chunks without upscaled work wait; completed chunks are
skipped via the shared sidecar ledger (`voyage.augment_sidecar`), so a
crash or restart resumes where it left off.

The interp poller re-derives the upscale plan dir (same `plan_hash_for`
call with `multiplier=1`: upscale outputs are multiplier-independent, so
different interp settings reuse them) and records `interpolated` entries
with the real multiplier alongside the `upscaled` ones.

Stdlib only at module scope (supervisor §12 GPU ban): torch enters
through the injected `interp_fn` seam or a function-local lazy import.

NOTE on co-tenancy: this worker and the upscale poller both default to
`cuda:1`. They are independent of *generation*, not of each other — run
them staggered (upscale drains first, interp follows) or schedule them so
both residents never exceed the 6 GB card (FILM pairs at the upscaled size
need ~5.9 GiB; see `augment_worker` VRAM notes).
"""

from __future__ import annotations

import inspect
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voyage.augment import chunk_windows, interpolated_frame_count
from voyage.augment_sidecar import (
    STAGE_INTERPOLATED,
    STAGE_UPSCALED,
    ChunkKey,
    append_chunk_record,
    chunk_output_complete,
    completed_stages,
    load_chunk_ledger,
    missing_chunk_indexes,
    plan_dir_for_segment,
    prune_stale_partials,
    stage_indexes_matching,
)
from voyage.augment_upscale_poller import committed_segment_sources
from voyage.errors import MediaError

INTERP_STAGE = STAGE_INTERPOLATED
"""Ledger stage this poller records (shared with the upscale poller)."""

DEFAULT_CHUNK_FRAMES = 32
"""Source frames per chunk; must match the upscale poller's value (windows align)."""

DEFAULT_INTERP_MULTIPLIER = 4
"""FILM interpolation factor (4x video-export recipe)."""

DEFAULT_INTERP_DEVICE = "cuda:1"
"""Augment device: the 2060, leaving video generation alone on `cuda:0`."""

DEFAULT_UPSCALE_FACTOR = 2
"""Pixel upscale factor recorded on interp keys (2x-only contract: never 4x-native)."""


@dataclass(frozen=True)
class InterpPollResult:
    """Counts from one poll pass (advisory; the ledger is the truth)."""

    segments_seen: int
    segments_skipped: int
    chunks_done: int
    chunks_skipped: int
    chunks_waiting: int
    partials_pruned: int


def _upscaled_frame_paths(plan_dir: Path, index: int) -> list[Path]:
    """Sorted upscaled PNGs for one chunk (empty when the dir is absent)."""
    output_dir = plan_dir / f"upscaled_{index:02d}"
    if not output_dir.is_dir():
        return []
    return sorted(output_dir.glob("frame_*.png"))


def _default_interp_pngs(
    frame_paths: list[Path],
    dest_dir: Path,
    multiplier: int,
    *,
    weights_path: Path,
    device: str,
    on_pair: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Interpolate PNGs via the resident FILM leg (lazy torch import).

    `on_pair`, when given, fires after each finished frame pair with
    `(pair_index, pair_count)` so the finalize bar advances live inside
    long chunks instead of jumping once per chunk at the end.
    """
    from voyage.augment import load_png_frames_as_tensors, write_tensors_as_png_frames
    from voyage.workers.augment_worker import interpolate_pair

    frames = load_png_frames_as_tensors(frame_paths)
    if len(frames) <= 1 or multiplier <= 1:
        return write_tensors_as_png_frames(frames, dest_dir)
    moments = [(position + 1) / multiplier for position in range(multiplier - 1)]
    pair_count = len(frames) - 1
    blended: list[object] = []
    for position in range(pair_count):
        blended.append(frames[position])
        for moment in moments:
            blended.append(
                interpolate_pair(
                    frames[position],
                    frames[position + 1],
                    weights_path,
                    moment=moment,
                    device=device,
                )
            )
        if on_pair is not None:
            on_pair(position, pair_count)
    blended.append(frames[-1])
    return write_tensors_as_png_frames(blended, dest_dir)


def _supports_on_pair(interp_fn: Callable[..., list[Path]]) -> bool | None:
    """Whether an injected `interp_fn` takes an `on_pair` keyword.

    True: call with `on_pair`. False: legacy 3-arg call. None: the
    callable is not introspectable (built-in, mock) — the caller tries
    the 4-arg form first and falls back on `TypeError`.
    """
    try:
        parameters = inspect.signature(interp_fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    return any(
        parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and parameter.name == "on_pair"
        for parameter in parameters
    )


def interp_poll_once(
    run_dir: Path,
    *,
    weights_path: Path,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int = DEFAULT_UPSCALE_FACTOR,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    multiplier: int = DEFAULT_INTERP_MULTIPLIER,
    device: str = DEFAULT_INTERP_DEVICE,
    crf: int = 15,
    preset: str = "veryfast",
    interp_fn: Callable[[list[Path], Path, int], list[Path]] | None = None,
    on_chunk: Callable[[str, int, int], None] | None = None,
    on_pair_frames: Callable[[str, float], None] | None = None,
) -> InterpPollResult:
    """Interpolate every upscaled-but-not-interpolated chunk (one pass).

    Each segment re-derives its upscale plan dir (plan hash with
    `multiplier=1`); chunks whose `upscaled` ledger entry is missing wait
    for the upscale poller. `out_fps` is the committed-segment (source)
    fps — shared with the upscale poller so both derive the same plan
    dir; interp ledger keys record `out_fps * multiplier` (the actual
    output fps). Returns counts; raises `MediaError` on a failed chunk
    (fail-loud, retry re-renders only the missing chunks). `weights_key`
    must cover both legs (ESRGAN + FILM): any leg change must miss old
    records. `on_chunk`, when given, fires after each rendered chunk
    with `(segment_id, chunk_index, chunk_count)`. `on_pair_frames`,
    when given, fires per finished FILM pair with `(segment_id,
    fractional_source_frames)` so long chunks advance the finalize bar
    live instead of jumping once at the end. An injected `interp_fn`
    that takes an `on_pair` keyword gets it (probed via
    `inspect.signature`, with a try-and-fall-back for unintrospectable
    callables); a legacy 3-arg `interp_fn` renders the whole chunk
    before the single `on_pair_frames` advance at chunk end.
    """
    if not weights_key:
        raise ValueError("weights_key must be a non-empty string")
    if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier < 1:
        raise ValueError(f"multiplier must be an int >= 1 (got {multiplier!r})")
    sources, skipped = committed_segment_sources(run_dir)
    chunks_done = 0
    chunks_skipped = 0
    chunks_waiting = 0
    pruned_total = 0
    final_fps = out_fps * multiplier
    for source in sources:
        plan_dir = plan_dir_for_segment(
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
        if not plan_dir.is_dir():
            chunks_waiting += len(list(chunk_windows(source.total_frames, chunk_frames)))
            continue
        pruned_total += prune_stale_partials(plan_dir)
        ledger_path = plan_dir / "chunks.jsonl"
        records = load_chunk_ledger(ledger_path)
        done = completed_stages(records)
        windows = list(chunk_windows(source.total_frames, chunk_frames))
        indexes = list(range(len(windows)))
        ledger_ready = [index for index in indexes if STAGE_UPSCALED in done.get(index, set())]
        # Output-truth on the upscaled inputs: a ledgered chunk whose
        # upscaled output is gone or short waits (the upscale poller
        # heals it) instead of failing loud here.
        ready = [
            index
            for index in ledger_ready
            if chunk_output_complete(plan_dir / f"upscaled_{index:02d}", windows[index][1])
        ]
        chunks_waiting += len(indexes) - len(ready)
        ready_set = set(ready)
        ledger_missing = missing_chunk_indexes(
            [record for record in records if record.get("stage") == INTERP_STAGE],
            ready,
            stage=INTERP_STAGE,
        )
        # Exact-window guard: an index counts as ledgered only when its
        # record tiles the same `(start, count)` window the current
        # `chunk_frames` produces (a re-tiled plan dir must re-render,
        # never skip wrong outputs).
        ledger_done = (
            set(
                stage_indexes_matching(
                    [record for record in records if record.get("stage") == INTERP_STAGE],
                    windows,
                    stage=INTERP_STAGE,
                    chunk_frames=chunk_frames,
                )
            )
            & ready_set
        )
        ledger_missing_set = set(ledger_missing) | (ready_set - ledger_done)
        # Ledger-truth plus output-truth on our own outputs: a ledgered
        # chunk whose interpolated dir is gone or short rejoins missing.
        missing = sorted(ledger_missing_set)
        missing_set = set(missing)
        for index in ready:
            if index in missing_set:
                continue
            expected = interpolated_frame_count(windows[index][1], multiplier)
            if not chunk_output_complete(plan_dir / f"interpolated_{index:02d}", expected):
                missing.append(index)
                missing_set.add(index)
        missing.sort()
        chunks_skipped += len(ready) - len(missing)
        for index in missing:
            start, count = windows[index]
            expected = interpolated_frame_count(count, multiplier)
            upscaled_paths = _upscaled_frame_paths(plan_dir, index)
            if len(upscaled_paths) != count:
                # The upscale input is corrupt or was pruned mid-round:
                # wait for the upscale poller to heal it instead of
                # failing loud (one corrupt chunk must not abort the
                # whole finalize; the stuck-detector still fires when
                # nothing progresses at all).
                chunks_waiting += 1
                continue
            output_dir = plan_dir / f"interpolated_{index:02d}"
            if output_dir.exists():
                # Ledger-truth rule: unledgered output is incomplete — drop it.
                if output_dir.is_dir() and not output_dir.is_symlink():
                    shutil.rmtree(output_dir)
                else:
                    output_dir.unlink()
            partial_dir = plan_dir / f"interpolated_{index:02d}.partial"
            if partial_dir.exists():
                if partial_dir.is_dir() and not partial_dir.is_symlink():
                    shutil.rmtree(partial_dir)
                else:
                    partial_dir.unlink()
            pair_advance = count / max(count - 1, 1)
            pairs_fired = 0

            def _fire_pair(
                _pair_index: int,
                _pair_count: int,
                _segment_id: str = source.segment_id,
                _advance: float = pair_advance,
            ) -> None:
                nonlocal pairs_fired
                pairs_fired += 1
                if on_pair_frames is not None:
                    on_pair_frames(_segment_id, _advance)

            if interp_fn is None:
                written = _default_interp_pngs(
                    upscaled_paths,
                    partial_dir,
                    multiplier,
                    weights_path=weights_path,
                    device=device,
                    on_pair=_fire_pair if on_pair_frames is not None else None,
                )
            elif _supports_on_pair(interp_fn) is False:
                written = interp_fn(upscaled_paths, partial_dir, multiplier)
            elif _supports_on_pair(interp_fn) is True:
                written = interp_fn(
                    upscaled_paths,
                    partial_dir,
                    multiplier,
                    on_pair=_fire_pair,  # type: ignore[call-arg]
                )
            else:
                # Not introspectable: try the 4-arg form first so mocks
                # and builtins keep working, fall back to 3-arg.
                try:
                    written = interp_fn(
                        upscaled_paths,
                        partial_dir,
                        multiplier,
                        on_pair=_fire_pair,  # type: ignore[call-arg]
                    )
                except TypeError:
                    written = interp_fn(upscaled_paths, partial_dir, multiplier)
            if len(written) != expected:
                raise MediaError(
                    f"interp rendered {len(written)} frames, "
                    f"expected {expected} ({source.segment_id} chunk {index})"
                )
            for frame_file in written:
                if not frame_file.is_file():
                    raise MediaError(
                        f"interp output missing: {frame_file} ({source.segment_id} chunk {index})"
                    )
            os.replace(partial_dir, output_dir)
            key = ChunkKey(
                chunk_index=index,
                start_frame=start,
                source_frames=count,
                expected_frames=expected,
                upscale_factor=upscale_factor,
                multiplier=multiplier,
                crf=crf,
                preset=preset,
                source_key=source.source_key,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=final_fps,
                chunk_frames=chunk_frames,
            )
            relative = os.path.relpath(output_dir, run_dir).replace(os.sep, "/")
            append_chunk_record(ledger_path, key, stage=INTERP_STAGE, path=relative)
            records = load_chunk_ledger(ledger_path)
            done = completed_stages(records)
            chunks_done += 1
            if on_chunk is not None:
                on_chunk(source.segment_id, index, len(windows))
            if on_pair_frames is not None and pairs_fired == 0:
                # No live pair fired (single-frame passthrough or a
                # legacy 3-arg `interp_fn`): advance the whole chunk at
                # once so the bar still reaches its total.
                on_pair_frames(source.segment_id, float(count))
    return InterpPollResult(
        segments_seen=len(sources),
        segments_skipped=skipped,
        chunks_done=chunks_done,
        chunks_skipped=chunks_skipped,
        chunks_waiting=chunks_waiting,
        partials_pruned=pruned_total,
    )
