"""Independent upscale poller (DESIGN §§56-57, augment plan §140).

Polls committed segments and upscales each missing chunk on the augment
device (`cuda:1` on the 2-GPU box, leaving `cuda:0` to video generation)
without synchronizing with the generation loop. Progress persists in the
durable sidecar ledger (`voyage.augment_sidecar`), so a crash or restart
resumes where it left off: ledgered chunks are skipped, unledgered output
(including `*.partial` leftovers) is re-rendered.

Stdlib only at module scope (supervisor §12 GPU ban): torch/ffmpeg enter
through injected seams (`decode_fn`/`upscale_fn`) or function-local lazy
imports, so this module is import-safe everywhere. The interp poller
consumes the `upscaled_<idx>/` PNG dirs this worker publishes.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voyage import paths
from voyage.augment import chunk_frames_match_size, chunk_windows, interpolated_frame_count
from voyage.augment_sidecar import (
    STAGE_UPSCALED,
    ChunkKey,
    append_chunk_record,
    chunk_mp4_complete,
    chunk_output_complete,
    find_upscaled_donor,
    load_chunk_ledger,
    missing_chunk_indexes,
    plan_dir_for_segment,
    prune_stale_partials,
    stage_indexes_matching,
)
from voyage.errors import MediaError
from voyage.hashing import sha256_file
from voyage.segment_manifest import load_segment_manifest

UPSCALE_STAGE = STAGE_UPSCALED
"""Ledger stage this poller records (shared with the interp poller)."""

DEFAULT_CHUNK_FRAMES = 32
"""Source frames per upscale chunk (mirrors the finalize chunk default)."""

DEFAULT_UPSCALE_FACTOR = 2
"""Pixel upscale factor (2x-only contract: never 4x-native output)."""

DEFAULT_UPSCALE_DEVICE = "cuda:1"
"""Augment device: the 2060, leaving video generation alone on `cuda:0`."""


@dataclass(frozen=True)
class SegmentSource:
    """One pollable committed segment (manifest is truth, never re-probed)."""

    segment_id: str
    segment_dir: Path
    video_path: Path
    source_key: str
    total_frames: int


@dataclass(frozen=True)
class UpscalePollResult:
    """Counts from one poll pass (advisory; the ledger is the truth).

    Frame counts are source frames (one per decoded input frame), so
    upscale and interp legs share one comparable unit for progress bars.
    """

    segments_seen: int
    segments_skipped: int
    chunks_done: int
    chunks_skipped: int
    partials_pruned: int
    frames_done: int = 0
    frames_skipped: int = 0


def _manifest_source(segment_dir: Path) -> SegmentSource | None:
    """Build a SegmentSource from a committed dir, or None when unusable.

    Unusable means: no DONE marker, no `video.mp4`, unreadable manifest,
    wrong manifest format, or no recorded video checksum / frame count.
    The poller skips such segments (finalize triage owns validity).
    """
    try:
        segment_id = segment_dir.name
        done_marker = segment_dir / paths.DONE_MARKER
        video_path = segment_dir / "video.mp4"
        if not done_marker.is_file() or not video_path.is_file():
            return None
        manifest = load_segment_manifest(segment_dir)
        # NOTE: the loader raises MediaError on a present-but-bad manifest,
        # so a returned dict is already format-validated; only sections
        # need checking here.
        checksums = manifest.get("checksums")
        metrics = manifest.get("metrics")
        if not isinstance(checksums, dict) or not isinstance(metrics, dict):
            return None
        source_key = checksums.get("video.mp4")
        total_frames = metrics.get("frames")
        if (
            not isinstance(source_key, str)
            or not source_key
            or isinstance(total_frames, bool)
            or not isinstance(total_frames, int)
            or total_frames < 1
        ):
            return None
        return SegmentSource(
            segment_id=segment_id,
            segment_dir=segment_dir,
            video_path=video_path,
            source_key=source_key,
            total_frames=total_frames,
        )
    except (OSError, ValueError, MediaError):
        return None


def committed_segment_sources(run_dir: Path) -> tuple[list[SegmentSource], int]:
    """Scan `segments/` for pollable segments; returns (sources, skipped)."""
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    sources: list[SegmentSource] = []
    skipped = 0
    if not segments_root.is_dir():
        return (sources, skipped)
    for child in sorted(segments_root.iterdir()):
        if not child.is_dir():
            continue
        source = _manifest_source(child)
        if source is None:
            skipped += 1
        else:
            sources.append(source)
    return (sources, skipped)


def _default_decode_fn(
    source_video: Path, dest_dir: Path, start_frame: int, frame_count: int
) -> list[Path]:
    """Decode one source window to PNGs (frame-exact, fresh-dir guarded)."""
    from voyage.augment import ffmpeg_decode_chunk

    return ffmpeg_decode_chunk(source_video, dest_dir, start_frame, frame_count, fps=None)


def _default_upscale_pngs(
    frame_paths: list[Path],
    dest_dir: Path,
    *,
    weights_path: Path,
    device: str,
    upscale_factor: int,
) -> list[Path]:
    """Upscale decoded PNGs via the resident ESRGAN leg (lazy torch import).

    Factor 1 (native-resolution finalize, DESIGN §140) is a CPU file
    copy: same names, identical bytes, no torch/PIL/model — the slim
    image carries none of them. Other factors take the model path
    (factors outside (1, 2, 4) fail here, before any GPU work).
    """
    if isinstance(upscale_factor, bool) or not isinstance(upscale_factor, int):
        raise TypeError(f"upscale factor must be an int (got {type(upscale_factor).__name__})")
    if upscale_factor not in (1, 2, 4):
        raise ValueError(f"upscale factor must be one of (1, 2, 4) (got {upscale_factor})")
    if upscale_factor == 1:
        # The decode stage already staged the source PNGs in dest_dir
        # (same contract as the model path, which overwrites them in
        # place) — a same-file copy would raise SameFileError, so only
        # copy across distinct paths (e.g. tests staging elsewhere).
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for frame_path in frame_paths:
            dest = dest_dir / frame_path.name
            if dest.resolve() != frame_path.resolve():
                shutil.copyfile(frame_path, dest)
            copied.append(dest)
        return copied
    from voyage.augment import load_png_frames_as_tensors, write_tensors_as_png_frames
    from voyage.workers.augment_worker import upscale_frames

    frames = load_png_frames_as_tensors(frame_paths)
    upscaled = upscale_frames(frames, weights_path, scale=upscale_factor, device=device)
    return write_tensors_as_png_frames(upscaled, dest_dir)


def _chunk_output_dir(plan_dir: Path, index: int) -> Path:
    return plan_dir / f"upscaled_{index:02d}"


def _adopt_upscaled_chunk(
    *,
    run_dir: Path,
    plan_dir: Path,
    ledger_path: Path,
    source_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    chunk_frames: int,
    weights_key: str,
    esrgan_sha: str,
    index: int,
    start: int,
    count: int,
    crf: int,
    preset: str,
) -> bool:
    """Copy a donor plan's upscaled PNGs for one missing chunk (else False).

    Cross-backend resume: the sidecar plan dir is keyed on BOTH weight
    legs, so a backend switch orphans byte-identical upscaled PNGs under
    another dir. The esrgan leg is backend-independent, so a donor whose
    record carries the same esrgan sha is adopted (hardlink, copy
    fallback) and recorded under the CURRENT key — the interp poller
    then proceeds without a re-render. Records with unparseable keys
    never adopt (unknown provenance). The interp leg never adopts.
    """
    donor = find_upscaled_donor(
        run_dir,
        exclude_dir=plan_dir,
        source_key=source_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        upscale_factor=upscale_factor,
        chunk_frames=chunk_frames,
        chunk_index=index,
        start_frame=start,
        source_frames=count,
        esrgan_sha=esrgan_sha,
    )
    if donor is None:
        return False
    output_dir = _chunk_output_dir(plan_dir, index)
    if output_dir.exists():
        # Mirror the render path: unledgered output is incomplete — drop it.
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    donor_dir = run_dir / str(donor["path"])
    try:
        for frame in sorted(donor_dir.glob("frame_*.png")):
            try:
                os.link(frame, output_dir / frame.name)
            except OSError:
                shutil.copy2(frame, output_dir / frame.name)
    except OSError:
        return False
    if not chunk_output_complete(output_dir, count) or not chunk_frames_match_size(
        output_dir, (out_width, out_height)
    ):
        return False
    key = ChunkKey(
        chunk_index=index,
        start_frame=start,
        source_frames=count,
        expected_frames=count,
        upscale_factor=upscale_factor,
        multiplier=1,
        crf=crf,
        preset=preset,
        source_key=source_key,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        chunk_frames=chunk_frames,
    )
    relative = os.path.relpath(output_dir, run_dir).replace(os.sep, "/")
    append_chunk_record(ledger_path, key, stage=STAGE_UPSCALED, path=relative)
    return True


def upscale_poll_once(
    run_dir: Path,
    *,
    weights_path: Path,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int = DEFAULT_UPSCALE_FACTOR,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    device: str = DEFAULT_UPSCALE_DEVICE,
    crf: int = 15,
    preset: str = "veryfast",
    decode_fn: Callable[[Path, Path, int, int], list[Path]] | None = None,
    upscale_fn: Callable[..., list[Path]] | None = None,
    on_chunk: Callable[[str, int, int], None] | None = None,
    on_chunk_frames: Callable[[str, int], None] | None = None,
    segment_ids: list[str] | None = None,
    chunk_ids: list[int] | None = None,
    prune_partials: bool = True,
    sources: list[SegmentSource] | None = None,
    interp_multiplier: int | None = None,
) -> UpscalePollResult:
    """Upscale every missing chunk of every committed segment (one pass).

    Each segment gets its own plan dir (`run/augment/<hash>/`) keyed on
    the segment checksum + weights + geometry + recipe, so a re-rendered
    segment or a settings change never falsely hits old outputs. Returns
    counts; raises `MediaError` on a failed chunk (fail-loud, retry
    re-runs only the missing chunks). `on_chunk`, when given, fires
    after each rendered chunk with `(segment_id, chunk_index,
    chunk_count)` — the finalize model-pass bar advances on it (main
    thread only; the background pre-warm passes None). `on_chunk_frames`,
    when given, fires alongside with `(segment_id, source_frames)` so
    progress bars can count frames instead of chunks. When
    `segment_ids` is given, only those committed segments are polled
    (segment-interleaved pipeline); None (default) polls every segment.
    `sources`, when given, replaces the `committed_segment_sources`
    scan (joint fix videos poll as first-class units through the same
    legs/ledgers); None (default) scans `segments/`. `segment_ids`
    still filters on top of an explicit list.
    When `chunk_ids` is given, only those chunk indexes are rendered by
    this call (parallel workers split disjoint index sets; unowned
    missing chunks count as skipped, never as this worker's `done`).
    None (default) renders every missing chunk.
    `prune_partials`
    (default True) sweeps stale `.partial` dirs per plan dir; the
    parallel driver passes False (it prunes once upfront — a
    per-call sweep would rmtree the other worker's live partial).
    `interp_multiplier`, when given, enables the durable-mp4 fast path:
    chunks with an exact-key `chunk_mp4` record (built with this
    multiplier, mirroring the interp poller's key) plus a non-empty
    `chunk_NN.mp4` skip before any PNG gate — their PNG dirs were
    pruned by design after the mp4 superseded them, so without this
    they would re-render pointlessly. None (default) keeps the legacy
    PNG-truth skip. Drivers pass the interp multiplier guarded by
    `_accepts_keyword`, so older fakes without the param keep working.
    """
    if not weights_key:
        raise ValueError("weights_key must be a non-empty string")
    if interp_multiplier is not None and (
        isinstance(interp_multiplier, bool)
        or not isinstance(interp_multiplier, int)
        or interp_multiplier < 1
    ):
        raise ValueError(
            f"interp_multiplier must be an int >= 1 or None (got {interp_multiplier!r})"
        )
    decode = decode_fn or _default_decode_fn
    upscale = upscale_fn
    try:
        esrgan_sha = sha256_file(weights_path)
    except (OSError, TypeError, ValueError):
        # Unresolvable esrgan file: donor adoption skips silently for
        # this poll (the render path fail-louds on missing weights).
        esrgan_sha = ""
    if sources is None:
        sources, skipped = committed_segment_sources(run_dir)
    else:
        if not isinstance(sources, list):
            raise TypeError(
                f"sources must be a list of SegmentSource or None (got {type(sources).__name__})"
            )
        skipped = 0
    if segment_ids is not None:
        if not isinstance(segment_ids, list) or not all(
            isinstance(item, str) for item in segment_ids
        ):
            raise TypeError("segment_ids must be a list of str or None")
        wanted = set(segment_ids)
        sources = [source for source in sources if source.segment_id in wanted]
    chunks_done = 0
    chunks_skipped = 0
    frames_done = 0
    frames_skipped = 0
    pruned_total = 0
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
        plan_dir.mkdir(parents=True, exist_ok=True)
        if prune_partials:
            pruned_total += prune_stale_partials(plan_dir)
        ledger_path = plan_dir / "chunks.jsonl"
        records = load_chunk_ledger(ledger_path)
        windows = list(chunk_windows(source.total_frames, chunk_frames))
        indexes = list(range(len(windows)))
        # Durable-mp4 fast path (see `interp_multiplier`): mp4-complete
        # chunks skip before any PNG gate — their PNG dirs were pruned by
        # design, so without this they would rejoin missing and
        # re-render pointlessly on every poll.
        if interp_multiplier is not None:
            mp4_done = {
                index
                for index, (start, count) in enumerate(windows)
                if chunk_mp4_complete(
                    plan_dir,
                    records,
                    ChunkKey(
                        chunk_index=index,
                        start_frame=start,
                        source_frames=count,
                        expected_frames=interpolated_frame_count(count, interp_multiplier),
                        upscale_factor=upscale_factor,
                        multiplier=interp_multiplier,
                        crf=crf,
                        preset=preset,
                        source_key=source.source_key,
                        weights_key=weights_key,
                        out_width=out_width,
                        out_height=out_height,
                        out_fps=out_fps * interp_multiplier,
                        chunk_frames=chunk_frames,
                    ),
                )
            }
            if mp4_done:
                chunks_skipped += len(mp4_done)
                frames_skipped += sum(windows[index][1] for index in mp4_done)
                indexes = [index for index in indexes if index not in mp4_done]
        ledger_missing = missing_chunk_indexes(records, indexes, stage=UPSCALE_STAGE)
        # Exact-window guard (see interp poller): a re-tiled plan dir
        # must re-render, never skip wrong outputs.
        ledger_done = set(
            stage_indexes_matching(
                records,
                windows,
                stage=UPSCALE_STAGE,
                chunk_frames=chunk_frames,
            )
        )
        ledger_missing_set = set(ledger_missing) | (set(indexes) - ledger_done)
        # Ledger-truth plus output-truth: a ledgered chunk whose output
        # is gone or short (deletion, corruption, crash after prune)
        # rejoins the missing set instead of deadlocking the skip.
        missing = sorted(ledger_missing_set)
        missing_set = set(missing)
        for index in indexes:
            if index in missing_set:
                continue
            _start, count = windows[index]
            chunk_dir = _chunk_output_dir(plan_dir, index)
            if not chunk_output_complete(chunk_dir, count) or not chunk_frames_match_size(
                chunk_dir, (out_width, out_height)
            ):
                missing.append(index)
                missing_set.add(index)
        missing.sort()
        if chunk_ids is not None:
            if not isinstance(chunk_ids, list) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in chunk_ids
            ):
                raise TypeError("chunk_ids must be a list of int or None")
            owned = set(chunk_ids)
            missing = [index for index in missing if index in owned]
            missing_set = set(missing)
        chunks_skipped += len(indexes) - len(missing)
        frames_skipped += sum(windows[index][1] for index in indexes if index not in missing_set)
        for index in missing:
            start, count = windows[index]
            output_dir = _chunk_output_dir(plan_dir, index)
            if _adopt_upscaled_chunk(
                run_dir=run_dir,
                plan_dir=plan_dir,
                ledger_path=ledger_path,
                source_key=source.source_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=out_fps,
                upscale_factor=upscale_factor,
                chunk_frames=chunk_frames,
                weights_key=weights_key,
                esrgan_sha=esrgan_sha,
                index=index,
                start=start,
                count=count,
                crf=crf,
                preset=preset,
            ):
                records = load_chunk_ledger(ledger_path)
                chunks_done += 1
                frames_done += count
                if on_chunk is not None:
                    on_chunk(source.segment_id, index, len(windows))
                if on_chunk_frames is not None:
                    on_chunk_frames(source.segment_id, count)
                continue
            if output_dir.exists():
                # Ledger-truth rule: unledgered output is incomplete — drop it.
                if output_dir.is_dir() and not output_dir.is_symlink():
                    shutil.rmtree(output_dir)
                else:
                    output_dir.unlink()
            partial_dir = plan_dir / f"upscaled_{index:02d}.partial"
            if partial_dir.exists():
                if partial_dir.is_dir() and not partial_dir.is_symlink():
                    shutil.rmtree(partial_dir)
                else:
                    partial_dir.unlink()
            decoded = decode(source.video_path, partial_dir, start, count)
            if len(decoded) != count:
                raise MediaError(
                    f"upscale decode delivered {len(decoded)} frames, "
                    f"expected {count} ({source.segment_id} chunk {index})"
                )
            if upscale is None:
                written = _default_upscale_pngs(
                    decoded,
                    partial_dir,
                    weights_path=weights_path,
                    device=device,
                    upscale_factor=upscale_factor,
                )
            else:
                written = upscale(decoded, partial_dir)
            if len(written) != count:
                raise MediaError(
                    f"upscale rendered {len(written)} frames, "
                    f"expected {count} ({source.segment_id} chunk {index})"
                )
            for frame_file in written:
                if not frame_file.is_file():
                    raise MediaError(
                        f"upscale output missing: {frame_file} ({source.segment_id} chunk {index})"
                    )
            os.replace(partial_dir, output_dir)
            key = ChunkKey(
                chunk_index=index,
                start_frame=start,
                source_frames=count,
                expected_frames=count,
                upscale_factor=upscale_factor,
                multiplier=1,
                crf=crf,
                preset=preset,
                source_key=source.source_key,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=out_fps,
                chunk_frames=chunk_frames,
            )
            relative = os.path.relpath(output_dir, run_dir).replace(os.sep, "/")
            append_chunk_record(ledger_path, key, stage=UPSCALE_STAGE, path=relative)
            records = load_chunk_ledger(ledger_path)
            chunks_done += 1
            frames_done += count
            if on_chunk is not None:
                on_chunk(source.segment_id, index, len(windows))
            if on_chunk_frames is not None:
                on_chunk_frames(source.segment_id, count)
    return UpscalePollResult(
        segments_seen=len(sources),
        segments_skipped=skipped,
        chunks_done=chunks_done,
        chunks_skipped=chunks_skipped,
        partials_pruned=pruned_total,
        frames_done=frames_done,
        frames_skipped=frames_skipped,
    )
