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
from voyage.augment import chunk_windows
from voyage.augment_sidecar import (
    STAGE_UPSCALED,
    ChunkKey,
    append_chunk_record,
    chunk_output_complete,
    load_chunk_ledger,
    missing_chunk_indexes,
    plan_dir_for_segment,
    prune_stale_partials,
    stage_indexes_matching,
)
from voyage.errors import MediaError
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
    """Counts from one poll pass (advisory; the ledger is the truth)."""

    segments_seen: int
    segments_skipped: int
    chunks_done: int
    chunks_skipped: int
    partials_pruned: int


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
    progress bars can count frames instead of chunks.
    """
    if not weights_key:
        raise ValueError("weights_key must be a non-empty string")
    decode = decode_fn or _default_decode_fn
    upscale = upscale_fn
    sources, skipped = committed_segment_sources(run_dir)
    chunks_done = 0
    chunks_skipped = 0
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
        pruned_total += prune_stale_partials(plan_dir)
        ledger_path = plan_dir / "chunks.jsonl"
        records = load_chunk_ledger(ledger_path)
        windows = list(chunk_windows(source.total_frames, chunk_frames))
        indexes = list(range(len(windows)))
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
            if not chunk_output_complete(_chunk_output_dir(plan_dir, index), count):
                missing.append(index)
                missing_set.add(index)
        missing.sort()
        chunks_skipped += len(indexes) - len(missing)
        for index in missing:
            start, count = windows[index]
            output_dir = _chunk_output_dir(plan_dir, index)
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
            if on_chunk is not None:
                on_chunk(source.segment_id, index, len(windows))
    return UpscalePollResult(
        segments_seen=len(sources),
        segments_skipped=skipped,
        chunks_done=chunks_done,
        chunks_skipped=chunks_skipped,
        partials_pruned=pruned_total,
    )
