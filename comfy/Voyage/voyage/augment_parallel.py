"""Bidirectional parallel model pass: work-stealing chunk deque (DESIGN §140).

Finalize-only, RIFE-only. Two in-process workers share one card pair —
worker A renders from the start on cuda:1 (2060), worker B from the end
on cuda:0 (4060) — meeting in the middle. The same worker renders a
chunk's upscale immediately followed by its interp, so no worker ever
waits on the other's ledger output. Different devices mean different
CUDA default streams, so no per-thread stream plumbing is needed; the
torch-level thread safety (locked cache get-or-load, prepare-once,
per-geometry warp-grid memo) lives in `workers.augment_worker`.

Work-stealing (not static halves): a shared `collections.deque` holds
one task per missing chunk in segment order; A `popleft`s, B `pop`s.
Fast chunks don't idle while slow ones render — whoever finishes first
takes the next task from its own end. Resume is ledger-truth: completed
chunks are exact-key ledger hits, so a retry only re-queues the still
missing chunks.

Ledger safety: `append_chunk_record` takes an `fcntl` exclusive lock, so
concurrent appends to the same `chunks.jsonl` never interleave. Each
worker passes `prune_partials=False` (the driver prunes once upfront —
a per-call sweep would rmtree the other worker's live `.partial` dir).
Seams are never rendered by the workers (the drain fallback owns every
segment joint — seam-early belongs to the single-driver path only).
"""

from __future__ import annotations

import collections
import inspect
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voyage.augment import (
    chunk_frames_match_size,
    chunk_windows,
    interpolated_frame_count,
)
from voyage.augment_sidecar import (
    STAGE_INTERPOLATED,
    STAGE_UPSCALED,
    chunk_output_complete,
    load_chunk_ledger,
    missing_chunk_indexes,
    plan_dir_for_segment,
    prune_stale_partials,
    stage_indexes_matching,
)
from voyage.console import VoyageConsole, optional_bar
from voyage.errors import MediaError

#: Cards for the two workers (A starts at chunk 0, B at the last chunk).
WORKER_A_DEVICE = "cuda:1"
WORKER_B_DEVICE = "cuda:0"

#: Backends eligible for the parallel driver (RIFE-only by design:
#: FILM needs ~5.9 GiB at upscaled size and never fits the 2060).
_PARALLEL_BACKENDS = frozenset({"rife"})


@dataclass
class ParallelChunkTask:
    """One missing chunk: segment identity + chunk window."""

    segment_id: str
    source_key: str
    chunk_index: int
    start: int
    count: int


def parallel_model_pass_armed(*, interp_backend: str, devices: tuple[str, ...]) -> bool:
    """Pure gate: RIFE-only, and both worker cards must be visible.

    `devices` is the visible-device set (e.g. `augment_devices()`).
    1-GPU boxes collapse to a single card, so the gate disarms and the
    single-driver interleaved pass runs instead — same outputs, no
    thread overhead.
    """
    if not isinstance(interp_backend, str) or not isinstance(devices, tuple):
        return False
    visible = set(devices)
    return (
        interp_backend in _PARALLEL_BACKENDS
        and WORKER_A_DEVICE in visible
        and WORKER_B_DEVICE in visible
    )


def _takes_keyword(fn: Callable[..., Any], name: str) -> bool:
    """Probe whether `fn` accepts `name` (explicit param or `**kwargs`)."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def build_parallel_tasks(
    run_dir: Path,
    *,
    weights_key: str,
    out_width: int,
    out_height: int,
    source_fps_key: int,
    upscale_factor: int,
    chunk_frames: int,
    multiplier: int,
    crf: int,
    preset: str,
) -> collections.deque[ParallelChunkTask]:
    """Enumerate every ledger-missing chunk as a stealable deque (segment order).

    Prunes each segment plan dir once upfront (workers pass
    `prune_partials=False`). Ledger-complete chunks are never queued, so
    a retry resumes instead of redoing.
    """
    from voyage.augment_upscale_poller import committed_segment_sources

    segment_sources, _skipped = committed_segment_sources(run_dir)
    ordered = sorted(segment_sources, key=lambda source: source.segment_id)
    tasks: collections.deque[ParallelChunkTask] = collections.deque()
    for source in ordered:
        plan_dir = plan_dir_for_segment(
            run_dir,
            source_key=source.source_key,
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            out_fps=source_fps_key,
            upscale_factor=upscale_factor,
            crf=crf,
            preset=preset,
        )
        plan_dir.mkdir(parents=True, exist_ok=True)
        prune_stale_partials(plan_dir)
        records = load_chunk_ledger(plan_dir / "chunks.jsonl")
        windows = list(chunk_windows(source.total_frames, chunk_frames))
        indexes = list(range(len(windows)))
        # Mirror the pollers: ledger-missing plus exact-window
        # mismatches rejoin, for each stage independently.
        pending: set[int] = (
            set(missing_chunk_indexes(records, indexes, stage=STAGE_UPSCALED))
            | (
                set(indexes)
                - set(
                    stage_indexes_matching(
                        records, windows, stage=STAGE_UPSCALED, chunk_frames=chunk_frames
                    )
                )
            )
            | set(missing_chunk_indexes(records, indexes, stage=STAGE_INTERPOLATED))
            | (
                set(indexes)
                - set(
                    stage_indexes_matching(
                        records, windows, stage=STAGE_INTERPOLATED, chunk_frames=chunk_frames
                    )
                )
            )
        )
        # Output-truth: a ledgered chunk whose output dir is gone
        # or short rejoins instead of deadlocking the skip.
        for index in sorted(set(indexes) - pending):
            _start, count = windows[index]
            up_dir = plan_dir / f"upscaled_{index:02d}"
            ip_dir = plan_dir / f"interpolated_{index:02d}"
            expected_ip = interpolated_frame_count(count, multiplier)
            if not (
                chunk_output_complete(up_dir, count)
                and chunk_frames_match_size(up_dir, (out_width, out_height))
            ) or not (
                chunk_output_complete(ip_dir, expected_ip)
                and chunk_frames_match_size(ip_dir, (out_width, out_height))
            ):
                pending.add(index)
        for index in sorted(pending):
            start, count = windows[index]
            tasks.append(
                ParallelChunkTask(
                    segment_id=source.segment_id,
                    source_key=source.source_key,
                    chunk_index=index,
                    start=start,
                    count=count,
                )
            )
    return tasks


def _preload_worker_models(
    weights: Any, weights_key: str, interp_backend: str, device: str
) -> None:
    """Resident-load this worker's nets once (locked get-or-load in the worker)."""
    from voyage.augment_finalize import _interp_weights_path
    from voyage.workers.augment_worker import (
        _ESRGAN_CACHE,
        _RIFE_CACHE,
        _get_prepared_model,
        _load_esrgan_net,
        _load_rife_net,
        _model_cache_key,
    )

    esrgan_path = weights.realesrgan
    if esrgan_path is None:
        raise MediaError("parallel model pass needs the Real-ESRGAN weights (missing)")
    _get_prepared_model(
        _ESRGAN_CACHE,
        _model_cache_key(esrgan_path, device),
        device,
        lambda: _load_esrgan_net(esrgan_path),
    )
    rife_path = _interp_weights_path(weights, interp_backend)
    if rife_path is None:
        raise MediaError("parallel model pass needs the RIFE weights (missing)")
    _get_prepared_model(
        _RIFE_CACHE,
        _model_cache_key(rife_path, device),
        device,
        lambda: _load_rife_net(rife_path),
    )


def run_parallel_model_pass(
    run_dir: Path,
    *,
    weights: Any,
    weights_key: str,
    out_width: int,
    out_height: int,
    source_fps: float,
    upscale_factor: int,
    multiplier: int,
    chunk_frames: int,
    crf: int,
    preset: str,
    interp_backend: str = "rife",
    upscale_poll_fn: Callable[..., Any] | None = None,
    interp_poll_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
    task_builder: Callable[..., collections.deque[ParallelChunkTask]] | None = None,
) -> None:
    """Render every queued chunk via two work-stealing workers, then settle.

    Worker A (`cuda:1`) steals from the front, worker B (`cuda:0`) from
    the back; each renders its chunk's upscale immediately followed by
    its interp. Settles on the global ledger (every queued chunk must
    show both stages) — anything still missing raises `MediaError`
    fail-loud, never a silent short video. `timings` accumulates into
    the same keys the single driver uses (`upscale_poll_s`,
    `interp_poll_s`, chunk/frame counts), so the elapsed-time report
    never KeyErrors on the parallel path. A `model pass chunks` bar
    tracks every queued chunk (one advance per finished chunk), so the
    default finalize path shows live model-pass progress.
    """
    if upscale_poll_fn is None:
        from voyage.augment_upscale_poller import upscale_poll_once

        upscale_poll_fn = upscale_poll_once
    if interp_poll_fn is None:
        from voyage.augment_interp_poller import interp_poll_once

        interp_poll_fn = interp_poll_once
    from voyage.augment_finalize import _interp_weights_path

    source_fps_key = int(round(source_fps))
    if task_builder is not None:
        tasks = task_builder(
            run_dir,
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            source_fps_key=source_fps_key,
            upscale_factor=upscale_factor,
            chunk_frames=chunk_frames,
            multiplier=multiplier,
            crf=crf,
            preset=preset,
        )
    else:
        tasks = build_parallel_tasks(
            run_dir,
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            source_fps_key=source_fps_key,
            upscale_factor=upscale_factor,
            chunk_frames=chunk_frames,
            multiplier=multiplier,
            crf=crf,
            preset=preset,
        )
    if not tasks:
        return
    interp_weights = _interp_weights_path(weights, interp_backend)
    up_takes_prune = _takes_keyword(upscale_poll_fn, "prune_partials")
    ip_takes_prune = _takes_keyword(interp_poll_fn, "prune_partials")
    up_takes_segments = _takes_keyword(upscale_poll_fn, "segment_ids")
    ip_takes_segments = _takes_keyword(interp_poll_fn, "segment_ids")
    up_takes_chunks = _takes_keyword(upscale_poll_fn, "chunk_ids")
    up_takes_progress = _takes_keyword(upscale_poll_fn, "progress")
    ip_takes_chunks = _takes_keyword(interp_poll_fn, "chunk_ids")
    ip_takes_progress = _takes_keyword(interp_poll_fn, "progress")
    queue_lock = threading.Lock()
    timing_lock = threading.Lock()
    errors: dict[str, BaseException] = {}
    started = time.monotonic()

    def _worker(device: str, from_front: bool, label: str) -> None:
        view: VoyageConsole | None = None
        stream_view = getattr(progress, "stream_view", None)
        if callable(stream_view):
            candidate = stream_view(label)
            if isinstance(candidate, VoyageConsole):
                view = candidate
        try:
            _preload_worker_models(weights, weights_key, interp_backend, device)
        except Exception as exc:  # noqa: BLE001 — collected, re-raised after join
            with queue_lock:
                errors[device] = exc
            return
        while True:
            with queue_lock:
                if errors or not tasks:
                    return
                task = tasks.popleft() if from_front else tasks.pop()
            upscale_seconds = 0.0
            interp_seconds = 0.0
            up_frames = 0
            ip_frames = 0
            try:
                upscale_start = time.monotonic()
                upscale_kwargs: dict[str, Any] = {
                    "weights_path": weights.realesrgan,
                    "weights_key": weights_key,
                    "out_width": out_width,
                    "out_height": out_height,
                    "out_fps": source_fps_key,
                    "upscale_factor": upscale_factor,
                    "chunk_frames": chunk_frames,
                    "device": device,
                    "crf": crf,
                    "preset": preset,
                }
                if up_takes_progress:
                    upscale_kwargs["progress"] = view
                if up_takes_segments:
                    upscale_kwargs["segment_ids"] = [task.segment_id]
                if up_takes_chunks:
                    upscale_kwargs["chunk_ids"] = [task.chunk_index]
                if up_takes_prune:
                    upscale_kwargs["prune_partials"] = False
                upscale_result = upscale_poll_fn(run_dir, **upscale_kwargs)
                upscale_seconds = time.monotonic() - upscale_start
                up_frames = int(getattr(upscale_result, "frames_done", 0) or 0)
                interp_start = time.monotonic()
                interp_kwargs: dict[str, Any] = {
                    "weights_path": interp_weights,
                    "weights_key": weights_key,
                    "out_width": out_width,
                    "out_height": out_height,
                    "out_fps": source_fps_key,
                    "upscale_factor": upscale_factor,
                    "chunk_frames": chunk_frames,
                    "multiplier": multiplier,
                    "device": device,
                    "crf": crf,
                    "preset": preset,
                    "interp_backend": interp_backend,
                }
                if ip_takes_progress:
                    interp_kwargs["progress"] = view
                if ip_takes_segments:
                    interp_kwargs["segment_ids"] = [task.segment_id]
                if ip_takes_chunks:
                    interp_kwargs["chunk_ids"] = [task.chunk_index]
                if ip_takes_prune:
                    interp_kwargs["prune_partials"] = False
                interp_result = interp_poll_fn(run_dir, **interp_kwargs)
                interp_seconds = time.monotonic() - interp_start
                ip_frames = int(getattr(interp_result, "frames_done", 0) or 0)
            except Exception as exc:  # noqa: BLE001 — collected, re-raised after join
                with queue_lock:
                    errors[device] = exc
                    tasks.clear()
                return
            if timings is not None:
                with timing_lock:
                    timings["upscale_poll_s"] = timings.get("upscale_poll_s", 0.0) + upscale_seconds
                    timings["interp_poll_s"] = timings.get("interp_poll_s", 0.0) + interp_seconds
                    timings["upscale_chunks_done"] = timings.get("upscale_chunks_done", 0.0) + 1.0
                    timings["interp_chunks_done"] = timings.get("interp_chunks_done", 0.0) + 1.0
                    timings["upscale_frames_done"] = timings.get(
                        "upscale_frames_done", 0.0
                    ) + float(up_frames)
                    timings["interp_frames_done"] = timings.get("interp_frames_done", 0.0) + float(
                        ip_frames
                    )
            with timing_lock:
                if model_bar is not None:
                    model_bar.update(1)

    bar_ctx = optional_bar(progress, "model pass chunks", total=len(tasks))
    model_bar = bar_ctx.__enter__()
    try:
        thread_a = threading.Thread(
            target=_worker, args=(WORKER_A_DEVICE, True, "model pass A"), daemon=True
        )
        thread_b = threading.Thread(
            target=_worker, args=(WORKER_B_DEVICE, False, "model pass B"), daemon=True
        )
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()
    finally:
        bar_ctx.__exit__(None, None, None)
    if timings is not None:
        with timing_lock:
            timings["parallel_model_pass_s"] = timings.get("parallel_model_pass_s", 0.0) + (
                time.monotonic() - started
            )
    if errors:
        first = next(iter(errors.values()))
        raise first
    _verify_parallel_settle(
        run_dir,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        source_fps_key=source_fps_key,
        upscale_factor=upscale_factor,
        chunk_frames=chunk_frames,
        crf=crf,
        preset=preset,
    )


def _verify_parallel_settle(
    run_dir: Path,
    *,
    weights_key: str,
    out_width: int,
    out_height: int,
    source_fps_key: int,
    upscale_factor: int,
    chunk_frames: int,
    crf: int,
    preset: str,
) -> None:
    """Fail loud when any committed chunk still misses a stage (never ship short)."""
    from voyage.augment_upscale_poller import committed_segment_sources

    segment_sources, _skipped = committed_segment_sources(run_dir)
    missing: list[str] = []
    for source in sorted(segment_sources, key=lambda item: item.segment_id):
        plan_dir = plan_dir_for_segment(
            run_dir,
            source_key=source.source_key,
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            out_fps=source_fps_key,
            upscale_factor=upscale_factor,
            crf=crf,
            preset=preset,
        )
        records = load_chunk_ledger(plan_dir / "chunks.jsonl")
        windows = list(chunk_windows(source.total_frames, chunk_frames))
        indexes = list(range(len(windows)))
        for stage in (STAGE_UPSCALED, STAGE_INTERPOLATED):
            pending = set(missing_chunk_indexes(records, indexes, stage=stage)) | (
                set(indexes)
                - set(
                    stage_indexes_matching(records, windows, stage=stage, chunk_frames=chunk_frames)
                )
            )
            for index in sorted(pending):
                missing.append(f"{source.segment_id}:{stage}:{index}")
    if missing:
        preview = ", ".join(missing[:8])
        raise MediaError(
            f"parallel model pass settled with {len(missing)} chunks missing "
            f"(rerun finalize to resume): {preview}"
        )
