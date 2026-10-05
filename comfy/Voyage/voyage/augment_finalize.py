"""Durable finalize model pass over sidecar plan dirs (DESIGN §§56-57, §140).

Finalize-side orchestration for the independent augment workers: poll the
upscale worker (cuda:1 ESRGAN) and the interp worker (cuda:1 FILM) to
completion, drain one intermediate per usable segment, and concat them in
segment order. Replaces the all-or-nothing `TemporaryDirectory`
`run_finalize_model_pass` flow — every expensive chunk persists under
`run_dir/augment/<plan-hash>/` with a ledger record, so a crashed finalize
retries only the missing chunks.

Stdlib only at module scope (supervisor §12 GPU ban): the pollers, drain,
and ffmpeg enter through injected seams or function-local lazy imports.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voyage.augment_seam import render_seam_once, seam_endpoints, seam_plan_dir
from voyage.augment_sidecar import plan_dir_for_segment
from voyage.console import optional_bar, optional_stage
from voyage.errors import MediaError
from voyage.hashing import sha256_file

if TYPE_CHECKING:
    from voyage.console import VoyageConsole

FINAL_INTERMEDIATE_FILENAME = "model_intermediate.mp4"
"""Concat output in the caller's work dir (cheap stream copy, redone per finalize)."""

_MAX_POLL_PASSES = 10
"""Poll-loop cap: each pass must finish chunks or the run is stuck (fail-loud)."""


def weights_key_for(weights: Any) -> str:
    """Ledger key covering both model legs (any leg change misses old records)."""
    film = weights.film
    realesrgan = weights.realesrgan
    if film is None or realesrgan is None:
        raise ValueError(
            "durable model pass needs both legs provisioned "
            f"(film={film!r}, realesrgan={realesrgan!r})"
        )
    return f"{sha256_file(film)}|{sha256_file(realesrgan)}"


def _require_run_dir(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"run_dir must be a Path (got {type(value).__name__})")
    return value


def _require_usable(value: list[Path]) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"usable needs at least one segment dir (got {value!r})")
    for segment in value:
        if not isinstance(segment, Path):
            raise TypeError(f"usable entries must be Paths (got {type(segment).__name__})")
    return value


def _require_box(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int (got {value!r})")
    return value


def _require_fps(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number (got {type(value).__name__})")
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{name} must be finite and > 0 (got {value!r})")
    return rate


def _require_multiplier(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"multiplier must be an int >= 1 (got {value!r})")
    return value


def _require_factor(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 4):
        raise ValueError(f"upscale_factor must be one of (1, 2, 4) (got {value!r})")
    return value


def _require_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1 (got {value!r})")
    return value


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string (got {value!r})")
    return value


def _poll_to_completion(
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
    device: str,
    crf: int,
    preset: str,
    upscale_poll_fn: Callable[..., Any] | None,
    interp_poll_fn: Callable[..., Any] | None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
) -> None:
    """Run both pollers until a full pass finishes nothing (fail-loud when stuck).

    `progress` renders one determinate `model-pass chunks` bar: both legs
    cover the same chunk windows 1:1, so the settled total is twice the
    per-leg count (learned after the first pass from done + skipped +
    interp-waiting) and every rendered chunk advances it via the pollers'
    `on_chunk` callbacks. A re-rendered chunk (output-truth rejoin)
    fires again but advances once — the seen-set dedupes it.
    """
    if upscale_poll_fn is None:
        from voyage.augment_upscale_poller import upscale_poll_once

        upscale_poll_fn = upscale_poll_once
    if interp_poll_fn is None:
        from voyage.augment_interp_poller import interp_poll_once

        interp_poll_fn = interp_poll_once
    source_fps_key = int(round(source_fps))
    seen: set[tuple[str, str, int]] = set()
    with optional_bar(progress, "model-pass chunks") as tracker:

        def _advance(leg: str, segment_id: str, index: int) -> None:
            key = (leg, segment_id, index)
            if key in seen:
                return
            seen.add(key)
            if tracker is not None:
                tracker.update()

        def _up_chunk(segment_id: str, index: int, _total: int) -> None:
            _advance("upscale", segment_id, index)

        def _ip_chunk(segment_id: str, index: int, _total: int) -> None:
            _advance("interp", segment_id, index)

        first_pass = True
        for _ in range(_MAX_POLL_PASSES):
            upscale_start = time.monotonic()
            upscale_result = upscale_poll_fn(
                run_dir,
                weights_path=weights.realesrgan,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=source_fps_key,
                upscale_factor=upscale_factor,
                chunk_frames=chunk_frames,
                device=device,
                crf=crf,
                preset=preset,
                on_chunk=_up_chunk,
            )
            if timings is not None:
                timings["upscale_poll_s"] += time.monotonic() - upscale_start
                timings["upscale_chunks_done"] += float(upscale_result.chunks_done)
            interp_start = time.monotonic()
            interp_result = interp_poll_fn(
                run_dir,
                weights_path=weights.film,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=source_fps_key,
                upscale_factor=upscale_factor,
                chunk_frames=chunk_frames,
                multiplier=multiplier,
                device=device,
                crf=crf,
                preset=preset,
                on_chunk=_ip_chunk,
            )
            if timings is not None:
                timings["interp_poll_s"] += time.monotonic() - interp_start
                timings["interp_chunks_done"] += float(interp_result.chunks_done)
            if first_pass:
                # Learn the settled total: rendered chunks advanced via
                # the callbacks above; already-ledgered ones never fire,
                # so count them here (later passes only repeat them).
                # `getattr` keeps legacy injected fakes (done-only
                # namespaces) working — the bar just learns a short total.
                first_pass = False
                if tracker is not None:
                    total = (
                        upscale_result.chunks_done
                        + getattr(upscale_result, "chunks_skipped", 0)
                        + interp_result.chunks_done
                        + getattr(interp_result, "chunks_skipped", 0)
                        + getattr(interp_result, "chunks_waiting", 0)
                    )
                    if total > 0:
                        tracker.set_total(total)
                    tracker.update(
                        getattr(upscale_result, "chunks_skipped", 0)
                        + getattr(interp_result, "chunks_skipped", 0)
                    )
            if upscale_result.chunks_done == 0 and interp_result.chunks_done == 0:
                if getattr(interp_result, "chunks_waiting", 0) > 0:
                    raise MediaError(
                        f"augment polling stuck: {interp_result.chunks_waiting} chunks "
                        "waiting with no progress (rerun the pollers, then finalize again)"
                    )
                return
    raise MediaError(
        f"augment polling made no settling pass in {_MAX_POLL_PASSES} rounds "
        "(rerun the pollers, then finalize again)"
    )


def run_durable_model_pass(
    run_dir: Path,
    usable: list[Path],
    *,
    out_width: int,
    out_height: int,
    source_fps: float,
    weights: Any,
    upscale_factor: int = 2,
    multiplier: int = 4,
    chunk_frames: int = 32,
    crf: int = 15,
    preset: str = "veryfast",
    device: str = "cuda:1",
    work_dir: Path,
    upscale_poll_fn: Callable[..., Any] | None = None,
    interp_poll_fn: Callable[..., Any] | None = None,
    drain_fn: Callable[..., Any] | None = None,
    concat_fn: Callable[[list[Path], Path], Path] | None = None,
    seam_interp_fn: Callable[..., Any] | None = None,
    morph_joints: bool = False,
    morph_interp_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
) -> tuple[Path, int]:
    """Poll, drain, and concat the durable sidecar path (issue: independent workers).

    `usable` holds committed segment dirs in presentation order; `out_fps`
    for the shared plan derivation is always `source_fps` (the interp
    worker records the lifted fps inside the same plan dir). Returns the
    final intermediate + its fps (`round(source_fps * multiplier)`),
    matching the legacy `run_finalize_model_pass` contract. Pollers that
    scan `run_dir` may also finish skipped (non-usable) segments — the
    drain below only consumes `usable`, in order.

    `timings` (optional out-param) records per-phase seconds plus chunk
    counts for the elapsed-time report: `upscale_poll_s`,
    `interp_poll_s`, `drain_s`, `concat_s`, `upscale_chunks_done`,
    `interp_chunks_done`, `chunks_drained`, plus `seam_s` seconds spent
    rendering seam mids and `seams_done` seams actually rendered
    (ledger hits resume without re-rendering). Keys are zero-initialized
    on entry so miners never KeyError, even when a later stage fails.

    Seams (`seam_interp_fn`, defaulting to the resident FILM leg):
    every adjacent usable pair gains its `multiplier - 1` mids between
    A's last and B's first interpolated frames, interleaved as
    [A, seam, B] in the final concat — no hard cuts, no dropped or
    duplicated endpoints. `multiplier=1` has zero mids and skips seams.

    Morph-cut joints (`morph_joints=True`, ltx25/ltx23): replaces the
    mids-insert seams with count-preserving morph-cuts — per joint,
    `A[-2:]+B[:2]` are replaced by 4 FILM bridge frames morphed between
    anchors `A[-3]` and `B[+2]` (see `voyage.augment_morph`). Frame total
    is unchanged so audio needs no work; `seam_interp_fn` is never called
    in this mode. `morph_interp_fn` defaults to the resident FILM leg.
    """
    if not isinstance(morph_joints, bool):
        raise TypeError(f"morph_joints must be a bool (got {type(morph_joints).__name__})")
    run_dir = _require_run_dir(run_dir)
    segments = _require_usable(usable)
    out_width = _require_box("out_width", out_width)
    out_height = _require_box("out_height", out_height)
    source_fps = _require_fps("source_fps", source_fps)
    upscale_factor = _require_factor(upscale_factor)
    multiplier = _require_multiplier(multiplier)
    chunk_frames = _require_count("chunk_frames", chunk_frames)
    device = _require_text("device", device)
    preset = _require_text("preset", preset)
    if not isinstance(work_dir, Path):
        raise TypeError(f"work_dir must be a Path (got {type(work_dir).__name__})")
    if timings is not None:
        for key in (
            "upscale_poll_s",
            "interp_poll_s",
            "drain_s",
            "concat_s",
            "upscale_chunks_done",
            "interp_chunks_done",
            "chunks_drained",
            "seam_s",
            "seams_done",
            "morph_s",
            "morphs_done",
        ):
            timings.setdefault(key, 0.0)
    weights_key = weights_key_for(weights)
    # Integer fps key shared by the pollers and the plan derivation below:
    # the hash formats it via str(), so float 24.0 vs int 24 would fork
    # plan dirs — one normalization keeps all three on the same dir.
    source_fps_key = int(round(source_fps))
    _poll_to_completion(
        run_dir,
        weights=weights,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        source_fps=source_fps,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        chunk_frames=chunk_frames,
        device=device,
        crf=crf,
        preset=preset,
        upscale_poll_fn=upscale_poll_fn,
        interp_poll_fn=interp_poll_fn,
        timings=timings,
        progress=progress,
    )
    if drain_fn is None:
        from voyage.augment_drain import drain_interpolated_plan

        drain_fn = drain_interpolated_plan
    if concat_fn is None:
        from voyage.augment_drain import concat_chunk_mp4s

        concat_fn = concat_chunk_mp4s
    from voyage.augment_upscale_poller import committed_segment_sources

    sources, _skipped = committed_segment_sources(run_dir)
    by_id = {source.segment_id: source for source in sources}
    intermediates: list[Path] = []
    previous: tuple[str, Path] | None = None
    drain_cm = optional_bar(progress, "drain segments", total=len(segments))
    with drain_cm as drain_tracker:
        for segment in segments:
            source = by_id.get(segment.name)
            if source is None:
                raise MediaError(
                    f"usable segment {segment.name} has no pollable source "
                    "(DONE + manifest with video checksum/frames required)"
                )
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
            if previous is not None and multiplier > 1 and not morph_joints:
                prev_key, prev_plan = previous
                seam_dir = seam_plan_dir(
                    run_dir,
                    key_a=prev_key,
                    key_b=source.source_key,
                    weights_key=weights_key,
                    out_width=out_width,
                    out_height=out_height,
                    out_fps=source_fps_key,
                    upscale_factor=upscale_factor,
                    crf=crf,
                    preset=preset,
                )
                endpoints = seam_endpoints(prev_plan, plan_dir)
                if endpoints is None:
                    raise MediaError(
                        f"seam endpoints missing between {prev_key} and "
                        f"{source.source_key} (interp incomplete after polling)"
                    )
                before_png, after_png = endpoints
                seam_start = time.monotonic()
                rendered = render_seam_once(
                    run_dir,
                    seam_dir,
                    before_png=before_png,
                    after_png=after_png,
                    multiplier=multiplier,
                    source_key=f"{prev_key}|{source.source_key}",
                    weights_key=weights_key,
                    out_width=out_width,
                    out_height=out_height,
                    out_fps=source_fps_key,
                    upscale_factor=upscale_factor,
                    crf=crf,
                    preset=preset,
                    weights_path=weights.film,
                    device=device,
                    interp_fn=seam_interp_fn,
                )
                if timings is not None:
                    timings["seam_s"] += time.monotonic() - seam_start
                    timings["seams_done"] += float(rendered)
                seam_drain_start = time.monotonic()
                seam_drained = drain_fn(seam_dir, out_fps=source_fps * multiplier)
                if timings is not None:
                    timings["drain_s"] += time.monotonic() - seam_drain_start
                    timings["chunks_drained"] += float(seam_drained.chunks_drained)
                intermediates.append(seam_drained.intermediate_mp4)
            drain_start = time.monotonic()
            drained = drain_fn(plan_dir, out_fps=source_fps * multiplier)
            if timings is not None:
                timings["drain_s"] += time.monotonic() - drain_start
                timings["chunks_drained"] += float(drained.chunks_drained)
            intermediates.append(drained.intermediate_mp4)
            previous = (source.source_key, plan_dir)
            if drain_tracker is not None:
                drain_tracker.update()
    work_dir.mkdir(parents=True, exist_ok=True)
    final = work_dir / FINAL_INTERMEDIATE_FILENAME
    if morph_joints:
        from voyage.augment_morph import assemble_morphed_timeline

        morph_cm = optional_stage(progress, "morph joints", f"{len(intermediates) - 1} joint(s)")
        morph_start = time.monotonic()
        with morph_cm:
            final = assemble_morphed_timeline(
                intermediates,
                joint_root=run_dir / "augment" / "morph_joints",
                fps=int(round(source_fps * multiplier)),
                crf=crf,
                preset=preset,
                pix_fmt="yuv420p",
                interp_fn=morph_interp_fn,
                weights=weights.film,
                device=device,
                concat_fn=concat_fn,
            )
        if timings is not None:
            timings["morph_s"] += time.monotonic() - morph_start
            timings["morphs_done"] += float(len(intermediates) - 1)
        if not final.exists() or final.stat().st_size == 0:
            raise MediaError(f"durable morph pass produced empty output {final}")
        return (final, round(source_fps * multiplier))
    concat_start = time.monotonic()
    concat_fn(intermediates, final)
    if timings is not None:
        timings["concat_s"] += time.monotonic() - concat_start
    if not final.exists() or final.stat().st_size == 0:
        raise MediaError(f"durable model pass produced empty output {final}")
    return (final, round(source_fps * multiplier))
