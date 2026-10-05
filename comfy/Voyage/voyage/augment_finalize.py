"""Durable finalize model pass over sidecar plan dirs (DESIGN §§56-57, §140).

Finalize-side orchestration for the independent augment workers: poll the
upscale worker (2060 ESRGAN) and the interp worker (4060 FILM, picking up
the ledgered upscale chunks) to completion, drain one intermediate per
usable segment, and concat them in segment order. Replaces the all-or-nothing `TemporaryDirectory`
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
    upscale_poll_fn: Callable[..., Any] | None = None,
    interp_poll_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
    upscale_device: str | None = None,
    interp_device: str | None = None,
    include_interp: bool = True,
    include_upscale: bool = True,
) -> None:
    """Run both pollers until a full pass finishes nothing (fail-loud when stuck).

    DESIGN §140 A/V stream: the upscale sweep runs on `upscale_device`
    (the 2060, cuda:1) while the interp sweep runs on `interp_device`
    (the 4060, cuda:0) — the interpolator picks up the ledgered upscale
    chunks the 2060 published, and the two legs never share a card. Both
    default to `device` (the legacy single-device behavior) so existing
    callers keep today's pinning. `include_interp=False` runs the
    upscale sweep only (the finalize Phase A + generation-time pre-warm:
    interpolation happens at finalize time, never during generation).
    `include_upscale=False` runs the interp sweep only (the finalize
    Phase C: the 4060 picks up the Phase A upscale ledger after the
    music takes and the SFX bed finished).

    `progress` renders one leg bar per sweep (`upscale frames`, then
    `interp frames` — sequential, never two concurrent Live displays).
    Bars count source frames, so both legs share one comparable unit
    (not the multiplied interp output); totals come from each sweep's
    result (done + skipped, plus waiting for interp) and rendered chunks
    advance live via the pollers' `on_chunk_frames` callbacks. Counts
    still accumulate into `timings` for the elapsed-time report.
    """
    if upscale_poll_fn is None:
        from voyage.augment_upscale_poller import upscale_poll_once

        upscale_poll_fn = upscale_poll_once
    if interp_poll_fn is None:
        from voyage.augment_interp_poller import interp_poll_once

        interp_poll_fn = interp_poll_once
    source_fps_key = int(round(source_fps))
    up_dev = upscale_device or device
    ip_dev = interp_device or device
    verbose = bool(progress is not None and progress.verbose)
    for _ in range(_MAX_POLL_PASSES):
        up_done = 0
        up_per_segment: dict[str, list[int]] = {}
        if include_upscale:
            upscale_start = time.monotonic()
            with optional_bar(progress, "upscale frames") as up_tracker:

                def _up_chunk(
                    segment_id: str,
                    index: int,
                    _total: int,
                    _into: dict[str, list[int]] = up_per_segment,
                ) -> None:
                    _into.setdefault(segment_id, []).append(index)

                def _up_frames(segment_id: str, frames: int) -> None:
                    del segment_id
                    if up_tracker is not None:
                        up_tracker.update(frames)

                upscale_result = upscale_poll_fn(
                    run_dir,
                    weights_path=weights.realesrgan,
                    weights_key=weights_key,
                    out_width=out_width,
                    out_height=out_height,
                    out_fps=source_fps_key,
                    upscale_factor=upscale_factor,
                    chunk_frames=chunk_frames,
                    device=up_dev,
                    crf=crf,
                    preset=preset,
                    on_chunk=_up_chunk,
                    on_chunk_frames=_up_frames,
                )
                upscale_seconds = time.monotonic() - upscale_start
                up_done = int(getattr(upscale_result, "chunks_done", 0) or 0)
                up_frames_done = int(getattr(upscale_result, "frames_done", 0) or 0)
                up_frames_skipped = int(getattr(upscale_result, "frames_skipped", 0) or 0)
                if up_tracker is not None:
                    up_total = up_frames_done + up_frames_skipped
                    if up_total > 0:
                        up_tracker.set_total(up_total)
                    if up_frames_skipped > 0:
                        up_tracker.update(up_frames_skipped)
                    if up_frames_done > 0 and upscale_seconds > 0:
                        up_tracker.set_extra(f"{up_frames_done / upscale_seconds:.1f} frames/s")
            if timings is not None:
                timings["upscale_poll_s"] = timings.get("upscale_poll_s", 0.0) + upscale_seconds
                timings["upscale_chunks_done"] = timings.get("upscale_chunks_done", 0.0) + float(
                    up_done
                )
                timings["upscale_frames_done"] = timings.get("upscale_frames_done", 0.0) + float(
                    up_frames_done
                )
            if verbose and up_per_segment and progress is not None:
                for segment_id in sorted(up_per_segment):
                    progress.info(
                        f"upscale {segment_id}: chunks {_format_ranges(up_per_segment[segment_id])}"
                    )
        ip_done = 0
        if include_interp:
            interp_start = time.monotonic()
            ip_per_segment: dict[str, list[int]] = {}
            with optional_bar(progress, "interp frames") as ip_tracker:

                def _ip_chunk(
                    segment_id: str,
                    index: int,
                    _total: int,
                    _into: dict[str, list[int]] = ip_per_segment,
                ) -> None:
                    _into.setdefault(segment_id, []).append(index)

                def _ip_frames(segment_id: str, frames: int) -> None:
                    del segment_id
                    if ip_tracker is not None:
                        ip_tracker.update(frames)

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
                    device=ip_dev,
                    crf=crf,
                    preset=preset,
                    on_chunk=_ip_chunk,
                    on_chunk_frames=_ip_frames,
                )
                interp_seconds = time.monotonic() - interp_start
                ip_done = int(getattr(interp_result, "chunks_done", 0) or 0)
                ip_frames_done = int(getattr(interp_result, "frames_done", 0) or 0)
                ip_frames_skipped = int(getattr(interp_result, "frames_skipped", 0) or 0)
                ip_frames_waiting = int(getattr(interp_result, "frames_waiting", 0) or 0)
                if ip_tracker is not None:
                    ip_total = ip_frames_done + ip_frames_skipped + ip_frames_waiting
                    if ip_total > 0:
                        ip_tracker.set_total(ip_total)
                    if ip_frames_skipped > 0:
                        ip_tracker.update(ip_frames_skipped)
                    if ip_frames_done > 0 and interp_seconds > 0:
                        ip_tracker.set_extra(f"{ip_frames_done / interp_seconds:.1f} frames/s")
            if timings is not None:
                timings["interp_poll_s"] = timings.get("interp_poll_s", 0.0) + interp_seconds
                timings["interp_chunks_done"] = timings.get("interp_chunks_done", 0.0) + float(
                    ip_done
                )
                timings["interp_frames_done"] = timings.get("interp_frames_done", 0.0) + float(
                    ip_frames_done
                )
            if verbose and ip_per_segment and progress is not None:
                for segment_id in sorted(ip_per_segment):
                    progress.info(
                        f"interp {segment_id}: chunks {_format_ranges(ip_per_segment[segment_id])}"
                    )
            if up_done == 0 and ip_done == 0:
                waiting = int(getattr(interp_result, "chunks_waiting", 0) or 0)
                if waiting > 0:
                    raise MediaError(
                        f"augment polling stuck: {waiting} chunks "
                        "waiting with no progress (rerun the pollers, then finalize again)"
                    )
                return
        elif up_done == 0:
            return
    raise MediaError(
        f"augment polling made no settling pass in {_MAX_POLL_PASSES} rounds "
        "(rerun the pollers, then finalize again)"
    )


def _format_ranges(indexes: list[int]) -> str:
    """Compact `0, 1, 2, 5` chunk indexes as `0-2, 5` (verbose lines)."""
    ordered = sorted(set(indexes))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


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
    upscale_device: str | None = None,
    interp_device: str | None = None,
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

    DESIGN §140 A/V stream: `upscale_device` (the 2060, cuda:1) runs the
    upscale sweep while `interp_device` (the 4060, cuda:0) runs the interp
    sweep plus the seam/morph FILM work — the interpolator picks up the
    ledgered upscale chunks the 2060 published, and the two legs never
    share a card. Both default to `device` (the legacy single-device
    behavior).
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
            "upscale_frames_done",
            "interp_frames_done",
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
        upscale_device=upscale_device,
        interp_device=interp_device,
    )
    return _drain_to_intermediate(
        run_dir,
        segments,
        weights=weights,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        source_fps=source_fps,
        source_fps_key=source_fps_key,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        crf=crf,
        preset=preset,
        device=interp_device or device,
        work_dir=work_dir,
        drain_fn=drain_fn,
        concat_fn=concat_fn,
        seam_interp_fn=seam_interp_fn,
        morph_joints=morph_joints,
        morph_interp_fn=morph_interp_fn,
        timings=timings,
        progress=progress,
    )


def _drain_to_intermediate(
    run_dir: Path,
    segments: list[Path],
    *,
    weights: Any,
    weights_key: str,
    out_width: int,
    out_height: int,
    source_fps: float,
    source_fps_key: int,
    upscale_factor: int,
    multiplier: int,
    crf: int,
    preset: str,
    device: str,
    work_dir: Path,
    drain_fn: Callable[..., Any] | None = None,
    concat_fn: Callable[[list[Path], Path], Path] | None = None,
    seam_interp_fn: Callable[..., Any] | None = None,
    morph_joints: bool = False,
    morph_interp_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
) -> tuple[Path, int]:
    """Drain ledgered interp chunks (+ seam/morph FILM joints) into one intermediate.

    Shared by `run_durable_model_pass` and `run_interp_phase`: consumes
    only `usable` in order, interleaving `multiplier - 1` seam mids (or
    count-preserving morph-cuts when `morph_joints=True`) between
    adjacent pairs. All FILM work here runs on `device` — the caller
    passes the 4060 interp device, never the 2060 upscale card.
    """
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


def run_upscale_phase(
    run_dir: Path,
    *,
    weights: Any,
    out_width: int,
    out_height: int,
    source_fps: float,
    upscale_factor: int = 2,
    chunk_frames: int = 32,
    crf: int = 15,
    preset: str = "veryfast",
    device: str = "cuda:1",
    upscale_poll_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
) -> None:
    """Poll the upscale leg to completion, never interpolating (A/V stream Phase A).

    DESIGN §140 finalize Phase A: runs on the 2060 (cuda:1) overlapping
    the 4060 music takes, publishing `upscaled_NN` chunk dirs plus ledger
    records the Phase C interp sweep picks up on the 4060. Interpolation
    (including seam/morph FILM work) happens only in the finalize interp
    phase — never here, never during generation. Fail-loud when the
    realesrgan leg is absent (the caller keeps the legacy flow then).
    """
    run_dir = _require_run_dir(run_dir)
    out_width = _require_box("out_width", out_width)
    out_height = _require_box("out_height", out_height)
    source_fps = _require_fps("source_fps", source_fps)
    upscale_factor = _require_factor(upscale_factor)
    chunk_frames = _require_count("chunk_frames", chunk_frames)
    device = _require_text("device", device)
    preset = _require_text("preset", preset)
    if weights is None or getattr(weights, "realesrgan", None) is None:
        raise MediaError("upscale phase needs the provisioned realesrgan leg (got none)")
    weights_key = weights_key_for(weights)
    from voyage.augment_interp_poller import DEFAULT_INTERP_MULTIPLIER

    _poll_to_completion(
        run_dir,
        weights=weights,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        source_fps=source_fps,
        upscale_factor=upscale_factor,
        multiplier=DEFAULT_INTERP_MULTIPLIER,
        chunk_frames=chunk_frames,
        device=device,
        crf=crf,
        preset=preset,
        upscale_poll_fn=upscale_poll_fn,
        interp_poll_fn=None,
        timings=timings,
        progress=progress,
        include_interp=False,
    )


def run_interp_phase(
    run_dir: Path,
    usable: list[Path],
    *,
    weights: Any,
    out_width: int,
    out_height: int,
    source_fps: float,
    upscale_factor: int = 2,
    multiplier: int = 4,
    chunk_frames: int = 32,
    crf: int = 15,
    preset: str = "veryfast",
    device: str = "cuda:0",
    work_dir: Path,
    interp_poll_fn: Callable[..., Any] | None = None,
    drain_fn: Callable[..., Any] | None = None,
    concat_fn: Callable[[list[Path], Path], Path] | None = None,
    seam_interp_fn: Callable[..., Any] | None = None,
    morph_joints: bool = False,
    morph_interp_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
) -> tuple[Path, int]:
    """Poll the interp leg + drain, never upscaling (A/V stream Phase C).

    DESIGN §140 finalize Phase C: runs on the 4060 (cuda:0) after the
    music takes and the SFX bed finished, picking up the Phase A upscale
    ledger the 2060 published — the 2060 card is free by now, and the
    two legs never share a card. Includes the seam/morph FILM joints
    (they render here on `device`, never on the upscale card). Returns
    the final intermediate + its fps (`round(source_fps * multiplier)`),
    matching the `run_durable_model_pass` contract. Fail-loud when the
    film leg is absent (the caller keeps the legacy flow then).
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
    if weights is None or getattr(weights, "film", None) is None:
        raise MediaError("interp phase needs the provisioned film leg (got none)")
    if timings is not None:
        for key in (
            "upscale_poll_s",
            "interp_poll_s",
            "drain_s",
            "concat_s",
            "upscale_chunks_done",
            "interp_chunks_done",
            "upscale_frames_done",
            "interp_frames_done",
            "chunks_drained",
            "seam_s",
            "seams_done",
            "morph_s",
            "morphs_done",
        ):
            timings.setdefault(key, 0.0)
    weights_key = weights_key_for(weights)
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
        interp_poll_fn=interp_poll_fn,
        timings=timings,
        progress=progress,
        interp_device=device,
        include_upscale=False,
    )
    return _drain_to_intermediate(
        run_dir,
        segments,
        weights=weights,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        source_fps=source_fps,
        source_fps_key=source_fps_key,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        crf=crf,
        preset=preset,
        device=device,
        work_dir=work_dir,
        drain_fn=drain_fn,
        concat_fn=concat_fn,
        seam_interp_fn=seam_interp_fn,
        morph_joints=morph_joints,
        morph_interp_fn=morph_interp_fn,
        timings=timings,
        progress=progress,
    )
