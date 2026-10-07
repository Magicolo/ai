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

import inspect
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voyage.augment_joints import ensure_joint_units
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


def weights_key_for(weights: Any, interp_backend: str = "rife") -> str:
    """Ledger key covering both model legs, legacy `sha|sha` shape.

    The key intentionally carries NO backend prefix: the interp-leg sha
    differs between backends (FILM and RIFE weights are different files),
    so a backend switch already misses old records by construction --
    while pre-backend runs (e.g. kaolin's FILM ledgers) keep hitting
    with zero re-render. Final freshness across a backend switch rides
    the skip-key backend component instead.
    """
    from voyage.workers import augment_worker

    backend = augment_worker.validate_interp_backend(interp_backend)
    # Legacy fakes carry the FILM leg as `.film` (no `interp_leg` method).
    interp = weights.interp_leg(backend) if hasattr(weights, "interp_leg") else weights.film
    realesrgan = weights.realesrgan
    if interp is None or realesrgan is None:
        raise ValueError(
            "durable model pass needs the interp leg and the upscale leg provisioned "
            f"(backend={backend}, interp={interp!r}, realesrgan={realesrgan!r})"
        )
    return f"{sha256_file(interp)}|{sha256_file(realesrgan)}"


def _interp_weights_path(weights: Any, interp_backend: str) -> Path | None:
    """Active interp-leg weights path for `interp_backend`.

    Delegates to the typed `augment.interp_leg_path` (backend validated
    inside; legacy fakes fall back to `.film`).
    """
    from voyage.augment import interp_leg_path

    return interp_leg_path(weights, interp_backend)


def _accepts_keyword(func: Callable[..., Any], name: str) -> bool:
    """True when `func` declares keyword `name` (injected-fake tolerance).

    Production pollers accept the backend keyword; older injected test
    doubles may not — probing keeps them working instead of TypeErroring.
    Mirrors the `_supports_on_pair` probe in `augment_interp_poller`.
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


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


def _expected_frame_totals(run_dir: Path, *, chunk_frames: int, multiplier: int) -> tuple[int, int]:
    """Upfront (source_frames, output_frames) totals over committed segments.

    Best-effort: any failure (no segments yet, unreadable manifest) yields
    `(0, 0)` and the bars fall back to end-of-pass accounting. Source
    frames tile via `chunk_windows`; output frames apply the unchunked
    per-window `(n-1)*m+1` recipe, so the finish line can show both units.
    """
    try:
        from voyage.augment import chunk_windows, interpolated_frame_count
        from voyage.augment_upscale_poller import committed_segment_sources

        sources, _skipped = committed_segment_sources(run_dir)
        source_total = 0
        output_total = 0
        for source in sources:
            for _start, count in chunk_windows(source.total_frames, chunk_frames):
                source_total += count
                output_total += interpolated_frame_count(count, multiplier)
    except (OSError, ValueError, TypeError):
        return (0, 0)
    else:
        return (source_total, output_total)


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
    interp_backend: str = "rife",
    joint_interp_fn: Callable[..., Any] | None = None,
) -> None:
    """Run both pollers segment-interleaved until a pass finishes nothing.

    Each pass enumerates the committed segments once, then runs every
    segment's upscale immediately followed by its interp — interp starts
    on committed frames instead of waiting for the full upscale sweep
    (the old full-sweep-then-full-sweep stalls long finalizes by ~1h).
    Both legs default to `device` (legacy single-device behavior);
    `upscale_device`/`interp_device` pin legs to cards (interp moved to
    the 2060, cuda:1, sharing the card with upscale — RIFE's 0.65 GiB
    peak fits beside the llama sidecar). `include_interp=False` runs the
    upscale leg only; `include_upscale=False` runs the interp leg only.

    `progress` renders one leg bar per segment (`upscale frames`, then
    `interp frames` — sequential, never two concurrent Live displays).
    Bars count source frames, so both legs share one comparable unit
    (not the multiplied interp output); totals come from each segment
    (elapsed + ETA from the first second) and reconcile at scoped-call
    end, rendered chunks advance live via the pollers' `on_chunk_frames`
    callbacks, and interp chunks additionally advance per finished
    interp pair via `on_pair_frames`.     Counts still accumulate into
    `timings` for the elapsed-time report. Joints follow fix, then the
    uniform pass (seam fix -> upscale -> interpolate): the fix stage
    (`ensure_joint_units`) renders every adjacent pair's 4 source-res
    morph bridges once, then each pass polls the joint units through the
    SAME upscale + interp pollers as the segments (via the pollers'
    `sources=` override — the joints re-interpolate by design). Joint
    units rebuild when the committed segment list changes mid-poll
    (cheap manifest-key comparison; the fix itself is ledger-hit).
    `joint_interp_fn`, when given, drives the fix-stage bridge render
    (tests stub it); else the resident interp leg.
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
    expected_source, expected_output = _expected_frame_totals(
        run_dir, chunk_frames=chunk_frames, multiplier=multiplier
    )
    # Fix stage (seam fix -> upscale -> interpolate): every adjacent pair's
    # 4 source-res morph bridges render once here; each pass below then
    # polls the joint units through the same upscale + interp pollers as
    # the segments. The unit list rebuilds only when the committed
    # segment list changes mid-poll (manifest-key comparison — no file
    # hashing per pass; the fix itself is ledger-hit idempotent).
    from voyage.augment_upscale_poller import committed_segment_sources

    interp_weights = _interp_weights_path(weights, interp_backend)
    joint_units: list[Any] = []
    joint_signature: tuple[tuple[str, str], ...] | None = None

    def _refresh_joint_units(ordered: list[Any]) -> list[Any]:
        nonlocal joint_units, joint_signature
        signature = tuple((source.segment_id, source.source_key) for source in ordered)
        if joint_signature is None or signature != joint_signature:
            joint_units = ensure_joint_units(
                run_dir,
                ordered,
                source_fps=source_fps_key,
                crf=crf,
                preset=preset,
                interp_fn=joint_interp_fn,
                weights=interp_weights,
                device=ip_dev,
                interp_backend=interp_backend,
            )
            joint_signature = signature
        return joint_units

    for _ in range(_MAX_POLL_PASSES):
        up_done = 0
        ip_done = 0
        ip_waiting = 0
        up_per_segment: dict[str, list[int]] = {}
        ip_per_segment: dict[str, list[int]] = {}
        # Segment-interleaved pipeline: enumerate once per pass, then run
        # each segment's upscale immediately followed by its interp, so
        # interp starts on committed frames instead of waiting for the
        # full upscale sweep. Pass totals below preserve the old settle
        # semantics exactly (a pass that finishes nothing returns, unless
        # interp chunks are still waiting on missing upscale outputs).
        segment_sources, _skipped_sources = committed_segment_sources(run_dir)
        ordered_sources = sorted(segment_sources, key=lambda source: source.segment_id)
        up_takes_segments = _accepts_keyword(upscale_poll_fn, "segment_ids")
        ip_takes_segments = _accepts_keyword(interp_poll_fn, "segment_ids")
        interp_weights = _interp_weights_path(weights, interp_backend)
        for source in ordered_sources:
            segment_id = source.segment_id
            segment_frames = source.total_frames or None
            if include_upscale:
                upscale_start = time.monotonic()
                with optional_bar(progress, "upscale frames", segment_frames) as up_tracker:

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

                    upscale_kwargs: dict[str, Any] = {
                        "weights_path": weights.realesrgan,
                        "weights_key": weights_key,
                        "out_width": out_width,
                        "out_height": out_height,
                        "out_fps": source_fps_key,
                        "upscale_factor": upscale_factor,
                        "chunk_frames": chunk_frames,
                        "device": up_dev,
                        "crf": crf,
                        "preset": preset,
                        "on_chunk": _up_chunk,
                        "on_chunk_frames": _up_frames,
                    }
                    if _accepts_keyword(upscale_poll_fn, "interp_multiplier"):
                        upscale_kwargs["interp_multiplier"] = multiplier
                    if up_takes_segments:
                        upscale_kwargs["segment_ids"] = [segment_id]
                    upscale_result = upscale_poll_fn(run_dir, **upscale_kwargs)
                    upscale_seconds = time.monotonic() - upscale_start
                    scoped_done = int(getattr(upscale_result, "chunks_done", 0) or 0)
                    scoped_frames = int(getattr(upscale_result, "frames_done", 0) or 0)
                    scoped_skipped = int(getattr(upscale_result, "frames_skipped", 0) or 0)
                    up_done += scoped_done
                    if up_tracker is not None:
                        scoped_total = scoped_frames + scoped_skipped
                        if scoped_total > 0:
                            up_tracker.set_total(scoped_total)
                        if scoped_skipped > 0:
                            up_tracker.update(scoped_skipped)
                        if scoped_frames > 0 and upscale_seconds > 0:
                            up_tracker.set_extra(f"{scoped_frames / upscale_seconds:.1f} frames/s")
                if timings is not None:
                    timings["upscale_poll_s"] = timings.get("upscale_poll_s", 0.0) + upscale_seconds
                    timings["upscale_chunks_done"] = timings.get(
                        "upscale_chunks_done", 0.0
                    ) + float(scoped_done)
                    timings["upscale_frames_done"] = timings.get(
                        "upscale_frames_done", 0.0
                    ) + float(scoped_frames)
            if include_interp:
                interp_start = time.monotonic()
                with optional_bar(progress, "interp frames", segment_frames) as ip_tracker:
                    _ip_carry: list[float] = [0.0]

                    def _ip_chunk(
                        segment_id: str,
                        index: int,
                        _total: int,
                        _into: dict[str, list[int]] = ip_per_segment,
                    ) -> None:
                        _into.setdefault(segment_id, []).append(index)

                    def _ip_pair(
                        segment_id: str,
                        fraction_done: float,
                        _carry: list[float] = _ip_carry,
                    ) -> None:
                        del segment_id
                        _carry[0] += fraction_done
                        whole, _carry[0] = divmod(_carry[0], 1.0)
                        if ip_tracker is not None and whole >= 1:
                            ip_tracker.update(int(whole))

                    interp_kwargs: dict[str, Any] = {
                        "weights_path": interp_weights,
                        "weights_key": weights_key,
                        "out_width": out_width,
                        "out_height": out_height,
                        "out_fps": source_fps_key,
                        "upscale_factor": upscale_factor,
                        "chunk_frames": chunk_frames,
                        "multiplier": multiplier,
                        "device": ip_dev,
                        "crf": crf,
                        "preset": preset,
                        "on_chunk": _ip_chunk,
                        "on_pair_frames": _ip_pair,
                    }
                    if ip_takes_segments:
                        interp_kwargs["segment_ids"] = [segment_id]
                    if _accepts_keyword(interp_poll_fn, "interp_backend"):
                        interp_kwargs["interp_backend"] = interp_backend
                    interp_result = interp_poll_fn(run_dir, **interp_kwargs)
                    interp_seconds = time.monotonic() - interp_start
                    scoped_done = int(getattr(interp_result, "chunks_done", 0) or 0)
                    scoped_frames = int(getattr(interp_result, "frames_done", 0) or 0)
                    scoped_skipped = int(getattr(interp_result, "frames_skipped", 0) or 0)
                    scoped_waiting = int(getattr(interp_result, "chunks_waiting", 0) or 0)
                    ip_done += scoped_done
                    ip_waiting += scoped_waiting
                    if ip_tracker is not None:
                        scoped_total = scoped_frames + scoped_skipped + scoped_waiting
                        if scoped_total > 0:
                            ip_tracker.set_total(scoped_total)
                        advanced = scoped_skipped + scoped_waiting
                        if advanced > 0:
                            ip_tracker.update(advanced)
                        if scoped_frames > 0 and interp_seconds > 0:
                            ip_tracker.set_extra(
                                f"{scoped_frames / interp_seconds:.1f} frames/s "
                                f"({expected_output} out frames)"
                            )
                        else:
                            ip_tracker.set_extra(f"{expected_output} out frames")
                if timings is not None:
                    timings["interp_poll_s"] = timings.get("interp_poll_s", 0.0) + interp_seconds
                    timings["interp_chunks_done"] = timings.get("interp_chunks_done", 0.0) + float(
                        scoped_done
                    )
                    timings["interp_frames_done"] = timings.get("interp_frames_done", 0.0) + float(
                        scoped_frames
                    )
        # Uniform pass over the fix-stage joints: the joint units poll
        # through the same legs as the segments (sources= override), so
        # bridge frames upscale + re-interpolate by design. Joints are new
        # units with no prior upscale, so an interp-only poll still runs
        # their upscale — otherwise they would strand as waiting forever.
        joint_here = _refresh_joint_units(ordered_sources)
        joint_sources = [unit.as_source() for unit in joint_here]

        def _joint_up_chunk(
            segment_id: str,
            index: int,
            _total: int,
            _into: dict[str, list[int]] = up_per_segment,
        ) -> None:
            _into.setdefault(segment_id, []).append(index)

        def _joint_up_frames(segment_id: str, frames: int) -> None:
            del segment_id, frames

        def _joint_ip_chunk(
            segment_id: str,
            index: int,
            _total: int,
            _into: dict[str, list[int]] = ip_per_segment,
        ) -> None:
            _into.setdefault(segment_id, []).append(index)

        def _joint_ip_pair(segment_id: str, fraction_done: float) -> None:
            del segment_id, fraction_done

        joint_up = include_upscale or (include_interp and bool(joint_sources))
        if joint_up and joint_sources:
            joint_up_start = time.monotonic()
            joint_up_kwargs: dict[str, Any] = {
                "weights_path": weights.realesrgan,
                "weights_key": weights_key,
                "out_width": out_width,
                "out_height": out_height,
                "out_fps": source_fps_key,
                "upscale_factor": upscale_factor,
                "chunk_frames": chunk_frames,
                "device": up_dev,
                "crf": crf,
                "preset": preset,
                "on_chunk": _joint_up_chunk,
                "on_chunk_frames": _joint_up_frames,
                "sources": joint_sources,
            }
            if _accepts_keyword(upscale_poll_fn, "interp_multiplier"):
                joint_up_kwargs["interp_multiplier"] = multiplier
            if up_takes_segments:
                joint_up_kwargs["segment_ids"] = [unit.segment_id for unit in joint_sources]
            joint_up_result = upscale_poll_fn(run_dir, **joint_up_kwargs)
            joint_up_seconds = time.monotonic() - joint_up_start
            up_done += int(getattr(joint_up_result, "chunks_done", 0) or 0)
            if timings is not None:
                timings["upscale_poll_s"] = timings.get("upscale_poll_s", 0.0) + joint_up_seconds
                timings["upscale_chunks_done"] = timings.get("upscale_chunks_done", 0.0) + float(
                    getattr(joint_up_result, "chunks_done", 0) or 0
                )
                timings["upscale_frames_done"] = timings.get("upscale_frames_done", 0.0) + float(
                    getattr(joint_up_result, "frames_done", 0) or 0
                )
        if include_interp and joint_sources:
            joint_ip_start = time.monotonic()
            joint_ip_kwargs: dict[str, Any] = {
                "weights_path": interp_weights,
                "weights_key": weights_key,
                "out_width": out_width,
                "out_height": out_height,
                "out_fps": source_fps_key,
                "upscale_factor": upscale_factor,
                "chunk_frames": chunk_frames,
                "multiplier": multiplier,
                "device": ip_dev,
                "crf": crf,
                "preset": preset,
                "on_chunk": _joint_ip_chunk,
                "on_pair_frames": _joint_ip_pair,
                "sources": joint_sources,
            }
            if ip_takes_segments:
                joint_ip_kwargs["segment_ids"] = [unit.segment_id for unit in joint_sources]
            if _accepts_keyword(interp_poll_fn, "interp_backend"):
                joint_ip_kwargs["interp_backend"] = interp_backend
            joint_ip_result = interp_poll_fn(run_dir, **joint_ip_kwargs)
            joint_ip_seconds = time.monotonic() - joint_ip_start
            ip_done += int(getattr(joint_ip_result, "chunks_done", 0) or 0)
            ip_waiting += int(getattr(joint_ip_result, "chunks_waiting", 0) or 0)
            if timings is not None:
                timings["interp_poll_s"] = timings.get("interp_poll_s", 0.0) + joint_ip_seconds
                timings["interp_chunks_done"] = timings.get("interp_chunks_done", 0.0) + float(
                    getattr(joint_ip_result, "chunks_done", 0) or 0
                )
                timings["interp_frames_done"] = timings.get("interp_frames_done", 0.0) + float(
                    getattr(joint_ip_result, "frames_done", 0) or 0
                )
        if verbose and up_per_segment and progress is not None:
            for segment_id in sorted(up_per_segment):
                progress.info(
                    f"upscale {segment_id}: chunks {_format_ranges(up_per_segment[segment_id])}"
                )
        if verbose and ip_per_segment and progress is not None:
            for segment_id in sorted(ip_per_segment):
                progress.info(
                    f"interp {segment_id}: chunks {_format_ranges(ip_per_segment[segment_id])}"
                )
        if include_interp:
            if up_done == 0 and ip_done == 0:
                if ip_waiting > 0:
                    raise MediaError(
                        f"augment polling stuck: {ip_waiting} chunks waiting "
                        "with no progress (rerun the pollers, then finalize again)"
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


def _ensure_model_pass_timings(timings: dict[str, float] | None) -> None:
    """Zero-initialize every model-pass timings key (miners never KeyError).

    Called on entry by both `run_durable_model_pass` and
    `_drain_to_intermediate` — the drain is also invoked directly by the
    bidirectional parallel branch (media.py), which never runs the
    durable entry's init, so the drain must not assume pre-initialized
    keys (live kaolin `KeyError: 'drain_s'`, 2026-10-06).

    Mastering keys (`mastering_poll_s` / `mastering_chunks_done` /
    `mastering_frames_done`, Track C) are zero-initialized here for the
    future `voyage/mastering.py` consumer (Track B owns that module) —
    miners and `_model_pass_stage_rows` never KeyError, even before any
    mastering work lands.
    """
    if timings is None:
        return
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
        "mastering_poll_s",
        "mastering_chunks_done",
        "mastering_frames_done",
    ):
        timings.setdefault(key, 0.0)


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
    interp_backend: str = "rife",
    upscale_device: str | None = None,
    interp_device: str | None = None,
    upscale_poll_fn: Callable[..., Any] | None = None,
    interp_poll_fn: Callable[..., Any] | None = None,
    chunk_encode_fn: Callable[..., Any] | None = None,
    drain_fn: Callable[..., Any] | None = None,
    concat_fn: Callable[[list[Path], Path], Path] | None = None,
    assemble_fn: Callable[..., Any] | None = None,
    joint_interp_fn: Callable[..., Any] | None = None,
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

    Finalize-start cleanup: `sweep_chunk_mp4s` first encodes (+ records
    + prunes PNGs) every fully-interpolated chunk still missing its
    durable mp4 — one-time catch-up for runs rendered before per-chunk
    mp4s existed, so their gigabytes of PNGs collapse to megabytes
    before polling starts — then `prune_orphan_plan_dirs` GCs plan dirs
    no current segment or joint references. `chunk_encode_fn`, when
    given, replaces the production chunk-ffmpeg encode inside the sweep
    (tests stub it); None (default) encodes for real.

    `timings` (optional out-param) records per-phase seconds plus chunk
    counts for the elapsed-time report: `upscale_poll_s`,
    `interp_poll_s`, `drain_s`, `concat_s`, `upscale_chunks_done`,
    `interp_chunks_done`, `chunks_drained`, plus `seam_s` seconds spent
    rendering fix-stage joints and `seams_done` joints actually rendered
    (ledger hits resume without re-rendering). Keys are zero-initialized
    on entry so miners never KeyError, even when a later stage fails.

    Joints (universal, all backends): every adjacent usable pair is fixed
    once at source resolution (morph 2+2 bridges, no upscale — see
    `voyage.augment_joints`), then the 4 bridge frames flow through the
    same upscale + interp pollers as the segments and drain interleaved
    as [trimA, joint, trimB] with source-derived trims. `joint_interp_fn`
    drives the fix-stage bridge render (tests stub it); when None,
    `morph_interp_fn` is the fallback (legacy name, same seam); both
    None defaults to the resident interp leg. `morph_joints` is a legacy
    flag (validated bool): both values yield the universal joints.

    Interleaved pass (DESIGN §140): the driver runs each segment's
    upscale immediately followed by its interp on one card — the
    caller pins both `upscale_device` and `interp_device` to the 2060
    (cuda:1; RIFE fits beside the llama sidecar), and the drain below
    inherits the interp card. Both default to `device` (the legacy
    single-device behavior).
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
    _ensure_model_pass_timings(timings)
    weights_key = weights_key_for(weights, interp_backend)
    # Finalize-start cleanup (see docstring): collapse any pre-cleanup
    # PNGs to durable chunk mp4s first, then GC orphan plan dirs — both
    # are idempotent, so a kill between them converges on retry.
    from voyage.augment_drain import prune_orphan_plan_dirs, sweep_chunk_mp4s

    sweep_chunk_mp4s(run_dir, encode_fn=chunk_encode_fn)
    # Integer fps key shared by the pollers and the plan derivation below:
    # the hash formats it via str(), so float 24.0 vs int 24 would fork
    # plan dirs — one normalization keeps all three on the same dir.
    source_fps_key = int(round(source_fps))
    prune_orphan_plan_dirs(
        run_dir,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=source_fps_key,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
    )
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
        interp_backend=interp_backend,
        joint_interp_fn=(joint_interp_fn if joint_interp_fn is not None else morph_interp_fn),
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
        interp_backend=interp_backend,
        drain_fn=drain_fn,
        concat_fn=concat_fn,
        assemble_fn=assemble_fn,
        joint_interp_fn=(joint_interp_fn if joint_interp_fn is not None else morph_interp_fn),
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
    interp_backend: str = "rife",
    drain_fn: Callable[..., Any] | None = None,
    concat_fn: Callable[[list[Path], Path], Path] | None = None,
    assemble_fn: Callable[..., Any] | None = None,
    joint_interp_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
) -> tuple[Path, int]:
    """Drain ledgered interp chunks (+ uniform-pass joints) into one intermediate.

    Shared entry for `run_durable_model_pass`: consumes
    only `usable` in order. The fix stage renders every boundary's
    source-res joints (ledger-hit when `run_durable_model_pass` already
    polled them), the joint units drain whole through the same
    `drain_fn`, and `assemble_joint_timeline` joins
    [trimA, joint, trimB, ...] with source-derived trims. All interp
    work here runs on `device` — the caller passes the 4060 interp
    device, never the 2060 upscale card.
    """
    if drain_fn is None:
        from voyage.augment_drain import drain_interpolated_plan

        drain_fn = drain_interpolated_plan
    if concat_fn is None:
        from voyage.augment_drain import concat_chunk_mp4s

        concat_fn = concat_chunk_mp4s
    from voyage.augment_upscale_poller import committed_segment_sources

    _ensure_model_pass_timings(timings)
    sources, _skipped = committed_segment_sources(run_dir)
    by_id = {source.segment_id: source for source in sources}
    ordered: list[Any] = []
    for segment in segments:
        source = by_id.get(segment.name)
        if source is None:
            raise MediaError(
                f"usable segment {segment.name} has no pollable source "
                "(DONE + manifest with video checksum/frames required)"
            )
        ordered.append(source)
    # Fix stage (idempotent: the poll sweep above already rendered these;
    # a direct drain call renders them here instead). Skipped entirely
    # for a lone segment — no pair means no joint, and the interp leg is
    # never even resolved (a lone drain must not need it).
    fix_start = time.monotonic()
    joint_units: list[Any] = []
    if len(ordered) > 1:
        joint_units = ensure_joint_units(
            run_dir,
            ordered,
            source_fps=source_fps_key,
            crf=crf,
            preset=preset,
            interp_fn=joint_interp_fn,
            weights=_interp_weights_path(weights, interp_backend),
            device=device,
            interp_backend=interp_backend,
        )
    if timings is not None:
        timings["seam_s"] += time.monotonic() - fix_start
        timings["seams_done"] += float(len(joint_units))
    segment_intermediates: list[Path] = []
    drain_cm = optional_bar(progress, "drain segments", total=len(segments))
    with drain_cm as drain_tracker:
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
            drain_start = time.monotonic()
            drained = drain_fn(plan_dir, out_fps=source_fps * multiplier)
            if timings is not None:
                timings["drain_s"] += time.monotonic() - drain_start
                timings["chunks_drained"] += float(drained.chunks_drained)
            segment_intermediates.append(drained.intermediate_mp4)
            if drain_tracker is not None:
                drain_tracker.update()
    joint_intermediates: list[Path] = []
    for unit in joint_units:
        joint_plan = plan_dir_for_segment(
            run_dir,
            source_key=unit.joint_key,
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            out_fps=source_fps_key,
            upscale_factor=upscale_factor,
            crf=crf,
            preset=preset,
        )
        joint_drain_start = time.monotonic()
        joint_drained = drain_fn(joint_plan, out_fps=source_fps * multiplier)
        if timings is not None:
            timings["drain_s"] += time.monotonic() - joint_drain_start
            timings["chunks_drained"] += float(joint_drained.chunks_drained)
        joint_intermediates.append(joint_drained.intermediate_mp4)
    work_dir.mkdir(parents=True, exist_ok=True)
    final = work_dir / FINAL_INTERMEDIATE_FILENAME
    if not joint_units:
        concat_start = time.monotonic()
        concat_fn(segment_intermediates, final)
        if timings is not None:
            timings["concat_s"] += time.monotonic() - concat_start
    else:
        if assemble_fn is None:
            from voyage.augment_joints import assemble_joint_timeline

            assemble_fn = assemble_joint_timeline
        joint_cm = optional_stage(progress, "joints", f"{len(joint_units)} joint(s)")
        joint_start = time.monotonic()
        with joint_cm:
            final = assemble_fn(
                segment_intermediates,
                joint_intermediates,
                source_counts=[source.total_frames for source in ordered],
                multiplier=multiplier,
                joint_root=run_dir / "augment" / "joint_timeline",
                fps=int(round(source_fps * multiplier)),
                crf=crf,
                preset=preset,
                pix_fmt="yuv420p",
                concat_fn=concat_fn,
            )
        if timings is not None:
            timings["morph_s"] += time.monotonic() - joint_start
            timings["morphs_done"] += float(len(joint_units))
    if not final.exists() or final.stat().st_size == 0:
        raise MediaError(f"durable model pass produced empty output {final}")
    return (final, round(source_fps * multiplier))
