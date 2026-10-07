"""Background model-pass pre-warm during generation (DESIGN §140).

While video renders on `cuda:0` and the llama director serves on `cuda:1`,
committed segments sit idle waiting for finalize. This driver runs the
same upscale + interp pollers finalize uses — with the same plan
derivation (target box/fps from `config.video`, floors from
`config.augment`, weights key over both legs, `source_fps_key` integer
normalization) — so every background-rendered chunk is a finalize ledger
hit and the finalize wait shrinks to drain + concat + encode.

Resume is trivial both ways: the sidecar ledger (`chunks.jsonl`) is the
truth, so a resumed generation re-polls (ledgered chunks skip,
unledgered partials re-render) and a resumed finalization polls to
completion over whatever the background already finished.

Stdlib only at module scope (supervisor §12 GPU ban): torch/ffmpeg enter
through the pollers' own seams or function-local lazy imports. Poller
failures never raise out of the background thread — generation must never
fail because the pre-warm did.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voyage import augment as augment_module
from voyage.augment import model_pass_devices
from voyage.media import (
    FINALIZE_CRF_DEFAULT,
    FINALIZE_PRESET_DEFAULT,
    plan_augmentation,
)

#: Source frames per pre-warm chunk (mirrors the poller/finalize default).
BACKGROUND_CHUNK_FRAMES = 32

#: Pre-warm runs one segment-interleaved upscale→interp pass (same
#: pipeline as finalize: the legs share `cuda:1`, never co-resident).
BACKGROUND_DEVICE_FALLBACK = "cuda:1"

#: Minimum free VRAM (GiB) on the pre-warm device for the upscale
#: sweep to start. SRVGGNetCompact peaks ~0.34 GiB (2060 probe: 58% with
#: the llama sidecar resident), so 1.0 leaves comfortable headroom while
#: letting upscale progress run alongside the director — the kaolin run
#: showed every pass silently skipped at ~2.8 GiB free under the old
#: single 3.0 floor for both legs.
UPSCALE_MIN_FREE_GIB = 1.0

#: Minimum free VRAM (GiB) on the pre-warm device for the interp sweep.
#: RIFE at 2x peaks ~0.65 GiB, so 1.0 leaves headroom while letting interp
#: progress run alongside the resident llama sidecar on the 6 GB 2060 —
#: the same llama-share budget the upscale sweep already runs under.
#: Unknown (no nvidia-smi / parse failure) allows the sweep: the pre-warm
#: must never go quiet on an unprobable box.
PREWARM_MIN_FREE_GIB = 1.0

#: Minimum free VRAM (GiB) when the leg's nets are already resident.
#: A cold pass needs the full floors above (H2D load + first-forward
#: workspace), but a pass whose nets are already prepared only needs
#: forward working set (~0.65 GiB RIFE peak, ~0.34 GiB SRVGG peak) —
#: without this carve-out the second pass probes cold, loads the nets,
#: and every later pass sees free net of its own resident footprint and
#: holds back forever (boba: 0.9 GiB free < 1.0 GiB needed on every
#: segment after the first successful sweep).
RESIDENT_MIN_FREE_GIB = 0.75

#: How long a notified pass waits for the director to go idle before
#: giving up the pass. Prefetch decides hold `cuda:1` for ~10-30s, so a
#: single-shot idle check at notify time loses the whole pass (kaolin:
#: zero pre-warm work during either segment render — the notify landed
#: while prefetch was in flight). 60s covers slow prefetches; past that
#: the pass is recorded director-busy and the next commit notifies again.
PREWARM_IDLE_WAIT_SECONDS = 60.0

#: Poll interval inside the idle wait (also the stop-flag latency).
PREWARM_IDLE_POLL_SECONDS = 1.0


def device_free_gib(device: str) -> float | None:
    """Free VRAM on `cuda:N` in GiB, None when unprobable (fail-open).

    Never raises: a missing binary, a non-CUDA device, or unparsable
    output all mean "unknown", and the caller treats unknown as allowed.
    """
    try:
        index = int(device.split(":")[1])
    except (IndexError, ValueError):
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip().split()[0]) / 1024.0
    except (IndexError, ValueError):
        return None


@dataclass(frozen=True)
class BackgroundPlan:
    """Finalize-equivalent plan for one pre-warm pass (all ledger keys)."""

    out_width: int
    out_height: int
    out_fps: int
    source_fps: float
    source_fps_key: int
    upscale_factor: int
    multiplier: int
    crf: int
    preset: str
    weights_key: str
    device: str
    interp_path: Path
    realesrgan_path: Path
    interp_backend: str = "rife"


@dataclass(frozen=True)
class PrewarmResult:
    """Counts from one pre-warm pass (advisory; the ledger is the truth).

    Frame counts are source frames per leg; seconds are per-leg wall time
    summed across the segment-interleaved pass (legs still run
    sequentially — never co-resident). `seams_done` counts
    boundary joints rendered early during the pass (surfaced on
    the verbose sweep line; the ledger stays frame-shaped).
    `skip_reason` is empty when both legs swept: it names the leg the
    VRAM guard held back (upscale or interp alone) or a busy-device pass
    the idle wait gave up on. Moot passes (no plan) return None instead —
    never a zero-count result — so the report can tell "nothing to do"
    (silent) from "tried but held back" (one compact note).
    """

    segments_seen: int
    upscale_chunks_done: int
    upscale_chunks_skipped: int
    interp_chunks_done: int
    interp_chunks_skipped: int
    interp_chunks_waiting: int
    upscale_frames_done: int = 0
    interp_frames_done: int = 0
    upscale_seconds: float = 0.0
    interp_seconds: float = 0.0
    skip_reason: str = ""
    seams_done: int = 0


PREWARM_NOTHING_LOUD_NOTE = "pre-warm did nothing — everything defers to finalize"
"""Loud nothing-all-pass line (DESIGN §§59/140).

A `prewarm_once` pass (or a whole generation run of passes) can do zero
work — VRAM guards hold back both legs, or the director-busy idle wait
gives up every pass — with only a per-leg `skip_reason` that the
post-commit report surfaces once and moot (`None`) passes staying silent
by design. That leaves the nothing-all-run case invisible on a compact
console: the operator cannot tell "pre-warm helped" from "finalize will
do everything". The loud line below names the outcome without gating on
`--verbose`; the per-leg reason stays verbose-only detail.
"""


def prewarm_pass_did_nothing(result: PrewarmResult | None) -> bool:
    """True when a finished pass rendered zero chunks and hit zero ledger entries.

    Why a helper (DESIGN §140): the post-commit report must tell
    "nothing to do" (moot `None` — knob off, no segments yet — silent by
    design) from "tried but did nothing" (a real zero-count result —
    loud). Only the latter returns True: every chunk counter (done +
    skipped + waiting on both legs) and every frame counter is zero.
    Skipped counts as activity (a ledger hit is useful work), so a pass
    that skipped everything is NOT nothing — finalize still benefits.
    """
    if result is None:
        return False
    return (
        result.upscale_chunks_done == 0
        and result.upscale_chunks_skipped == 0
        and result.interp_chunks_done == 0
        and result.interp_chunks_skipped == 0
        and result.interp_chunks_waiting == 0
        and result.upscale_frames_done == 0
        and result.interp_frames_done == 0
    )


def report_prewarm_pass(
    result: PrewarmResult | None,
    progress: Any,
    *,
    verbose: bool = False,
) -> None:
    """Emit the loud nothing line for a did-nothing pass (DESIGN §§59/140).

    Why this exists: `prewarm_once` callers (and the supervisor
    post-commit report — FOLLOW-UP: wire this helper into
    `Supervisor._report_background_prewarm`, which is under concurrent
    migration and intentionally untouched here) need one non-verbose-gated
    `progress.note` when a full pass did nothing, so the operator learns
    everything defers to finalize. Useful work (any chunk/frame activity)
    and moot (`None`) passes stay silent. The per-leg `skip_reason` rides
    a second line only when verbose (`--verbose` or an explicit
    `verbose=True`), keeping the compact console to one line.
    """
    if progress is None:
        return
    if not prewarm_pass_did_nothing(result):
        return
    progress.note(PREWARM_NOTHING_LOUD_NOTE)
    if result is None:  # Narrowing for mypy (unreachable: None never did nothing).
        return
    show_detail = verbose or bool(getattr(progress, "verbose", False))
    if show_detail and result.skip_reason:
        progress.note(f"pre-warm detail: {result.skip_reason}")


def probe_segment_source(video_path: Path) -> tuple[int, int, float]:
    """Source (width, height, fps) for one committed segment video.

    Separated for testability (the suite has no ffmpeg): finalize probes
    via `media.probe` + `_probe_video_fps`/`_probe_video_geometry`, and so
    does this default. Raises when the source is unprobable — the caller
    treats that as "nothing to pre-warm yet", never as an error.
    """
    from voyage.media import _probe_video_fps, _probe_video_geometry, probe

    info = probe(video_path)
    width, height = _probe_video_geometry(info)
    fps = _probe_video_fps(info)
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"unprobable segment source {video_path} ({width}x{height}@{fps})")
    return (width, height, float(fps))


def _first_committed_video(run_dir: Path) -> Path | None:
    """Video path of the oldest committed segment (finalize probes seg-0 too)."""
    from voyage.augment_upscale_poller import committed_segment_sources

    sources, _skipped = committed_segment_sources(run_dir)
    if not sources:
        return None
    return sources[0].video_path


def resolve_background_plan(run_dir: Path, config: Any) -> BackgroundPlan | None:
    """Derive the finalize-equivalent plan, or None when pre-warm is moot.

    None means: no work demanded (upscale == 1 and interpolate == 1),
    models dir unset, either weight leg absent (the durable finalize path
    needs both — partial legs keep the legacy all-or-nothing flow with
    nothing resumable to pre-warm), no model device visible, no committed
    segments yet, or the source unprobable. Every check is fail-soft:
    background work is advisory, never load-bearing.
    """
    augment = config.augment
    if augment.upscale <= 1 and augment.interpolate <= 1:
        return None
    models_dir = config.video.models_dir
    if models_dir is None:
        return None
    try:
        weights = augment_module.resolve_augment_weights(models_dir)
    except (TypeError, ValueError, OSError):
        return None
    interp_backend = getattr(config.augment, "interp_backend", "rife") or "rife"
    active_leg = augment_module.interp_leg_path(weights, interp_backend)
    if active_leg is None or weights.realesrgan is None:
        return None
    try:
        devices = model_pass_devices()
    except Exception:  # noqa: BLE001 - visibility probe must never fail generation
        return None
    if not devices:
        return None
    first_video = _first_committed_video(run_dir)
    if first_video is None:
        return None
    try:
        source_w, source_h, source_fps = probe_segment_source(first_video)
    except Exception:  # noqa: BLE001 - unprobable source means "not yet", not error
        return None
    multiplier = augment.interpolate
    plan = plan_augmentation(
        source_w,
        source_h,
        source_fps,
        upscale=augment.upscale,
        interpolate=multiplier,
        presentation_fps=augment.presentation_fps,
    )
    try:
        from voyage.augment_finalize import weights_key_for

        weights_key = weights_key_for(weights, interp_backend)
    except (TypeError, ValueError, OSError):
        return None
    return BackgroundPlan(
        out_width=plan.out_w,
        out_height=plan.out_h,
        out_fps=plan.out_fps,
        source_fps=source_fps,
        source_fps_key=int(round(source_fps)),
        upscale_factor=augment.upscale,
        multiplier=multiplier,
        crf=FINALIZE_CRF_DEFAULT,
        preset=FINALIZE_PRESET_DEFAULT,
        weights_key=weights_key,
        device=devices[0],
        interp_path=active_leg,
        realesrgan_path=weights.realesrgan,
        interp_backend=interp_backend,
    )


def _prewarm_resident_ready(plan: BackgroundPlan) -> tuple[bool, bool]:
    """Whether each leg's nets are already resident on the plan device.

    Reads the in-process augment-worker caches (no torch import, no
    weights touched — cache keys are (weights path, device) strings).
    A resident leg skips the cold-load floor in `prewarm_once` and is
    gated on `RESIDENT_MIN_FREE_GIB` instead: the H2D load already
    happened, so only forward working set must fit. Any lookup failure
    reads as not-resident (fail-safe toward the full floor).
    """
    try:
        from voyage.workers.augment_worker import (
            _ESRGAN_CACHE,
            _FILM_CACHE,
            _RIFE_CACHE,
            _model_cache_key,
        )
    except Exception:  # noqa: BLE001 - worker module shape is not this gate's contract
        return (False, False)
    try:
        up_ready = _model_cache_key(plan.realesrgan_path, plan.device) in _ESRGAN_CACHE
    except Exception:  # noqa: BLE001 - key shape failure reads as not-resident
        up_ready = False
    try:
        interp_cache = _FILM_CACHE if plan.interp_backend == "film" else _RIFE_CACHE
        ip_ready = _model_cache_key(plan.interp_path, plan.device) in interp_cache
    except Exception:  # noqa: BLE001 - key shape failure reads as not-resident
        ip_ready = False
    return (up_ready, ip_ready)


def prewarm_once(
    run_dir: Path,
    config: Any,
    *,
    upscale_poll_fn: Callable[..., Any] | None = None,
    interp_poll_fn: Callable[..., Any] | None = None,
    should_stop: Callable[[], bool] | None = None,
    joint_interp_fn: Callable[..., Any] | None = None,
    on_upscale_chunk: Callable[[str, int, int], None] | None = None,
    on_upscale_frames: Callable[[str, int], None] | None = None,
    on_interp_chunk: Callable[[str, int, int], None] | None = None,
    on_interp_frames: Callable[[str, int], None] | None = None,
    include_interp: bool = True,
) -> PrewarmResult | None:
    """Run one segment-interleaved upscale + interp pass (unless disabled).

    Each pass enumerates the committed segments once, then runs every
    segment's upscale immediately followed by its interp — interp starts
    on committed frames instead of waiting for the full upscale sweep
    (mirrors the finalize driver). Direct callers keep the default
    `include_interp=True` path; `include_interp=False` runs the upscale
    leg only (upscale-only by design — never a skip, never interp
    progress).

    Returns None when pre-warm is moot (see `resolve_background_plan`)
    or when `should_stop` fires after a segment's upscale (lets `stop()`
    abandon a pass without racing a live CUDA sweep at interpreter
    exit — accumulated counts are discarded, same as the old
    between-sweeps check). Raises `MediaError` on a failed chunk
    (fail-loud like finalize — the background thread catches it; direct
    callers such as tests see it).

    VRAM gating is per leg: the pass needs `UPSCALE_MIN_FREE_GIB` to
    start (SRVGG fits beside the resident sidecar), the interp leg
    probes once before the loop and needs `PREWARM_MIN_FREE_GIB` (RIFE
    fits beside it too). Legs whose nets are already resident in this
    process gate on the smaller `RESIDENT_MIN_FREE_GIB` instead — the
    H2D load already happened, so only forward working set must fit
    (without this, the first pass loads the nets and every later pass
    holds back on free net of its own resident footprint). A held-back
    leg yields a zero-count result carrying `skip_reason` instead of
    None, so the post-commit report can say what waited and why; both
    legs held back means no poller runs at all. Unknown free space
    stays fail-open on both legs.

    After the segment sweep, the pass runs the uniform joint sweep: the
    fix stage above rendered every boundary's source-res joints, and the
    joint units poll through the same legs via the pollers' `sources=`
    override (bridge frames upscale + re-interpolate by design).

    The `on_*` callbacks forward to the pollers' `on_chunk` /
    `on_chunk_frames` (fired per rendered chunk on the calling thread —
    skipped chunks never fire, so pass-end ledger deltas stay the source
    of truth for skipped frames). The background driver passes None
    unless the generation loop supplies live callbacks; any callback must
    never touch display code (it runs on the pre-warm thread — the main
    thread drains them into progress bars).
    """
    plan = resolve_background_plan(run_dir, config)
    if plan is None:
        return None
    if upscale_poll_fn is None:
        from voyage.augment_upscale_poller import upscale_poll_once

        upscale_poll_fn = upscale_poll_once
    if interp_poll_fn is None and include_interp:
        from voyage.augment_interp_poller import interp_poll_once

        interp_poll_fn = interp_poll_once
    up_ready, ip_ready = _prewarm_resident_ready(plan)
    upscale_floor = RESIDENT_MIN_FREE_GIB if up_ready else UPSCALE_MIN_FREE_GIB
    upscale_free = device_free_gib(plan.device)
    if upscale_free is not None and upscale_free < upscale_floor:
        return PrewarmResult(
            0,
            0,
            0,
            0,
            0,
            0,
            skip_reason=(
                f"upscale skipped: {upscale_free:.1f} GiB free on {plan.device} < "
                f"{upscale_floor:.1f} GiB needed"
            ),
        )
    upscale_seconds = 0.0
    skip_reason = ""
    interp_armed = bool(include_interp and interp_poll_fn is not None)
    interp_floor = RESIDENT_MIN_FREE_GIB if ip_ready else PREWARM_MIN_FREE_GIB
    if (
        interp_armed
        and (interp_free := device_free_gib(plan.device)) is not None
        and (interp_free < interp_floor)
    ):
        skip_reason = (
            f"interp skipped: {interp_free:.1f} GiB free on {plan.device} < "
            f"{interp_floor:.1f} GiB needed (waits for finalize)"
        )
        interp_armed = False
    from voyage.augment_finalize import _accepts_keyword
    from voyage.augment_joints import ensure_joint_units, existing_joint_videos
    from voyage.augment_upscale_poller import committed_segment_sources

    # Joints first (seam fix -> upscale -> interpolate): every adjacent
    # joint renders from the committed source videos before any segment
    # chunk work (ledger-hit idempotent with the in-loop attempts and
    # the finalize drain). Skipped while the interp leg is held back.
    # (Pre-pass runs below, after the counters, so fresh renders count
    # into `seams_early` for the post-commit report.)
    # Segment-interleaved pipeline (mirrors the finalize driver): enumerate
    # once per pass, then run each segment's upscale immediately followed
    # by its interp, so interp starts on committed frames instead of
    # waiting for the full upscale sweep. `should_stop` is honored after
    # each segment's upscale (same discard-on-stop shape as the old
    # between-sweeps check, but responsive mid-pass).
    segment_sources, _skipped_sources = committed_segment_sources(run_dir)
    ordered_sources = sorted(segment_sources, key=lambda source: source.segment_id)
    up_takes_segments = _accepts_keyword(upscale_poll_fn, "segment_ids")
    ip_takes_segments = (
        _accepts_keyword(interp_poll_fn, "segment_ids") if interp_poll_fn is not None else False
    )
    segments_seen = 0
    up_chunks_done = 0
    up_chunks_skipped = 0
    up_frames_done = 0
    ip_chunks_done = 0
    ip_chunks_skipped = 0
    ip_chunks_waiting = 0
    ip_frames_done = 0
    seams_early = 0
    ip_seconds = 0.0
    # Fix stage (seam fix -> upscale -> interpolate): fresh joints are the
    # joint videos absent before this ensure; ledger-hits cost nothing.
    seen_joint_videos = set(map(str, existing_joint_videos(run_dir)))
    joint_units = ensure_joint_units(
        run_dir,
        ordered_sources,
        source_fps=plan.source_fps_key,
        crf=plan.crf,
        preset=plan.preset,
        interp_fn=joint_interp_fn,
        weights=plan.interp_path,
        device=plan.device,
        interp_backend=plan.interp_backend,
    )
    seams_early += sum(1 for unit in joint_units if str(unit.joint_video) not in seen_joint_videos)
    joint_sources = [unit.as_source() for unit in joint_units]
    for source in ordered_sources:
        segment_id = source.segment_id
        up_kwargs: dict[str, Any] = {
            "weights_path": plan.realesrgan_path,
            "weights_key": plan.weights_key,
            "out_width": plan.out_width,
            "out_height": plan.out_height,
            "out_fps": plan.source_fps_key,
            "upscale_factor": plan.upscale_factor,
            "chunk_frames": BACKGROUND_CHUNK_FRAMES,
            "device": plan.device,
            "crf": plan.crf,
            "preset": plan.preset,
            "on_chunk": on_upscale_chunk,
            "on_chunk_frames": on_upscale_frames,
        }
        if up_takes_segments:
            up_kwargs["segment_ids"] = [segment_id]
        up_started = time.monotonic()
        up_result = upscale_poll_fn(run_dir, **up_kwargs)
        upscale_seconds += time.monotonic() - up_started
        segments_seen += int(getattr(up_result, "segments_seen", 0) or 0)
        up_chunks_done += int(getattr(up_result, "chunks_done", 0) or 0)
        up_chunks_skipped += int(getattr(up_result, "chunks_skipped", 0) or 0)
        up_frames_done += int(getattr(up_result, "frames_done", 0) or 0)
        if should_stop is not None and should_stop():
            return None
        if interp_armed and interp_poll_fn is not None:
            ip_kwargs: dict[str, Any] = {
                "weights_path": plan.interp_path,
                "weights_key": plan.weights_key,
                "out_width": plan.out_width,
                "out_height": plan.out_height,
                "out_fps": plan.source_fps_key,
                "upscale_factor": plan.upscale_factor,
                "chunk_frames": BACKGROUND_CHUNK_FRAMES,
                "multiplier": plan.multiplier,
                "device": plan.device,
                "crf": plan.crf,
                "preset": plan.preset,
                "interp_backend": plan.interp_backend,
                "on_chunk": on_interp_chunk,
                "on_chunk_frames": on_interp_frames,
            }
            if ip_takes_segments:
                ip_kwargs["segment_ids"] = [segment_id]
            ip_started = time.monotonic()
            ip_result = interp_poll_fn(run_dir, **ip_kwargs)
            ip_seconds += time.monotonic() - ip_started
            ip_chunks_done += int(getattr(ip_result, "chunks_done", 0) or 0)
            ip_chunks_skipped += int(getattr(ip_result, "chunks_skipped", 0) or 0)
            ip_chunks_waiting += int(getattr(ip_result, "chunks_waiting", 0) or 0)
            ip_frames_done += int(getattr(ip_result, "frames_done", 0) or 0)
    # Uniform pass over the fix-stage joints (sources= override): joints
    # upscale + re-interpolate by design. Joints are new units, so their
    # upscale runs even when the segment sweep was interp-only.
    if joint_sources:
        joint_up_kwargs: dict[str, Any] = {
            "weights_path": plan.realesrgan_path,
            "weights_key": plan.weights_key,
            "out_width": plan.out_width,
            "out_height": plan.out_height,
            "out_fps": plan.source_fps_key,
            "upscale_factor": plan.upscale_factor,
            "chunk_frames": BACKGROUND_CHUNK_FRAMES,
            "device": plan.device,
            "crf": plan.crf,
            "preset": plan.preset,
            "on_chunk": on_upscale_chunk,
            "on_chunk_frames": on_upscale_frames,
            "sources": joint_sources,
        }
        if up_takes_segments:
            joint_up_kwargs["segment_ids"] = [unit.segment_id for unit in joint_sources]
        joint_up_started = time.monotonic()
        joint_up_result = upscale_poll_fn(run_dir, **joint_up_kwargs)
        upscale_seconds += time.monotonic() - joint_up_started
        up_chunks_done += int(getattr(joint_up_result, "chunks_done", 0) or 0)
        up_chunks_skipped += int(getattr(joint_up_result, "chunks_skipped", 0) or 0)
        up_frames_done += int(getattr(joint_up_result, "frames_done", 0) or 0)
        if should_stop is not None and should_stop():
            return None
        if interp_armed and interp_poll_fn is not None:
            joint_ip_kwargs: dict[str, Any] = {
                "weights_path": plan.interp_path,
                "weights_key": plan.weights_key,
                "out_width": plan.out_width,
                "out_height": plan.out_height,
                "out_fps": plan.source_fps_key,
                "upscale_factor": plan.upscale_factor,
                "chunk_frames": BACKGROUND_CHUNK_FRAMES,
                "multiplier": plan.multiplier,
                "device": plan.device,
                "crf": plan.crf,
                "preset": plan.preset,
                "interp_backend": plan.interp_backend,
                "on_chunk": on_interp_chunk,
                "on_chunk_frames": on_interp_frames,
                "sources": joint_sources,
            }
            if ip_takes_segments:
                joint_ip_kwargs["segment_ids"] = [unit.segment_id for unit in joint_sources]
            joint_ip_started = time.monotonic()
            joint_ip_result = interp_poll_fn(run_dir, **joint_ip_kwargs)
            ip_seconds += time.monotonic() - joint_ip_started
            ip_chunks_done += int(getattr(joint_ip_result, "chunks_done", 0) or 0)
            ip_chunks_skipped += int(getattr(joint_ip_result, "chunks_skipped", 0) or 0)
            ip_chunks_waiting += int(getattr(joint_ip_result, "chunks_waiting", 0) or 0)
            ip_frames_done += int(getattr(joint_ip_result, "frames_done", 0) or 0)
    return PrewarmResult(
        segments_seen=segments_seen,
        upscale_chunks_done=up_chunks_done,
        upscale_chunks_skipped=up_chunks_skipped,
        interp_chunks_done=ip_chunks_done,
        interp_chunks_skipped=ip_chunks_skipped,
        interp_chunks_waiting=ip_chunks_waiting,
        upscale_frames_done=up_frames_done,
        interp_frames_done=ip_frames_done,
        upscale_seconds=round(upscale_seconds, 3),
        interp_seconds=round(ip_seconds, 3),
        seams_done=seams_early,
        skip_reason=skip_reason,
    )


class BackgroundPrewarm:
    """Single-thread pre-warm driver owned by the generation loop.

    The supervisor starts this in `start_workers`, notifies it after each
    commit, and stops it in `stop_workers`. Each notification triggers at
    most one `prewarm_once` pass, and only while `idle_fn` reports the
    augment device free (director wins every contention — the pre-warm
    never runs while a prefetch decide occupies `cuda:1`). Failures are
    swallowed: a busy GPU, a torn segment, or a missing binary just means
    finalize does the work later.

    The `on_upscale_*` callbacks (both None by default = today's silent
    behavior) forward per-rendered-chunk upscale events on THIS thread —
    they must never touch display code. The generation loop supplies
    queue-appending callbacks and drains them into progress bars on the
    main thread while the video render blocks. The `on_interp_*` callbacks
    (also None by default) forward per-rendered-chunk interp events the
    same way. The background pass runs both legs when VRAM allows and is
    silent unless the caller passes callbacks.
    """

    def __init__(
        self,
        run_dir: Path,
        config: Any,
        *,
        prewarm_fn: Callable[[Path, Any], PrewarmResult | None] | None = None,
        idle_fn: Callable[[], bool] | None = None,
        on_upscale_chunk: Callable[[str, int, int], None] | None = None,
        on_upscale_frames: Callable[[str, int], None] | None = None,
        on_interp_chunk: Callable[[str, int, int], None] | None = None,
        on_interp_frames: Callable[[str, int], None] | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._config = config
        if prewarm_fn is not None:
            # Test seam: custom drivers own the whole pass, so live
            # callbacks cannot flow through them (documented, not
            # forwarded — tests asserting callbacks use the default path
            # with fake poll fns).
            self._prewarm_fn = prewarm_fn
        else:
            self._prewarm_fn = lambda run_dir, config: prewarm_once(
                run_dir,
                config,
                should_stop=self._is_stopping,
                on_upscale_chunk=on_upscale_chunk,
                on_upscale_frames=on_upscale_frames,
                on_interp_chunk=on_interp_chunk,
                on_interp_frames=on_interp_frames,
                include_interp=True,
            )
        self._idle_fn = idle_fn or (lambda: True)
        self._condition = threading.Condition()
        self._pending = False
        self._stopping = False
        self._thread: threading.Thread | None = None
        # Cumulative ledgered work finished by passes so far — the
        # generation loop reads this after each commit and reports newly
        # ledgered work on the console (the background thread itself never
        # touches display code). (passes, upscale chunks, interp chunks,
        # upscale frames, interp frames, upscale seconds, interp seconds.)
        # One tuple store keeps the read GIL-atomic.
        self._ledgered: tuple[int, int, int, int, int, float, float] = (0, 0, 0, 0, 0, 0.0, 0.0)
        # Latest finished pass (for --verbose sweep lines); None before
        # the first pass or when the latest pass was moot/skipped.
        self._last_result: PrewarmResult | None = None

    def start(self) -> None:
        """Spawn the daemon thread (idempotent; an initial sweep is queued)."""
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._pending = True
            self._thread = threading.Thread(
                target=self._loop, name="voyage-background-prewarm", daemon=True
            )
            self._thread.start()

    def notify_committed(self) -> bool:
        """Wake the thread after a commit; False when not running."""
        with self._condition:
            if self._thread is None or not self._thread.is_alive():
                return False
            self._pending = True
            self._condition.notify()
            return True

    def is_alive(self) -> bool:
        """Whether the background thread is currently running."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def ledgered_totals(self) -> tuple[int, int, int]:
        """Cumulative (passes, upscale chunks, interp chunks) finished so far.

        Main-thread read for the post-commit pre-warm report; idle or
        moot passes still count (their result holds no new chunks).
        Kept chunk-shaped for older readers — prefer `ledgered_frames()`.
        """
        passes, up, ip, _upf, _ipf, _ups, _ips = self._ledgered
        return (passes, up, ip)

    def ledgered_frames(self) -> tuple[int, int, int, int, int, float, float]:
        """Cumulative (passes, up chunks, ip chunks, up frames, ip frames, up s, ip s).

        Main-thread read for the frame-unit pre-warm report and the
        persistent model-pass bar. Frame counts are source frames per
        leg, so both legs share one comparable unit.
        """
        return self._ledgered

    @property
    def last_result(self) -> PrewarmResult | None:
        """Latest finished pass (None before the first pass or when moot).

        Main-thread read for --verbose sweep lines. GIL-atomic attribute
        read; the background thread only ever replaces the object.
        """
        return self._last_result

    def did_nothing_all_run(self) -> bool:
        """True when every finished pass did nothing (DESIGN §140).

        Why cumulative, not per-pass: a single held-back pass already
        reports once via `skip_reason`, but a run where VRAM guards or
        the director-busy deadline held back EVERY pass looks identical
        to "pre-warm helped" on a compact console. Cumulative chunk and
        frame counters stay zero while at least one real (non-`None`)
        pass finished — moot-only runs (`_last_result is None`: knob off,
        no segments) stay silent by design, since nothing was demanded.
        GIL-atomic tuple read; main-thread only.
        """
        passes, upscale_chunks, interpolation_chunks = self.ledgered_totals()
        _passes, _up, _ip, upscale_frames, interpolation_frames, _ups, _ips = self._ledgered
        if passes <= 0:
            return False
        if (
            upscale_chunks != 0
            or interpolation_chunks != 0
            or upscale_frames != 0
            or interpolation_frames != 0
        ):
            return False
        last = self._last_result
        if last is None:
            return False
        return prewarm_pass_did_nothing(last)

    def report_at_stop(
        self,
        progress: Any,
        *,
        verbose: bool = False,
    ) -> None:
        """Emit the loud nothing-all-run line when pre-warm never helped.

        Why a stop report (DESIGN §§59/140): per-pass notes describe one
        sweep, but the operator's real question at the end of generation
        is "did pre-warm help at all?" — when `did_nothing_all_run`
        holds, one non-verbose-gated `progress.note` says everything
        defers to finalize. The per-leg reason stays verbose-only.
        FOLLOW-UP: wire this into the supervisor stop path (under
        concurrent migration, intentionally untouched here) — today it
        is a background-module helper its future caller owns.
        """
        if progress is None:
            return
        if not self.did_nothing_all_run():
            return
        progress.note(PREWARM_NOTHING_LOUD_NOTE)
        last = self._last_result
        show_detail = verbose or bool(getattr(progress, "verbose", False))
        if show_detail and last is not None and last.skip_reason:
            progress.note(f"pre-warm detail: {last.skip_reason}")

    def _is_stopping(self) -> bool:
        """Stop flag read for the between-sweeps abandon check (GIL-atomic)."""
        return self._stopping

    def stop(self) -> None:
        """Stop the thread (safe without `start`; joins briefly)."""
        with self._condition:
            thread = self._thread
            self._thread = None
            if thread is None:
                return
            self._stopping = True
            self._condition.notify()
        thread.join(timeout=5.0)

    def _loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait(timeout=1.0)
                if self._stopping:
                    return
                self._pending = False
            if not self._idle_fn():
                # The notify landed while the director holds the device
                # (prefetch decide) — wait for it to go idle instead of
                # losing the whole pass on a single-shot check. Past the
                # deadline the pass is recorded director-busy (a zero-count
                # result, so the report says so once) and the next commit
                # notifies again.
                deadline = time.monotonic() + PREWARM_IDLE_WAIT_SECONDS
                while not self._stopping:
                    if self._idle_fn():
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(PREWARM_IDLE_POLL_SECONDS)
                if self._stopping:
                    return
                if not self._idle_fn():
                    self._last_result = PrewarmResult(
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        skip_reason=(
                            f"director busy past {PREWARM_IDLE_WAIT_SECONDS:.0f}s idle wait"
                        ),
                    )
                    passes, up, ip, upf, ipf, ups, ips = self._ledgered
                    self._ledgered = (passes + 1, up, ip, upf, ipf, ups, ips)
                    continue
            try:
                result = self._prewarm_fn(self._run_dir, self._config)
                passes, up, ip, upf, ipf, ups, ips = self._ledgered
                if result is not None:
                    up += result.upscale_chunks_done
                    ip += result.interp_chunks_done
                    upf += result.upscale_frames_done
                    ipf += result.interp_frames_done
                    ups += result.upscale_seconds
                    ips += result.interp_seconds
                    self._last_result = result
                self._ledgered = (passes + 1, up, ip, upf, ipf, ups, ips)
            except Exception:  # noqa: BLE001 - pre-warm must never fail generation
                continue
