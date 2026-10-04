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
    upscale_factor_for,
)

#: Source frames per pre-warm chunk (mirrors the poller/finalize default).
BACKGROUND_CHUNK_FRAMES = 32

#: Pre-warm runs one upscale sweep then one interp sweep per pass (same
#: stagger as finalize: the legs share `cuda:1`, never co-resident).
BACKGROUND_DEVICE_FALLBACK = "cuda:1"

#: Minimum free VRAM (GiB) on the pre-warm device for a sweep to start.
#: The llama director sidecar permanently holds ~5GB of the 6GB 2060, so
#: an ungated sweep allocates torch tensors with ~13MB free and spams
#: CUDACachingAllocator expandable-segments warnings (poualh 2026-10-04:
#: 25x on device 1). Below this the pass is skipped — finalize covers the
#: same ledger keys later. Unknown (no nvidia-smi / parse failure) allows
#: the sweep: the pre-warm must never go quiet on an unprobable box.
PREWARM_MIN_FREE_GIB = 3.0


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
    film_path: Path
    realesrgan_path: Path


@dataclass(frozen=True)
class PrewarmResult:
    """Counts from one pre-warm pass (advisory; the ledger is the truth)."""

    segments_seen: int
    upscale_chunks_done: int
    upscale_chunks_skipped: int
    interp_chunks_done: int
    interp_chunks_skipped: int
    interp_chunks_waiting: int


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

    None means: knob off, models dir unset, either weight leg absent (the
    durable finalize path needs both — partial legs keep the legacy
    all-or-nothing flow with nothing resumable to pre-warm), no model
    device visible, no committed segments yet, or the source unprobable.
    Every check is fail-soft: background work is advisory, never load-bearing.
    """
    augment = config.augment
    if not augment.use_model_pass:
        return None
    models_dir = config.video.models_dir
    if models_dir is None:
        return None
    try:
        weights = augment_module.resolve_augment_weights(models_dir)
    except (TypeError, ValueError, OSError):
        return None
    if weights.film is None or weights.realesrgan is None:
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
    multiplier = augment.interp_multiplier
    plan = plan_augmentation(
        source_w,
        source_h,
        source_fps,
        config.video.width,
        config.video.height,
        config.video.fps,
        augment.min_fps,
        augment.min_width,
        augment.min_height,
        interp_multiplier=multiplier,
    )
    try:
        from voyage.augment_finalize import weights_key_for

        weights_key = weights_key_for(weights)
    except (ValueError, OSError):
        return None
    return BackgroundPlan(
        out_width=plan.out_w,
        out_height=plan.out_h,
        out_fps=plan.out_fps,
        source_fps=source_fps,
        source_fps_key=int(round(source_fps)),
        upscale_factor=upscale_factor_for(source_w, source_h, plan.out_w, plan.out_h),
        multiplier=multiplier,
        crf=FINALIZE_CRF_DEFAULT,
        preset=FINALIZE_PRESET_DEFAULT,
        weights_key=weights_key,
        device=devices[0],
        film_path=weights.film,
        realesrgan_path=weights.realesrgan,
    )


def prewarm_once(
    run_dir: Path,
    config: Any,
    *,
    upscale_poll_fn: Callable[..., Any] | None = None,
    interp_poll_fn: Callable[..., Any] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> PrewarmResult | None:
    """Run one upscale sweep + one interp sweep (staggered, `cuda:1`-only).

    Returns None when pre-warm is moot (see `resolve_background_plan`),
    when the device lacks `PREWARM_MIN_FREE_GIB` headroom (finalize covers
    the same ledger keys later), or when `should_stop` fires between the
    sweeps (lets `stop()` abandon a pass without racing a live CUDA sweep
    at interpreter exit). Raises `MediaError` on a failed chunk
    (fail-loud like finalize — the background thread catches it; direct
    callers such as tests see it).
    """
    plan = resolve_background_plan(run_dir, config)
    if plan is None:
        return None
    free_gib = device_free_gib(plan.device)
    if free_gib is not None and free_gib < PREWARM_MIN_FREE_GIB:
        return None
    if upscale_poll_fn is None:
        from voyage.augment_upscale_poller import upscale_poll_once

        upscale_poll_fn = upscale_poll_once
    if interp_poll_fn is None:
        from voyage.augment_interp_poller import interp_poll_once

        interp_poll_fn = interp_poll_once
    upscale_result = upscale_poll_fn(
        run_dir,
        weights_path=plan.realesrgan_path,
        weights_key=plan.weights_key,
        out_width=plan.out_width,
        out_height=plan.out_height,
        out_fps=plan.source_fps_key,
        upscale_factor=plan.upscale_factor,
        chunk_frames=BACKGROUND_CHUNK_FRAMES,
        device=plan.device,
        crf=plan.crf,
        preset=plan.preset,
    )
    if should_stop is not None and should_stop():
        return None
    interp_result = interp_poll_fn(
        run_dir,
        weights_path=plan.film_path,
        weights_key=plan.weights_key,
        out_width=plan.out_width,
        out_height=plan.out_height,
        out_fps=plan.source_fps_key,
        upscale_factor=plan.upscale_factor,
        chunk_frames=BACKGROUND_CHUNK_FRAMES,
        multiplier=plan.multiplier,
        device=plan.device,
        crf=plan.crf,
        preset=plan.preset,
    )
    return PrewarmResult(
        segments_seen=upscale_result.segments_seen,
        upscale_chunks_done=upscale_result.chunks_done,
        upscale_chunks_skipped=upscale_result.chunks_skipped,
        interp_chunks_done=interp_result.chunks_done,
        interp_chunks_skipped=interp_result.chunks_skipped,
        interp_chunks_waiting=interp_result.chunks_waiting,
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
    """

    def __init__(
        self,
        run_dir: Path,
        config: Any,
        *,
        prewarm_fn: Callable[[Path, Any], PrewarmResult | None] | None = None,
        idle_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._config = config
        self._prewarm_fn = prewarm_fn or (
            lambda run_dir, config: prewarm_once(run_dir, config, should_stop=self._is_stopping)
        )
        self._idle_fn = idle_fn or (lambda: True)
        self._condition = threading.Condition()
        self._pending = False
        self._stopping = False
        self._thread: threading.Thread | None = None

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
            try:
                if self._idle_fn():
                    self._prewarm_fn(self._run_dir, self._config)
            except Exception:  # noqa: BLE001 - pre-warm must never fail generation
                continue
