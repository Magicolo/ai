"""ACE-Step 1.5 music backend compat (DESIGN §37).

Every ACE-Step API touchpoint lives in this module — never in the worker
or the supervisor. Upstream's handler/inference API drifts between
releases (handler method vs `inference.generate_music`, GenerationParams
field names), so exactly one file needs auditing when the pin moves.

Checkpoint layout (populated by `voyage models download audio-acestep`):
`<models_dir>/acestep/` is the ACE project root holding `checkpoints/`
with all four MAIN_MODEL_COMPONENTS (turbo DiT + VAE + text encoder +
the 1.7B default LM, which only satisfies the handler's gate) plus the
0.6B planner LM submodel. The stack is resident while held (~12.3GB on
cuda:0) and must be evicted before the video DiT reloads — sequential
residency, never co-resident with LongLive.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voyage.errors import VoyageError

TURBO_CONFIG = "acestep-v15-turbo"
PLANNER_MODEL = "acestep-5Hz-lm-0.6B"
"""LLM path relative to checkpoint_dir (upstream joins the two)."""
PROJECT_SUBDIR = "acestep"
MIN_DURATION_SECONDS = 1.0
"""ACE-Step rejects anything below 1.0s (the floor zoomy hit in §10)."""
VALID_TASK_TYPES = ("text2music", "repaint")
"""ACE-Step generation modes used by the voyage (fresh take vs continuation)."""
MIN_BPM = 1
MAX_BPM = 300
"""Explicit-tempo bounds: 0 bypassed the energy clamp, absurd values fail late."""


def validate_bpm(bpm: int | None) -> None:
    """Reject explicit tempos outside the musical range (issue 063)."""
    if bpm is not None and (bpm < MIN_BPM or bpm > MAX_BPM):
        raise ValueError(f"bpm must be None or {MIN_BPM}..{MAX_BPM} (got {bpm})")


def validate_duration_seconds(duration_seconds: float) -> None:
    """Reject non-positive/non-finite durations (issue 063).

    Positive-but-short values still hit the ACE 1.0 s floor in
    `render_take` (an upstream requirement, not a silent coercion);
    zero/negative/NaN/inf are caller bugs and fail here instead.
    """
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(f"duration_seconds must be finite and > 0 (got {duration_seconds})")


def validate_task_type(task_type: str) -> None:
    """Reject unknown ACE-Step generation modes (issue 063)."""
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {VALID_TASK_TYPES} (got {task_type!r})")


def validate_reference_audio(src_audio: str | Path | None) -> None:
    """Reject non-path or missing repaint sources (issue 063)."""
    if src_audio is None:
        return
    if not isinstance(src_audio, (str, Path)) or not Path(src_audio).is_file():
        raise ValueError(
            f"reference_audio must be an existing audio file or None (got {src_audio!r})"
        )


def is_oom(failure: BaseException) -> bool:
    """True when `failure` is an out-of-memory (issue 052).

    Matches two shapes without importing torch (this module keeps heavy
    imports function-local): the OOM exception class itself, by name
    (`torch.cuda.OutOfMemoryError` and the `torch.OutOfMemoryError` alias
    both end there), and any error whose message carries the allocator's
    `out of memory` text (e.g. a RuntimeError re-raised by upstream code).
    OOMs stay retryable — callers re-raise them unwrapped so the worker
    loop maps them to WORKER_ERROR instead of fatal VoyageError.
    """
    if type(failure).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(failure).lower()


@dataclass
class AceStepStack:
    diffusion: Any
    language: Any
    device: str


def bpm_for_energy(energy: float) -> int:
    """Map the 0..1 director energy knob onto a musical tempo (60..140 BPM)."""
    return min(140, max(60, 80 + round(energy * 60.0)))


def initialize(models_dir: str | Path, device: str) -> AceStepStack:
    """Load the turbo DiT + planner LM and return the resident stack."""
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    project_root = Path(models_dir) / PROJECT_SUBDIR
    try:
        diffusion = AceStepHandler()
        diffusion.initialize_service(
            project_root=str(project_root), config_path=TURBO_CONFIG, device=device
        )
        language = LLMHandler()
        # offload_to_cpu=True keeps the 0.6B planner LM on CPU when idle and
        # moves it back after each planning phase (upstream _load_model_context).
        # Without it the LM stays resident (~2.4GB) and the DiT VRAM preflight
        # fails with "only 0.7 GB available" (probe-verified 2026-09-22).
        language.initialize(
            checkpoint_dir=str(project_root / "checkpoints"),
            lm_model_path=PLANNER_MODEL,
            backend="pt",
            device=device,
            offload_to_cpu=True,
        )
    except Exception as failure:
        # OOM during load stays retryable (issue 052): wrapping it as the
        # base VoyageError would bypass the supervisor restart budget.
        if is_oom(failure):
            raise
        raise VoyageError(f"ACE-Step stack init failed: {failure}") from failure
    # Upstream swallows load-time OOMs internally (model left as None) and
    # only fails at first render with "Model not fully initialized" — fail
    # fast here instead, mirroring its own readiness gate.
    missing = [
        name
        for name in ("model", "vae", "text_tokenizer", "text_encoder")
        if getattr(diffusion, name, None) is None
    ]
    if missing:
        raise VoyageError(
            "ACE-Step stack init incomplete"
            f" (missing: {', '.join(missing)} — likely OOM during load)"
        )
    return AceStepStack(diffusion=diffusion, language=language, device=device)


def render_take(
    stack: AceStepStack,
    caption: str,
    duration_seconds: float,
    seed: int,
    save_path: str | Path,
    energy: float = 0.5,
    task_type: str = "text2music",
    src_audio: str | Path | None = None,
    repaint_start: float = 0.0,
    repaint_end: float = -1.0,
    bpm: int | None = None,
) -> Path:
    """Render one FLAC take; continuation via task_type=repaint + src_audio.

    Repaint semantics (probe-verified): the 0..repaint_start head is
    preserved near bit-exact while repaint_start..repaint_end is
    regenerated — the mechanism the slow loop uses to extend music.

    `bpm` overrides the energy-derived tempo (beat-grid takes pass the
    grid BPM so segment cuts land on beats); None keeps the legacy
    energy mapping.
    """
    # Validate before the upstream import (issue 038): a bad call must
    # fail as ValueError on CPU, never as ModuleNotFoundError, and never
    # after paying for a heavy import.
    validate_bpm(bpm)
    validate_duration_seconds(duration_seconds)
    validate_task_type(task_type)
    validate_reference_audio(src_audio)
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    params = GenerationParams(
        caption=caption,
        lyrics="",
        bpm=bpm if bpm is not None else bpm_for_energy(energy),
        duration=max(MIN_DURATION_SECONDS, duration_seconds),
        seed=seed,
        task_type=task_type,
        src_audio=None if src_audio is None else str(src_audio),
        repainting_start=repaint_start,
        repainting_end=repaint_end,
        repaint_wav_crossfade_sec=1.0 if task_type == "repaint" else 0.0,
    )
    try:
        result = generate_music(
            stack.diffusion,
            stack.language,
            params,
            GenerationConfig(audio_format="flac"),
            save_dir=str(save_path.parent),
        )
    except Exception as failure:
        # OOM during the take render stays retryable (issue 052): the
        # 45 s DiT render is the known transient-OOM site on 16 GiB cards,
        # and wrapping it as the base VoyageError would fail the run with
        # no restart instead of evicting + retrying.
        if is_oom(failure):
            raise
        raise VoyageError(f"ACE-Step render failed: {failure}") from failure
    if not result.success or not result.audios:
        raise VoyageError(f"ACE-Step render failed: {result.error}")
    try:
        shutil.move(result.audios[0]["path"], save_path)
    except OSError as failure:
        raise VoyageError(f"ACE-Step take move failed: {failure}") from failure
    return save_path


def evict(stack: AceStepStack) -> None:
    """Release the resident stack (sequential residency with video)."""
    import gc

    import torch

    del stack.diffusion
    del stack.language
    # gc first: reference cycles keep GPU tensors alive past `del` (same
    # lesson as the video session evict — empty_cache alone frees nothing).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
