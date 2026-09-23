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
) -> Path:
    """Render one FLAC take; continuation via task_type=repaint + src_audio.

    Repaint semantics (probe-verified): the 0..repaint_start head is
    preserved near bit-exact while repaint_start..repaint_end is
    regenerated — the mechanism the slow loop uses to extend music.
    """
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    params = GenerationParams(
        caption=caption,
        lyrics="",
        bpm=bpm_for_energy(energy),
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
