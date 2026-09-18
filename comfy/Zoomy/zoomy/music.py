"""ACE-Step instrumental music rendering for finalize windows.

The diffusion plus language-model stack stays resident across windows of
one finalize (previously every window loaded and evicted it, paying N
multi-GB loads per finalize). Long finalizes additionally render one
global track and slice per-window stems out of it instead of running one
diffusion pass per window.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zoomy.engine_protocol import MUSIC_BEATS_PER_MINUTE
from zoomy.errors import EngineConfigurationError, EngineExecutionError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def load_music_stack(music_project_directory: Path, device: str) -> dict[str, Any]:
    """Initialize the ACE-Step diffusion plus language-model handlers."""
    from zoomy.engine_protocol import (  # noqa: PLC0415
        ACE_DIFFUSION_CONFIG_NAME,
        ACE_LANGUAGE_BACKEND,
        ACE_LANGUAGE_MODEL_NAME,
    )
    from zoomy.vendor_compat import apply_transformers5_compat  # noqa: PLC0415

    apply_transformers5_compat()
    try:
        from acestep.handler import AceStepHandler  # noqa: PLC0415
        from acestep.llm_inference import LLMHandler  # noqa: PLC0415
    except ImportError as failure:
        message = (
            "ACE-Step package is not installed; the GPU image clones it "
            f"to /opt/ACE-Step-1.5: {failure}"
        )
        raise EngineConfigurationError(message) from failure
    try:
        diffusion_handler = AceStepHandler()
        diffusion_handler.initialize_service(
            project_root=str(music_project_directory),
            config_path=ACE_DIFFUSION_CONFIG_NAME,
            device=device,
        )
        language_handler = LLMHandler()
        language_handler.initialize(
            checkpoint_dir=str(music_project_directory / "checkpoints"),
            lm_model_path=ACE_LANGUAGE_MODEL_NAME,
            backend=ACE_LANGUAGE_BACKEND,
            device=device,
        )
    except Exception as failure:
        message = f"ACE-Step initialization failed: {failure}"
        raise EngineExecutionError("music-load", message) from failure
    return {
        "diffusion_handler": diffusion_handler,
        "language_handler": language_handler,
    }


@dataclass(slots=True)
class MusicRenderRequest:
    """Everything one instrumental stem needs besides the loaded stack."""

    caption: str
    duration_seconds: float
    stem_path: Path
    seed: int


def render_music_track(
    music_stack: dict[str, Any],
    request: MusicRenderRequest,
    raise_if_interrupted: Callable[[], None],
) -> None:
    """Render one instrumental stem with an already-loaded ACE-Step stack.

    The caller owns stack residency (load once per finalize, evict after
    the last window or before the effects stack loads) so this stays a
    pure render step with no load/evict side effects.
    """
    try:
        from acestep.inference import (  # noqa: PLC0415
            GenerationConfig,
            GenerationParams,
            generate_music,
        )
    except ImportError as failure:
        message = (
            "ACE-Step package is not installed; the GPU image clones it "
            f"to /opt/ACE-Step-1.5: {failure}"
        )
        raise EngineConfigurationError(message) from failure
    raise_if_interrupted()
    try:
        parameters = GenerationParams(
            caption=request.caption,
            lyrics="",
            bpm=MUSIC_BEATS_PER_MINUTE,
            duration=request.duration_seconds,
            seed=request.seed,
        )
        config = GenerationConfig(audio_format="flac")
        result = generate_music(
            music_stack["diffusion_handler"],
            music_stack["language_handler"],
            parameters,
            config,
            save_dir=str(request.stem_path.parent),
        )
    except Exception as failure:
        message = f"Music generation failed: {failure}"
        raise EngineExecutionError("music", message) from failure
    raise_if_interrupted()
    if not result.success or not result.audios:
        message = f"Music generation failed: {result.error}"
        raise EngineExecutionError("music", message)
    try:
        shutil.move(result.audios[0]["path"], request.stem_path)
    except OSError as failure:
        message = f"Music stem move failed: {failure}"
        raise EngineExecutionError("music", message) from failure
