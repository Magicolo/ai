"""Frame diffusion pipeline loading, LoRA selection, and scheduler wiring.

One resident frame pipeline stays loaded (evicted when the family changes):
each holds gigabytes of CPU weights under offload, and several resident
families would overflow host RAM. LoRA adapters stay registered after
loading but only the selected names influence a pass, with a dirty check
so unchanged selections skip the weight swap entirely.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zoomy.errors import EngineConfigurationError, EngineExecutionError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from zoomy.family_catalog import FamilyDefinition, LoraDefinition

DIFFUSION_MODELS_DIRECTORY_NAME = "diffusion_models"
LORAS_DIRECTORY_NAME = "loras"

ERNIE_HUB_REPOSITORY = "baidu/ERNIE-Image-Turbo"
Z_TURBO_HUB_REPOSITORY = "Tongyi-MAI/Z-Image-Turbo"
Z_BASE_HUB_REPOSITORY = "Tongyi-MAI/Z-Image"
Z_HUB_REPOSITORY_BY_FAMILY_KEY = {
    "z_fast": Z_TURBO_HUB_REPOSITORY,
    "z_quality": Z_BASE_HUB_REPOSITORY,
}

_cached_transformer_config: str | None = None


@dataclass(slots=True)
class LoadedFramePipeline:
    """One resident frame pipeline plus its LoRA bookkeeping."""

    family_key: str
    pipeline: Any
    text_encoder_device: Any = None
    loaded_lora_names_by_file: dict[str, list[str]] = field(default_factory=dict)
    active_adapter_key: tuple[tuple[str, float], ...] | None = None
    # (family key, prompt) -> (text memory, text lengths) for Ernie passes.
    text_embedding_cache: dict[tuple[str, str], tuple[Any, Any]] = field(default_factory=dict)


def cached_transformer_config() -> str:
    """Return the Ernie transformer config path, downloading only once."""
    global _cached_transformer_config  # noqa: PLW0603
    if _cached_transformer_config is None:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        downloaded_config = hf_hub_download(ERNIE_HUB_REPOSITORY, "transformer/config.json")
        if not downloaded_config:
            message = f"Transformer config download failed: {ERNIE_HUB_REPOSITORY}"
            raise EngineConfigurationError(message)
        _cached_transformer_config = downloaded_config
    return _cached_transformer_config


def _enable_fast_offload(pipeline: Any) -> None:
    """Page whole models instead of streaming every layer every step.

    ``enable_model_cpu_offload`` keeps one model resident and pages the
    rest, which is markedly faster than per-layer sequential streaming on
    a 16 GB card. Sequential offload remains the fallback when the faster
    mode is unavailable (older diffusers, CPU-only test doubles).
    """
    if hasattr(pipeline, "enable_model_cpu_offload"):
        try:
            pipeline.enable_model_cpu_offload()
        except Exception:  # noqa: BLE001
            pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.enable_sequential_cpu_offload()


def _enable_vae_memory_savers(pipeline: Any) -> None:
    """Enable VAE slicing/tiling when the loaded autoencoder supports it."""
    autoencoder = getattr(pipeline, "vae", None)
    for method in ("enable_slicing", "enable_tiling"):
        enable = getattr(autoencoder, method, None)
        if callable(enable):
            with contextlib.suppress(Exception):
                enable()


def _compile_transformer(pipeline: Any) -> None:
    """Compile the transformer full-graph when torch.compile is available.

    Best-effort only: test doubles and older torch builds lack ``compile``,
    and a failed compile must never block a render — eager still works.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(torch, "compile"):
        return
    try:
        pipeline.transformer = torch.compile(transformer)
    except Exception:  # noqa: BLE001
        return


def _load_single_file_transformer(
    model_class: Any,
    diffusion_path: Path,
    torch_module: Any,
    **load_kwargs: Any,
) -> Any:
    """Load a local DiT single-file checkpoint, preferring fp8 dtypes.

    The shipped primaries are fp8-pruned; loading them directly in fp8
    halves the transient versus upcasting to bfloat16. A corrupt local
    file fails loud (naming the file) instead of silently swapping in
    official weights plus a surprise multi-GB download.
    """
    last_error: Exception | None = None
    for dtype in _transformer_dtype_candidates(torch_module):
        try:
            return model_class.from_single_file(
                str(diffusion_path), torch_dtype=dtype, **load_kwargs
            )
        except Exception as failure:  # noqa: BLE001
            last_error = failure
    cause = last_error
    if cause is None:
        cause = RuntimeError("no transformer dtype candidate succeeded")
    message = f"Local transformer failed to load ({diffusion_path}): {cause}"
    failure_error = EngineConfigurationError(message)
    raise failure_error from cause


def _transformer_dtype_candidates(torch_module: Any) -> list[Any]:
    """Prefer fp8 for the DiT, falling back to bfloat16 on failure.

    The shipped primaries are fp8-pruned; loading them directly in fp8
    halves the transient versus upcasting to bfloat16. Older torch builds
    lack the fp8 dtype, so the candidates degrade gracefully.
    """
    candidates: list[Any] = []
    for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
        dtype = getattr(torch_module, name, None)
        if dtype is not None:
            candidates.append(dtype)
    candidates.append(torch_module.bfloat16)
    return candidates


def load_ernie_pipeline(models_directory: Path, family: FamilyDefinition) -> Any:
    """Assemble the Ernie pipeline: local fp8 DiT, official everything else.

    The Comfy-format text encoder is not HuggingFace-format, so text
    encoder, VAE, tokenizer, and scheduler always come from the official
    ``baidu/ERNIE-Image-Turbo`` repository (cached after the first
    download). The transformer config downloads once per process — never
    the official bf16 weights.
    """
    import torch  # noqa: PLC0415
    from diffusers import ErnieImagePipeline, ErnieImageTransformer2DModel  # noqa: PLC0415

    diffusion_path = models_directory / DIFFUSION_MODELS_DIRECTORY_NAME / family.base_model_file
    if not diffusion_path.is_file():
        message = f"Diffusion model file is missing: {diffusion_path}"
        raise EngineConfigurationError(message)
    try:
        transformer_config = cached_transformer_config()
        transformer = _load_single_file_transformer(
            ErnieImageTransformer2DModel, diffusion_path, torch, config=transformer_config
        )
        pipeline = ErnieImagePipeline.from_pretrained(  # type: ignore[no-untyped-call]
            ERNIE_HUB_REPOSITORY,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
        _enable_fast_offload(pipeline)
        _enable_vae_memory_savers(pipeline)
        _compile_transformer(pipeline)
    except EngineConfigurationError:
        raise
    except Exception as failure:
        message = f"Ernie pipeline load failed: {failure}"
        raise EngineExecutionError("ernie-load", message) from failure
    return pipeline


def _load_local_autoencoder_or_none(
    models_directory: Path, family: FamilyDefinition, torch_module: Any
) -> Any:
    """Load the local VAE, returning None (official-VAE fallback) on failure.

    The shipped ae.safetensors is known-unloadable (32ch weights vs the
    8ch config), so any local VAE failure keeps the documented official
    fallback — announced in the server log instead of vanishing silently.
    """
    from diffusers import AutoencoderKL  # noqa: PLC0415

    autoencoder_path = models_directory / "vae" / family.autoencoder_file
    if not autoencoder_path.is_file():
        return None
    try:
        return AutoencoderKL.from_single_file(
            str(autoencoder_path), torch_dtype=torch_module.bfloat16
        )
    except Exception as failure:  # noqa: BLE001
        warnings.warn(
            f"Local autoencoder failed to load ({autoencoder_path}): "
            f"{failure}; using the official VAE instead.",
            stacklevel=2,
        )
        return None


def load_z_pipeline(models_directory: Path, family: FamilyDefinition) -> Any:
    """Assemble the Z-Image pipeline: local DiT, official fallback chain.

    The Juggernaut single-file tensors load plainly in fp8 when possible;
    the local VAE only fits the pipeline's autoencoder when its channel
    count matches, so a mismatch falls back to the official VAE. Text
    encoder, tokenizer, and scheduler always come from the official Z-Image
    repository matching the family (Turbo for fast, base for quality).
    """
    import torch  # noqa: PLC0415
    from diffusers import ZImageImg2ImgPipeline, ZImageTransformer2DModel  # noqa: PLC0415

    repository = Z_HUB_REPOSITORY_BY_FAMILY_KEY[family.key]
    diffusion_path = models_directory / DIFFUSION_MODELS_DIRECTORY_NAME / family.base_model_file
    if not diffusion_path.is_file():
        message = f"Diffusion model file is missing: {diffusion_path}"
        raise EngineConfigurationError(message)
    try:
        transformer = _load_single_file_transformer(ZImageTransformer2DModel, diffusion_path, torch)
        autoencoder = _load_local_autoencoder_or_none(models_directory, family, torch)
        pipeline_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "transformer": transformer,
        }
        if autoencoder is not None:
            pipeline_kwargs["vae"] = autoencoder
        pipeline = ZImageImg2ImgPipeline.from_pretrained(  # type: ignore[no-untyped-call]
            repository, **pipeline_kwargs
        )
        configure_z_scheduler(pipeline, family)
        _enable_fast_offload(pipeline)
        _enable_vae_memory_savers(pipeline)
        _compile_transformer(pipeline)
    except EngineConfigurationError:
        raise
    except Exception as failure:
        message = f"Z-Image pipeline load failed: {failure}"
        raise EngineExecutionError("z-load", message) from failure
    return pipeline


def configure_z_scheduler(pipeline: Any, family: FamilyDefinition) -> None:
    """Apply the catalog sampler recipe to a Z-Image pipeline, best-effort.

    ``z_fast`` ships DDIM/normal while ``z_quality`` ships
    res_multistep/beta with shift 3.0; previously the call passed only
    steps and cfg, so both variants diffused with the repository default
    scheduler. This sets the shift when the scheduler exposes it and swaps
    the scheduler class when diffusers offers the named one — silently
    keeping the default when it does not, so unknown scheduler builds
    still render.
    """
    scheduler = getattr(pipeline, "scheduler", None)
    if scheduler is None:
        return
    if family.model_shift is not None:
        config = getattr(scheduler, "config", None)
        if config is not None and hasattr(config, "shift"):
            with contextlib.suppress(Exception):
                config.shift = family.model_shift
    sampler_name = (family.sampler_name or "").lower()
    if "ddim" in sampler_name:
        scheduler_class_name = "DDIMScheduler"
    elif "multistep" in sampler_name or "res" in sampler_name:
        scheduler_class_name = "FlowMatchEulerDiscreteScheduler"
    else:
        return
    try:
        from diffusers import (  # noqa: PLC0415
            DDIMScheduler,
            FlowMatchEulerDiscreteScheduler,
        )
    except ImportError:
        return
    wanted = {
        "DDIMScheduler": DDIMScheduler,
        "FlowMatchEulerDiscreteScheduler": (FlowMatchEulerDiscreteScheduler),
    }[scheduler_class_name]
    if isinstance(scheduler, wanted):
        return
    try:
        pipeline.scheduler = wanted.from_config(scheduler.config)
    except Exception:  # noqa: BLE001
        return


def apply_lora_selection(
    loaded: LoadedFramePipeline,
    models_directory: Path,
    lora_selections: Sequence[tuple[LoraDefinition, float]],
) -> None:
    """Load newly selected LoRAs once, then activate exactly the selection.

    Adapters stay registered after loading but only the names passed to
    ``set_adapters`` influence the pass, so deselecting is a weight swap —
    no reload, no pipeline duplication. Unchanged selections skip the swap
    entirely via the active-adapter dirty check.
    """
    adapter_key = tuple(
        (definition.file_name, strength) for definition, strength in lora_selections
    )
    if not adapter_key:
        loaded.active_adapter_key = ()
        return
    if loaded.active_adapter_key == adapter_key:
        return
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    for definition, strength in lora_selections:
        if definition.file_name not in loaded.loaded_lora_names_by_file:
            lora_path = models_directory / LORAS_DIRECTORY_NAME / definition.file_name
            if not lora_path.is_file():
                message = f"LoRA file is missing: {lora_path}"
                raise EngineConfigurationError(message)
            try:
                loaded.pipeline.load_lora_weights(str(lora_path))
                registered = next(iter(loaded.pipeline.get_list_adapters().values()))
            except Exception as failure:
                message = f"LoRA load failed for {definition.file_name}: {failure}"
                raise EngineExecutionError("lora", message) from failure
            loaded.loaded_lora_names_by_file[definition.file_name] = (
                registered if isinstance(registered, list) else [registered]
            )
        names = loaded.loaded_lora_names_by_file[definition.file_name]
        adapter_names.extend(names)
        adapter_weights.extend([strength] * len(names))
    try:
        loaded.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
    except Exception as failure:
        message = f"LoRA activation failed: {failure}"
        raise EngineExecutionError("lora", message) from failure
    loaded.active_adapter_key = adapter_key
