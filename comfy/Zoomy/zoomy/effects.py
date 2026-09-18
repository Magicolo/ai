"""MMAudio video-synced sound-effects rendering for finalize windows.

The stack (transformer, autoencoder, synchformer, CLIP features) stays
resident across every window of one finalize — previously each window loaded
it from disk and evicted it, paying N multi-GB loads per finalize — and is
evicted before assembly or when the music stack loads (the two never fit
together on a 16 GB card). The flow-matching sampler is built once with the
stack, not once per window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zoomy import final_assembly
from zoomy.engine_protocol import (
    SOUND_EFFECT_CLASSIFIER_FREE_GUIDANCE,
    SOUND_EFFECT_CLIP_FILE,
    SOUND_EFFECT_MODE,
    SOUND_EFFECT_MODEL_FILE,
    SOUND_EFFECT_STEPS,
    SOUND_EFFECT_SYNC_FRAME_PIXELS,
    SOUND_EFFECT_SYNC_FRAMES_PER_SECOND,
    SOUND_EFFECT_SYNCHFORMER_CONFIG,
    SOUND_EFFECT_SYNCHFORMER_FILE,
    SOUND_EFFECT_VAE_FILE,
)
from zoomy.errors import EngineConfigurationError, EngineExecutionError, RenderInterruptedError
from zoomy.video_io import pad_frames_to_minimum

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SOUND_EFFECTS_DIRECTORY_NAME = "mmaudio"
# Second text-embedding width marking a v2 MMAudio checkpoint (v1 uses 768).
_MMAUDIO_V2_TEXT_WIDTH = 896


@dataclass(slots=True)
class EffectsLoaders:
    """Lazily imported primitives shared by the MMAudio loader helpers."""

    torch_module: Any
    init_empty_weights: Any
    set_module_tensor_to_device: Any
    load_file: Any
    vendor: dict[str, Any]
    effects_directory: Path


@dataclass(slots=True)
class LoadedEffectsStack:
    """One resident MMAudio stack, shared by every window of a finalize."""

    model: Any
    feature_utils: Any
    sync_transform: Any
    flow_matching: Any
    generate: Any


def load_effects_stack(models_directory: Path) -> LoadedEffectsStack:
    """Load the MMAudio fp16 stack once, mirroring the node loaders."""
    torch_module, init_empty_weights, set_module_tensor_to_device, load_file = (
        _import_effects_loaders()
    )
    vendor = _import_effects_vendor()
    _require_effects_files(models_directory)
    effects_directory = models_directory / SOUND_EFFECTS_DIRECTORY_NAME
    loaders = EffectsLoaders(
        torch_module=torch_module,
        init_empty_weights=init_empty_weights,
        set_module_tensor_to_device=set_module_tensor_to_device,
        load_file=load_file,
        vendor=vendor,
        effects_directory=effects_directory,
    )
    try:
        model = _load_effects_transformer(loaders)
        autoencoder, synchformer = _load_effects_autoencoder(loaders)
        clip_model = _load_effects_clip_model(loaders)
        feature_utils = vendor["FeaturesUtils"](
            vae=autoencoder,
            synchformer=synchformer,
            enable_conditions=True,
            clip_model=clip_model,
        )
        sync_transform = _build_effects_sync_transform(torch_module, vendor["v2"])
        flow_matching = vendor["FlowMatching"](
            min_sigma=0, inference_mode="euler", num_steps=SOUND_EFFECT_STEPS
        )
    except EngineConfigurationError:
        raise
    except Exception as failure:
        message = f"Sound-effect model load failed: {failure}"
        raise EngineExecutionError("sound-effects-load", message) from failure
    return LoadedEffectsStack(
        model=model,
        feature_utils=feature_utils,
        sync_transform=sync_transform,
        flow_matching=flow_matching,
        generate=vendor["generate"],
    )


@dataclass(slots=True)
class SoundEffectsRenderRequest:
    """Everything one video-synced effects stem needs besides the stack."""

    prompt: str
    negative_prompt: str | None
    frame_arrays: Sequence[Any]
    duration_seconds: float
    stem_path: Path
    device: str
    seed: int
    minimum_sync_frames: int


def render_sound_effects_track(
    stack: LoadedEffectsStack,
    request: SoundEffectsRenderRequest,
    raise_if_interrupted: Callable[[], None],
) -> None:
    """Render one video-synced effects stem with an already-loaded stack.

    The caller owns stack residency (load once per finalize, evict after the
    last window or before the music stack loads) so this stays a pure render
    step with no load/evict side effects. ``request.frame_arrays`` are the
    shared interpolated numpy buffers — stacked directly, never re-opened or
    re-converted — padded to the synchformer minimum, then truncated to the
    needed sync frames *before* the per-frame transform runs, so padded
    excess on long renders is never transformed just to be discarded.
    """
    raise_if_interrupted()
    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        padded_arrays = pad_frames_to_minimum(
            list(request.frame_arrays), request.minimum_sync_frames
        )
        needed_sync_frames = min(
            len(padded_arrays),
            int(SOUND_EFFECT_SYNC_FRAMES_PER_SECOND * request.duration_seconds),
        )
        video = torch.from_numpy(np.stack(padded_arrays[:needed_sync_frames])).float().div_(255.0)
        frames_channel_first = video.permute(0, 3, 1, 2)
        sync_frames = torch.stack([stack.sync_transform(frame) for frame in frames_channel_first])
        resolved_seconds = sync_frames.shape[0] / SOUND_EFFECT_SYNC_FRAMES_PER_SECOND
        model = stack.model
        model.seq_cfg.duration = resolved_seconds
        model.update_seq_lengths(
            model.seq_cfg.latent_seq_len,
            model.seq_cfg.clip_seq_len,
            model.seq_cfg.sync_seq_len,
        )
        generator = torch.Generator(device=request.device).manual_seed(request.seed)
        feature_utils = stack.feature_utils
        torch_device = torch.device(request.device)
        feature_utils.to(torch_device)
        model.to(torch_device)
        with torch.no_grad():
            audios = stack.generate(
                None,
                sync_frames.unsqueeze(0).to(torch_device),
                [request.prompt],
                negative_text=[request.negative_prompt or ""],
                feature_utils=feature_utils,
                net=model,
                fm=stack.flow_matching,
                rng=generator,
                cfg_strength=SOUND_EFFECT_CLASSIFIER_FREE_GUIDANCE,
            )
        import soundfile  # noqa: PLC0415

        waveform = audios.float().cpu()
        soundfile.write(
            str(request.stem_path), waveform[0].T.numpy(), final_assembly.OUTPUT_SAMPLE_RATE
        )
    except RenderInterruptedError:
        raise
    except Exception as failure:
        message = f"Sound-effect generation failed: {failure}"
        raise EngineExecutionError("sound-effects", message) from failure
    raise_if_interrupted()


def _require_effects_files(models_directory: Path) -> None:
    """Reject missing sound-effect files before any model loads."""
    effects_directory = models_directory / SOUND_EFFECTS_DIRECTORY_NAME
    for file_name in (
        SOUND_EFFECT_MODEL_FILE,
        SOUND_EFFECT_VAE_FILE,
        SOUND_EFFECT_SYNCHFORMER_FILE,
        SOUND_EFFECT_CLIP_FILE,
    ):
        if not (effects_directory / file_name).is_file():
            message = f"Sound-effect model file is missing: {effects_directory / file_name}"
            raise EngineConfigurationError(message)


def _import_effects_loaders() -> tuple[Any, Any, Any, Any]:
    """Import the torch/accelerate/safetensors loading primitives for MMAudio."""
    try:
        import torch  # noqa: PLC0415
        from accelerate import init_empty_weights  # noqa: PLC0415
        from accelerate.utils import set_module_tensor_to_device  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415
    except ImportError as failure:
        message = f"MMAudio loader dependencies are missing: {failure}"
        raise EngineConfigurationError(message) from failure
    return (torch, init_empty_weights, set_module_tensor_to_device, load_file)


def _import_effects_vendor() -> dict[str, Any]:
    """Import the vendored MMAudio classes, naming the GPU image location."""
    try:
        from mmaudio.eval_utils import generate  # noqa: PLC0415
        from mmaudio.ext.autoencoder import AutoEncoderModule  # noqa: PLC0415
        from mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2  # noqa: PLC0415
        from mmaudio.ext.synchformer import Synchformer  # noqa: PLC0415
        from mmaudio.model.flow_matching import FlowMatching  # noqa: PLC0415
        from mmaudio.model.networks import MMAudio  # noqa: PLC0415
        from mmaudio.model.sequence_config import CONFIG_44K  # noqa: PLC0415
        from mmaudio.model.utils.features_utils import FeaturesUtils  # noqa: PLC0415
        from open_clip import CLIP  # noqa: PLC0415
        from torchvision.transforms import v2  # noqa: PLC0415
    except ImportError as failure:
        message = (
            "MMAudio package is not installed; the GPU image clones it "
            f"to /opt/ComfyUI-MMAudio: {failure}"
        )
        raise EngineConfigurationError(message) from failure
    return {
        "generate": generate,
        "AutoEncoderModule": AutoEncoderModule,
        "BigVGANv2": BigVGANv2,
        "Synchformer": Synchformer,
        "FlowMatching": FlowMatching,
        "MMAudio": MMAudio,
        "CONFIG_44K": CONFIG_44K,
        "FeaturesUtils": FeaturesUtils,
        "CLIP": CLIP,
        "v2": v2,
    }


def _load_effects_transformer(loaders: EffectsLoaders) -> Any:
    """Load the large MMAudio transformer in fp16 on CPU."""
    torch_module = loaders.torch_module
    data_type = torch_module.float16
    weights = loaders.load_file(
        str(loaders.effects_directory / SOUND_EFFECT_MODEL_FILE), device="cpu"
    )
    with loaders.init_empty_weights():
        model = loaders.vendor["MMAudio"](
            latent_dim=40,
            clip_dim=1024,
            sync_dim=768,
            text_dim=1024,
            hidden_dim=64 * 14,
            depth=21,
            fused_depth=14,
            num_heads=14,
            latent_seq_len=345,
            clip_seq_len=64,
            sync_seq_len=192,
            v2=weights["t_embed.mlp.0.weight"].shape[1] == _MMAUDIO_V2_TEXT_WIDTH,
        )
    model = model.eval()
    for name, _parameter in model.named_parameters():
        loaders.set_module_tensor_to_device(
            model, name, device="cpu", dtype=data_type, value=weights[name]
        )
    del weights
    model.seq_cfg = loaders.vendor["CONFIG_44K"]
    return model


def _load_effects_autoencoder(loaders: EffectsLoaders) -> tuple[Any, Any]:
    """Load the MMAudio VAE (with BigVGAN vocoder) and synchformer in fp16."""
    torch_module = loaders.torch_module
    data_type = torch_module.float16
    synchformer_weights = loaders.load_file(
        str(loaders.effects_directory / SOUND_EFFECT_SYNCHFORMER_FILE), device="cpu"
    )
    with loaders.init_empty_weights():
        synchformer = loaders.vendor["Synchformer"]().eval()
    for name, _parameter in synchformer.named_parameters():
        loaders.set_module_tensor_to_device(
            synchformer, name, device="cpu", dtype=data_type, value=synchformer_weights[name]
        )
    del synchformer_weights
    vocoder = (
        loaders.vendor["BigVGANv2"]
        .from_pretrained(
            str(loaders.effects_directory / "nvidia" / "bigvgan_v2_44khz_128band_512x")
        )
        .eval()
        .to("cpu", data_type)
    )
    autoencoder_weights = loaders.load_file(
        str(loaders.effects_directory / SOUND_EFFECT_VAE_FILE), device="cpu"
    )
    autoencoder = loaders.vendor["AutoEncoderModule"](
        vae_state_dict=autoencoder_weights,
        bigvgan_vocoder=vocoder,
        mode=SOUND_EFFECT_MODE,
    )
    autoencoder = autoencoder.eval().to("cpu", data_type)
    del autoencoder_weights
    return (autoencoder, synchformer)


def _load_effects_clip_model(loaders: EffectsLoaders) -> Any:
    """Load the DFN5B CLIP model feeding MMAudio text conditioning."""
    import torch  # noqa: PLC0415

    clip_config_path = _mmaudio_config_path(SOUND_EFFECT_SYNCHFORMER_CONFIG)
    with clip_config_path.open() as config_file:
        clip_config = json.load(config_file)
    with loaders.init_empty_weights():
        try:
            clip_model = loaders.vendor["CLIP"](**clip_config["model_cfg"]).eval()
        except TypeError:
            clip_config["model_cfg"]["nonscalar_logit_scale"] = True
            clip_model = loaders.vendor["CLIP"](**clip_config["model_cfg"]).eval()
    clip_weights = loaders.load_file(
        str(loaders.effects_directory / SOUND_EFFECT_CLIP_FILE), device="cpu"
    )
    for name, _parameter in clip_model.named_parameters():
        loaders.set_module_tensor_to_device(
            clip_model, name, device="cpu", dtype=torch.float16, value=clip_weights[name]
        )
    del clip_weights
    return clip_model


def _build_effects_sync_transform(torch_module: Any, transforms_module: Any) -> Any:
    """Build the normalized sync-frame transform for MMAudio video."""
    # Only the sync transform runs: mask_away_clip stays enabled, so the
    # video CLIP features it would mask are never computed.
    return transforms_module.Compose(
        [
            transforms_module.Resize(
                SOUND_EFFECT_SYNC_FRAME_PIXELS,
                interpolation=transforms_module.InterpolationMode.BICUBIC,
            ),
            transforms_module.CenterCrop(SOUND_EFFECT_SYNC_FRAME_PIXELS),
            transforms_module.ToPILImage(),
            transforms_module.ToTensor(),
            transforms_module.ConvertImageDtype(torch_module.float32),
            transforms_module.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _mmaudio_config_path(file_name: str) -> Path:
    """Locate an MMAudio config beside the imported package, else by path."""
    import mmaudio  # noqa: PLC0415

    package_file = mmaudio.__file__
    if package_file is None:
        message = f"MMAudio config is missing: {file_name}"
        raise EngineConfigurationError(message)
    package_directory = Path(package_file).parent
    for candidate in (
        package_directory.parent / "configs" / file_name,
        package_directory / "configs" / file_name,
    ):
        if candidate.is_file():
            return candidate
    message = f"MMAudio config is missing: {file_name}"
    raise EngineConfigurationError(message)
