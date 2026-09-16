"""In-process generation engine: frames, interpolation, music, effects, video.

This is the pure-code replacement for the ComfyUI REST path: every stage the
old workflow graphs expressed as nodes now runs here as plain Python over
``diffusers`` (frames), ``ccvfi``/RIFE (interpolation), the native ACE-Step
package (music), vendored MMAudio (sound effects), and ffmpeg (twins and the
segmented assembly in :mod:`zoomy.final_assembly`).

Residency budget (RTX 4060 Ti 16 GB, 62 GB host RAM): model weights live on
CPU under sequential offload and stream through VRAM layer by layer, so only
one heavyweight stays resident per stage group — one frame pipeline (evicted
when the family changes), one interpolation model, one music stack, one
effects stack. Stages never run concurrently; each finalize window runs
interpolate → music → effects → twin mux in order, which is what keeps the
per-window peak inside the verified envelope (~13 GB with music resident).

Heavy third-party packages import lazily inside the methods that need them,
so the module (and the slim test image) loads without torch installed.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from zoomy import final_assembly
from zoomy.engine_protocol import (
    ACE_DIFFUSION_CONFIG_NAME,
    ACE_LANGUAGE_BACKEND,
    ACE_LANGUAGE_MODEL_NAME,
    CROP_BORDER_PIXELS,
    CROP_HEIGHT_PIXELS,
    CROP_WIDTH_PIXELS,
    FRAME_HEIGHT_PIXELS,
    FRAME_WIDTH_PIXELS,
    INTERPOLATION_MULTIPLIER,
    MINIMUM_SYNC_FRAMES,
    MUSIC_BEATS_PER_MINUTE,
    MUSIC_SEED,
    SOUND_EFFECT_CLASSIFIER_FREE_GUIDANCE,
    SOUND_EFFECT_CLIP_FILE,
    SOUND_EFFECT_MODE,
    SOUND_EFFECT_MODEL_FILE,
    SOUND_EFFECT_SEED,
    SOUND_EFFECT_STEPS,
    SOUND_EFFECT_SYNCHFORMER_CONFIG,
    SOUND_EFFECT_SYNCHFORMER_FILE,
    SOUND_EFFECT_VAE_FILE,
    VIDEO_CONSTANT_RATE_FACTOR,
    VIDEO_FRAMES_PER_SECOND,
    VIDEO_PIXEL_FORMAT,
    EngineStatistics,
    FinalizeRequest,
    FrameRenderRequest,
    ProgressUpdate,
    SegmentWindow,
    compute_segment_music_seconds,
    compute_segment_sound_seconds,
    compute_segment_video_seconds,
    compute_segment_windows,
    needs_segmentation,
)
from zoomy.errors import (
    AssemblyError,
    EmptyFrameSequenceError,
    EngineConfigurationError,
    EngineExecutionError,
    RenderInterruptedError,
)
from zoomy.final_assembly import AssemblyRequest, SegmentSoundtrack, assemble_final_video

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from PIL.Image import Image

    from zoomy.family_catalog import FamilyDefinition
    from zoomy.frame_repository import FrameRepository

DIFFUSION_MODELS_DIRECTORY_NAME = "diffusion_models"
TEXT_ENCODERS_DIRECTORY_NAME = "text_encoders"
AUTOENCODERS_DIRECTORY_NAME = "vae"
LORAS_DIRECTORY_NAME = "loras"
SOUND_EFFECTS_DIRECTORY_NAME = "mmaudio"

ERNIE_HUB_REPOSITORY = "baidu/ERNIE-Image-Turbo"
Z_TURBO_HUB_REPOSITORY = "Tongyi-MAI/Z-Image-Turbo"
Z_BASE_HUB_REPOSITORY = "Tongyi-MAI/Z-Image"
_Z_HUB_REPOSITORY_BY_FAMILY_KEY = {
    "z_fast": Z_TURBO_HUB_REPOSITORY,
    "z_quality": Z_BASE_HUB_REPOSITORY,
}

_RIFE_MODEL_NAME = "RIFE_IFNet_v426_heavy"
# Bisection levels deriving the x4 multiplier: each level doubles the frames.
_INTERPOLATION_BISECTION_DEPTH = int(math.log2(INTERPOLATION_MULTIPLIER))
# Second text-embedding width marking a v2 MMAudio checkpoint (v1 uses 768).
_MMAUDIO_V2_TEXT_WIDTH = 896
# A /proc/meminfo line always holds name, value, and unit ("kB").
_MEMINFO_PART_COUNT = 3

_MUSIC_STEM_SUFFIX = "_music_stem.flac"
_SOUND_STEM_SUFFIX = "_sfx_stem.flac"
_MUSIC_TWIN_SUFFIX = "_music-audio.mp4"
_SOUND_TWIN_SUFFIX = "_sfx-audio.mp4"


@dataclass(slots=True)
class _LoadedFramePipeline:
    """One resident frame pipeline plus its LoRA bookkeeping."""

    family_key: str
    pipeline: Any
    text_encoder_device: Any
    loaded_lora_names_by_file: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class _EffectsLoaders:
    """Lazily imported primitives shared by the MMAudio loader helpers."""

    torch_module: Any
    init_empty_weights: Any
    set_module_tensor_to_device: Any
    load_file: Any
    vendor: dict[str, Any]
    effects_directory: Path


class LocalEngine:
    """Generate zoomy artifacts in-process on one CUDA device.

    Args:
        models_directory: Comfy-style model tree (diffusion_models,
            text_encoders, vae, loras, mmaudio subdirectories).
        seed_directory: Cold-start seed images named by the family catalog.
        repository: Frame/video artifact store the engine reads and writes.
        device: Torch device string for generation (default ``cuda:0``).
        music_project_directory: Writable directory where ACE-Step keeps its
            downloaded checkpoints across restarts (created when missing).
    """

    def __init__(
        self,
        *,
        models_directory: Path,
        seed_directory: Path,
        repository: FrameRepository,
        device: str = "cuda:0",
        music_project_directory: Path,
    ) -> None:
        """Validate directories and initialize empty lazy model caches."""
        music_project_directory.mkdir(parents=True, exist_ok=True)
        for label, directory in (
            ("models", models_directory),
            ("seed", seed_directory),
            ("music project", music_project_directory),
        ):
            if not directory.is_dir():
                message = f"LocalEngine {label} directory does not exist: {directory}"
                raise EngineConfigurationError(message)
        self._models_directory = models_directory
        self._seed_directory = seed_directory
        self._repository = repository
        self._device = device
        self._music_project_directory = music_project_directory
        self._interrupt = threading.Event()
        self._frame_pipeline: _LoadedFramePipeline | None = None
        self._rife_model: Any | None = None
        self._effects_stack: dict[str, Any] | None = None
        self._music_stack: dict[str, Any] | None = None

    def request_interrupt(self) -> None:
        """Flag the running job to abort at its next checkpoint."""
        self._interrupt.set()

    def is_ready(self) -> bool:
        """Return True when the model and seed directories are reachable."""
        return self._models_directory.is_dir() and self._seed_directory.is_dir()

    def engine_statistics(self) -> EngineStatistics:
        """Return live system RAM and render-device VRAM figures."""
        system_free, system_total = _read_system_memory_bytes()
        video_free: int | None = None
        video_total: int | None = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(self._device)
                video_free, video_total = int(free), int(total)
        except (ImportError, RuntimeError, ValueError):
            video_free, video_total = None, None
        return EngineStatistics(
            system_memory_free_bytes=system_free,
            system_memory_total_bytes=system_total,
            video_memory_free_bytes=video_free,
            video_memory_total_bytes=video_total,
        )

    def render_frame(self, request: FrameRenderRequest) -> Image:
        """Render one frame image for ``request`` (the caller saves it).

        Raises:
            EngineConfigurationError: A model, seed, or source file is
                missing, or the family has no engine recipe.
            EngineExecutionError: A generation stage failed.
            RenderInterruptedError: An interrupt was requested mid-render.
        """
        self._interrupt.clear()
        renderer = self._frame_renderer(request.family.key)
        try:
            source_image = self._load_source_image(request)
            return renderer(request, source_image)
        except (RenderInterruptedError, EngineConfigurationError, EngineExecutionError):
            raise
        except Exception as failure:
            message = f"Frame render failed: {failure}"
            raise EngineExecutionError("frame", message) from failure

    def _frame_renderer(self, family_key: str) -> Callable[[FrameRenderRequest, Image], Image]:
        """Return the frame recipe for a family, refusing unknown keys."""
        if family_key == "ernie_turbo":
            return self._render_ernie_frame
        if family_key in _Z_HUB_REPOSITORY_BY_FAMILY_KEY:
            return self._render_z_frame
        message = f"No engine recipe for family {family_key!r}"
        raise EngineConfigurationError(message)

    def finalize_sequence(self, request: FinalizeRequest) -> Iterator[ProgressUpdate]:
        """Turn the sequence into a video, yielding progress until it lands.

        Short sequences finalize in one pass; long ones render window by
        window (bounded VRAM per job) and assemble in Python. Single-pass
        windows carry no audio extension while segment windows over-generate
        exactly the assembly's crossfade overlap, so stems stay sample-locked
        to their video with no drift.

        Raises:
            EmptyFrameSequenceError: The sequence has no frames.
            EngineConfigurationError: A model or frame file is missing.
            EngineExecutionError: A generation stage failed.
            RenderInterruptedError: An interrupt was requested mid-finalize.
            AssemblyError: Segment outputs are missing or ffmpeg failed.
        """
        self._interrupt.clear()
        sequence_key = request.family.sequence_key
        if request.frame_count < 1:
            raise EmptyFrameSequenceError(
                "Cannot finalize a video: the sequence has no frames yet. "
                "Render at least one frame first."
            )
        operation_started = time.monotonic()
        if needs_segmentation(request.frame_count):
            windows = compute_segment_windows(request.frame_count)
            extension = True
        else:
            windows = (
                SegmentWindow(index=0, skip_first_images=0, frame_count=request.frame_count),
            )
            extension = False
        yield ProgressUpdate(
            message=(
                f"Finalizing {request.frame_count} frames "
                f"in {len(windows)} pass{'es' if len(windows) > 1 else ''}…"
            ),
            frame_count=request.frame_count,
        )
        soundtracks: list[SegmentSoundtrack] = []
        for window in windows:
            self._raise_if_interrupted()
            yield ProgressUpdate(
                message=f"Finalizing segment {window.index + 1}/{len(windows)}…",
                frame_count=request.frame_count,
            )
            soundtrack = self._finalize_window(
                request.family, sequence_key, window, extension=extension
            )
            soundtracks.append(soundtrack)
            yield ProgressUpdate(
                message=f"Segment {window.index + 1}/{len(windows)} complete.",
                frame_count=request.frame_count,
            )
        self._raise_if_interrupted()
        video_stem = self._repository.next_video_stem(sequence_key)
        assembly_request = AssemblyRequest(
            segments=tuple(soundtracks),
            work_directory=self._repository.assembly_directory(sequence_key),
            output_video_path=self._repository.output_directory / f"{video_stem}.mp4",
            output_audio_video_path=self._repository.output_directory / f"{video_stem}-audio.mp4",
        )
        try:
            outputs = assemble_final_video(assembly_request)
        except AssemblyError:
            raise
        except Exception as failure:
            message = f"Final video assembly failed: {failure}"
            raise AssemblyError(message) from failure
        self._repository.remove_segment_files(sequence_key)
        elapsed_seconds = time.monotonic() - operation_started
        yield ProgressUpdate(
            message=f"Video complete in {elapsed_seconds:.0f} s.",
            video_path=outputs.audio_video_path,
            frame_count=request.frame_count,
            elapsed_seconds=elapsed_seconds,
        )

    def _finalize_window(
        self,
        family: FamilyDefinition,
        sequence_key: str,
        window: SegmentWindow,
        *,
        extension: bool,
    ) -> SegmentSoundtrack:
        """Render one window's twins and stems, verifying every artifact."""
        frame_paths = self._repository.frame_paths(sequence_key)[
            window.skip_first_images : window.skip_first_images + window.frame_count
        ]
        if len(frame_paths) != window.frame_count:
            message = (
                f"Segment {window.index} needs {window.frame_count} frames, "
                f"found {len(frame_paths)}"
            )
            raise EngineConfigurationError(message)
        video_seconds = compute_segment_video_seconds(window.frame_count)
        if extension:
            music_seconds = compute_segment_music_seconds(window.frame_count)
            sound_seconds = compute_segment_sound_seconds(window.frame_count)
        else:
            music_seconds = video_seconds
            sound_seconds = video_seconds
        segment_stem = f"Zoomy_{sequence_key}_seg{window.index:03d}"
        work_directory = self._repository.assembly_directory(sequence_key)
        work_directory.mkdir(parents=True, exist_ok=True)
        silent_video_path = work_directory / f"seg{window.index:03d}_silent.mp4"
        music_stem_path = self._repository.output_directory / f"{segment_stem}{_MUSIC_STEM_SUFFIX}"
        sound_stem_path = self._repository.output_directory / f"{segment_stem}{_SOUND_STEM_SUFFIX}"
        music_twin_path = self._repository.output_directory / f"{segment_stem}{_MUSIC_TWIN_SUFFIX}"
        sound_twin_path = self._repository.output_directory / f"{segment_stem}{_SOUND_TWIN_SUFFIX}"
        try:
            from PIL import Image as PillowImage  # noqa: PLC0415

            source_frames = [PillowImage.open(path).convert("RGB") for path in frame_paths]
            interpolated_frames = self._interpolate_frames(source_frames)
            _write_silent_video(interpolated_frames, silent_video_path)
            self._render_music(
                family.music_prompt,
                music_seconds,
                music_stem_path,
                seed=MUSIC_SEED + window.index,
            )
            self._render_sound_effects(
                family.sound_effect_prompt,
                family.sound_effect_negative_prompt,
                interpolated_frames,
                sound_seconds,
                sound_stem_path,
            )
            _mux_audio_twin(silent_video_path, music_stem_path, music_twin_path)
            _mux_audio_twin(silent_video_path, sound_stem_path, sound_twin_path)
            silent_video_path.unlink(missing_ok=True)
        except (
            RenderInterruptedError,
            EngineConfigurationError,
            EngineExecutionError,
            AssemblyError,
        ):
            raise
        except Exception as failure:
            message = f"Segment {window.index} failed: {failure}"
            raise EngineExecutionError(f"segment-{window.index}", message) from failure
        _verify_segment_artifacts(
            window.index,
            (music_twin_path, sound_twin_path, music_stem_path, sound_stem_path),
        )
        return SegmentSoundtrack(
            music_video_path=music_twin_path,
            music_stem_path=music_stem_path,
            sound_effect_stem_path=sound_stem_path,
            music_seconds=music_seconds,
            sound_effect_seconds=sound_seconds,
        )

    def _load_source_image(self, request: FrameRenderRequest) -> Image:
        """Return the cold-start seed (frame zero) or the latest frame."""
        from PIL import Image as PillowImage  # noqa: PLC0415

        if request.frame_count == 0:
            seed_path = self._seed_directory / request.family.cold_start_image
            if not seed_path.is_file():
                message = f"Cold-start seed image is missing: {seed_path}"
                raise EngineConfigurationError(message)
            return PillowImage.open(seed_path).convert("RGB")
        latest_path = self._repository.latest_frame_path(request.family.sequence_key)
        if latest_path is None:
            message = (
                f"Sequence {request.family.sequence_key!r} reports "
                f"{request.frame_count} frames but none are on disk"
            )
            raise EngineConfigurationError(message)
        return PillowImage.open(latest_path).convert("RGB")

    @staticmethod
    def _zoomed_frame(source_image: Image) -> Image:
        """Crop the centered dive window and rescale it back to full size."""
        crop_box = (
            CROP_BORDER_PIXELS,
            CROP_BORDER_PIXELS,
            CROP_BORDER_PIXELS + CROP_WIDTH_PIXELS,
            CROP_BORDER_PIXELS + CROP_HEIGHT_PIXELS,
        )
        cropped = source_image.crop(crop_box)
        from PIL import Image as PillowImage  # noqa: PLC0415

        return cropped.resize(
            (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS), PillowImage.Resampling.BICUBIC
        )

    def _render_ernie_frame(self, request: FrameRenderRequest, source_image: Image) -> Image:
        """Run the Ernie custom img2img loop (TXT2IMG-only pipeline).

        ``ErnieImagePipeline`` exposes text-to-image only, so the loop
        replicates the KSampler path by hand: VAE-encode the zoomed frame,
        patchify and batch-normalize the latents, flow-match noise at the
        denoise sigma, run the transformer with the padded text memory (no
        classifier-free guidance — the recipe's cfg is 1.0), then unnormalize,
        unpatchify, and decode.
        """
        import torch  # noqa: PLC0415

        loaded = self._frame_pipeline_for(request)
        pipeline = loaded.pipeline
        device = torch.device(self._device)
        generator = torch.Generator(device=self._device).manual_seed(request.seed)
        frame = self._zoomed_frame(source_image)
        try:
            with torch.no_grad():
                # Private diffusers helpers: the Ernie pipeline exposes no
                # public img2img entry point, so the loop below is pinned to
                # diffusers==0.40 (see requirements-gpu.txt).
                text_hidden = pipeline.encode_prompt(request.prompt, device)
                text_memory, text_lengths = pipeline._pad_text(  # noqa: SLF001
                    text_hidden, device, torch.bfloat16, pipeline.transformer.config.text_in_dim
                )
                image_tensor = pipeline.image_processor.preprocess(frame).to(
                    device, pipeline.vae.dtype
                )
                image_latents = pipeline.vae.encode(image_tensor).latent_dist.sample(
                    generator=generator
                )
                image_latents = pipeline._patchify_latents(image_latents)  # noqa: SLF001
                batch_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(
                    device, image_latents.dtype
                )
                batch_variance = pipeline.vae.bn.running_var.view(1, -1, 1, 1).to(
                    device, image_latents.dtype
                )
                image_latents = (
                    (image_latents - batch_mean) / torch.sqrt(batch_variance + 1e-5)
                ).to(torch.bfloat16)
                total_steps = request.family.sampler_steps
                sigmas = torch.linspace(1.0, 0.0, total_steps + 1)
                start_index = total_steps - int(total_steps * request.family.denoise_strength)
                pipeline.scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)
                pipeline.scheduler.set_begin_index(start_index)
                noise = torch.randn(
                    image_latents.shape,
                    generator=generator,
                    device=device,
                    dtype=image_latents.dtype,
                )
                start_sigma = sigmas[start_index].to(device, image_latents.dtype)
                latents = start_sigma * noise + (1.0 - start_sigma) * image_latents
                for timestep in pipeline.scheduler.timesteps[start_index:]:
                    self._raise_if_interrupted()
                    timestep_batch = torch.full(
                        (1,), timestep.item(), device=device, dtype=latents.dtype
                    )
                    prediction = pipeline.transformer(
                        hidden_states=latents,
                        timestep=timestep_batch,
                        text_bth=text_memory,
                        text_lens=text_lengths,
                        return_dict=False,
                    )[0]
                    latents = pipeline.scheduler.step(prediction, timestep, latents).prev_sample
                latents = latents * torch.sqrt(batch_variance + 1e-5).to(
                    latents.dtype
                ) + batch_mean.to(latents.dtype)
                latents = pipeline._unpatchify_latents(latents)  # noqa: SLF001
                decoded = pipeline.vae.decode(latents, return_dict=False)[0]
                frame = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
                return cast("Image", frame)
        except RenderInterruptedError:
            raise
        except Exception as failure:
            message = f"Ernie frame render failed: {failure}"
            raise EngineExecutionError("ernie-frame", message) from failure

    def _render_z_frame(self, request: FrameRenderRequest, source_image: Image) -> Image:
        """Run the native Z-Image img2img pipeline at the family recipe."""
        import torch  # noqa: PLC0415

        loaded = self._frame_pipeline_for(request)
        pipeline = loaded.pipeline
        generator = torch.Generator(device=self._device).manual_seed(request.seed)
        frame = self._zoomed_frame(source_image)
        call: dict[str, Any] = {
            "prompt": request.prompt,
            "image": frame,
            "strength": request.family.denoise_strength,
            "height": FRAME_HEIGHT_PIXELS,
            "width": FRAME_WIDTH_PIXELS,
            "num_inference_steps": request.family.sampler_steps,
            "guidance_scale": request.family.classifier_free_guidance,
            "generator": generator,
        }
        if request.family.negative_prompt is not None:
            call["negative_prompt"] = request.negative_prompt or ""
        try:
            with torch.no_grad():
                frame = pipeline(**call).images[0]
                return cast("Image", frame)
        except RenderInterruptedError:
            raise
        except Exception as failure:
            message = f"Z-Image frame render failed: {failure}"
            raise EngineExecutionError("z-frame", message) from failure

    def _frame_pipeline_for(self, request: FrameRenderRequest) -> _LoadedFramePipeline:
        """Return the resident pipeline for the family, loading on change.

        Only one frame pipeline stays resident (evicted when the family
        changes): each holds ~20 GB of CPU weights under sequential offload,
        and three resident families would overflow the 62 GB host.
        """
        resident = self._frame_pipeline
        if resident is not None and resident.family_key == request.family.key:
            self._apply_lora_selection(resident, request)
            return resident
        self._frame_pipeline = None
        _collect_free_video_memory()
        if request.family.key == "ernie_turbo":
            pipeline = self._load_ernie_pipeline(request.family)
        elif request.family.key in _Z_HUB_REPOSITORY_BY_FAMILY_KEY:
            pipeline = self._load_z_pipeline(request.family)
        else:
            message = f"No engine recipe for family {request.family.key!r}"
            raise EngineConfigurationError(message)
        loaded = _LoadedFramePipeline(
            family_key=request.family.key,
            pipeline=pipeline,
            text_encoder_device=None,
            loaded_lora_names_by_file={},
        )
        self._frame_pipeline = loaded
        self._apply_lora_selection(loaded, request)
        return loaded

    def _apply_lora_selection(
        self, loaded: _LoadedFramePipeline, request: FrameRenderRequest
    ) -> None:
        """Load newly selected LoRAs once, then activate exactly the selection.

        Adapters stay registered after loading but only the names passed to
        ``set_adapters`` influence the pass, so deselecting is a weight swap —
        no reload, no pipeline duplication.
        """
        adapter_names: list[str] = []
        adapter_weights: list[float] = []
        for definition, strength in request.lora_selections:
            if definition.file_name not in loaded.loaded_lora_names_by_file:
                lora_path = self._models_directory / LORAS_DIRECTORY_NAME / definition.file_name
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
        if adapter_names:
            try:
                loaded.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
            except Exception as failure:
                message = f"LoRA activation failed: {failure}"
                raise EngineExecutionError("lora", message) from failure

    def _load_ernie_pipeline(self, family: FamilyDefinition) -> Any:
        """Assemble the Ernie pipeline: local fp8 DiT, official everything else.

        The Comfy-format text encoder is not HuggingFace-format, so text
        encoder, VAE, tokenizer, and scheduler always come from the official
        ``baidu/ERNIE-Image-Turbo`` repository (cached after the first
        download). The transformer config travels with that repository, so
        only its ``config.json`` downloads — never the official bf16 weights.
        """
        import torch  # noqa: PLC0415
        from diffusers import ErnieImagePipeline, ErnieImageTransformer2DModel  # noqa: PLC0415
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        diffusion_path = (
            self._models_directory / DIFFUSION_MODELS_DIRECTORY_NAME / family.base_model_file
        )
        if not diffusion_path.is_file():
            message = f"Diffusion model file is missing: {diffusion_path}"
            raise EngineConfigurationError(message)
        try:
            transformer_config = hf_hub_download(ERNIE_HUB_REPOSITORY, "transformer/config.json")
            transformer: Any | None = None
            try:
                transformer = ErnieImageTransformer2DModel.from_single_file(
                    str(diffusion_path),
                    config=transformer_config,
                    torch_dtype=torch.bfloat16,
                )
            except Exception:  # noqa: BLE001
                # Any load failure (missing keys, dtype mismatch) falls back
                # to the official weights below; the error resurfaces there.
                transformer = None
            pipeline_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
            if transformer is not None:
                pipeline_kwargs["transformer"] = transformer
            pipeline = ErnieImagePipeline.from_pretrained(  # type: ignore[no-untyped-call]
                ERNIE_HUB_REPOSITORY, **pipeline_kwargs
            )
            # The 8B bf16 transformer cannot sit resident next to anything on
            # a 16 GB card; per-layer streaming is slower than resident fp8
            # would be but always fits.
            pipeline.enable_sequential_cpu_offload()
        except EngineConfigurationError:
            raise
        except Exception as failure:
            message = f"Ernie pipeline load failed: {failure}"
            raise EngineExecutionError("ernie-load", message) from failure
        return pipeline

    def _load_z_pipeline(self, family: FamilyDefinition) -> Any:
        """Assemble the Z-Image pipeline: local DiT, official fallback chain.

        The Juggernaut single-file tensors load plainly; the local VAE only
        fits the pipeline's autoencoder when its channel count matches, so a
        mismatch falls back to the official VAE. Text encoder, tokenizer, and
        scheduler always come from the official Z-Image repository matching
        the family (Turbo for fast, base for quality).
        """
        import torch  # noqa: PLC0415
        from diffusers import (  # noqa: PLC0415
            AutoencoderKL,
            ZImageImg2ImgPipeline,
            ZImageTransformer2DModel,
        )

        repository = _Z_HUB_REPOSITORY_BY_FAMILY_KEY[family.key]
        diffusion_path = (
            self._models_directory / DIFFUSION_MODELS_DIRECTORY_NAME / family.base_model_file
        )
        if not diffusion_path.is_file():
            message = f"Diffusion model file is missing: {diffusion_path}"
            raise EngineConfigurationError(message)
        try:
            transformer: Any | None = None
            try:
                transformer = ZImageTransformer2DModel.from_single_file(
                    str(diffusion_path), torch_dtype=torch.bfloat16
                )
            except Exception:  # noqa: BLE001
                # Any load failure falls back to the official weights below.
                transformer = None
            autoencoder: Any | None = None
            autoencoder_path = (
                self._models_directory / AUTOENCODERS_DIRECTORY_NAME / family.autoencoder_file
            )
            if autoencoder_path.is_file():
                try:
                    autoencoder = AutoencoderKL.from_single_file(
                        str(autoencoder_path), torch_dtype=torch.bfloat16
                    )
                except Exception:  # noqa: BLE001
                    # Channel-count mismatches fall back to the official VAE.
                    autoencoder = None
            pipeline_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
            if transformer is not None:
                pipeline_kwargs["transformer"] = transformer
            if autoencoder is not None:
                pipeline_kwargs["vae"] = autoencoder
            pipeline = ZImageImg2ImgPipeline.from_pretrained(  # type: ignore[no-untyped-call]
                repository, **pipeline_kwargs
            )
            pipeline.enable_sequential_cpu_offload()
        except Exception as failure:
            message = f"Z-Image pipeline load failed: {failure}"
            raise EngineExecutionError("z-load", message) from failure
        return pipeline

    def _interpolate_frames(self, source_frames: list[Image]) -> list[Any]:
        """Expand frames 4x with RIFE recursive bisection (depth 2)."""
        import numpy as np  # noqa: PLC0415
        from ccvfi import AutoModel, ConfigType  # noqa: PLC0415

        if self._rife_model is None:
            try:
                self._rife_model = AutoModel.from_pretrained(
                    pretrained_model_name=ConfigType[_RIFE_MODEL_NAME]
                )
            except Exception as failure:
                message = f"RIFE model load failed: {failure}"
                raise EngineExecutionError("interpolate-load", message) from failure
        model = self._rife_model
        try:
            arrays = [np.asarray(frame) for frame in source_frames]
            interpolated: list[Any] = []
            for first, second in pairwise(arrays):
                self._raise_if_interrupted()
                sequence = self._bisect(model, first, second, _INTERPOLATION_BISECTION_DEPTH)
                interpolated.extend(sequence if not interpolated else sequence[1:])
        except RenderInterruptedError:
            raise
        except Exception as failure:
            message = f"Frame interpolation failed: {failure}"
            raise EngineExecutionError("interpolate", message) from failure
        else:
            return interpolated

    def _bisect(self, model: Any, first: Any, second: Any, depth: int) -> list[Any]:
        """Bisect one frame pair: depth 2 yields ``[a, m1, m, m2, b]``."""
        middle = model.inference_image_list([first, second])[0]
        if depth <= 1:
            return [first, middle, second]
        left = self._bisect(model, first, middle, depth - 1)
        right = self._bisect(model, middle, second, depth - 1)
        return [*left, *right[1:]]

    def _render_music(
        self, caption: str, duration_seconds: float, stem_path: Path, *, seed: int
    ) -> None:
        """Render the instrumental music stem with ACE-Step (2B-turbo + LM)."""
        from zoomy.vendor_compat import apply_transformers5_compat  # noqa: PLC0415

        apply_transformers5_compat()
        try:
            from acestep.handler import AceStepHandler  # noqa: PLC0415
            from acestep.inference import (  # noqa: PLC0415
                GenerationConfig,
                GenerationParams,
                generate_music,
            )
            from acestep.llm_inference import LLMHandler  # noqa: PLC0415
        except ImportError as failure:
            message = (
                "ACE-Step package is not installed; the GPU image clones it "
                f"to /opt/ACE-Step-1.5: {failure}"
            )
            raise EngineConfigurationError(message) from failure
        if self._music_stack is None:
            try:
                diffusion_handler = AceStepHandler()
                diffusion_handler.initialize_service(
                    project_root=str(self._music_project_directory),
                    config_path=ACE_DIFFUSION_CONFIG_NAME,
                    device=self._device,
                )
                language_handler = LLMHandler()
                language_handler.initialize(
                    checkpoint_dir=str(self._music_project_directory / "checkpoints"),
                    lm_model_path=ACE_LANGUAGE_MODEL_NAME,
                    backend=ACE_LANGUAGE_BACKEND,
                    device=self._device,
                )
            except Exception as failure:
                message = f"ACE-Step initialization failed: {failure}"
                raise EngineExecutionError("music-load", message) from failure
            self._music_stack = {
                "diffusion_handler": diffusion_handler,
                "language_handler": language_handler,
            }
        self._raise_if_interrupted()
        try:
            parameters = GenerationParams(
                caption=caption,
                lyrics="",
                bpm=MUSIC_BEATS_PER_MINUTE,
                duration=duration_seconds,
                seed=seed,
            )
            config = GenerationConfig(audio_format="flac")
            result = generate_music(
                self._music_stack["diffusion_handler"],
                self._music_stack["language_handler"],
                parameters,
                config,
                save_dir=str(stem_path.parent),
            )
        except Exception as failure:
            message = f"Music generation failed: {failure}"
            raise EngineExecutionError("music", message) from failure
        self._raise_if_interrupted()
        if not result.success or not result.audios:
            message = f"Music generation failed: {result.error}"
            raise EngineExecutionError("music", message)
        try:
            shutil.move(result.audios[0]["path"], stem_path)
        except OSError as failure:
            message = f"Music stem move failed: {failure}"
            raise EngineExecutionError("music", message) from failure
        # ACE (~9 GB with its language models) and MMAudio (~5 GB) never fit
        # together on the 16 GB card, so each audio stage evicts its own
        # stack once its stem lands; the next window reloads on demand.
        self._unload_music_stack()

    def _render_sound_effects(
        self,
        prompt: str,
        negative_prompt: str | None,
        interpolated_frames: list[Any],
        duration_seconds: float,
        stem_path: Path,
    ) -> None:
        """Render the video-synced effects stem with MMAudio (fp16, 44k)."""
        stack = self._effects_stack_for()
        self._raise_if_interrupted()
        try:
            import torch  # noqa: PLC0415

            device = torch.device(self._device)
            video = stack["video_tensor"](
                _pad_frames_to_minimum(interpolated_frames, MINIMUM_SYNC_FRAMES)
            )
            frames_channel_first = video.permute(0, 3, 1, 2)
            sync_frames = torch.stack(
                [stack["sync_transform"](frame) for frame in frames_channel_first]
            )[: int(25 * duration_seconds)]
            duration_seconds = sync_frames.shape[0] / 25.0
            model = stack["model"]
            model.seq_cfg.duration = duration_seconds
            model.update_seq_lengths(
                model.seq_cfg.latent_seq_len,
                model.seq_cfg.clip_seq_len,
                model.seq_cfg.sync_seq_len,
            )
            generator = torch.Generator(device=self._device).manual_seed(SOUND_EFFECT_SEED)
            feature_utils = stack["feature_utils"]
            feature_utils.to(device)
            model.to(device)
            with torch.no_grad():
                audios = stack["generate"](
                    None,
                    sync_frames.unsqueeze(0).to(device),
                    [prompt],
                    negative_text=[negative_prompt or ""],
                    feature_utils=feature_utils,
                    net=model,
                    fm=stack["flow_matching"](),
                    rng=generator,
                    cfg_strength=SOUND_EFFECT_CLASSIFIER_FREE_GUIDANCE,
                )
            import soundfile  # noqa: PLC0415

            waveform = audios.float().cpu()
            soundfile.write(str(stem_path), waveform[0].T.numpy(), 44100)
        except RenderInterruptedError:
            raise
        except Exception as failure:
            message = f"Sound-effect generation failed: {failure}"
            raise EngineExecutionError("sound-effects", message) from failure
        self._raise_if_interrupted()
        # Same VRAM pact as the music stage above: evict so the next
        # window's music reload (or the assembler) finds a free card.
        self._unload_effects_stack()

    def _unload_music_stack(self) -> None:
        """Drop the resident ACE-Step stack and return its VRAM."""
        self._music_stack = None
        _collect_free_video_memory()

    def _unload_effects_stack(self) -> None:
        """Drop the resident MMAudio stack and return its VRAM."""
        self._effects_stack = None
        _collect_free_video_memory()

    def _effects_stack_for(self) -> dict[str, Any]:
        """Load the MMAudio fp16 stack once, mirroring the node loaders."""
        if self._effects_stack is not None:
            return self._effects_stack
        torch_module, init_empty_weights, set_module_tensor_to_device, load_file = (
            _import_effects_loaders()
        )
        vendor = _import_effects_vendor()
        self._require_effects_files()
        effects_directory = self._models_directory / SOUND_EFFECTS_DIRECTORY_NAME
        loaders = _EffectsLoaders(
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
        except EngineConfigurationError:
            raise
        except Exception as failure:
            message = f"Sound-effect model load failed: {failure}"
            raise EngineExecutionError("sound-effects-load", message) from failure

        def build_video_tensor(frames: list[Any]) -> Any:
            import numpy as np  # noqa: PLC0415

            stacked = np.stack([np.asarray(frame) for frame in frames])
            return torch_module.from_numpy(stacked).float() / 255.0

        def build_flow_matching() -> Any:
            return vendor["FlowMatching"](
                min_sigma=0, inference_mode="euler", num_steps=SOUND_EFFECT_STEPS
            )

        self._effects_stack = {
            "model": model,
            "feature_utils": feature_utils,
            "sync_transform": sync_transform,
            "video_tensor": build_video_tensor,
            "flow_matching": build_flow_matching,
            "generate": vendor["generate"],
        }
        return self._effects_stack

    def _require_effects_files(self) -> None:
        """Reject missing sound-effect files before any model loads."""
        effects_directory = self._models_directory / SOUND_EFFECTS_DIRECTORY_NAME
        for file_name in (
            SOUND_EFFECT_MODEL_FILE,
            SOUND_EFFECT_VAE_FILE,
            SOUND_EFFECT_SYNCHFORMER_FILE,
            SOUND_EFFECT_CLIP_FILE,
        ):
            if not (effects_directory / file_name).is_file():
                message = f"Sound-effect model file is missing: {effects_directory / file_name}"
                raise EngineConfigurationError(message)

    def _raise_if_interrupted(self) -> None:
        """Abort the running job when an interrupt was requested."""
        if self._interrupt.is_set():
            raise RenderInterruptedError("The engine job was interrupted.")


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


def _load_effects_transformer(loaders: _EffectsLoaders) -> Any:
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


def _load_effects_autoencoder(loaders: _EffectsLoaders) -> tuple[Any, Any]:
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


def _load_effects_clip_model(loaders: _EffectsLoaders) -> Any:
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
    """Build the 224px normalized sync-frame transform for MMAudio video."""
    # Only the sync transform runs: mask_away_clip stays enabled, so the
    # video CLIP features it would mask are never computed.
    return transforms_module.Compose(
        [
            transforms_module.Resize(
                224, interpolation=transforms_module.InterpolationMode.BICUBIC
            ),
            transforms_module.CenterCrop(224),
            transforms_module.ToPILImage(),
            transforms_module.ToTensor(),
            transforms_module.ConvertImageDtype(torch_module.float32),
            transforms_module.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _verify_segment_artifacts(window_index: int, artifacts: tuple[Path, ...]) -> None:
    """Reject missing or empty window outputs at the step that caused them."""
    for artifact in artifacts:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            message = f"Segment {window_index} produced no output: {artifact}"
            raise AssemblyError(message)


def _mmaudio_config_path(file_name: str) -> Path:
    """Locate an MMAudio config beside the imported package, else by path."""
    import mmaudio  # noqa: PLC0415

    package_directory = Path(mmaudio.__file__).parent
    for candidate in (
        package_directory.parent / "configs" / file_name,
        package_directory / "configs" / file_name,
    ):
        if candidate.is_file():
            return candidate
    message = f"MMAudio config is missing: {file_name}"
    raise EngineConfigurationError(message)


def _read_system_memory_bytes() -> tuple[int, int]:
    """Return free and total system RAM, parsed from /proc on Linux."""
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return (0, 0)
    values: dict[str, int] = {}
    for line in meminfo.splitlines():
        parts = line.split()
        if len(parts) == _MEMINFO_PART_COUNT and parts[2] == "kB":
            try:
                values[parts[0].rstrip(":")] = int(parts[1]) * 1024
            except ValueError:
                continue
    return (values.get("MemAvailable", 0), values.get("MemTotal", 0))


def _collect_free_video_memory() -> None:
    """Collect garbage and hand freed blocks back to the CUDA allocator."""
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pad_frames_to_minimum(frames: list[Any], minimum_frames: int) -> list[Any]:
    """Tile a short frame batch up to ``minimum_frames`` (passthrough above).

    MMAudio's synchformer encodes video in 16-frame sync segments and crashes
    on an empty segment list, so short renders repeat the whole batch (the
    retired BatchPadToMin custom node did exactly this). Callers guarantee a
    non-empty batch: there is nothing sensible to tile from zero frames.
    """
    padded = list(frames)
    while len(padded) < minimum_frames:
        padded.extend(frames)
    return padded


def _write_silent_video(frames: list[Any], destination: Path) -> None:
    """Encode interpolated frames as h264 (crf 19, yuv420p, 32 fps)."""
    import numpy as np  # noqa: PLC0415

    ffmpeg = final_assembly.ffmpeg_binary()
    height, width, _channels = np.asarray(frames[0]).shape
    command = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(VIDEO_FRAMES_PER_SECOND),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        VIDEO_PIXEL_FORMAT,
        "-crf",
        str(VIDEO_CONSTANT_RATE_FACTOR),
        str(destination),
    )
    try:
        # Tuple argv, no shell: every argument comes from our own builders.
        process = subprocess.Popen(command, stdin=subprocess.PIPE)  # noqa: S603
    except OSError as failure:
        message = f"Silent video encode failed for {destination}: {failure}"
        raise AssemblyError(message) from failure
    if process.stdin is None:
        message = f"Silent video encode failed for {destination}: no input pipe"
        raise AssemblyError(message)
    try:
        for frame in frames:
            process.stdin.write(np.asarray(frame).tobytes())
        process.stdin.close()
        process.wait()
    except (OSError, ValueError) as failure:
        message = f"Silent video encode failed for {destination}: {failure}"
        raise AssemblyError(message) from failure
    if process.returncode != 0:
        message = f"Silent video encode failed for {destination}"
        raise AssemblyError(message)
    if not destination.is_file() or destination.stat().st_size == 0:
        message = f"Silent video encode produced no output: {destination}"
        raise AssemblyError(message)


def _mux_audio_twin(silent_video: Path, stem: Path, twin: Path) -> None:
    """Mux one stem onto the silent video (aac 192k, trimmed to video)."""
    ffmpeg = final_assembly.ffmpeg_binary()
    command = final_assembly.build_mux_command(ffmpeg, silent_video, stem, twin)
    try:
        # Tuple argv, no shell: every argument comes from our own builders.
        subprocess.run(list(command), check=True)  # noqa: S603
    except (OSError, subprocess.CalledProcessError) as failure:
        message = f"Audio twin mux failed for {twin}: {failure}"
        raise AssemblyError(message) from failure
    if not twin.is_file() or twin.stat().st_size == 0:
        message = f"Audio twin mux produced no output: {twin}"
        raise AssemblyError(message)
