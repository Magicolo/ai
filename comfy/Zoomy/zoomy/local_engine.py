"""In-process generation engine: frames, interpolation, music, effects, video.

This is the pure-code replacement for the ComfyUI REST path, and it stays a
thin orchestrator: each generation stage lives in its own module
(:mod:`zoomy.frame_pipelines`, :mod:`zoomy.interpolation`,
:mod:`zoomy.music`, :mod:`zoomy.effects`, :mod:`zoomy.video_io`) with shared
retry policy in :mod:`zoomy.retry`. The engine owns residency and sequencing
— which stack stays loaded when — while the stages own their model calls.

Residency budget (RTX 4060 Ti 16 GB, 62 GB host RAM): model weights live on
CPU under offload and stream through VRAM layer by layer, so only one
heavyweight stays resident per stage group — one frame pipeline (evicted when
the family changes, and dropped wholesale when finalize starts because no
more frames render after it), one interpolation model, one music stack, one
effects stack. Frame diffusion and finalize never interleave, and within a
finalize the music stack renders one global track up front, is evicted, and
only then does the effects stack load for the per-window loop — music and
effects never share the card. Per-window peak stays inside the verified
envelope (~13 GB with music resident).

Heavy third-party packages import lazily inside the methods that need them,
so the module (and the slim test image) loads without torch installed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from zoomy import effects, frame_pipelines, interpolation, music, video_io
from zoomy.effects import SoundEffectsRenderRequest
from zoomy.engine_protocol import (
    CROP_BORDER_PIXELS,
    FRAME_SIZE_ALIGNMENT_PIXELS,
    MAXIMUM_DENOISE_STRENGTH,
    MINIMUM_DENOISE_STRENGTH,
    MINIMUM_FRAMES_FOR_INTERPOLATION,
    MINIMUM_SYNC_FRAMES,
    MUSIC_SEED,
    SEGMENT_MUSIC_EXTENSION_SECONDS,
    SOUND_EFFECT_SEED,
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
from zoomy.frame_pipelines import LoadedFramePipeline
from zoomy.music import MusicRenderRequest
from zoomy.retry import MAXIMUM_STAGE_ATTEMPTS, run_stage_with_retries

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from PIL.Image import Image

    from zoomy.effects import LoadedEffectsStack
    from zoomy.family_catalog import FamilyDefinition
    from zoomy.frame_repository import FrameRepository

_MUSIC_STEM_SUFFIX = "_music_stem.flac"
_SOUND_STEM_SUFFIX = "_sfx_stem.flac"
_GLOBAL_MUSIC_FILE_NAME = "global_music.flac"
# A single window renders its music stem directly; the global take only pays
# off once a finalize splits into several windows.
_MINIMUM_WINDOWS_FOR_GLOBAL_MUSIC = 2


@dataclass(slots=True)
class GlobalMusicSlice:
    """One window's view into the finalize's single global music take."""

    track_path: Path
    start_seconds: float = 0.0


# Re-exported stage helpers so existing imports keep working: the
# implementations live in video_io/retry, but tests and callers address them
# through the engine module as before.
_pad_frames_to_minimum = video_io.pad_frames_to_minimum
_read_system_memory_bytes = video_io.read_system_memory_bytes
_write_silent_video = video_io.write_silent_video

__all__ = [
    "MAXIMUM_STAGE_ATTEMPTS",
    "LocalEngine",
    "_pad_frames_to_minimum",
    "_read_system_memory_bytes",
    "_write_silent_video",
    "run_stage_with_retries",
]


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
        self._interrupt_lock = threading.Lock()
        self._interrupt_pending = False
        self._abort_job_at_start = False
        self._frame_pipeline: LoadedFramePipeline | None = None
        self._rife_model: Any | None = None
        self._effects_stack: LoadedEffectsStack | None = None
        self._music_stack: dict[str, Any] | None = None

    def request_interrupt(self) -> None:
        """Flag the running job — or the next one — to abort at its next checkpoint.

        The flag is consume-on-observe under a lock, so a request racing a
        job's start is never wiped: it either lands before the job takes
        ownership (and aborts it at the first checkpoint) or after (and
        aborts it at the next one).
        """
        with self._interrupt_lock:
            self._interrupt_pending = True

    def _begin_job(self) -> None:
        """Take ownership of any idle interrupt request for the starting job."""
        with self._interrupt_lock:
            self._abort_job_at_start = self._interrupt_pending
            self._interrupt_pending = False

    def is_ready(self) -> bool:
        """Return True when the model and seed directories are reachable."""
        return self._models_directory.is_dir() and self._seed_directory.is_dir()

    def engine_statistics(self) -> EngineStatistics:
        """Return live system RAM and render-device VRAM figures."""
        system_free, system_total = video_io.read_system_memory_bytes()
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
                missing, the frame size breaks alignment, or the family has
                no engine recipe.
            EngineExecutionError: A generation stage failed.
            RenderInterruptedError: An interrupt was requested mid-render.
        """
        self._begin_job()
        self._validate_frame_size(request)
        self._validate_denoise_strength(request)
        renderer = self._frame_renderer(request.family.key)
        try:
            source_image = self._load_source_image(request)
            return run_stage_with_retries(
                "frame",
                lambda: renderer(request, source_image),
                evict_resident_stacks=self._evict_all_stacks,
            )
        except (RenderInterruptedError, EngineConfigurationError, EngineExecutionError):
            raise
        except Exception as failure:
            message = f"Frame render failed: {failure}"
            raise EngineExecutionError("frame", message) from failure

    @staticmethod
    def _validate_frame_size(request: FrameRenderRequest) -> None:
        """Refuse frame sizes the autoencoders cannot encode.

        Both VAE families downsample in powers of two, so each side must be
        a positive multiple of 16 (the HD default and 512-coding sizes pass).
        """
        for label, size in (("width", request.frame_width), ("height", request.frame_height)):
            if size <= 0 or size % FRAME_SIZE_ALIGNMENT_PIXELS != 0:
                message = (
                    f"Frame {label} must be a positive multiple of "
                    f"{FRAME_SIZE_ALIGNMENT_PIXELS}, received {size}"
                )
                raise EngineConfigurationError(message)

    @staticmethod
    def _validate_denoise_strength(request: FrameRenderRequest) -> None:
        """Refuse denoise strengths outside the verified img2img span.

        The interface coerces its coherence slider into range, so a failure
        here is a programmatic caller bug, not a slider edge.
        """
        denoise = request.denoise_strength
        if not MINIMUM_DENOISE_STRENGTH <= denoise <= MAXIMUM_DENOISE_STRENGTH:
            message = (
                "Denoise strength must be within "
                f"{MINIMUM_DENOISE_STRENGTH}..{MAXIMUM_DENOISE_STRENGTH}, "
                f"received {denoise}"
            )
            raise EngineConfigurationError(message)

    def _evict_all_stacks(self) -> None:
        """Drop every resident stack so a retried stage finds a free card.

        The frame pipeline rebuilds lazily on the next render; evicting it
        too covers OOMs where the diffusion weights themselves fragment the
        allocator.
        """
        self._frame_pipeline = None
        self._rife_model = None
        self._music_stack = None
        self._effects_stack = None
        video_io.collect_free_video_memory()

    def _frame_renderer(self, family_key: str) -> Callable[[FrameRenderRequest, Image], Image]:
        """Return the frame recipe for a family, refusing unknown keys."""
        if family_key == "ernie_turbo":
            return self._render_ernie_frame
        if family_key in frame_pipelines.Z_HUB_REPOSITORY_BY_FAMILY_KEY:
            return self._render_z_frame
        message = f"No engine recipe for family {family_key!r}"
        raise EngineConfigurationError(message)

    def finalize_sequence(self, request: FinalizeRequest) -> Iterator[ProgressUpdate]:
        """Turn the sequence into a video, yielding progress until it lands.

        Short sequences finalize in one pass; long ones render window by
        window (bounded VRAM per job) and assemble in Python. Segment windows
        over-generate exactly the assembly's crossfade overlap, so stems stay
        sample-locked to their video with no drift.

        Music renders once as a single global take covering the whole
        timeline (previously one diffusion pass per window): each window then
        slices its stem — overlap included — out of it, so a window retry
        re-slices instead of re-running ACE-Step. Sound effects stay
        per-window because they are conditioned on each window's own frames.

        Raises:
            EmptyFrameSequenceError: The sequence has no frames.
            EngineConfigurationError: A model or frame file is missing.
            EngineExecutionError: A generation stage failed.
            RenderInterruptedError: An interrupt was requested mid-finalize.
            AssemblyError: Segment outputs are missing or ffmpeg failed.
        """
        self._begin_job()
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
        # Frame diffusion is done: drop the ~20 GB CPU pipeline before the
        # audio stacks load, and size every window's durations once.
        self._frame_pipeline = None
        video_io.collect_free_video_memory()
        video_seconds = [compute_segment_video_seconds(window.frame_count) for window in windows]
        work_directory = self._repository.assembly_directory(sequence_key)
        work_directory.mkdir(parents=True, exist_ok=True)
        global_music = self._prepare_global_music(
            request.family.music_prompt, windows, video_seconds, work_directory
        )
        try:
            soundtracks: list[SegmentSoundtrack] = []
            for position, window in enumerate(windows):
                self._raise_if_interrupted()
                yield ProgressUpdate(
                    message=f"Finalizing segment {window.index + 1}/{len(windows)}…",
                    frame_count=request.frame_count,
                )
                soundtrack = self._finalize_window(
                    request.family,
                    window,
                    extension=extension,
                    global_music=global_music[position],
                )
                soundtracks.append(soundtrack)
                yield ProgressUpdate(
                    message=f"Segment {window.index + 1}/{len(windows)} complete.",
                    frame_count=request.frame_count,
                )
        finally:
            # The music stack is already gone on the multi-window path, but a
            # failed global render (or a single-window music stem) can leave
            # one behind; either audio stack must be gone before assembly.
            self._unload_music_stack()
            self._unload_effects_stack()
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

    def _prepare_global_music(
        self,
        caption: str,
        windows: tuple[SegmentWindow, ...],
        video_seconds: list[float],
        work_directory: Path,
    ) -> tuple[GlobalMusicSlice | None, ...]:
        """Render the whole timeline's music in one ACE-Step pass, retrying OOMs.

        The single take is the finalize's checkpoint: window retries slice
        from it instead of re-running diffusion, so it gets its own
        evict-and-retry budget like any other stage. Single-pass finalizes
        skip the take (windows render their stem directly) and get Nones.
        """
        if len(windows) < _MINIMUM_WINDOWS_FOR_GLOBAL_MUSIC:
            return (None,) * len(windows)
        track_path = work_directory / _GLOBAL_MUSIC_FILE_NAME
        total_music_seconds = sum(video_seconds) + SEGMENT_MUSIC_EXTENSION_SECONDS
        run_stage_with_retries(
            "music-global",
            lambda: self._render_music(caption, total_music_seconds, track_path, seed=MUSIC_SEED),
            evict_resident_stacks=self._evict_all_stacks,
        )
        self._unload_music_stack()
        slices: list[GlobalMusicSlice | None] = []
        running_start = 0.0
        for seconds in video_seconds:
            slices.append(GlobalMusicSlice(track_path=track_path, start_seconds=running_start))
            running_start += seconds
        return tuple(slices)

    def _finalize_window(
        self,
        family: FamilyDefinition,
        window: SegmentWindow,
        *,
        extension: bool,
        global_music: GlobalMusicSlice | None = None,
    ) -> SegmentSoundtrack:
        """Render one window's silent video and stems, verifying every artifact.

        The window keeps one silent segment video (the assembly concatenates
        these directly — the former per-stem twin muxes are gone) plus its
        music and effects stems. With a global music track the window slices
        its stem out of it; otherwise (single pass) it renders the stem
        directly, which keeps the old stub surface for tests.
        """
        sequence_key = family.sequence_key
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
        segment_stem = f"{sequence_key}_seg{window.index:03d}"
        work_directory = self._repository.assembly_directory(sequence_key)
        work_directory.mkdir(parents=True, exist_ok=True)
        silent_video_path = work_directory / f"seg{window.index:03d}_silent.mp4"
        music_stem_path = self._repository.output_directory / f"{segment_stem}{_MUSIC_STEM_SUFFIX}"
        sound_stem_path = self._repository.output_directory / f"{segment_stem}{_SOUND_STEM_SUFFIX}"

        def render_window() -> SegmentSoundtrack:
            """Run this window's interpolation, stems, and silent encode."""
            window_artifacts = (
                silent_video_path,
                music_stem_path,
                sound_stem_path,
            )
            completed = False
            try:
                frames, arrays = video_io.frame_arrays(frame_paths)
                interpolated_frames = self._interpolate_frames(arrays)
                del frames, arrays
                video_io.write_silent_video(interpolated_frames, silent_video_path)
                if global_music is None:
                    self._render_music(
                        family.music_prompt,
                        music_seconds,
                        music_stem_path,
                        seed=MUSIC_SEED + window.index,
                    )
                else:
                    self._raise_if_interrupted()
                    video_io.slice_audio(
                        global_music.track_path,
                        music_stem_path,
                        start_seconds=global_music.start_seconds,
                        duration_seconds=music_seconds,
                    )
                self._render_sound_effects(
                    family.sound_effect_prompt,
                    family.sound_effect_negative_prompt,
                    interpolated_frames,
                    sound_seconds,
                    sound_stem_path,
                )
                del interpolated_frames
                video_io.verify_paths(window.index, window_artifacts)
                completed = True
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
            finally:
                if not completed:
                    video_io.discard_partial_artifacts(window_artifacts)
            return SegmentSoundtrack(
                music_video_path=silent_video_path,
                music_stem_path=music_stem_path,
                sound_effect_stem_path=sound_stem_path,
                music_seconds=music_seconds,
                sound_effect_seconds=sound_seconds,
            )

        return run_stage_with_retries(
            f"segment-{window.index}",
            render_window,
            evict_resident_stacks=self._evict_all_stacks,
        )

    def _load_source_image(self, request: FrameRenderRequest) -> Image:
        """Return the cold-start seed (frame zero) or the latest frame."""
        if request.frame_count == 0:
            seed_path = self._seed_directory / request.family.cold_start_image
            if not seed_path.is_file():
                message = f"Cold-start seed image is missing: {seed_path}"
                raise EngineConfigurationError(message)
            return video_io.converted_frame(seed_path)
        latest_path = self._repository.latest_frame_path(request.family.sequence_key)
        if latest_path is None:
            message = (
                f"Sequence {request.family.sequence_key!r} reports "
                f"{request.frame_count} frames but none are on disk"
            )
            raise EngineConfigurationError(message)
        return video_io.converted_frame(latest_path)

    @staticmethod
    def _zoomed_frame(source_image: Image, frame_width: int, frame_height: int) -> Image:
        """Crop the centered dive window and rescale it back to full size.

        The 10 px border rule that centers the dive at HD scales down with
        the request: the crop keeps a 10 px margin while the frame stays
        larger than twice the margin, shrinking proportionally below that.
        """
        border = min(CROP_BORDER_PIXELS, frame_width // 4, frame_height // 4)
        crop_box = (
            border,
            border,
            frame_width - border,
            frame_height - border,
        )
        cropped = source_image.crop(crop_box)
        from PIL import Image as PillowImage  # noqa: PLC0415

        return cropped.resize((frame_width, frame_height), PillowImage.Resampling.BICUBIC)

    def _render_ernie_frame(self, request: FrameRenderRequest, source_image: Image) -> Image:
        """Run the Ernie custom img2img loop (TXT2IMG-only pipeline).

        ``ErnieImagePipeline`` exposes text-to-image only, so the loop
        replicates the KSampler path by hand: VAE-encode the zoomed frame,
        patchify and batch-normalize the latents, flow-match noise at the
        denoise sigma, run the transformer with the padded text memory (no
        classifier-free guidance — the recipe's cfg is 1.0), then unnormalize,
        unpatchify, and decode. Text embeddings are cached per (family,
        prompt) on the resident pipeline: sequences reuse one prompt across
        frames, so only the first frame pays the 7B text encoder.
        """
        import torch  # noqa: PLC0415

        loaded = self._frame_pipeline_for(request)
        pipeline = loaded.pipeline
        device = torch.device(self._device)
        generator = torch.Generator(device=self._device).manual_seed(request.seed)
        frame = self._zoomed_frame(source_image, request.frame_width, request.frame_height)
        try:
            with torch.no_grad():
                # Private diffusers helpers: the Ernie pipeline exposes no
                # public img2img entry point, so the loop below is pinned to
                # diffusers==0.40 (see requirements-gpu.txt).
                cache_key = (request.family.key, request.prompt)
                cached_embeddings = loaded.text_embedding_cache.get(cache_key)
                if cached_embeddings is None:
                    text_hidden = pipeline.encode_prompt(request.prompt, device)
                    text_memory, text_lengths = pipeline._pad_text(  # noqa: SLF001
                        text_hidden,
                        device,
                        torch.bfloat16,
                        pipeline.transformer.config.text_in_dim,
                    )
                    loaded.text_embedding_cache[cache_key] = (text_memory, text_lengths)
                else:
                    text_memory, text_lengths = cached_embeddings
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
                start_index = total_steps - int(total_steps * request.denoise_strength)
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
        frame = self._zoomed_frame(source_image, request.frame_width, request.frame_height)
        call: dict[str, Any] = {
            "prompt": request.prompt,
            "image": frame,
            "strength": request.denoise_strength,
            "height": request.frame_height,
            "width": request.frame_width,
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

    def _frame_pipeline_for(self, request: FrameRenderRequest) -> LoadedFramePipeline:
        """Return the resident pipeline for the family, loading on change.

        Only one frame pipeline stays resident (evicted when the family
        changes): each holds ~20 GB of CPU weights under offload, and three
        resident families would overflow the 62 GB host.
        """
        resident = self._frame_pipeline
        if resident is not None and resident.family_key == request.family.key:
            frame_pipelines.apply_lora_selection(
                resident, self._models_directory, request.lora_selections
            )
            return resident
        self._frame_pipeline = None
        video_io.collect_free_video_memory()
        if request.family.key == "ernie_turbo":
            pipeline = self._load_ernie_pipeline(request.family)
        elif request.family.key in frame_pipelines.Z_HUB_REPOSITORY_BY_FAMILY_KEY:
            pipeline = self._load_z_pipeline(request.family)
        else:
            message = f"No engine recipe for family {request.family.key!r}"
            raise EngineConfigurationError(message)
        loaded = LoadedFramePipeline(family_key=request.family.key, pipeline=pipeline)
        self._frame_pipeline = loaded
        frame_pipelines.apply_lora_selection(
            loaded, self._models_directory, request.lora_selections
        )
        return loaded

    def _load_ernie_pipeline(self, family: FamilyDefinition) -> Any:
        """Assemble the Ernie pipeline: local fp8 DiT, official everything else.

        Delegates to :mod:`zoomy.frame_pipelines` (kept as a method so loader
        tests keep their surface): the Comfy-format text encoder is not
        HuggingFace-format, so text encoder, VAE, tokenizer, and scheduler
        always come from the official repository while the transformer config
        downloads once per process — never the official bf16 weights.
        """
        return frame_pipelines.load_ernie_pipeline(self._models_directory, family)

    def _load_z_pipeline(self, family: FamilyDefinition) -> Any:
        """Assemble the Z-Image pipeline: local DiT, official fallback chain.

        Delegates to :mod:`zoomy.frame_pipelines` (kept as a method so loader
        tests keep their surface).
        """
        return frame_pipelines.load_z_pipeline(self._models_directory, family)

    def _interpolate_frames(self, source_frames: list[Any]) -> list[Any]:
        """Expand frame buffers 4x with RIFE recursive bisection (depth 2).

        Fewer than two frames have no pairs to bisect, so they pass through
        unchanged (the retired FILM node behaved the same); this also skips
        the pointless RIFE download for single-frame sequences. Accepts PIL
        frames or numpy arrays — the window path passes the shared arrays.
        """
        if len(source_frames) < MINIMUM_FRAMES_FOR_INTERPOLATION:
            return list(source_frames)
        if self._rife_model is None:
            self._rife_model = interpolation.load_rife_model()
        try:
            import numpy as np  # noqa: PLC0415

            arrays = [np.asarray(frame) for frame in source_frames]
            return interpolation.interpolate_arrays(
                self._rife_model, arrays, self._raise_if_interrupted
            )
        except RenderInterruptedError:
            raise
        except EngineExecutionError:
            raise
        except Exception as failure:
            message = f"Frame interpolation failed: {failure}"
            raise EngineExecutionError("interpolate", message) from failure

    def _render_music(
        self, caption: str, duration_seconds: float, stem_path: Path, *, seed: int
    ) -> None:
        """Render the instrumental music stem with the resident ACE-Step stack.

        The stack loads on first use and stays resident for the finalize's
        remaining windows (or the single global take); the caller evicts it
        before the effects stack loads, since the two never fit together.
        """
        if self._music_stack is None:
            self._music_stack = music.load_music_stack(self._music_project_directory, self._device)
        music.render_music_track(
            self._music_stack,
            MusicRenderRequest(
                caption=caption,
                duration_seconds=duration_seconds,
                stem_path=stem_path,
                seed=seed,
            ),
            self._raise_if_interrupted,
        )

    def _render_sound_effects(
        self,
        prompt: str,
        negative_prompt: str | None,
        interpolated_frames: list[Any],
        duration_seconds: float,
        stem_path: Path,
    ) -> None:
        """Render the video-synced effects stem with the resident MMAudio stack.

        The stack loads on the first window and stays resident for the rest;
        the caller evicts it after the last window or before assembly.
        """
        if self._effects_stack is None:
            self._effects_stack = effects.load_effects_stack(self._models_directory)
        effects.render_sound_effects_track(
            self._effects_stack,
            SoundEffectsRenderRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                frame_arrays=interpolated_frames,
                duration_seconds=duration_seconds,
                stem_path=stem_path,
                device=self._device,
                seed=SOUND_EFFECT_SEED,
                minimum_sync_frames=MINIMUM_SYNC_FRAMES,
            ),
            self._raise_if_interrupted,
        )

    def _unload_music_stack(self) -> None:
        """Drop the resident ACE-Step stack and return its VRAM."""
        self._music_stack = None
        video_io.collect_free_video_memory()

    def _unload_effects_stack(self) -> None:
        """Drop the resident MMAudio stack and return its VRAM."""
        self._effects_stack = None
        video_io.collect_free_video_memory()

    def _raise_if_interrupted(self) -> None:
        """Abort the running job when an interrupt was requested.

        Observation consumes the request, so one press aborts exactly one
        job: a stale press from a previous job can never kill the next one.
        """
        with self._interrupt_lock:
            abort = self._abort_job_at_start or self._interrupt_pending
            self._abort_job_at_start = False
            self._interrupt_pending = False
        if abort:
            raise RenderInterruptedError("The engine job was interrupted.")
