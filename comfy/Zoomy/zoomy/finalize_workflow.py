"""Builder for the finalize workflow: frames to a music video.

The heavy lifting ComfyUI is good at stays in the graph — FILM frame
interpolation, ACE-Step music generation, MMAudio video-synced sound effects,
and the VideoHelperSuite video encode — while arithmetic lives in Python:

    interpolated frames = (frame_count - 1) * INTERPOLATION_MULTIPLIER + 1
    audio seconds = max(interpolated frames / VIDEO_FRAMES_PER_SECOND,
                        MINIMUM_AUDIO_SECONDS)

The 1.0-second floor is the minimum EmptyAceStep1.5LatentAudio accepts, and it
also keeps MMAudio's synchformer happy on short sequences (it needs at least
16 sync frames). The video node trims any audio excess with ``-shortest`` —
so the floor is safe in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zoomy.graph import ComfyWorkflow, NodeReference

if TYPE_CHECKING:
    from zoomy.family_catalog import FamilyDefinition

INTERPOLATION_MULTIPLIER = 4
VIDEO_FRAMES_PER_SECOND = 32
# Lowest audio duration any audio node accepts: EmptyAceStep1.5LatentAudio
# enforces seconds >= 1.0 (TextEncodeAceStepAudio1.5 allows >= 0.0 and
# MMAudioSampler has no minimum). Overshoot is safe because the video mux
# always applies -shortest, so a floored audio track is trimmed to the video.
MINIMUM_AUDIO_SECONDS = 1.0
MINIMUM_SYNC_FRAMES = 17
# Source frames per finalize segment: 48 frames interpolate to 189 frames and
# about 5.9 s of audio, inside the envelope verified live on the 16 GB card —
# while the 300-frame single pass (1197 interpolated frames) OOMs MMAudio.
SEGMENT_SOURCE_FRAMES = 48
# Extra audio each segment generates beyond its video span: the assembly
# overlaps neighbors by exactly the crossfade length, so generating the
# overlap keeps every stem sample-locked to its video (no drift) while the
# final mux trims the last tail. Each must equal the assembly's matching
# crossfade — pinned by test_extension_matches_assembly_overlaps.
SEGMENT_MUSIC_EXTENSION_SECONDS = 1.0
SEGMENT_SOUND_EXTENSION_SECONDS = 0.25

INTERPOLATION_MODEL_FILE = "film_net_fp16.safetensors"

MUSIC_MODEL_FILE = "acestep_v1.5_turbo.safetensors"
MUSIC_MODEL_SHIFT = 3.0
MUSIC_TEXT_ENCODER_PRIMARY_FILE = "qwen_0.6b_ace15.safetensors"
MUSIC_TEXT_ENCODOR_SECONDARY_FILE = "qwen_1.7b_ace15.safetensors"
MUSIC_TEXT_ENCODOR_TYPE = "ace"
MUSIC_AUTOENCODER_FILE = "ace_1.5_vae.safetensors"
MUSIC_SEED = 31
MUSIC_STEPS = 8
MUSIC_CLASSIFIER_FREE_GUIDANCE = 1.0
MUSIC_SAMPLER = "euler"
MUSIC_SCHEDULER = "simple"
MUSIC_BEATS_PER_MINUTE = 100
MUSIC_TIME_SIGNATURE = "4"
MUSIC_LANGUAGE = "en"
MUSIC_KEY_SCALE = "E minor"
MUSIC_GENERATE_AUDIO_CODES = True
MUSIC_TOP_K = 2
MUSIC_TOP_P = 0.85
MUSIC_TEMPERATURE = 0.9
MUSIC_MIN_P = 0.0
MUSIC_CLASSIFIER_FREE_GUIDANCE_SCALE = 0.0

SOUND_EFFECT_MODEL_FILE = "mmaudio_large_44k_v2_fp16.safetensors"
SOUND_EFFECT_MODEL_PRECISION = "fp16"
SOUND_EFFECT_VAE_FILE = "mmaudio_vae_44k_fp16.safetensors"
SOUND_EFFECT_SYNCHFORMER_FILE = "mmaudio_synchformer_fp16.safetensors"
SOUND_EFFECT_CLIP_FILE = "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"
SOUND_EFFECT_MODE = "44k"
SOUND_EFFECT_PRECISION = "fp16"
SOUND_EFFECT_SEED = 7
SOUND_EFFECT_STEPS = 25
SOUND_EFFECT_CLASSIFIER_FREE_GUIDANCE = 4.5
SOUND_EFFECT_MASK_AWAY_CLIP = True
SOUND_EFFECT_FORCE_OFFLOAD = True
SOUND_EFFECT_VOLUME_DECIBELS = -6

VIDEO_FORMAT = "video/h264-mp4"
VIDEO_LOOP_COUNT = 0
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_CONSTANT_RATE_FACTOR = 19
VIDEO_SAVE_METADATA = True
VIDEO_TRIM_TO_AUDIO = False


@dataclass(frozen=True, slots=True)
class FinalizeRequest:
    """Everything the finalize workflow builder needs.

    Attributes:
        family: Family whose music and sound-effect prompts shape the audio.
        frame_count: Frames currently in the sequence; drives the duration
            math.
        output_directory: ComfyUI-side path of the output directory.
    """

    family: FamilyDefinition
    frame_count: int
    output_directory: str


@dataclass(frozen=True, slots=True)
class SegmentWindow:
    """One contiguous slice of a frame sequence finalized as a single job.

    Attributes:
        index: Zero-based position among the sequence's windows.
        skip_first_images: Source frames to skip in ``VHS_LoadImagesPath``.
        frame_count: Source frames this window holds (at most
            :data:`SEGMENT_SOURCE_FRAMES`).
    """

    index: int
    skip_first_images: int
    frame_count: int


def compute_interpolated_frame_count(frame_count: int) -> int:
    """Return the frame count after interpolation: ``(n - 1) * 4 + 1``."""
    return (frame_count - 1) * INTERPOLATION_MULTIPLIER + 1


def compute_audio_seconds(interpolated_frame_count: int) -> float:
    """Return the audio duration for a video of ``interpolated_frame_count``.

    Floored at :data:`MINIMUM_AUDIO_SECONDS`, the lowest value the ACE-Step
    latent node accepts, so short sequences still give MMAudio enough material
    for its 16-frame sync segments.
    """
    raw_seconds = interpolated_frame_count / VIDEO_FRAMES_PER_SECOND
    return max(raw_seconds, MINIMUM_AUDIO_SECONDS)


def compute_segment_video_seconds(window_frame_count: int) -> float:
    """Return the video duration one segment window produces."""
    return compute_audio_seconds(compute_interpolated_frame_count(window_frame_count))


def compute_segment_music_seconds(window_frame_count: int) -> float:
    """Return the music duration a segment generates: video plus the overlap."""
    return compute_segment_video_seconds(window_frame_count) + SEGMENT_MUSIC_EXTENSION_SECONDS


def compute_segment_sound_seconds(window_frame_count: int) -> float:
    """Return the effects duration a segment generates: video plus overlap."""
    return compute_segment_video_seconds(window_frame_count) + SEGMENT_SOUND_EXTENSION_SECONDS


def needs_segmentation(frame_count: int) -> bool:
    """Return True when a sequence is too long for one finalize pass.

    A single pass stays within the verified VRAM envelope only while its
    interpolated frame count fits what :data:`SEGMENT_SOURCE_FRAMES` source
    frames produce; anything longer must be finalized window by window.
    """
    single_pass_budget = compute_interpolated_frame_count(SEGMENT_SOURCE_FRAMES)
    return compute_interpolated_frame_count(frame_count) > single_pass_budget


def compute_segment_windows(
    frame_count: int, *, segment_size: int = SEGMENT_SOURCE_FRAMES
) -> tuple[SegmentWindow, ...]:
    """Split a sequence into contiguous windows of at most ``segment_size``.

    The windows tile ``[0, frame_count)`` without gaps or overlaps, so a
    segmented finalize covers exactly the source frames — boundary frames of
    adjacent windows are consecutive renders, which is why interpolation
    needs no overlap between windows.

    Raises:
        ValueError: If ``frame_count`` is below one or ``segment_size`` is
            below one.
    """
    if frame_count < 1:
        message = f"Cannot segment a sequence with {frame_count} frames"
        raise ValueError(message)
    if segment_size < 1:
        message = f"Segment size must be at least 1, received {segment_size}"
        raise ValueError(message)
    windows = []
    for index, first_frame in enumerate(range(0, frame_count, segment_size)):
        windows.append(
            SegmentWindow(
                index=index,
                skip_first_images=first_frame,
                frame_count=min(segment_size, frame_count - first_frame),
            )
        )
    return tuple(windows)


def build_finalize_workflow(request: FinalizeRequest) -> dict[str, dict[str, Any]]:
    """Build the API-format finalize workflow for one frame sequence.

    Raises:
        ValueError: If ``frame_count`` is below one; the caller (rendering
            layer) translates this into a user-facing error.
    """
    if request.frame_count < 1:
        message = f"Cannot finalize a sequence with {request.frame_count} frames"
        raise ValueError(message)
    sequence_key = request.family.sequence_key
    return _assemble_finalize_document(
        request.family,
        _DocumentSpec(
            frames_directory=f"{request.output_directory}/Zoomy/{sequence_key}",
            skip_first_images=0,
            image_load_cap=0,
            music_seconds=compute_segment_music_seconds(request.frame_count),
            sound_seconds=compute_segment_sound_seconds(request.frame_count),
            music_seed=MUSIC_SEED,
            filename_prefix=f"Zoomy_{sequence_key}",
        ),
    )


def build_finalize_segment_workflow(
    request: FinalizeRequest, window: SegmentWindow
) -> dict[str, dict[str, Any]]:
    """Build the finalize workflow for one segment window of a long sequence.

    The graph mirrors :func:`build_finalize_workflow` but reads only the
    window's frames and scores only the window's seconds of audio, so peak
    VRAM depends on the window size — never on the total frame count. Music
    evolves across windows (``MUSIC_SEED + window.index``) while tags, tempo,
    and key stay shared; the Python-side assembly crossfades the boundary.

    Raises:
        ValueError: If the sequence or the window holds no frames.
    """
    if request.frame_count < 1:
        message = f"Cannot finalize a sequence with {request.frame_count} frames"
        raise ValueError(message)
    if window.frame_count < 1:
        message = (
            f"Cannot finalize a segment window with {window.frame_count} frames "
            f"(index {window.index})"
        )
        raise ValueError(message)
    sequence_key = request.family.sequence_key
    return _assemble_finalize_document(
        request.family,
        _DocumentSpec(
            frames_directory=f"{request.output_directory}/Zoomy/{sequence_key}",
            skip_first_images=window.skip_first_images,
            image_load_cap=window.frame_count,
            music_seconds=compute_segment_music_seconds(window.frame_count),
            sound_seconds=compute_segment_sound_seconds(window.frame_count),
            music_seed=MUSIC_SEED + window.index,
            filename_prefix=f"Zoomy_{sequence_key}_seg{window.index:03d}",
            separate_soundtrack=True,
        ),
    )


@dataclass(frozen=True, slots=True)
class _DocumentSpec:
    """The values that vary between a full and a segment finalize graph."""

    frames_directory: str
    skip_first_images: int
    image_load_cap: int
    music_seconds: float
    sound_seconds: float
    music_seed: int
    filename_prefix: str
    separate_soundtrack: bool = False


def _assemble_finalize_document(
    family: FamilyDefinition, spec: _DocumentSpec
) -> dict[str, dict[str, Any]]:
    """Assemble the shared interpolate → music + SFX → mux graph."""
    workflow = ComfyWorkflow()
    frame_sequence = workflow.add(
        "load_frame_sequence",
        "VHS_LoadImagesPath",
        {
            "directory": spec.frames_directory,
            "image_load_cap": spec.image_load_cap,
            "skip_first_images": spec.skip_first_images,
            "select_every_nth": 1,
        },
    )
    interpolation_model = workflow.add(
        "load_interpolation_model",
        "FrameInterpolationModelLoader",
        {"model_name": INTERPOLATION_MODEL_FILE},
    )
    interpolated_frames = workflow.add(
        "interpolate_frames",
        "FrameInterpolate",
        {
            "images": frame_sequence,
            "interp_model": interpolation_model,
            "multiplier": INTERPOLATION_MULTIPLIER,
        },
    )
    workflow.add(
        "pad_interpolated_frames",
        "BatchPadToMin",
        {"images": interpolated_frames, "min_frames": MINIMUM_SYNC_FRAMES},
    )
    music_audio = _add_music_chain(
        workflow, family.music_prompt, spec.music_seconds, music_seed=spec.music_seed
    )
    sound_effect_audio = _add_sound_effect_chain(
        workflow,
        family.sound_effect_prompt,
        family.sound_effect_negative_prompt,
        spec.sound_seconds,
    )
    if spec.separate_soundtrack:
        _add_video_node(
            workflow,
            "render_video_music",
            interpolated_frames,
            music_audio,
            f"{spec.filename_prefix}_music",
        )
        _add_video_node(
            workflow,
            "render_video_effects",
            interpolated_frames,
            sound_effect_audio,
            f"{spec.filename_prefix}_sfx",
        )
        _add_stem_node(
            workflow, "save_music_stem", music_audio, f"{spec.filename_prefix}_music_stem"
        )
        _add_stem_node(
            workflow, "save_sound_stem", sound_effect_audio, f"{spec.filename_prefix}_sfx_stem"
        )
        return workflow.build()
    merged_audio = workflow.add(
        "merge_audio_tracks",
        "AudioMerge",
        {
            "audio1": music_audio,
            "audio2": sound_effect_audio,
            "merge_method": "add",
        },
    )
    _add_video_node(
        workflow, "render_video", interpolated_frames, merged_audio, spec.filename_prefix
    )
    return workflow.build()


def _add_video_node(
    workflow: ComfyWorkflow,
    key: str,
    images: NodeReference,
    audio: NodeReference,
    filename_prefix: str,
) -> None:
    """Add a VideoHelperSuite combine node with the proven encode settings."""
    workflow.add(
        key,
        "VHS_VideoCombine",
        {
            "images": images,
            "audio": audio,
            "frame_rate": VIDEO_FRAMES_PER_SECOND,
            "loop_count": VIDEO_LOOP_COUNT,
            "filename_prefix": filename_prefix,
            "format": VIDEO_FORMAT,
            "pix_fmt": VIDEO_PIXEL_FORMAT,
            "crf": VIDEO_CONSTANT_RATE_FACTOR,
            "save_metadata": VIDEO_SAVE_METADATA,
            "trim_to_audio": VIDEO_TRIM_TO_AUDIO,
            "pingpong": False,
            "save_output": True,
        },
    )


def _add_stem_node(
    workflow: ComfyWorkflow, key: str, audio: NodeReference, filename_prefix: str
) -> None:
    """Save one full-length soundtrack stem as lossless FLAC.

    The video twins trim audio to the video span (VHS applies ``-shortest``),
    so the over-generated overlap tails the assembly blends only survive in
    these stem files.
    """
    workflow.add(
        key,
        "SaveAudioAdvanced",
        {"audio": audio, "filename_prefix": filename_prefix, "format": "flac"},
    )


def _add_music_chain(
    workflow: ComfyWorkflow, music_prompt: str, audio_seconds: float, *, music_seed: int
) -> NodeReference:
    """Build the ACE-Step music bed chain and return its audio output."""
    music_model = workflow.add(
        "load_music_model",
        "UNETLoader",
        {"unet_name": MUSIC_MODEL_FILE, "weight_dtype": "default"},
    )
    shifted_music_model = workflow.add(
        "shift_music_model",
        "ModelSamplingAuraFlow",
        {"model": music_model, "shift": MUSIC_MODEL_SHIFT},
    )
    music_text_encoders = workflow.add(
        "load_music_text_encoders",
        "DualCLIPLoader",
        {
            "clip_name1": MUSIC_TEXT_ENCODER_PRIMARY_FILE,
            "clip_name2": MUSIC_TEXT_ENCODOR_SECONDARY_FILE,
            "type": MUSIC_TEXT_ENCODOR_TYPE,
        },
    )
    encoded_music_prompt = workflow.add(
        "encode_music_prompt",
        "TextEncodeAceStepAudio1.5",
        {
            "clip": music_text_encoders,
            "tags": music_prompt,
            "lyrics": "",
            "seed": music_seed,
            "bpm": MUSIC_BEATS_PER_MINUTE,
            "duration": audio_seconds,
            "timesignature": MUSIC_TIME_SIGNATURE,
            "language": MUSIC_LANGUAGE,
            "keyscale": MUSIC_KEY_SCALE,
            "generate_audio_codes": MUSIC_GENERATE_AUDIO_CODES,
            "top_k": MUSIC_TOP_K,
            "top_p": MUSIC_TOP_P,
            "temperature": MUSIC_TEMPERATURE,
            "min_p": MUSIC_MIN_P,
            "cfg_scale": MUSIC_CLASSIFIER_FREE_GUIDANCE_SCALE,
        },
    )
    zeroed_music_negative = workflow.add(
        "zero_music_negative",
        "ConditioningZeroOut",
        {"conditioning": encoded_music_prompt},
    )
    music_noise = workflow.add(
        "create_music_noise",
        "EmptyAceStep1.5LatentAudio",
        {"batch_size": 1, "seconds": audio_seconds},
    )
    sampled_music = workflow.add(
        "sample_music",
        "KSampler",
        {
            "model": shifted_music_model,
            "positive": encoded_music_prompt,
            "negative": zeroed_music_negative,
            "latent_image": music_noise,
            "seed": music_seed,
            "steps": MUSIC_STEPS,
            "cfg": MUSIC_CLASSIFIER_FREE_GUIDANCE,
            "sampler_name": MUSIC_SAMPLER,
            "scheduler": MUSIC_SCHEDULER,
            "denoise": 1.0,
        },
    )
    music_autoencoder = workflow.add(
        "load_music_autoencoder",
        "VAELoader",
        {"vae_name": MUSIC_AUTOENCODER_FILE},
    )
    return workflow.add(
        "decode_music",
        "VAEDecodeAudio",
        {"samples": sampled_music, "vae": music_autoencoder},
    )


def _add_sound_effect_chain(
    workflow: ComfyWorkflow,
    sound_effect_prompt: str,
    sound_effect_negative_prompt: str,
    audio_seconds: float,
) -> NodeReference:
    """Build the MMAudio video-synced sound-effect chain and return its audio.

    The sampler conditions on the *padded* interpolated frames (MMAudio needs
    at least 16 sync frames), while the video renders from the unpadded
    interpolation — that is why ``pad_interpolated_frames`` exists.
    """
    sound_effect_model = workflow.add(
        "load_sound_effect_model",
        "MMAudioModelLoader",
        {
            "mmaudio_model": SOUND_EFFECT_MODEL_FILE,
            "base_precision": SOUND_EFFECT_MODEL_PRECISION,
        },
    )
    sound_effect_utilities = workflow.add(
        "load_sound_feature_utilities",
        "MMAudioFeatureUtilsLoader",
        {
            "vae_model": SOUND_EFFECT_VAE_FILE,
            "synchformer_model": SOUND_EFFECT_SYNCHFORMER_FILE,
            "clip_model": SOUND_EFFECT_CLIP_FILE,
            "mode": SOUND_EFFECT_MODE,
            "precision": SOUND_EFFECT_PRECISION,
        },
    )
    sampled_sound_effects = workflow.add(
        "sample_sound_effects",
        "MMAudioSampler",
        {
            "mmaudio_model": sound_effect_model,
            "feature_utils": sound_effect_utilities,
            "images": NodeReference(key="pad_interpolated_frames"),
            "prompt": sound_effect_prompt,
            "negative_prompt": sound_effect_negative_prompt,
            "steps": SOUND_EFFECT_STEPS,
            "cfg": SOUND_EFFECT_CLASSIFIER_FREE_GUIDANCE,
            "seed": SOUND_EFFECT_SEED,
            "duration": audio_seconds,
            "mask_away_clip": SOUND_EFFECT_MASK_AWAY_CLIP,
            "force_offload": SOUND_EFFECT_FORCE_OFFLOAD,
        },
    )
    return workflow.add(
        "soften_sound_effects",
        "AudioAdjustVolume",
        {
            "audio": sampled_sound_effects,
            "volume": SOUND_EFFECT_VOLUME_DECIBELS,
        },
    )
