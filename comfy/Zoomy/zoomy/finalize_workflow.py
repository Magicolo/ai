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


def build_finalize_workflow(request: FinalizeRequest) -> dict[str, dict[str, Any]]:
    """Build the API-format finalize workflow for one frame sequence.

    Raises:
        ValueError: If ``frame_count`` is below one; the caller (rendering
            layer) translates this into a user-facing error.
    """
    if request.frame_count < 1:
        message = f"Cannot finalize a sequence with {request.frame_count} frames"
        raise ValueError(message)
    interpolated_frame_count = compute_interpolated_frame_count(request.frame_count)
    audio_seconds = compute_audio_seconds(interpolated_frame_count)
    sequence_key = request.family.sequence_key
    frames_directory = f"{request.output_directory}/Zoomy/{sequence_key}"

    workflow = ComfyWorkflow()
    frame_sequence = workflow.add(
        "load_frame_sequence",
        "VHS_LoadImagesPath",
        {
            "directory": frames_directory,
            "image_load_cap": 0,
            "skip_first_images": 0,
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
    music_audio = _add_music_chain(workflow, request.family.music_prompt, audio_seconds)
    sound_effect_audio = _add_sound_effect_chain(
        workflow,
        request.family.sound_effect_prompt,
        request.family.sound_effect_negative_prompt,
        audio_seconds,
    )
    merged_audio = workflow.add(
        "merge_audio_tracks",
        "AudioMerge",
        {
            "audio1": music_audio,
            "audio2": sound_effect_audio,
            "merge_method": "add",
        },
    )
    workflow.add(
        "render_video",
        "VHS_VideoCombine",
        {
            "images": interpolated_frames,
            "audio": merged_audio,
            "frame_rate": VIDEO_FRAMES_PER_SECOND,
            "loop_count": VIDEO_LOOP_COUNT,
            "filename_prefix": f"Zoomy_{sequence_key}",
            "format": VIDEO_FORMAT,
            "pix_fmt": VIDEO_PIXEL_FORMAT,
            "crf": VIDEO_CONSTANT_RATE_FACTOR,
            "save_metadata": VIDEO_SAVE_METADATA,
            "trim_to_audio": VIDEO_TRIM_TO_AUDIO,
            "pingpong": False,
            "save_output": True,
        },
    )
    return workflow.build()


def _add_music_chain(
    workflow: ComfyWorkflow, music_prompt: str, audio_seconds: float
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
            "seed": MUSIC_SEED,
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
            "seed": MUSIC_SEED,
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
