"""Backend-neutral generation contracts for zoomy.

The image, music, and sound-effect engines run in-process (``diffusers``,
ACE-Step, MMAudio, RIFE), so this module holds everything the frame and
finalize paths share without referencing any execution backend:

- frame geometry (the ~1.5% per-frame zoom dive),
- video/audio duration math (interpolation multiplier, floors, segments),
- the per-family generation recipes (sampler settings live on the family;
  engine recipe constants live here),
- the request, progress, and statistics dataclasses,
- :class:`EngineProtocol`, the structural contract rendering and interface
  depend on (tests supply scripted fakes without GPU libraries).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from PIL.Image import Image

    from zoomy.family_catalog import FamilyDefinition, LoraDefinition

FRAME_WIDTH_PIXELS = 1376
FRAME_HEIGHT_PIXELS = 768
# Diffusion autoencoders downsample in powers of two, so odd frame sizes
# break the encode/decode round trip — every render size must be a multiple.
FRAME_SIZE_ALIGNMENT_PIXELS = 16
CROP_BORDER_PIXELS = 10
CROP_WIDTH_PIXELS = FRAME_WIDTH_PIXELS - 2 * CROP_BORDER_PIXELS
CROP_HEIGHT_PIXELS = FRAME_HEIGHT_PIXELS - 2 * CROP_BORDER_PIXELS
# Pixels per frame the view dives: the centered crop rescaled back to full
# size, so 1376 / 1356 is about 1.0147 with the (1376 - 1356) / 2 = 10 rule.
ZOOM_FACTOR_PER_FRAME = FRAME_WIDTH_PIXELS / CROP_WIDTH_PIXELS

INTERPOLATION_MULTIPLIER = 4
VIDEO_FRAMES_PER_SECOND = 32
# Source frames below this have no pairs to bisect, so they pass through
# interpolation unchanged (the retired FILM node behaved the same).
MINIMUM_FRAMES_FOR_INTERPOLATION = 2
# Lowest audio duration any audio stage accepts: the ACE-Step latent stage
# enforces seconds >= 1.0, and the floor also keeps MMAudio's synchformer
# happy on short sequences (it needs at least 16 sync frames). Overshoot is
# safe because the video mux always applies -shortest, trimming floored audio
# back to the video span.
MINIMUM_AUDIO_SECONDS = 1.0
MINIMUM_SYNC_FRAMES = 17
# Source frames per finalize segment: 48 frames interpolate to 189 frames and
# about 5.9 s of audio, inside the envelope verified live on the 16 GB card —
# while a 300-frame single pass (1197 interpolated frames) exhausts MMAudio.
SEGMENT_SOURCE_FRAMES = 48
# Extra audio each segment generates beyond its video span: the assembly
# overlaps neighbors by exactly the crossfade length, so generating the
# overlap keeps every stem sample-locked to its video (no drift) while the
# final mux trims the last tail. Each must equal the assembly's matching
# crossfade — pinned by test_extension_matches_assembly_overlaps.
SEGMENT_MUSIC_EXTENSION_SECONDS = 1.0
SEGMENT_SOUND_EXTENSION_SECONDS = 0.25

# Twin video encode, matching the former VideoHelperSuite settings: h264,
# yuv420p, constant-rate-factor 19, no loop, audio muxed with -shortest.
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_CONSTANT_RATE_FACTOR = 19

MUSIC_SEED = 31
MUSIC_BEATS_PER_MINUTE = 100
ACE_DIFFUSION_CONFIG_NAME = "acestep-v15-turbo"
ACE_LANGUAGE_MODEL_NAME = "acestep-5Hz-lm-0.6B"
ACE_LANGUAGE_BACKEND = "pt"

SOUND_EFFECT_MODEL_FILE = "mmaudio_large_44k_v2_fp16.safetensors"
SOUND_EFFECT_VAE_FILE = "mmaudio_vae_44k_fp16.safetensors"
SOUND_EFFECT_SYNCHFORMER_FILE = "mmaudio_synchformer_fp16.safetensors"
SOUND_EFFECT_CLIP_FILE = "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"
SOUND_EFFECT_SYNCHFORMER_CONFIG = "DFN5B-CLIP-ViT-H-14-384.json"
SOUND_EFFECT_MODE = "44k"
SOUND_EFFECT_SEED = 7
SOUND_EFFECT_STEPS = 25
SOUND_EFFECT_CLASSIFIER_FREE_GUIDANCE = 4.5
SOUND_EFFECT_MASK_AWAY_CLIP = True
SOUND_EFFECT_VOLUME_DECIBELS = -6
# Sync-video frame rate the synchformer encodes at. Unrelated to
# SOUND_EFFECT_STEPS (diffusion steps that happen to share the value): the
# SFX slice math must divide by this, never a literal.
SOUND_EFFECT_SYNC_FRAMES_PER_SECOND = 25
# Sync-frame edge length the vendor recipe normalizes to before encoding.
SOUND_EFFECT_SYNC_FRAME_PIXELS = 224


@dataclass(frozen=True, slots=True)
class FrameRenderRequest:
    """Everything the frame engine needs for one frame.

    Attributes:
        family: The model family to render with.
        prompt: Positive prompt text for this frame.
        negative_prompt: Negative prompt text; only used when the family
            defines one (families without a negative use zeroed conditioning).
        frame_count: Frames already in the sequence; zero means cold start.
        lora_selections: Selected LoRA styles with strengths, applied in this
            order.
        seed: Seed for the diffusion pass; the interface draws a fresh one
            per frame.
        frame_width: Frame width in pixels; must be a positive multiple of
            :data:`FRAME_SIZE_ALIGNMENT_PIXELS`.
        frame_height: Frame height in pixels, same constraint as the width.
    """

    family: FamilyDefinition
    prompt: str
    negative_prompt: str | None
    frame_count: int
    lora_selections: Sequence[tuple[LoraDefinition, float]]
    seed: int
    frame_width: int = FRAME_WIDTH_PIXELS
    frame_height: int = FRAME_HEIGHT_PIXELS


@dataclass(frozen=True, slots=True)
class FinalizeRequest:
    """Everything the finalize engine needs for one frame sequence.

    Attributes:
        family: Family whose music and sound-effect prompts shape the audio.
        frame_count: Frames currently in the sequence; drives the duration
            math.
    """

    family: FamilyDefinition
    frame_count: int


@dataclass(frozen=True, slots=True)
class SegmentWindow:
    """One contiguous slice of a frame sequence finalized as a single job.

    Attributes:
        index: Zero-based position among the sequence's windows.
        skip_first_images: Source frames to skip before this window.
        frame_count: Source frames this window holds (at most
            :data:`SEGMENT_SOURCE_FRAMES`).
    """

    index: int
    skip_first_images: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """One status update yielded to the interface.

    Attributes:
        message: Human-readable progress line.
        frame_path: Newest frame when one was produced this operation.
        video_path: Finished video when one was produced this operation.
        frame_count: Frames in the sequence after this operation.
        elapsed_seconds: Queue-to-completion seconds on terminal updates;
            ``None`` on interim progress updates.
    """

    message: str
    frame_path: Path | None = None
    video_path: Path | None = None
    frame_count: int = 0
    elapsed_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class EngineStatistics:
    """Live memory figures reported by the in-process engine.

    Attributes:
        system_memory_free_bytes: Free system RAM in bytes, or ``None``
            when /proc is unreadable.
        system_memory_total_bytes: Total system RAM in bytes, or ``None``
            when /proc is unreadable.
        video_memory_free_bytes: Free VRAM of the render device, or ``None``
            when no CUDA device is available.
        video_memory_total_bytes: Total VRAM of the render device, or ``None``
            when no CUDA device is available.
    """

    system_memory_free_bytes: int | None
    system_memory_total_bytes: int | None
    video_memory_free_bytes: int | None
    video_memory_total_bytes: int | None


class EngineProtocol(Protocol):
    """Structural contract for anything that generates zoomy artifacts."""

    def render_frame(self, request: FrameRenderRequest) -> Image:
        """Render one frame image for ``request`` (caller saves it)."""
        ...

    def finalize_sequence(self, request: FinalizeRequest) -> Iterator[ProgressUpdate]:
        """Turn the sequence into a video, yielding progress until it lands.

        The terminal update carries the finished (audio-muxed) video path.
        """
        ...

    def request_interrupt(self) -> None:
        """Ask the engine to abort the running job at the next checkpoint."""
        ...

    def is_ready(self) -> bool:
        """Return True when the engine can accept work (models reachable)."""
        ...

    def engine_statistics(self) -> EngineStatistics:
        """Return live memory figures for the stats line and health badge."""
        ...


def compute_interpolated_frame_count(frame_count: int) -> int:
    """Return the frame count after interpolation: ``(n - 1) * 4 + 1``.

    Raises:
        ValueError: If ``frame_count`` is below one.
    """
    if not frame_count >= 1:
        message = f"Interpolation needs at least 1 source frame, received {frame_count}"
        raise ValueError(message)
    return (frame_count - 1) * INTERPOLATION_MULTIPLIER + 1


def compute_frames_for_seconds(target_seconds: float) -> int:
    """Return the smallest source count whose video covers ``target_seconds``.

    Inverts :func:`compute_interpolated_frame_count` against the video frame
    rate, so a fixed-length run knows how many frames to render before the
    interpolation and audio stages run.

    Raises:
        ValueError: If ``target_seconds`` is not positive.
    """
    if not target_seconds > 0:
        message = f"Video duration must be positive, received {target_seconds}"
        raise ValueError(message)
    needed_interpolated = math.ceil(target_seconds * VIDEO_FRAMES_PER_SECOND)
    frames = math.ceil((needed_interpolated - 1) / INTERPOLATION_MULTIPLIER) + 1
    return max(1, frames)


def compute_audio_seconds(interpolated_frame_count: int) -> float:
    """Return the audio duration for a video of ``interpolated_frame_count``.

    Floored at :data:`MINIMUM_AUDIO_SECONDS`, the lowest value the ACE-Step
    latent stage accepts, so short sequences still give MMAudio enough
    material for its 16-frame sync segments.

    Raises:
        ValueError: If ``interpolated_frame_count`` is below one.
    """
    if not interpolated_frame_count >= 1:
        message = f"Audio needs at least 1 frame, received {interpolated_frame_count}"
        raise ValueError(message)
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
