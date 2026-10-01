"""LTX-2.5 spatial/frame validators (DESIGN §140; backend `ltx25`).

Same split as `voyage.workers.video_ltxv_validators` (issue 036): pure,
GPU-free, zero outward cross-refs as a block. `video_ltx25` re-exports
every name via explicit-`as` facade so importers hold one path.

Geometry note: the committed Mode A segment is 1216x704 (both divisible
by 64 — the two-stage latent-upscale path wants /64, one-stage wants
/32). Stage 1 runs at half resolution (608x352, also /32-clean), so the
commit size must additionally be even-halvable (/64). Voyage refuses
anything else instead of silently padding (cf. ltxv 768x432 -> 768x448).
"""

from __future__ import annotations

SPATIAL_GRANULARITY = 32
"""One-stage granularity (matches the LTXV family contract)."""

SPATIAL_GRANULARITY_TWO_STAGE = 64
"""Mode A commit geometry must clear /64 (stage 1 runs at half size)."""


def padded_size(value: int, multiple: int = 32) -> int:
    """Round `value` up to a multiple (latents pad to /32)."""
    return ((value - 1) // multiple + 1) * multiple


def validate_spatial_size(width: int, height: int) -> None:
    """Reject sizes the Mode A graph would silently pad or halve badly.

    Both axes must clear /64: /32 for the latent contract plus even
    halving for the 608x352-class stage 1. 1216x704 passes (19x11 tiles
    of 64); anything else fails loud here instead of mid-render.
    """
    for dimension_name, dimension in (("width", width), ("height", height)):
        if dimension <= 0:
            raise ValueError(f"LTX25 {dimension_name} must be positive (got {dimension})")
        if dimension % SPATIAL_GRANULARITY_TWO_STAGE != 0:
            padded = padded_size(dimension, SPATIAL_GRANULARITY_TWO_STAGE)
            raise ValueError(
                f"LTX25 {dimension_name} {dimension} is not divisible by "
                f"{SPATIAL_GRANULARITY_TWO_STAGE} (would pad to {padded})"
            )


def validate_frame_count(frame_count: int) -> None:
    """Enforce the upstream (F-1)%8==0 temporal contract."""
    if frame_count <= 0:
        raise ValueError(f"LTX25 frame count must be positive (got {frame_count})")
    if (frame_count - 1) % 8 != 0:
        raise ValueError(
            f"LTX25 frame count {frame_count} violates the 8n+1 constraint "
            "((F-1)%8 must be 0; e.g. 25, 121, 257)"
        )


def validate_fps(fps: int) -> None:
    """Reject non-positive frame rates before they reach tape/ffmpeg (issue 064)."""
    if fps <= 0:
        raise ValueError(f"LTX25 fps must be positive (got {fps})")


def validate_conditioning_start(start_frame: int, target_frames: int) -> None:
    """Enforce the upstream multiple-of-8 target-frame rule for extensions."""
    if start_frame < 0 or start_frame >= target_frames:
        raise ValueError(
            f"LTX25 conditioning start {start_frame} out of range [0, {target_frames - 1}]"
        )
    if start_frame % 8 != 0:
        raise ValueError(f"LTX25 conditioning start {start_frame} must be a multiple of 8")
