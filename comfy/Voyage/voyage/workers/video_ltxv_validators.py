"""LTXV spatial/frame validators (issue 036 split, DESIGN §5.3).

Move-verbatim from `voyage.workers.video_ltxv` (padded-size + spatial,
frame-count, fps, conditioning-start + `SPATIAL_GRANULARITY`): pure,
GPU-free, zero outward cross-refs as a block. `video_ltxv` re-exports
every name via explicit-`as` facade so existing importers hold.
"""

from __future__ import annotations

SPATIAL_GRANULARITY = 32


def padded_size(value: int, multiple: int = 32) -> int:
    """Round `value` up to a multiple (LTXV pads latents to /32)."""
    return ((value - 1) // multiple + 1) * multiple


def validate_spatial_size(width: int, height: int) -> None:
    """Reject sizes the pipeline would silently pad (DESIGN §5.3).

    Upstream pads non-conforming sizes with -1 then crops; Voyage refuses
    them instead so runs never silently bin/pad (768x432 -> 768x448).
    """
    for dimension_name, dimension in (("width", width), ("height", height)):
        if dimension <= 0:
            raise ValueError(f"LTXV {dimension_name} must be positive (got {dimension})")
        if dimension % SPATIAL_GRANULARITY != 0:
            padded = padded_size(dimension, SPATIAL_GRANULARITY)
            raise ValueError(
                f"LTXV {dimension_name} {dimension} is not divisible by "
                f"{SPATIAL_GRANULARITY} (would pad to {padded})"
            )


def validate_frame_count(frame_count: int) -> None:
    """Enforce the upstream (F-1)%8==0 temporal contract."""
    if frame_count <= 0:
        raise ValueError(f"LTXV frame count must be positive (got {frame_count})")
    if (frame_count - 1) % 8 != 0:
        raise ValueError(
            f"LTXV frame count {frame_count} violates the 8n+1 constraint "
            "((F-1)%8 must be 0; e.g. 25, 121, 257)"
        )


def validate_fps(fps: int) -> None:
    """Reject non-positive frame rates before they reach `mimsave`/tape (issue 064)."""
    if fps <= 0:
        raise ValueError(f"LTXV fps must be positive (got {fps})")


def validate_conditioning_start(start_frame: int, target_frames: int) -> None:
    """Enforce the upstream multiple-of-8 target-frame rule for extensions."""
    if start_frame < 0 or start_frame >= target_frames:
        raise ValueError(
            f"LTXV conditioning start {start_frame} out of range [0, {target_frames - 1}]"
        )
    if start_frame % 8 != 0:
        raise ValueError(f"LTXV conditioning start {start_frame} must be a multiple of 8")
