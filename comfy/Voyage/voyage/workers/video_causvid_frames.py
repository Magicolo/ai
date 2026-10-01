"""CausVid rollout frame accounting (issue 036 split, DESIGN §5.4).

Move-verbatim from `voyage.workers.video_causvid` (dropped/novel/split/
reencode): pure integer arithmetic, zero outward cross-refs as a block.
`video_causvid` re-exports every name via explicit-`as` facade so existing
importers hold.
"""

from __future__ import annotations


def dropped_tail_frames(overlap_frames: int) -> int:
    """Frames dropped per rollout: ``4 * (overlap - 1) + 1`` (9 at overlap 3)."""
    return 4 * (overlap_frames - 1) + 1


def novel_frames_per_rollout(decoded_frames: int, overlap_frames: int) -> int:
    """Committed frames per rollout: decoded minus the dropped tail."""
    dropped = dropped_tail_frames(overlap_frames)
    novel = decoded_frames - dropped
    if novel <= 0:
        raise ValueError(
            f"CausVid rollout decodes {decoded_frames} frames but drops {dropped} "
            f"(overlap {overlap_frames}) — nothing would be committed"
        )
    return novel


def split_tail_novel(decoded_frames: int, overlap_frames: int) -> tuple[int, int]:
    """Return (dropped_tail, novel_committed) for one rollout's decoded video."""
    dropped = dropped_tail_frames(overlap_frames)
    return (dropped, novel_frames_per_rollout(decoded_frames, overlap_frames))


def reencode_window_frames(overlap_frames: int) -> int:
    """Committed tail frames needed to VAE-re-encode ``overlap`` latent frames.

    Numerically identical to :func:`dropped_tail_frames` by construction:
    ``(window - 1) / 4 + 1 == overlap`` always, so re-encoding the window
    yields exactly ``overlap`` latent frames for a resume rebuild.
    """
    return dropped_tail_frames(overlap_frames)
