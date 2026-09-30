"""Duration, frame-budget, and CUDA-preflight helpers (DESIGN §§5.1, 58).

Leaf module of the issue-080 split: everything `cmd_generate` and the
TUI planners need to turn a human duration into segment counts, plus
the vocabulary-correct CUDA-stack preflight (issue 021). Duration
math uses the `backends.FLOAT_DUST_EPSILON` single source (issue 085) —
never a restated literal. `voyage.cli` re-exports every name below.
"""

from __future__ import annotations

import math
import re
import sys

from voyage.backends import FLOAT_DUST_EPSILON
from voyage.config import BACKEND_REGISTRY, ProjectConfig

_DURATION_EXAMPLES = "'5s', '90', '1m30s', '2m', '1h', '1h2m3.5s'"

_DURATION_PATTERN = re.compile(
    r"(?:(?P<hours>-?\d+(?:\.\d+)?)h)?"
    r"(?:(?P<minutes>-?\d+(?:\.\d+)?)m)?"
    r"(?:(?P<seconds>-?\d+(?:\.\d+)?)s?)?"
)


def parse_duration(raw: str) -> float:
    """Human-readable duration -> seconds (e.g. '5s', '90', '1m30s', '2m').

    Accepts hours (`1h`), fractional (`2.5m`, `1.5h`), combined
    (`1h2m3.5s`), bare (`90`) and whitespace-padded values. Signed
    numbers parse so negatives reach the positivity error below
    instead of the regex error. Rounds up to whole segments downstream
    (`segments_for_duration`), so the video never runs short.
    """
    match = _DURATION_PATTERN.fullmatch(raw.strip())
    if match is None or not any(match.groupdict().values()):
        raise ValueError(f"invalid duration {raw!r} (examples: {_DURATION_EXAMPLES})")
    total = 0.0
    for name, scale in (("hours", 3600.0), ("minutes", 60.0), ("seconds", 1.0)):
        value = match.group(name)
        if value is not None:
            total += float(value) * scale
    if total <= 0:
        raise ValueError(f"duration must be positive, got {raw!r}")
    return total


# Stream-A accounting (DESIGN §5.3, measured from the real tensors in
# voyage/workers/video_ltxv.py): every clip renders 121 frames; fresh blocks
# commit all 121, conditioned blocks drop the 25-frame prefix and commit 96
# novel. Duration planning uses the steady-state minimum (96 per block) so
# `generate --duration` never runs short however the fresh/extension mix
# lands (worker-reported frames remain the timeline truth).
_LTXV_NOVEL_BLOCK_FRAMES = 96

# LongLive Wan temporal VAE: L latents decode to (L-1)*4+1 frames, one block
# appends 8 latents (measured: 29f per 1-block segment, 93f per 3-block
# segment — qual-longlive2 + Phase-2 E2E, 2026-09-24/22).
_LONGLIVE_LATENTS_PER_BLOCK = 8

_LONGLIVE_DECODE_EXPANSION = 4

# CausVid DMD rollout: 81 decoded frames per rollout, the last
# 4*(overlap-1)+1 are the conditioning tail (9 at overlap 3) — 72 novel
# committed per rollout, uniform including rollout 0 (upstream long-video
# script parity). Duration planning uses the steady-state 72 so `generate
# --duration` never runs short (worker-reported frames stay the truth).
_CAUSVID_NOVEL_PER_ROLLOUT = 72


def _frames_per_segment(config: ProjectConfig) -> int:
    """Committed frames per segment for duration math (backend-specific)."""
    if config.video.backend == "ltxv":
        return _LTXV_NOVEL_BLOCK_FRAMES * config.video.blocks_per_segment
    if config.video.backend == "longlive2":
        latents = _LONGLIVE_LATENTS_PER_BLOCK * config.video.blocks_per_segment
        return (latents - 1) * _LONGLIVE_DECODE_EXPANSION + 1
    if config.video.backend == "causvid":
        return _CAUSVID_NOVEL_PER_ROLLOUT * config.video.blocks_per_segment
    return config.video.segment_frames


def segments_for_duration(duration_seconds: float, fps: int, frames_per_segment: int) -> int:
    """Segments needed to reach at least duration_seconds (rounds up, min 1)."""
    return max(1, math.ceil(duration_seconds * fps / frames_per_segment - FLOAT_DUST_EPSILON))


_CUDA_VIDEO_BACKENDS: frozenset[str] = frozenset(
    name for name, record in BACKEND_REGISTRY.items() if record.device.startswith("cuda")
)
"""Video backends needing the CUDA worker stack (derived, issue 021)."""

_CUDA_AUDIO_BACKENDS: frozenset[str] = frozenset(
    record.audio_backend
    for record in BACKEND_REGISTRY.values()
    if record.audio_device.startswith("cuda")
)
"""Audio backends needing the CUDA worker stack (derived, issue 021)."""

_CUDA_SFX_BACKENDS: frozenset[str] = frozenset(
    record.sfx_backend
    for record in BACKEND_REGISTRY.values()
    if record.sfx_device.startswith("cuda")
)
"""SFX backends needing the CUDA worker stack (derived, issue 021)."""

_CUDA_BACKENDS = _CUDA_VIDEO_BACKENDS | _CUDA_AUDIO_BACKENDS | _CUDA_SFX_BACKENDS
"""Legacy union across the video/audio/sfx vocabularies (issue 021).

Kept for the TUI warning import (tui_state.gpu_warning) and any reader
that only needs "does this name need CUDA". Per-branch checks above are
the vocabulary-correct source — never test a video backend against the
union (``acestep``/``mmaudio`` are not video backends).
"""


def _torch_available() -> bool:
    """Whether torch is importable (find_spec locates without importing)."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _cuda_stack_error(backend: str) -> str:
    return (
        f"error: backend {backend!r} needs the CUDA worker stack (torch), "
        "but torch is not importable in this container; re-run with "
        "VOYAGE_IMAGE=voyage-video:latest and VOYAGE_GPUS=1 (run.sh selects "
        "both automatically for CUDA backends and GPU-box bare launches)"
    )


def _cuda_offenders(config: ProjectConfig) -> list[str]:
    """CUDA backends configured on this run, video/audio/sfx qualified (021).

    The old message always blamed the video backend even when only the
    audio stack needed CUDA (e.g. acestep-audio + fake-video on CPU);
    the old set also never inspected the SFX backend, so a
    fake/fake/mmaudio run passed preflight and died late in the worker.
    Each branch checks its own vocabulary set (issue 021).
    """
    offenders: list[str] = []
    if config.video.backend in _CUDA_VIDEO_BACKENDS:
        offenders.append(f"video {config.video.backend!r}")
    if config.audio.backend in _CUDA_AUDIO_BACKENDS:
        offenders.append(f"audio {config.audio.backend!r}")
    if config.sfx.backend in _CUDA_SFX_BACKENDS:
        offenders.append(f"sfx {config.sfx.backend!r}")
    return offenders


def _require_cuda_stack(config: ProjectConfig) -> bool:
    """Fast-fail when a CUDA backend is configured but torch is unavailable.

    Workers are in-container subprocesses, so the container image must carry
    the worker stack (run.sh selects voyage-video automatically for CUDA
    backends and GPU-box bare launches; direct `docker run` users must
    pass the image + --gpus all themselves).
    find_spec locates torch without importing it — the supervisor never
    imports GPU libraries (§83).
    """
    # Seam dispatch (issue 080): torch presence resolves through the
    # voyage.cli namespace at call time (this shadows the module-global
    # definition below on purpose), so patching voyage.cli._torch_available
    # keeps intercepting the preflight exactly as pre-split.
    from voyage.cli import _torch_available as _seam_torch_available

    needs_cuda = (
        config.video.backend in _CUDA_VIDEO_BACKENDS
        or config.audio.backend in _CUDA_AUDIO_BACKENDS
        or config.sfx.backend in _CUDA_SFX_BACKENDS
    )
    if not needs_cuda or _seam_torch_available():
        return True
    offenders = _cuda_offenders(config)
    label = " + ".join(offenders) if offenders else config.video.backend
    print(_cuda_stack_error(label), file=sys.stderr)
    return False


def _warn_if_no_cuda(config: ProjectConfig) -> None:
    """Preflight: a CUDA device with no visible GPU fails late at worker init."""
    if not config.video.device.startswith("cuda"):
        return
    from voyage.doctor import probe as doctor_probe

    if not doctor_probe().get("nvidia_smi"):
        print(
            "warning: video device is CUDA but no GPU is visible; "
            "the worker will fail at init (use VOYAGE_GPUS=1 with run.sh)",
            file=sys.stderr,
        )
