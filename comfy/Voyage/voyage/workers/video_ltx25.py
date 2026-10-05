"""LTX-2.5 video worker: Q3 GGUF via in-process ComfyUI (DESIGN §140).

Backend `ltx25` (C1 primary, LTX2-REPORT.md recommendation). The worker
drives the pinned ComfyUI stack (ComfyUI @2f35f4a + ComfyUI-GGUF @6ea2651
+ gemma4 patch, image `voyage-ltx`) in-process through
`execution.PromptExecutor` on a fake server — no ComfyUI server process,
single resident GPU session like the other streaming workers.

Generation recipe (validated: LTX2.md Phase-0 Spike A, 14933 MiB peak):
Mode A two-stage — stage 1 608x352x121 distilled 8-sigma euler_ancestral
CFG 1.0, 2x latent upscale, stage-2 1216x704x121 3-step euler refine,
conv video VAE tiled decode, Gemma4 Q2_K text encoder. Continuation is
the native frozen-prefix mechanism (Phase-0 Spike B, seam 0.93x/1.02x
vs the 3x qual gate): trailing 25 frames pinned via
`LTXVImgToVideoInplace` strength 1.0, so 121f windows commit 96 novel
frames — the same 25+96 accounting as `video_ltxv`.

All parameterization is implicit (no user-facing quant/TE/VAE knobs):
Q3_K_M DiT only (user decision — OOM is a clean failure, no fallback
rung), fixed Gemma4 Q2_K encoder, fixed conv/audio VAEs. Audio is
all-deferred: the worker commits video only (the joint AV latent's
audio branch is never decoded — the executor runs the video save node
alone), while ACE-Step music takes carry the continuous mood across
segments and the finalize SFX dub owns effects (DESIGN §140 audio
continuity).
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voyage.hashing import sha256_file as shared_sha256_file
from voyage.registry_ltx25 import (
    LTX25_AUDIO_VAE_FILE,
    LTX25_AUDIO_VAE_SUBFOLDER,
    LTX25_COMMIT,
    LTX25_DIT_FILE,
    LTX25_DIT_REVISION,
    LTX25_GGUF_COMMIT,
    LTX25_SUBDIR,
    LTX25_TE_FILE,
    LTX25_TE_REVISION,
    LTX25_UPSC_FILE,
    LTX25_UPSC_SUBFOLDER,
    LTX25_VAE_REVISION,
    LTX25_VIDEO_VAE_FILE,
    LTX25_VIDEO_VAE_SUBFOLDER,
)
from voyage.workers import video_common
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts
from voyage.workers.video_common import TAIL_FILENAME as TAIL_FILENAME
from voyage.workers.video_common import TAPE_FILENAME as TAPE_FILENAME
from voyage.workers.video_ltx25_validators import SPATIAL_GRANULARITY as SPATIAL_GRANULARITY
from voyage.workers.video_ltx25_validators import (
    validate_conditioning_start as validate_conditioning_start,
)
from voyage.workers.video_ltx25_validators import validate_fps as validate_fps
from voyage.workers.video_ltx25_validators import validate_frame_count as validate_frame_count
from voyage.workers.video_ltx25_validators import validate_spatial_size as validate_spatial_size

RECOVERY_PROFILE = "ltx25"

# Mode A segment accounting: 121 = 8*15+1 and 25 = 8*3+1 satisfy the
# upstream (F-1)%8==0 contract; 121-25 = 96 novel frames (4.0 s at
# 24 fps). Baked constants, not tunables (Phase-0 Spike A locks them).
SEGMENT_TARGET_FRAMES = 121
CONDITIONING_TAIL_FRAMES = 25
COMMITTED_NOVEL_FRAMES = SEGMENT_TARGET_FRAMES - CONDITIONING_TAIL_FRAMES

# Mode A geometry (Phase-0 default): stage 1 at half resolution, commit
# at 1216x704. Both clear /64 (two-stage contract). The worker accepts
# the configured commit sizes only (high 1216x704 + low 768x448, see
# COMMIT_SIZE_OPTIONS) and rejects anything else loudly.
STAGE1_WIDTH = 608
STAGE1_HEIGHT = 352
COMMIT_WIDTH = 1216
COMMIT_HEIGHT = 704
NATIVE_FPS = 24

# Validated schedules (Spike A graph s0_121_B.json): stage-1 distilled
# 8-sigma euler_ancestral CFG 1.0, stage-2 3-step euler refine.
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.45, 0.3, 0.15, 0.0"
NEGATIVE_PROMPT = ""
QUANT_RUNG = "Q3_K_M"

STATE_MODE = "reconstructable_prefix"

FAST_PROFILE_ENV = "VOYAGE_LTX_FAST"
"""Set to `1` for the Phase-1 fast-iteration profile (never production)."""

CARRY_OVERRIDE_ENV = "VOYAGE_LTX_CARRY"
"""Carry-frame override for the Phase-1 carry sweep (validated, fail loud)."""

CONTINUATION_STRENGTH_ENV = "VOYAGE_LTX_STRENGTH"
"""Frozen-prefix strength override for bake-offs (default 1.0; ltx23 parity)."""

# Fast-iteration profile (Phase-1 mechanism work only): stage 1 384x224,
# commit 768x448, 49f windows, 9f carry. Every contract holds — commit
# clears /64 (12x7 tiles), stage 1 clears /32, (49-1)%8==0, (9-1)%8==0 —
# so mechanism reads transfer; winners get full-res confirmation before
# any production claim.
FAST_STAGE1_WIDTH = 384
FAST_STAGE1_HEIGHT = 224
FAST_COMMIT_WIDTH = 768
FAST_COMMIT_HEIGHT = 448
FAST_TARGET_FRAMES = 49
FAST_TAIL_FRAMES = 9

# Configured commit sizes (`--low-definition` / `--high-definition`): the
# low tier reuses the fast experiment profile's 768x448 commit geometry
# (and its 384x224 stage 1) with production 121f/25-carry accounting.
# The worker accepts exactly these two sizes and fails loud otherwise.
COMMIT_SIZE_OPTIONS = frozenset(
    {
        (COMMIT_WIDTH, COMMIT_HEIGHT),
        (FAST_COMMIT_WIDTH, FAST_COMMIT_HEIGHT),
    }
)

STAGE1_FOR_COMMIT_SIZE = {
    (COMMIT_WIDTH, COMMIT_HEIGHT): (STAGE1_WIDTH, STAGE1_HEIGHT),
    (FAST_COMMIT_WIDTH, FAST_COMMIT_HEIGHT): (FAST_STAGE1_WIDTH, FAST_STAGE1_HEIGHT),
}
"""Stage-1 (half-resolution) size per configured commit size."""


@dataclass(frozen=True)
class ExperimentProfile:
    """Resolved render geometry + segment accounting for one process run.

    Production defaults mirror the baked constants; the fast profile and
    the carry override are experiment-only (Phase 1) and resolve from the
    environment per call so one process can run sequential conditions.
    The tape profile hash covers the resolved values, so a fast tape
    never resumes on a production worker.
    """

    stage1_width: int
    stage1_height: int
    commit_width: int
    commit_height: int
    target_frames: int
    tail_frames: int


def resolve_experiment_profile() -> ExperimentProfile:
    """Resolve the render profile from the environment (fail loud)."""
    if os.environ.get(FAST_PROFILE_ENV, "0") == "1":
        profile = ExperimentProfile(
            stage1_width=FAST_STAGE1_WIDTH,
            stage1_height=FAST_STAGE1_HEIGHT,
            commit_width=FAST_COMMIT_WIDTH,
            commit_height=FAST_COMMIT_HEIGHT,
            target_frames=FAST_TARGET_FRAMES,
            tail_frames=FAST_TAIL_FRAMES,
        )
    else:
        profile = ExperimentProfile(
            stage1_width=STAGE1_WIDTH,
            stage1_height=STAGE1_HEIGHT,
            commit_width=COMMIT_WIDTH,
            commit_height=COMMIT_HEIGHT,
            target_frames=SEGMENT_TARGET_FRAMES,
            tail_frames=CONDITIONING_TAIL_FRAMES,
        )
    raw_carry = os.environ.get(CARRY_OVERRIDE_ENV)
    if raw_carry is None:
        return profile
    try:
        carry = int(raw_carry)
    except ValueError as exc:
        raise ValueError(f"{CARRY_OVERRIDE_ENV} must be an integer (got {raw_carry!r})") from exc
    if carry <= 0 or carry >= profile.target_frames or (carry - 1) % 8 != 0:
        raise ValueError(
            f"{CARRY_OVERRIDE_ENV} must satisfy 0 < carry < {profile.target_frames} "
            f"with (carry-1)%8==0 (got {raw_carry!r})"
        )
    return ExperimentProfile(
        stage1_width=profile.stage1_width,
        stage1_height=profile.stage1_height,
        commit_width=profile.commit_width,
        commit_height=profile.commit_height,
        target_frames=profile.target_frames,
        tail_frames=carry,
    )


def continuation_strength_from_env() -> float:
    """Read the frozen-prefix strength override for bake-offs (default 1.0).

    Production stays 1.0 (env unset). Fail loud on non-finite/out-of-range
    values — a silent clamp would invalidate the bake-off comparison.
    """
    raw = os.getenv(CONTINUATION_STRENGTH_ENV, "1.0")
    try:
        strength = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{CONTINUATION_STRENGTH_ENV} must be a float in (0, 1] (got {raw!r})"
        ) from exc
    if not 0.0 < strength <= 1.0 or strength != strength:
        raise ValueError(f"{CONTINUATION_STRENGTH_ENV} must be in (0, 1] (got {raw!r})")
    return strength


MILLISECONDS_PER_SECOND = 1000.0
"""`perf_counter` seconds → wall milliseconds for `stage_ms` telemetry."""

STAGE_MILLISECONDS_KEYS: tuple[str, ...] = ("encode_ms", "denoise_ms", "save_ms", "tape_ms")
"""Keys of the `stage_ms` mapping in every `generate_blocks` result (Stage A)."""

COMFYUI_PATH_ENV = "LTX_COMFYUI_PATH"
COMFYUI_PATH_DEFAULT = "/opt/comfyui"


def is_oom(failure: BaseException) -> bool:
    """True when `failure` is a CUDA out-of-memory (issue 049 idiom).

    Torch-free (no torch import — the slim gates image has none): matches
    the OOM exception class by name plus the allocator's `out of memory`
    message text, which is how OOMs surface when upstream code re-raises
    them as plain RuntimeError. Mirrors `video_ltxv.is_oom`; kept local
    so this fix touches only this worker.
    """
    if type(failure).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(failure).lower()


def split_prefix_novel(generated_frames: int, conditioning_frames: int) -> tuple[int, int]:
    """Return (prefix_discarded, novel_committed) for one extension clip.

    Fresh clips (conditioning 0) commit everything; conditioned clips drop
    the prefix and commit the remainder. Pure accounting — the caller must
    measure the real file count and never assume it matches.
    """
    if conditioning_frames < 0 or conditioning_frames > generated_frames:
        raise ValueError(f"conditioning {conditioning_frames} out of range [0, {generated_frames}]")
    return (conditioning_frames, generated_frames - conditioning_frames)


def extend_conditioning_tail(old_tail: list[Any] | None, novel: list[Any], carry: int) -> list[Any]:
    """Return the next clip's conditioning tail: the full window's last `carry` frames.

    The full window is (staged prefix + novel); the staged prefix IS the old
    tail, so the window tail is `(old_tail + novel)[-carry:]`. Production
    (novel 96 >= carry 25) reduces to `novel[-carry:]`; wide carries on
    short fast windows (novel 24 < carry 25) keep one stale frame instead of
    silently shortening the tail (which broke the next `_block_prefix` with
    `tail holds 24 frames, need 25`). Pure — pinned by host tests.
    """
    if carry <= 0:
        raise ValueError(f"LTX25 carry must be positive (got {carry})")
    if not novel:
        raise ValueError("LTX25 novel clip is empty")
    if old_tail is None:
        return list(novel[-carry:])
    return list((old_tail[-carry:] + novel)[-carry:])


def generation_profile_hash(
    width: int, height: int, fps: int, segment_target: int, conditioning_tail: int
) -> str:
    """Hash the reproducible generation profile (recovery tape)."""
    profile_text = (
        f"ltx25|{width}x{height}@{fps}|target={segment_target}"
        f"|tail={conditioning_tail}|dit={LTX25_DIT_FILE}"
        f"|first={STAGE1_SIGMAS}|second={STAGE2_SIGMAS}"
    )
    return hashlib.sha256(profile_text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Chunked SHA-256 (constant memory — tails are small, videos are not).

    Delegates to :func:`voyage.hashing.sha256_file` (issue 021); kept under
    the worker-local name so the tape code below is untouched.
    """
    return shared_sha256_file(path)


def build_recovery_tape(
    *,
    source_segment_id: str,
    conditioning_tail_path: str,
    conditioning_tail_sha256: str | None = None,
    prompts: list[str],
    seeds: list[int],
    width: int,
    height: int,
    fps: int,
    segment_target_frames: int = SEGMENT_TARGET_FRAMES,
    conditioning_tail_frames: int = CONDITIONING_TAIL_FRAMES,
    prompt_plan_digest: str | None = None,
) -> dict[str, Any]:
    """Build the §5.3 JSON recovery record (no GPU tensors — prefix replay).

    Extra ``last_prompt`` field (beyond the spec minimum) lets a resumed
    session keep reporting prompt changes truthfully. Model revisions pin
    the exact validated stack (ComfyUI + GGUF commits, DiT + TE revisions)
    so a tape never resumes across numerics.
    """
    tape: dict[str, Any] = {
        "backend": RECOVERY_PROFILE,
        "state_mode": STATE_MODE,
        "source_segment_id": source_segment_id,
        "conditioning_tail_path": conditioning_tail_path,
        "prompt_plan_digest": prompt_plan_digest,
        "seed": seeds[0] if seeds else 0,
        "seeds": list(seeds),
        "last_prompt": prompts[-1] if prompts else "",
        "model_revision": LTX25_DIT_REVISION,
        "text_encoder_revision": LTX25_TE_REVISION,
        "vae_revision": LTX25_VAE_REVISION,
        "pipeline_revision": f"{LTX25_COMMIT}+gguf@{LTX25_GGUF_COMMIT}",
        "profile_hash": generation_profile_hash(
            width, height, fps, segment_target_frames, conditioning_tail_frames
        ),
        "width": width,
        "height": height,
        "fps": fps,
        "segment_target_frames": segment_target_frames,
        "conditioning_tail_frames": conditioning_tail_frames,
    }
    if conditioning_tail_sha256 is not None:
        tape["conditioning_tail_sha256"] = conditioning_tail_sha256
    return tape


def parse_recovery_tape(tape: dict[str, Any]) -> dict[str, Any]:
    """Validate a §5.3 JSON tape; reject foreign tapes loudly.

    A missing tail *file* is not a parse error (run-file pruning): the
    tape records the would-be path, and resume derives it from the
    sibling segment video. Only a missing tail *path* fails here.
    """
    if not isinstance(tape, dict):
        raise ValueError("LTX25 recovery tape must be a JSON object")
    if tape.get("backend") != RECOVERY_PROFILE or tape.get("state_mode") != STATE_MODE:
        raise ValueError(
            "LTX25 recovery tape is a foreign format "
            f"(backend={tape.get('backend')!r}) — unresumable by design; re-render from seed"
        )
    tail_path = tape.get("conditioning_tail_path")
    if not isinstance(tail_path, str) or not tail_path:
        raise ValueError("LTX25 recovery tape has no conditioning tail path")
    return tape


class _FakeServer:
    """Minimal `execution.PromptExecutor` server surface (no ComfyUI server process).

    The executor only needs `client_id`/`send_sync`/`queue_updated`
    (verified against pinned `execution.py`); statuses are recorded and
    dropped — the supervisor reads artifacts from disk, never the wire.
    """

    def __init__(self) -> None:
        """Create a detached fake server (no client attached)."""
        self.client_id: str | None = None

    def send_sync(self, event: str, data: dict[str, Any], sid: str | None) -> None:
        """Accept a status message and drop it (disk is the record)."""
        del event, data, sid

    def queue_updated(self) -> None:
        """Accept a queue-progress ping and drop it."""


_COMFY_BOOTSTRAPPED = False
"""Process-wide ComfyUI node registry guard (`init_extra_nodes` runs once)."""


def _comfy_bootstrap(models_dir: Path, work_root: Path) -> None:
    """Import ComfyUI once and point it at the volume + scratch dirs.

    `folder_paths` output/input/temp land under a session scratch dir
    (system tmp, cleaned on evict — never the run dir, so `validate_run`
    orphan scans stay clean). Model folders point at the consolidated
    `/models/ltx25` volume (Hub layout); the `unet_gguf`/`clip_gguf`
    aliases follow their backing paths automatically (GGUF nodes.py).
    """
    global _COMFY_BOOTSTRAPPED
    comfy_root = Path(os.environ.get(COMFYUI_PATH_ENV, COMFYUI_PATH_DEFAULT))
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    import folder_paths
    import nodes

    for dirname in ("comfy_output", "comfy_input", "comfy_temp"):
        (work_root / dirname).mkdir(parents=True, exist_ok=True)
    folder_paths.set_output_directory(str(work_root / "comfy_output"))
    folder_paths.set_input_directory(str(work_root / "comfy_input"))
    folder_paths.set_temp_directory(str(work_root / "comfy_temp"))
    stack_dir = models_dir / LTX25_SUBDIR
    folder_paths.add_model_folder_path("unet", str(stack_dir), False)
    folder_paths.add_model_folder_path("clip", str(stack_dir), False)
    folder_paths.add_model_folder_path("vae", str(stack_dir / LTX25_VIDEO_VAE_SUBFOLDER), False)
    folder_paths.add_model_folder_path(
        "latent_upscale_models", str(stack_dir / LTX25_UPSC_SUBFOLDER), False
    )
    if not _COMFY_BOOTSTRAPPED:
        asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
        _COMFY_BOOTSTRAPPED = True


def build_mode_a_graph(
    *,
    prompt: str,
    seed: int,
    save_prefix: str,
    prefix_filenames: list[str] | None = None,
    strength: float = 1.0,
    stage1_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build the validated Mode A prompt-format graph (pure — CPU-testable).

    Mirrors Spike A `s0_121_B.json` node-for-node (29 nodes): Q3 DiT +
    Gemma4 TE + conv/audio VAEs, stage-1 608x352x121 distilled 8-sigma
    euler_ancestral CFG 1.0, 2x latent upscale, stage-2 1216x704x121
    3-step euler refine, tiled VAE decode into SaveImage. The graph
    retains the audio VAE/empty-latent/decode/SaveAudio branch (nodes
    7/9/27/29) only because the joint AV latent is the validated denoise
    path — the executor runs the video save node alone, so the audio
    branch never decodes (all-deferred audio: ACE-Step takes own the
    soundtrack).

    With `prefix_filenames` (input-root PNGs, Spike B convention) the graph
    gains LoadImage xN (ids 30+) + BatchImagesNode + LTXVImgToVideoInplace
    (strength frozen by default) and node 10 consumes the pinned latent
    instead of the empty one. Geometry/accounting resolve from the
    experiment profile (production 121f/25-carry unless VOYAGE_LTX_FAST
    or VOYAGE_LTX_CARRY select the Phase-1 fast profile); `stage1_size`
    overrides the profile stage-1 node for the low-definition tier.
    """
    profile = resolve_experiment_profile()
    stage1_width, stage1_height = (
        stage1_size if stage1_size is not None else (profile.stage1_width, profile.stage1_height)
    )
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": LTX25_DIT_FILE},
        },
        "2": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": LTX25_TE_FILE, "type": "ltxv"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": prompt},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": NEGATIVE_PROMPT},
        },
        "5": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "frame_rate": float(NATIVE_FPS),
            },
        },
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": LTX25_VIDEO_VAE_FILE}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": LTX25_AUDIO_VAE_FILE}},
        "8": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": stage1_width,
                "height": stage1_height,
                "length": profile.target_frames,
                "batch_size": 1,
            },
        },
        "9": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "frames_number": profile.target_frames,
                "frame_rate": float(NATIVE_FPS),
                "batch_size": 1,
                "audio_vae": ["7", 0],
            },
        },
        "10": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["8", 0], "audio_latent": ["9", 0]},
        },
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["5", 1],
                "cfg": 1.0,
            },
        },
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "14": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE1_SIGMAS}},
        "15": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["11", 0],
                "guider": ["12", 0],
                "sampler": ["13", 0],
                "sigmas": ["14", 0],
                "latent_image": ["10", 0],
            },
        },
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
        "17": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LTX25_UPSC_FILE}},
        "18": {
            "class_type": "LTXVLatentUpsampler",
            "inputs": {"samples": ["16", 0], "upscale_model": ["17", 0], "vae": ["6", 0]},
        },
        "19": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["18", 0], "audio_latent": ["16", 1]},
        },
        "20": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "21": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["5", 1],
                "cfg": 1.0,
            },
        },
        "22": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "23": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE2_SIGMAS}},
        "24": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["20", 0],
                "guider": ["21", 0],
                "sampler": ["22", 0],
                "sigmas": ["23", 0],
                "latent_image": ["19", 0],
            },
        },
        "25": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["24", 0]}},
        "26": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["25", 0],
                "vae": ["6", 0],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 128,
                "temporal_overlap": 32,
            },
        },
        "27": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {"samples": ["25", 1], "audio_vae": ["7", 0]},
        },
        "28": {
            "class_type": "SaveImage",
            "inputs": {"images": ["26", 0], "filename_prefix": f"{save_prefix}/frames"},
        },
        "29": {
            "class_type": "SaveAudio",
            "inputs": {"audio": ["27", 0], "filename_prefix": f"{save_prefix}/audio"},
        },
    }
    if prefix_filenames is not None:
        if len(prefix_filenames) != profile.tail_frames:
            raise ValueError(
                f"LTX25 prefix needs {profile.tail_frames} frames (got {len(prefix_filenames)})"
            )
        batch_inputs: dict[str, Any] = {}
        for index, filename in enumerate(prefix_filenames):
            node_id = str(30 + index)
            graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
            batch_inputs[f"images.image{index}"] = [node_id, 0]
        graph["55"] = {"class_type": "BatchImagesNode", "inputs": batch_inputs}
        graph["56"] = {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["6", 0],
                "image": ["55", 0],
                "latent": ["8", 0],
                "strength": strength,
                "bypass": False,
            },
        }
        video_latent = graph["10"]["inputs"]
        assert isinstance(video_latent, dict)
        video_latent["video_latent"] = ["56", 0]
        # The upsampler (18) drops the Inplace noise mask, so the stage-2
        # refine would repaint the frozen prefix. Re-attach the freeze
        # through the shared pack node (prefix branch only); K derives
        # from the carry via the LTX causal 8x grid.
        frozen_latent_frames = (len(prefix_filenames) - 1) // 8 + 1
        graph["57"] = {
            "class_type": "LTXPrefixFreeze",
            "inputs": {
                "samples": ["18", 0],
                "prefix_latent_frames": frozen_latent_frames,
            },
        }
        stage2_latent = graph["19"]["inputs"]
        assert isinstance(stage2_latent, dict)
        stage2_latent["video_latent"] = ["57", 0]
    return graph


def _run_ffmpeg(argv: list[str], purpose: str) -> None:
    """Run ffmpeg (arg-list, never shell) and fail loud on error."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("LTX25 needs an ffmpeg binary on PATH")
    completed = subprocess.run(
        [ffmpeg, *argv], capture_output=True, text=True, check=False, timeout=300
    )
    if completed.returncode != 0:
        raise RuntimeError(f"LTX25 {purpose} failed: {completed.stderr[-500:]}")


def _decode_tail_frames(
    tail_path: Path, frame_count: int, commit_size: tuple[int, int] | None = None
) -> list[Any]:
    """Decode a tail mp4 to RGB uint8 arrays (oldest-first).

    `commit_size` overrides the profile commit dims for the
    low-definition tier (defaults to the experiment profile).
    """
    import numpy as np

    profile = resolve_experiment_profile()
    commit_width, commit_height = (
        commit_size if commit_size is not None else (profile.commit_width, profile.commit_height)
    )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("LTX25 needs an ffmpeg binary on PATH")
    completed = subprocess.run(
        [
            ffmpeg,
            "-i",
            str(tail_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"LTX25 tail decode failed: {completed.stderr.decode()[-500:]}")
    pixel_bytes = bytes(commit_width * commit_height * 3)
    raw = completed.stdout
    if len(raw) != frame_count * len(pixel_bytes):
        raise ValueError(
            f"LTX25 tail {tail_path} decoded to {len(raw)} bytes, "
            f"expected {frame_count} frames of {commit_width}x{commit_height}"
        )
    return [
        np.frombuffer(raw[offset : offset + len(pixel_bytes)], dtype=np.uint8).reshape(
            (commit_height, commit_width, 3)
        )
        for offset in range(0, len(raw), len(pixel_bytes))
    ]


def _save_mp4(frames: list[Any], path: Path, fps: int) -> None:
    """Write RGB uint8 frames as h264 mp4 via system ffmpeg.

    The voyage-ltx image carries no imageio (by design — system ffmpeg
    is the guaranteed muxer, also used for tail decodes), so this
    mirrors `video_common.save_mp4` (libx264) through a temp PNG sequence
    instead of imageio.mimsave.
    """
    from PIL import Image

    if fps <= 0:
        raise ValueError(f"mp4 frame rate must be positive (got {fps})")
    if not frames:
        raise ValueError("mp4 needs at least one frame")
    with tempfile.TemporaryDirectory(prefix="ltx25-mux-") as tmp:
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(Path(tmp) / f"frame_{index:05d}.png")
        _run_ffmpeg(
            [
                "-framerate",
                str(fps),
                "-i",
                str(Path(tmp) / "frame_%05d.png"),
                "-frames:v",
                str(len(frames)),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            "segment mp4 mux",
        )
    if path.stat().st_size == 0:
        raise ValueError(f"LTX25 muxed mp4 is empty: {path}")


class LTX25Session:
    """Resident LTX-2.5 stack: ComfyUI in-process, Q3 DiT + VAEs on CUDA, TE on CPU.

    The executor owns a RAM-pressure node cache (server default), so
    text-encoder outputs reuse across blocks/segments of identical
    prompts (S21). Prefix frames ride the input dir (Spike B LoadImage-
    root convention); stage timings are coarse — ComfyUI in-process does
    not split TE-encode wall time, so `encode_ms` stays 0.0 (documented,
    not hidden) while execute/save/tape walls are measured.
    """

    def __init__(self, models_dir: Path, device: str, work_root: Path) -> None:
        """Load the validated stack and build the executor (fails loud on missing files)."""
        import torch

        stack_dir = models_dir / LTX25_SUBDIR
        needed = {
            "DiT": stack_dir / LTX25_DIT_FILE,
            "text encoder": stack_dir / LTX25_TE_FILE,
            "video VAE": stack_dir / LTX25_VIDEO_VAE_SUBFOLDER / LTX25_VIDEO_VAE_FILE,
            "audio VAE": stack_dir / LTX25_AUDIO_VAE_SUBFOLDER / LTX25_AUDIO_VAE_FILE,
            "upscaler": stack_dir / LTX25_UPSC_SUBFOLDER / LTX25_UPSC_FILE,
        }
        for label, path in needed.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"missing LTX25 {label} {path} — run `voyage models download` first"
                )
        if not device.startswith("cuda"):
            raise RuntimeError(f"video_ltx25 requires a CUDA device (got {device!r})")
        if not torch.cuda.is_available():
            raise RuntimeError("video_ltx25 requires a CUDA GPU")
        torch.set_grad_enabled(False)
        _comfy_bootstrap(models_dir, work_root)
        import comfy.model_management as model_management
        import execution

        print("LTX25 ComfyUI stack ready (Q3 DiT, Gemma4 TE, conv/audio VAE)", file=sys.stderr)
        self._models_dir = models_dir
        self._device = device
        self._work_root = work_root
        self._input_dir = work_root / "comfy_input"
        self._output_dir = work_root / "comfy_output"
        self._executor = execution.PromptExecutor(
            _FakeServer(),
            cache_type=execution.CacheType.RAM_PRESSURE,
            cache_args={
                "lru": 0,
                "ram": min(10.0, max(2.0, float(model_management.total_ram) / 10240.0)),
                "ram_inactive": min(128.0, float(model_management.total_ram) / 1024.0),
            },
        )
        self._torch = torch
        self._conditioning_tail_path: str | None = None
        self._tail_frames: list[Any] | None = None
        self._last_prompt: str | None = None

    def _execute_graph(self, graph: dict[str, Any], prompt_id: str) -> float:
        """Run one graph with a single OOM retry; return execute wall seconds.

        Q3-only (user decision): no fallback rung — a second OOM raises
        cleanly so the supervisor restarts or fails the run loudly.
        Only the video save node executes — the audio branch (27/29) is
        pruned (all-deferred audio).
        """
        started = time.perf_counter()
        try:
            self._executor.execute(graph, prompt_id, {}, ["28"])
        except Exception as first_failure:
            # OOM check below, anything else re-raises untouched.
            if not is_oom(first_failure):
                raise
            print("LTX25 OOM — evicting caches and retrying once", file=sys.stderr)
            import comfy.model_management as model_management

            model_management.unload_all_models()
            gc.collect()
            self._torch.cuda.empty_cache()
            self._executor.execute(graph, prompt_id, {}, ["28"])
        return time.perf_counter() - started

    def _block_prefix(
        self,
        conditioning_frames: int,
        segment_id: str,
        block_index: int,
        commit_size: tuple[int, int],
    ) -> list[str]:
        """Materialize prefix PNGs in the input dir (Spike B LoadImage-root rule)."""
        from PIL import Image

        if self._tail_frames is None:
            if self._conditioning_tail_path is None:
                raise RuntimeError("LTX25 continued block has no conditioning tail")
            tail_length = resolve_experiment_profile().tail_frames
            self._tail_frames = _decode_tail_frames(
                Path(self._conditioning_tail_path), tail_length, commit_size
            )
        tail = self._tail_frames
        if len(tail) != conditioning_frames:
            raise ValueError(f"LTX25 tail holds {len(tail)} frames, need {conditioning_frames}")
        filenames: list[str] = []
        for index, frame in enumerate(tail):
            filename = f"ltx25_prefix_{segment_id}_{block_index}_{index:02d}.png"
            Image.fromarray(frame).save(self._input_dir / filename)
            filenames.append(filename)
        return filenames

    def _read_block_images(self, save_prefix: str) -> list[Any]:
        """Read one executed block's PNGs oldest-first (fail loud on short reads)."""
        import numpy as np
        from PIL import Image

        block_dir = self._output_dir / save_prefix
        paths = sorted(block_dir.glob("frames_*.png"))
        target_frames = resolve_experiment_profile().target_frames
        if len(paths) != target_frames:
            raise ValueError(
                f"LTX25 block {save_prefix} produced {len(paths)} frames, expected {target_frames}"
            )
        frames: list[Any] = []
        for path in paths:
            with Image.open(path) as handle:
                frames.append(np.asarray(handle.convert("RGB")))
        return frames

    def generate_blocks(
        self,
        *,
        prompts: list[str],
        seeds: list[int],
        scene_cuts: list[bool],
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        segment_id: str,
        prompt_plan_digest: str | None = None,
        requested_frames: int | None = None,
    ) -> dict[str, Any]:
        """Render one segment (one Mode A clip per block), video only.

        Block 0 uses the resident tail unless fresh/`scene_cuts[0]`/missing;
        later blocks chain the in-memory tail (no mp4 roundtrip). Fresh
        clips commit the whole window; conditioned clips drop the carry
        prefix and commit the novel remainder. Counts are measured from
        disk, never assumed. Geometry/accounting follow the experiment
        profile (production 121f/25-carry unless VOYAGE_LTX_FAST or
        VOYAGE_LTX_CARRY select Phase-1 fast values). Audio is
        all-deferred — no audio is rendered or committed here.
        """
        profile = resolve_experiment_profile()
        validate_spatial_size(width, height)
        commit_size = (width, height)
        if commit_size not in COMMIT_SIZE_OPTIONS:
            options = ", ".join(
                f"{option[0]}x{option[1]}" for option in sorted(COMMIT_SIZE_OPTIONS)
            )
            raise ValueError(f"LTX25 commits one of {options} Mode A (got {width}x{height})")
        stage1_size = STAGE1_FOR_COMMIT_SIZE[commit_size]
        if fps != NATIVE_FPS:
            raise ValueError(f"LTX25 runs fixed at {NATIVE_FPS} fps (got {fps})")
        if not prompts:
            raise ValueError("LTX25 needs at least one prompt")
        if len(seeds) != len(prompts) or len(scene_cuts) != len(prompts):
            raise ValueError("LTX25 prompts/seeds/scene_cuts must align")

        segment_target_frames = profile.target_frames
        validate_frame_count(segment_target_frames)
        validate_conditioning_start(0, segment_target_frames)
        continuation_strength = continuation_strength_from_env()

        novel_frames_all: list[Any] = []
        generated_total = 0
        conditioning_total = 0
        fresh_blocks = 0
        prompt_changed = self._last_prompt is not None and prompts[0] != self._last_prompt
        denoise_total_ms = 0.0

        for block_index, (prompt, seed, scene_cut) in enumerate(
            zip(prompts, seeds, scene_cuts, strict=True)
        ):
            has_tail = self._tail_frames is not None or self._conditioning_tail_path is not None
            continued = has_tail and not scene_cut
            conditioning_frames = profile.tail_frames if continued else 0
            save_prefix = f"seg{segment_id}-b{block_index}"
            prefix_filenames = (
                self._block_prefix(conditioning_frames, segment_id, block_index, commit_size)
                if continued
                else None
            )
            graph = build_mode_a_graph(
                prompt=prompt,
                seed=seed,
                save_prefix=save_prefix,
                prefix_filenames=prefix_filenames,
                strength=continuation_strength,
                stage1_size=stage1_size,
            )
            prompt_id = f"ltx25-{segment_id}-{block_index}"
            denoise_total_ms += self._execute_graph(graph, prompt_id) * MILLISECONDS_PER_SECOND
            block_frames = self._read_block_images(save_prefix)
            generated_total += len(block_frames)
            conditioning_total += conditioning_frames
            _prefix_discarded, novel_count = split_prefix_novel(
                len(block_frames), conditioning_frames
            )
            novel = block_frames[conditioning_frames:]
            if len(novel) != novel_count:
                raise ValueError("LTX25 novel slice miscounted (unreachable)")
            novel_frames_all.extend(novel)
            if not continued:
                fresh_blocks += 1

            self._tail_frames = extend_conditioning_tail(
                self._tail_frames if continued else None, novel, profile.tail_frames
            )
            self._last_prompt = prompt
            if continued and prefix_filenames is not None:
                for filename in prefix_filenames:
                    (self._input_dir / filename).unlink(missing_ok=True)

        save_started = time.perf_counter()
        committed_frames = len(novel_frames_all)
        # The caller may pass an output path whose parent does not exist yet
        # (e.g. a fresh run directory); create it before any commit write.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_mp4(novel_frames_all, output_path, fps)
        tail_path = output_path.parent / TAIL_FILENAME
        tail_frames = novel_frames_all[-profile.tail_frames :]
        _save_mp4(tail_frames, tail_path, fps)
        save_total_ms = (time.perf_counter() - save_started) * MILLISECONDS_PER_SECOND

        tape = build_recovery_tape(
            source_segment_id=segment_id,
            conditioning_tail_path=str(tail_path),
            conditioning_tail_sha256=sha256_file(tail_path),
            prompts=prompts,
            seeds=seeds,
            width=width,
            height=height,
            fps=fps,
            segment_target_frames=segment_target_frames,
            conditioning_tail_frames=profile.tail_frames,
            prompt_plan_digest=prompt_plan_digest,
        )
        tape_path = output_path.parent / TAPE_FILENAME
        tape_started = time.perf_counter()
        video_common.write_tape_atomic(tape_path, tape)
        tape_total_ms = (time.perf_counter() - tape_started) * MILLISECONDS_PER_SECOND
        self._conditioning_tail_path = str(tail_path)
        prefix_discarded = generated_total - committed_frames
        del novel_frames_all
        return {
            "frames": committed_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "requested_frames": requested_frames
            if requested_frames is not None
            else segment_target_frames,
            "generated_frames": generated_total,
            "conditioning_frames": conditioning_total,
            "novel_frames": committed_frames,
            "committed_frames": committed_frames,
            "prefix_discarded_frames": prefix_discarded,
            "segment_target_frames": segment_target_frames,
            "conditioning_tail_frames": profile.tail_frames,
            "conditioning_start_frame": 0,
            "native_fps": fps,
            "prompt_changed": prompt_changed,
            "fresh_blocks": fresh_blocks,
            "resume_fallback": False,
            "conditioning_tail_path": str(tail_path),
            "recovery_path": str(tape_path),
            "quant_rung": QUANT_RUNG,
            "stage_ms": {
                "encode_ms": 0.0,
                "denoise_ms": denoise_total_ms,
                "save_ms": save_total_ms,
                "tape_ms": tape_total_ms,
            },
        }

    def resume_from_tape(self, tape: dict[str, Any]) -> dict[str, Any]:
        """Adopt the previous segment's tail video as the conditioning anchor.

        A missing tail file is derived from the sibling segment video
        (run-file pruning) and written to the recorded path, so later
        resumes hit it directly. The tape hash stays advisory: a derived
        tail re-hashes the tape in memory when it carries a tail hash
        (tapes without one are left alone), and an existing tail is
        adopted untouched — resume never hard-fails on a hash mismatch.
        """
        parsed = parse_recovery_tape(tape)
        tail_path = str(parsed["conditioning_tail_path"])
        raw_tail_frames = parsed.get("conditioning_tail_frames", CONDITIONING_TAIL_FRAMES)
        tail_frames = (
            int(raw_tail_frames)
            if isinstance(raw_tail_frames, int) and not isinstance(raw_tail_frames, bool)
            else CONDITIONING_TAIL_FRAMES
        )
        outcome = video_common.ensure_conditioning_tail(Path(tail_path), tail_frames=tail_frames)
        if outcome.derived:
            print(
                f"ltx25 derived missing tail from the segment video: {tail_path}",
                file=sys.stderr,
            )
            if "conditioning_tail_sha256" in parsed:
                parsed["conditioning_tail_sha256"] = sha256_file(outcome.path)
        self._conditioning_tail_path = tail_path
        self._tail_frames = None
        last_prompt = parsed.get("last_prompt")
        self._last_prompt = str(last_prompt) if isinstance(last_prompt, str) else None
        return {"resumed": True, "conditioning_tail_path": tail_path}

    def evict(self) -> None:
        """Unload the stack and scratch previous outputs (resume re-derives)."""
        import comfy.model_management as model_management

        model_management.unload_all_models()
        self._tail_frames = None
        self._conditioning_tail_path = None
        self._last_prompt = None
        del self._executor
        gc.collect()
        self._torch.cuda.empty_cache()
        scratch_output = self._output_dir
        scratch_input = self._input_dir
        for leftover in (scratch_output, scratch_input):
            for child in leftover.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)


_SESSION: LTX25Session | None = None
_INIT_PARAMS: dict[str, Any] = {}


def _build_session(work_root: Path) -> LTX25Session:
    models_dir = _INIT_PARAMS["models_dir"]
    assert isinstance(models_dir, Path)
    return LTX25Session(models_dir, str(_INIT_PARAMS["device"]), work_root)


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    """Load the resident stack (fails loud on missing weights / CPU-only)."""
    global _SESSION
    import torch

    checked_request(payload, models_dir=str)
    models_dir = Path(str(payload["models_dir"]))
    device = str(payload.get("device", "cuda:0"))
    if not device.startswith("cuda"):
        raise RuntimeError(f"video_ltx25 requires a CUDA device (got {device!r})")
    if not torch.cuda.is_available():
        raise RuntimeError("video_ltx25 requires a CUDA GPU")
    started = time.monotonic()
    _INIT_PARAMS.update({"models_dir": models_dir, "device": device})
    work_root = Path(tempfile.mkdtemp(prefix="voyage-ltx25-"))
    _SESSION = _build_session(work_root)
    name = torch.cuda.get_device_name(0)
    free_gib, total_gib = torch.cuda.mem_get_info()
    return {
        "status": "READY",
        "backend": RECOVERY_PROFILE,
        "gpu": name,
        "load_seconds": round(time.monotonic() - started, 1),
        "vram_free_gib": round(free_gib / 1024**3, 1),
        "vram_total_gib": round(total_gib / 1024**3, 1),
    }


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    """Report readiness plus VRAM headroom (never touches the session)."""
    del payload
    import torch

    ready = _SESSION is not None
    info: dict[str, Any] = {"status": "READY" if ready else "IDLE"}
    if torch.cuda.is_available():
        free_gib, total_gib = torch.cuda.mem_get_info()
        info["vram_free_gib"] = round(free_gib / 1024**3, 1)
        info["vram_total_gib"] = round(total_gib / 1024**3, 1)
    return info


def handle_generate_blocks(payload: dict[str, Any]) -> dict[str, Any]:
    """Render one segment (Mode A clips, video only) and commit it."""
    # Reject a bad frame rate before the session check so the boundary
    # fails fast (and stays CPU-testable without a GPU session).
    validate_fps(int(payload["fps"]))
    if _SESSION is None:
        raise RuntimeError("video_ltx25 not initialized — send `init` first")
    # One validated struct (issue 045): payload forms + shape checks live
    # in GenerateBlocksRequest.from_payload — no inline asserts. Geometry
    # defaults follow the active experiment profile (production commit
    # size unless VOYAGE_LTX_FAST selects the Phase-1 fast profile).
    profile = resolve_experiment_profile()
    request = video_common.GenerateBlocksRequest.from_payload(
        payload, width_default=profile.commit_width, height_default=profile.commit_height
    )
    output = request.output_path
    if request.width is None or request.height is None:
        raise ValueError("video_ltx25 requires width/height geometry (got native defaults)")
    result = _SESSION.generate_blocks(
        prompts=list(request.prompts),
        seeds=list(request.seeds),
        scene_cuts=list(request.scene_cuts),
        output_path=output,
        width=request.width,
        height=request.height,
        fps=request.fps,
        segment_id=request.segment_id,
        prompt_plan_digest=request.prompt_plan_digest,
        requested_frames=request.requested_frames,
    )
    artifacts = [str(output), str(result["conditioning_tail_path"])]
    artifacts.append(str(result["recovery_path"]))
    return {
        "blocks_generated": len(request.prompts),
        "artifacts": artifacts,
        "video": result,
    }


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured single-segment probes with VRAM peaks (§104).

    Probes are fresh text-to-video renders (scene cut, no resident tail), so
    this does NOT advance any stream — safe to run on a live
    session between segments (still prefer scratch). Each probe commits a
    full fresh window (no prefix to drop) at the active profile geometry.
    """
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    validate_benchmark_counts(warmup, measured)
    if _SESSION is None:
        raise RuntimeError("video_ltx25 not initialized — send `init` first")
    import torch

    profile = resolve_experiment_profile()
    session = _SESSION
    saved_tail = session._conditioning_tail_path
    saved_frames = session._tail_frames
    saved_prompt = session._last_prompt
    session._conditioning_tail_path = None
    session._tail_frames = None
    session._last_prompt = None
    benchmark_prompt = str(payload.get("prompt", "benchmark probe"))
    benchmark_seed = int(payload.get("seed", 0))
    benchmark_width = int(payload.get("width", profile.commit_width))
    benchmark_height = int(payload.get("height", profile.commit_height))
    benchmark_fps = int(payload.get("fps", NATIVE_FPS))
    committed = 0
    generated = 0

    def probe(output_path: Path, measured: bool) -> None:
        nonlocal committed, generated
        result = session.generate_blocks(
            prompts=[benchmark_prompt],
            seeds=[benchmark_seed],
            scene_cuts=[True],
            output_path=output_path,
            width=benchmark_width,
            height=benchmark_height,
            fps=benchmark_fps,
            segment_id="benchmark",
        )
        if measured:
            committed = int(result["committed_frames"])
            generated = int(result["generated_frames"])

    try:
        outcome = video_common.run_benchmark_harness(
            warmup,
            measured,
            "voyage-ltx25-bench-",
            probe,
            reset_peak_memory=torch.cuda.reset_peak_memory_stats,
            read_peak_gib=lambda: torch.cuda.max_memory_allocated() / 1024**3,
        )
    finally:
        session._conditioning_tail_path = saved_tail
        session._tail_frames = saved_frames
        session._last_prompt = saved_prompt
    walls = outcome.wall_seconds
    peaks = outcome.peak_gib
    mean = sum(walls) / len(walls)
    return {
        "backend": RECOVERY_PROFILE,
        "warmup_blocks": warmup,
        "measured_blocks": measured,
        "generated_frames_per_segment": generated,
        "committed_frames_per_segment": committed,
        "segment_wall_seconds": [round(wall, 3) for wall in walls],
        "segments_per_second": round(1 / mean, 3),
        "novel_fps_equivalent": round(committed / mean, 1),
        "vram_peak_gib": round(max(peaks), 2),
        "vram_avg_gib": round(sum(peaks) / len(peaks), 2),
    }


def _load_tape_json(recovery_path: str) -> dict[str, Any]:
    """Read a §5.3 JSON tape; foreign tapes fail with a clean-break error."""
    try:
        with open(recovery_path, encoding="utf-8") as handle:
            loaded: Any = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "LTX25 recovery tape is not a JSON object "
            "(foreign or pre-§5.3 format) — unresumable by design; re-render from seed"
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError("LTX25 recovery tape must be a JSON object")
    return loaded


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    """Adopt the latest committed tail video after a restart (§27.1)."""
    if _SESSION is None:
        raise RuntimeError("video_ltx25 not initialized — send `init` first")

    checked_request(payload, recovery_path=str)
    tape = _load_tape_json(str(payload["recovery_path"]))
    return {"resumed": True, **_SESSION.resume_from_tape(tape)}


def handle_evict_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    """Unload the video stack so audio can own the GPU (§40 pattern)."""
    del payload
    global _SESSION
    if _SESSION is not None:
        work_root = _SESSION._work_root
        _SESSION.evict()
        _SESSION = None
        shutil.rmtree(work_root, ignore_errors=True)
    return {"evicted": True}


def handle_rebuild(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the session after an eviction and adopt the tape (§40)."""
    global _SESSION

    checked_request(payload, recovery_path=str)
    if not _INIT_PARAMS:
        raise RuntimeError("video_ltx25 rebuilt before init")
    if _SESSION is not None:
        work_root = _SESSION._work_root
        _SESSION.evict()
        _SESSION = None
        shutil.rmtree(work_root, ignore_errors=True)
    work_root = Path(tempfile.mkdtemp(prefix="voyage-ltx25-"))
    _SESSION = _build_session(work_root)
    tape = _load_tape_json(str(payload["recovery_path"]))
    return {"rebuilt": True, **_SESSION.resume_from_tape(tape)}


def main() -> None:
    """Serve the LTX-2.5 video worker over JSONL-RPC (issue 019 map)."""
    serve(
        video_common.standard_serve_map(
            "ltx25",
            handle_init=handle_init,
            handle_health=handle_health,
            handle_generate_blocks=handle_generate_blocks,
            handle_benchmark=handle_benchmark,
            handle_evict_gpu=handle_evict_gpu,
            handle_rebuild=handle_rebuild,
            handle_resume=handle_resume,
        )
    )


if __name__ == "__main__":
    main()
