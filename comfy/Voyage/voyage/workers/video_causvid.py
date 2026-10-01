"""Video worker: CausVid DMD causal generator backend (Track 2.3, DESIGN §5.4).

Upstream mirror, not a reinvention: the rollout loop follows the pinned
long-video script
(``minimal_inference/longvideo_autoregressive_inference.py`` @ ``adb6a5e``)
exactly — fresh ``torch.randn([1, 21, 16, 60, 104])`` per rollout,
``pipeline.inference(noise, text_prompts, return_latents=True,
start_latents=...)``, commit all but the last ``4 * (overlap - 1) + 1``
decoded frames (overlap 3 → drop 9 → **72 novel per rollout, uniform
including rollout 0**), then
``start_latents = cat([VAE-re-encoded tail slice, latents[:, -(overlap-1):]])``.
Config comes from the repo's ``configs/wan_causal_dmd.yaml`` @ pin via
omegaconf; the checkpoint strict-loads ``['generator']`` from
``causvid/autoregressive_checkpoint/model.pt`` into a resident
``InferencePipeline(config, device).to(device, bf16)`` on the session
device (issue 124 — every placement site threads `device`, never bare
`"cuda"`).

Deliberate deviations from the script (each documented where it happens):

- Per-rollout ``torch.Generator(device).manual_seed(seed)`` passed to
  ``randn`` — the script uses global RNG state, so identical seeds give
  identical rollouts here (deterministic replay).
- Scene cuts: ``scene_cuts[i]`` (or a missing/unreadable tail) restarts
  with ``start_latents=None``, like the ltxv worker's fresh-block rule.
- Resume rebuilds ``start_latents`` purely from the committed tail video
  plus overlap metadata (no latent sidecars — smallest reconstructable
  artifact). That is an approximation of the in-flight upstream state
  (re-encoded head + raw latent tail); the qualify track must A/B resumed
  vs uninterrupted continuity before trusting long resumed runs.
- The tail file itself (``video_tail.mp4``) is derived on demand at
  resume from the sibling segment video — ``generate_blocks`` no longer
  persists it (run-file pruning); the tape records the would-be path,
  and resume writes the derived file there for later resumes.
- Media is imageio mp4 @ native 16 fps (never relabeled — the worker
  refuses any requested fps other than 16; the 24 fps finalize stage is a
  later orchestrator track).

CRITICAL import constraint: ``causvid.models.wan.causal_inference``
transitively requires a CUDA GPU at import time, so every heavy import
(torch, causvid, omegaconf, imageio) is lazy inside functions — the module
top level stays slim-safe and CPU tests stub the pipeline with fakes.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from voyage.hashing import sha256_file as shared_sha256_file
from voyage.hashing import sha256_text as shared_sha256_text
from voyage.model_registry import (
    CAUSVID_CHECKPOINT_FILE as CAUSVID_CHECKPOINT_FILE,
)
from voyage.model_registry import (
    CAUSVID_COMMIT as CAUSVID_COMMIT,
)
from voyage.model_registry import (
    CAUSVID_HF_REPO,
    CAUSVID_HF_REVISION,
    CAUSVID_LICENSE,
    CAUSVID_SUBDIR,
    WAN21_SUBDIR,
    verify_checkpoint_against_manifest,
)
from voyage.workers import video_common
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts
from voyage.workers.video_common import TAIL_FILENAME as TAIL_FILENAME
from voyage.workers.video_common import TAPE_FILENAME as TAPE_FILENAME

RECOVERY_PROFILE = "causvid"
STATE_MODE = "reconstructable_prefix"
NATIVE_FPS = 16
NATIVE_WIDTH = 832
NATIVE_HEIGHT = 480
# Pinned config `image_or_video_shape` (configs/wan_causal_dmd.yaml @ adb6a5e):
# [batch, latent frames, channels, latent height, latent width].
LATENT_SHAPE = [1, 21, 16, 60, 104]
NUM_FRAME_PER_BLOCK = 3
DEFAULT_OVERLAP_FRAMES = 3
# Wan VAE compression: 4x temporal ((21 - 1) * 4 + 1 = 81 pixel frames),
# 8x spatial (60 * 8 = 480, 104 * 8 = 832).
TEMPORAL_COMPRESSION = 4
VAE_SPATIAL_FACTOR = 8
CONFIG_RELATIVE_PATH = "configs/wan_causal_dmd.yaml"
DEFAULT_CAUSVID_DIR = "/opt/causvid"
START_FROM_UPSTREAM = "vae_reencoded_head_plus_raw_latent_tail"
START_FROM_RESUME = "committed_tail_reencode"


def default_config_path() -> Path:
    """Config inside the pinned repo clone (overridable via init payload)."""
    return Path(os.environ.get("VOYAGE_CAUSVID_DIR", DEFAULT_CAUSVID_DIR)) / CONFIG_RELATIVE_PATH


def _enter_causvid_tree(models_dir: Path) -> None:
    """Reproduce upstream's CWD contract: hardcoded relative `wan_models/` lookups.

    Upstream hardcodes ``WanModel.from_pretrained("wan_models/Wan2.1-T2V-1.3B/")``
    (DiT config + weights), ``torch.load("wan_models/.../models_t5_...pth")``
    (T5) and ``pretrained_path="wan_models/.../Wan2.1_VAE.pth"`` (VAE) — all
    resolved against the process CWD (repo root). The supervisor spawns
    workers with CWD=run_dir, so anchor the repo root, link its
    ``wan_models/Wan2.1-T2V-1.3B/`` at our ``Wan2.1-T2V-1.3B`` models volume,
    and chdir there. All voyage
    paths are absolute, so the chdir is side-effect free.
    """
    repo_root = Path(os.environ.get("VOYAGE_CAUSVID_DIR", DEFAULT_CAUSVID_DIR))
    anchor = repo_root / "wan_models" / "Wan2.1-T2V-1.3B"
    target = models_dir / WAN21_SUBDIR
    if anchor.is_symlink() and Path(os.readlink(anchor)) == target:
        # Already correct (pre-created at image build): zero writes, so a
        # root-owned checkout stays usable for the host-mapped worker user.
        # The check runs before the mkdir below, so this path mutates
        # nothing — `is_symlink` is False (not an error) when the parent
        # is missing entirely.
        os.chdir(repo_root)
        return
    anchor.parent.mkdir(parents=True, exist_ok=True)
    if anchor.is_symlink():
        anchor.unlink()
    if not anchor.exists():
        anchor.symlink_to(target)
    os.chdir(repo_root)


def validate_overlap_frames(overlap_frames: int, num_frame_per_block: int) -> None:
    """Mirror the upstream assert: overlap must be a positive block multiple."""
    if overlap_frames <= 0:
        raise ValueError(f"CausVid overlap frames must be positive (got {overlap_frames})")
    if num_frame_per_block <= 0:
        raise ValueError(f"CausVid frames-per-block must be positive (got {num_frame_per_block})")
    if overlap_frames % num_frame_per_block != 0:
        raise ValueError(
            "CausVid num_overlap_frames must be divisible by num_frame_per_block "
            f"(got overlap {overlap_frames}, block {num_frame_per_block})"
        )


def dropped_tail_frames(overlap_frames: int) -> int:
    """Frames dropped per rollout: ``4 * (overlap - 1) + 1`` (9 at overlap 3)."""
    return 4 * (overlap_frames - 1) + 1


def decoded_frames_for_latents(latent_frames: int) -> int:
    """Pixel frames decoded from ``latent_frames`` at 4x temporal compression."""
    if latent_frames <= 0:
        raise ValueError(f"CausVid latent frames must be positive (got {latent_frames})")
    return (latent_frames - 1) * TEMPORAL_COMPRESSION + 1


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


def validate_latent_shape(latent_shape: list[int]) -> list[int]:
    """Accept a 5-dim [B, T, C, H, W] latent shape with positive dims."""
    if len(latent_shape) != 5 or not all(isinstance(dim, int) and dim > 0 for dim in latent_shape):
        raise ValueError(
            f"CausVid latent shape must be 5 positive ints [B, T, C, H, W] (got {latent_shape!r})"
        )
    return list(latent_shape)


def validate_native_geometry(width: int, height: int, latent_shape: list[int]) -> None:
    """Require width/height to match the configured latent spatial dims at 8x VAE.

    The noise tensor — and therefore the decoded resolution — is fixed by the
    config; anything else would silently change the generation profile, so it
    is refused instead of padded or relabeled.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"CausVid geometry must be positive (got {width}x{height})")
    if width % VAE_SPATIAL_FACTOR != 0 or height % VAE_SPATIAL_FACTOR != 0:
        raise ValueError(
            f"CausVid geometry {width}x{height} is not divisible by "
            f"{VAE_SPATIAL_FACTOR} (Wan VAE spatial factor)"
        )
    expected = (latent_shape[3] * VAE_SPATIAL_FACTOR, latent_shape[4] * VAE_SPATIAL_FACTOR)
    if (height, width) != (expected[0], expected[1]):
        raise ValueError(
            f"CausVid geometry {width}x{height} does not match the configured "
            f"latent shape {latent_shape} (expects {expected[1]}x{expected[0]} @8x VAE)"
        )


def select_tail_window(frames: NDArray[np.uint8], overlap_frames: int) -> NDArray[np.uint8]:
    """Last ``reencode_window_frames(overlap)`` frames (resume re-encode input)."""
    window = reencode_window_frames(overlap_frames)
    if frames.shape[0] < window:
        raise ValueError(
            f"CausVid tail has {frames.shape[0]} frames, need {window} "
            f"to re-encode {overlap_frames} latent frames"
        )
    return frames[-window:]


def commit_novel_frames(frames: NDArray[np.uint8], overlap_frames: int) -> NDArray[np.uint8]:
    """First ``novel`` frames — the dropped tail is never committed (all rollouts)."""
    _dropped, novel = split_tail_novel(int(frames.shape[0]), overlap_frames)
    return frames[:novel]


def prompt_plan_hash(prompts: list[str]) -> str:
    """Deterministic hash of the segment's prompt sequence (recovery tape)."""
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def generation_profile_hash(
    *,
    width: int,
    height: int,
    fps: int,
    latent_shape: list[int],
    overlap_frames: int,
    num_frame_per_block: int,
    config_sha256: str,
) -> str:
    """Hash the reproducible generation profile (recovery tape)."""
    profile_text = (
        f"causvid|{width}x{height}@{fps}|latent={latent_shape}"
        f"|overlap={overlap_frames}|block={num_frame_per_block}"
        f"|ckpt={CAUSVID_HF_REVISION}|code={CAUSVID_COMMIT}|cfg={config_sha256}"
    )
    return hashlib.sha256(profile_text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Chunked SHA-256 (constant memory — tails are small, videos are not).

    Delegates to :func:`voyage.hashing.sha256_file` (issue 021); kept under
    the worker-local name so the tape code below is untouched.
    """
    return shared_sha256_file(path)


def sha256_text(text: str) -> str:
    """SHA-256 of a short string (config file content for the tape).

    Delegates to :func:`voyage.hashing.sha256_text` (issue 021).
    """
    return shared_sha256_text(text)


def build_recovery_tape(
    *,
    source_segment_id: str,
    conditioning_tail_path: str,
    conditioning_tail_sha256: str | None = None,
    overlap_frames: int,
    num_frame_per_block: int,
    decoded_frames_per_rollout: int,
    novel_frames_per_rollout_count: int,
    rollouts: int,
    latent_shape: list[int],
    prompts: list[str],
    seeds: list[int],
    width: int,
    height: int,
    fps: int,
    config_sha256: str,
    start_latents_from: str = START_FROM_UPSTREAM,
    prompt_plan_digest: str | None = None,
) -> dict[str, Any]:
    """Build the §5.4 JSON recovery record (no GPU tensors — tail replay).

    Covers the DESIGN §5.4 minimum: ``num_overlap_frames``, actual decoded
    overlap (``dropped_tail_frames``) and newly committed
    (``novel_frames_per_rollout``) counts, latent-tail shape/dtype,
    ``start_latents_from`` (encoded video vs generated latent provenance),
    and the exact CausVid commit/config hashes plus the license record.

    The tail hash rides along only when the caller hashed a file for it:
    run-file pruning records the would-be tail path with no hash (nothing
    is persisted to hash), and resume fills it in after deriving.
    """
    dropped = dropped_tail_frames(overlap_frames)
    tape: dict[str, Any] = {
        "profile": RECOVERY_PROFILE,
        "backend": RECOVERY_PROFILE,
        "state_mode": STATE_MODE,
        "source_segment_id": source_segment_id,
        "conditioning_tail_path": conditioning_tail_path,
        "num_overlap_frames": overlap_frames,
        "num_frame_per_block": num_frame_per_block,
        "decoded_frames_per_rollout": decoded_frames_per_rollout,
        "dropped_tail_frames": dropped,
        "novel_frames_per_rollout": novel_frames_per_rollout_count,
        "rollouts": rollouts,
        "latent_shape": list(latent_shape),
        "latent_tail_shape": [latent_shape[0], overlap_frames - 1, *latent_shape[2:]],
        "latent_dtype": "bfloat16",
        "start_latents_from": start_latents_from,
        "prompt_plan_hash": prompt_plan_digest or prompt_plan_hash(prompts),
        "seed": seeds[0] if seeds else 0,
        "seeds": list(seeds),
        "last_prompt": prompts[-1] if prompts else "",
        "checkpoint_repo": CAUSVID_HF_REPO,
        "checkpoint_revision": CAUSVID_HF_REVISION,
        "checkpoint_license": CAUSVID_LICENSE,
        "code_commit": CAUSVID_COMMIT,
        "config_sha256": config_sha256,
        "profile_hash": generation_profile_hash(
            width=width,
            height=height,
            fps=fps,
            latent_shape=list(latent_shape),
            overlap_frames=overlap_frames,
            num_frame_per_block=num_frame_per_block,
            config_sha256=config_sha256,
        ),
        "width": width,
        "height": height,
        "fps": fps,
    }
    if conditioning_tail_sha256 is not None:
        tape["conditioning_tail_sha256"] = conditioning_tail_sha256
    return tape


def parse_recovery_tape(tape: dict[str, Any]) -> dict[str, Any]:
    """Validate a §5.4 JSON tape; reject everything else loudly.

    Clean break: this is the first CausVid worker, so there are no legacy
    CausVid tapes — any JSON without the causvid marker, and any non-JSON
    bytes (old torch pickles), are unresumable by design.

    A missing tail *file* is not a parse error (run-file pruning): the
    tape records the would-be path, and resume derives it from the
    sibling segment video. Only a missing tail *path* fails here.
    """
    if not isinstance(tape, dict):
        raise ValueError("CausVid recovery tape must be a JSON object")
    if tape.get("profile") != RECOVERY_PROFILE or tape.get("backend") != RECOVERY_PROFILE:
        raise ValueError(
            "CausVid recovery tape is foreign or legacy (missing "
            "'profile/backend: causvid') — unresumable by design; re-render from seed"
        )
    if tape.get("state_mode") != STATE_MODE:
        raise ValueError(
            f"CausVid recovery tape has unexpected state_mode {tape.get('state_mode')!r} "
            f"(expects {STATE_MODE!r})"
        )
    tail_path = tape.get("conditioning_tail_path")
    if not isinstance(tail_path, str) or not tail_path:
        raise ValueError("CausVid recovery tape has no conditioning tail path")
    overlap = tape.get("num_overlap_frames")
    block = tape.get("num_frame_per_block")
    if not isinstance(overlap, int) or not isinstance(block, int):
        raise ValueError("CausVid recovery tape has no overlap/block metadata")
    validate_overlap_frames(overlap, block)
    return tape


def _load_tape_json(recovery_path: str) -> dict[str, Any]:
    """Read a §5.4 JSON tape; non-JSON bytes fail with a clean-break error."""
    try:
        with open(recovery_path, encoding="utf-8") as handle:
            loaded: Any = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "CausVid recovery tape is not JSON (e.g. a torch pickle) — "
            "unresumable by design; re-render from seed"
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError("CausVid recovery tape must be a JSON object")
    return loaded


class _Rollout(NamedTuple):
    """One rollout's decoded output: uint8 frames plus the raw GPU tensors."""

    frames: NDArray[np.uint8]
    video: Any
    latents: Any
    decoded: int


_stub_encoder_classes: dict[Any, Any] = {}
"""One stub-encoder subclass per torch module object (issue 030).

Was `_StubEncoder`, a fresh `nn.Module` subclass defined per `_infer`
call (one class object per rollout): the concrete subclass is now built
once per torch module by `_stub_encoder_class` and only holds the
precomputed embeds for its call.
"""


def _stub_encoder_class(module_base: Any) -> Any:
    """Return the cached precomputed-encoder subclass for `module_base`.

    Caches by the base object itself (strong ref — one entry per torch
    module in practice), so rollouts share one class instead of
    allocating a new subclass per `_infer` call.
    """
    cached = _stub_encoder_classes.get(module_base)
    if cached is not None:
        return cached

    class _PrecomputedEncoder(module_base):  # type: ignore[misc]
        """Quacks like WanTextEncoder for one inference call."""

        def __init__(self, precomputed: dict[str, Any]) -> None:
            super().__init__()
            self._precomputed = precomputed

        def forward(self, text_prompts: Any = None) -> dict[str, Any]:
            del text_prompts
            return self._precomputed

    _stub_encoder_classes[module_base] = _PrecomputedEncoder
    return _PrecomputedEncoder


def _imageio_v2() -> Any:
    """Single lazy imageio accessor (slim-safe; one import site for mypy)."""
    import imageio.v2 as imageio  # type: ignore[import-not-found]

    return imageio


def _save_mp4(frames: NDArray[np.uint8], path: Path, fps: int) -> None:
    """Write (T,H,W,C) uint8 frames as h264 @ native fps (upstream writes 16).

    Frames arrive uint8 already (clipped in `_frames_from_video`); the
    mimsave mechanics live in :mod:`video_common` (issue 019).
    """
    video_common.save_mp4([frames[index] for index in range(frames.shape[0])], path, fps)


def _vae_encode_window(wrapper: Any, scaled: Any, dtype: Any) -> Any:
    """Encode a ``[1, C, T, H, W]`` [-1, 1] window via the raw VAE model.

    ``WanVAEWrapper`` exposes no ``.encode`` (only ``decode_to_pixel``) —
    the upstream script reaches through to ``vae.model.encode(x, scale)``
    with ``scale = [mean, 1/std]`` on the window's device/dtype. The raw
    ``encode`` chunks time internally (1, 4, 4, …), so one call handles any
    window length. Returns ``[1, T', C, H, W]`` latents.
    """
    device = scaled.device
    scale = [
        wrapper.mean.to(device=device, dtype=dtype),
        1.0 / wrapper.std.to(device=device, dtype=dtype),
    ]
    return wrapper.model.encode(scaled, scale).transpose(2, 1).to(dtype)


def _vae_encode_slice(vae: Any, video: Any, overlap_frames: int, dtype: Any) -> Any:
    """Mirror the script's tail-slice VAE re-encode for the next start_latents.

    Slice ``video[:, -4*(overlap-1)-1 : -4*(overlap-1), :]`` (at overlap 3:
    the single frame at index -9 — the head of the dropped tail), scale
    [0,1] → [-1,1], transpose into the vae's [B,C,T,H,W] layout, encode.
    Yields one latent frame; the caller cats the raw
    ``latents[:, -(overlap-1):]`` tail beside it.

    Overlap 1 is the degenerate case (issue 128): ``end`` is 0, and
    ``video[:, -1:0, :]`` selects nothing (negative start, zero stop,
    positive step) — an empty window that fails deep inside the VAE or,
    worse, poisons `start_latents`. Take the last frame explicitly so the
    slice agrees with `reencode_window_frames(1) == 1`.
    """
    if overlap_frames < 1:
        raise ValueError(f"CausVid overlap frames must be positive (got {overlap_frames})")
    end = -4 * (overlap_frames - 1)
    if end == 0:
        window = video[:, -1:, :]
    else:
        start = end - 1
        window = video[:, start:end, :]
    if int(window.shape[1]) < 1:
        raise ValueError(
            f"CausVid tail slice is empty for overlap {overlap_frames} "
            "(refusing to encode a zero-frame window)"
        )
    scaled = (window * 2.0 - 1.0).transpose(2, 1).to(dtype)
    return _vae_encode_window(vae, scaled, dtype)


def require_weight_files(models_dir: Path) -> dict[str, Path]:
    """Fail fast on a missing CausVid stack (pure — no torch/CUDA needed).

    Checks the DMD checkpoint plus the Wan2.1 base subset the worker loads
    underneath it (DiT shard, VAE, T5, tokenizer dir, config).
    """
    needed: dict[str, Path] = {
        "causvid DMD checkpoint": models_dir / CAUSVID_SUBDIR / CAUSVID_CHECKPOINT_FILE,
        "Wan2.1 DiT shard": models_dir / WAN21_SUBDIR / "diffusion_pytorch_model.safetensors",
        "Wan2.1 config": models_dir / WAN21_SUBDIR / "config.json",
        "Wan2.1 VAE": models_dir / WAN21_SUBDIR / "Wan2.1_VAE.pth",
        "Wan2.1 T5": models_dir / WAN21_SUBDIR / "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1 tokenizer": models_dir / WAN21_SUBDIR / "google" / "umt5-xxl",
    }
    for label, path in needed.items():
        if label.endswith("tokenizer"):
            if not path.is_dir() or not any(path.iterdir()):
                raise FileNotFoundError(
                    f"missing {label} {path} — run `voyage models download` first"
                )
        elif not path.is_file():
            raise FileNotFoundError(f"missing {label} {path} — run `voyage models download` first")
    return needed


class CausvidSession:
    """Resident CausVid stack: DMD generator + Wan2.1 VAE/T5 on CUDA, bf16.

    Continuation state (``_start_latents``) is GPU-resident across rollouts
    and segments; crash recovery re-derives it from the committed tail video
    (see :meth:`resume_from_tape` — a documented approximation of the
    in-flight upstream state, flagged for qualify-track A/B).
    """

    def __init__(
        self,
        models_dir: Path,
        device: str,
        latent_shape: list[int] | None = None,
        overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
        config_path: Path | None = None,
    ) -> None:
        import torch
        from causvid.models.wan.causal_inference import (  # type: ignore[import-not-found]
            InferencePipeline,
        )
        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        self._torch = torch
        self._device = device
        self._latent_shape = validate_latent_shape(
            list(latent_shape) if latent_shape is not None else list(LATENT_SHAPE)
        )
        resolved_config = config_path if config_path is not None else default_config_path()
        if not resolved_config.is_file():
            raise FileNotFoundError(
                f"missing CausVid config {resolved_config} "
                "(pinned repo clone should provide configs/wan_causal_dmd.yaml)"
            )
        config = OmegaConf.load(str(resolved_config))
        self._config_sha256 = sha256_text(Path(str(resolved_config)).read_text(encoding="utf-8"))
        block = int(getattr(config, "num_frame_per_block", NUM_FRAME_PER_BLOCK))
        validate_overlap_frames(overlap_frames, block)
        self._overlap_frames = overlap_frames
        self._num_frame_per_block = block
        weights = require_weight_files(models_dir)
        torch.set_grad_enabled(False)
        print("loading CausVid pipeline (bf16) ...", file=sys.stderr)
        pipeline = InferencePipeline(config, device=self._device)
        pipeline.to(device=self._device, dtype=torch.bfloat16)
        checkpoint = weights["causvid DMD checkpoint"]
        print(f"loading CausVid generator {checkpoint} ...", file=sys.stderr)
        # 005: sha256 against the download manifest first, then
        # weights_only (the snapshot holds tensors + metadata only).
        verify_checkpoint_against_manifest(models_dir, "causvid", checkpoint)
        loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        try:
            generator_state = loaded["generator"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"CausVid checkpoint {checkpoint} has no ['generator'] entry "
                "(expects the autoregressive DMD training snapshot)"
            ) from exc
        pipeline.generator.load_state_dict(generator_state, strict=True)
        # 16 GB fitment: upstream puts the 11 GB T5 on GPU via pipeline.to(),
        # which leaves no room for the DiT forward + ~6 GB KV prealloc
        # (measured: 14.87 GiB resident, forward OOMs). Park T5 on CPU —
        # `_infer` shuttles it back for the encode only (see below).
        pipeline.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        self._pipeline = pipeline
        self._start_latents: Any | None = None
        self._pending_tail_path: str | None = None
        self._pending_overlap: int = overlap_frames
        self._last_prompt: str | None = None

    def _fresh_noise(self, seed: int) -> Any:
        """Fresh noise per rollout on a per-seed generator (deterministic)."""
        torch = self._torch
        generator = torch.Generator(device=self._device).manual_seed(seed)
        return torch.randn(
            list(self._latent_shape),
            device=self._device,
            dtype=torch.bfloat16,
            generator=generator,
        )

    def _encode_conditionals(self, prompts: list[str]) -> list[dict[str, Any]]:
        """Encode every prompt on ONE T5 roundtrip (issue 029).

        The 11 GB text encoder moves to CUDA once, encodes each prompt
        with the same per-prompt call as before, and parks back on CPU
        once. Repeated alloc/free cycles fragment the allocator into
        OOMs — the old per-prompt shuttle paid that per prompt.

        The collect + empty_cache before the move matters on resident
        sessions: the previous segment leaves cached-but-unused blocks
        that fragment the 11 GB T5 placement (measured: 2nd-segment
        shuttle OOM at 14.72 GiB in use without it).
        """
        if not prompts:
            raise ValueError("causvid needs at least one prompt to encode")
        torch = self._torch
        pipeline = self._pipeline
        gc.collect()
        torch.cuda.empty_cache()
        pipeline.text_encoder.to(self._device)
        try:
            return [pipeline.text_encoder([prompt]) for prompt in prompts]
        finally:
            pipeline.text_encoder.to("cpu")
            torch.cuda.empty_cache()

    def _encode_conditional(self, prompt: str) -> dict[str, Any]:
        """Encode one prompt on a brief T5 GPU visit; T5 parks on CPU after.

        Single-prompt form (benchmark probes); segment renders use
        `_encode_conditionals` so N rollouts cost one 11 GB roundtrip.
        """
        return self._encode_conditionals([prompt])[0]

    def _infer(
        self, noise: Any, prompt: str, start: Any | None, conditional: dict[str, Any]
    ) -> tuple[Any, Any]:
        """One ``pipeline.inference`` call; returns (video, latents) tensors.

        16 GB fitment shim: the T5 text encoder (≈11 GB) is parked on CPU
        (see ``__init__``), but upstream ``inference()`` encodes inline with
        no skip flag — and the DiT consumes the embeds with no device
        transfer, so they must be CUDA. The caller precomputes them via
        ``_encode_conditional``; here they are reused through a stub encoder
        swapped onto our resident pipeline only (no upstream fork), always
        restored in ``finally``.
        """
        pipeline = self._pipeline
        real_encoder = pipeline.text_encoder
        stub_class = _stub_encoder_class(self._torch.nn.Module)
        pipeline.text_encoder = stub_class(conditional)
        try:
            video, latents = pipeline.inference(
                noise=noise,
                text_prompts=[prompt],
                return_latents=True,
                start_latents=start,
            )
        finally:
            pipeline.text_encoder = real_encoder
        return video, latents

    def _frames_from_video(self, video: Any) -> NDArray[np.uint8]:
        """Decode a [1,T,C,H,W] float video in [0,1] to (T,H,W,C) uint8 frames.

        Layout mirror of the script's
        ``video[0].permute(0, 2, 3, 1).cpu().numpy()``; the ``* 2.0 - 1.0``
        in the re-encode path confirms the [0,1] range assumption (qualify
        track confirms against a live rollout).
        """
        frames_any: Any = video[0].permute(0, 2, 3, 1).cpu().numpy()
        clipped: NDArray[np.uint8] = np.clip(frames_any * 255.0, 0, 255).astype(np.uint8)
        return clipped

    def _run_rollout(
        self, prompt: str, seed: int, start: Any | None, conditional: dict[str, Any]
    ) -> _Rollout:
        """Fresh noise → inference → uint8 frames (no commit, no state change).

        Seeds the global torch RNG (CPU + all CUDA) from the rollout seed
        first: upstream ``pipeline.inference`` draws unseeded global-CUDA
        randomness per call (measured live 2026-09-25: one rollout advances
        the cuda0 RNG state; same-seed/same-conditional fresh rollouts
        diverged ~0.5 latent mean-abs while T5 double-encodes bit-identical).
        Without this, every rollout is irreproducible across processes and
        segment boundaries cut. The explicit per-seed Generator in
        ``_fresh_noise`` is unaffected (separate stream); the single-threaded
        worker loop (§46) makes the global seeding race-free. Deliberate
        deviation from the upstream script (which rides the global RNG) —
        same class as the per-rollout Generator.
        """
        self._torch.manual_seed(seed)
        noise = self._fresh_noise(seed)
        video, latents = self._infer(noise, prompt, start, conditional)
        decoded = int(video.shape[1])
        return _Rollout(
            frames=self._frames_from_video(video), video=video, latents=latents, decoded=decoded
        )

    def _advance_start_latents(self, video: Any, latents: Any) -> Any:
        """Next ``start_latents`` via the upstream cat formula (in-session path)."""
        torch = self._torch
        head = _vae_encode_slice(self._pipeline.vae, video, self._overlap_frames, torch.bfloat16)
        tail = latents[:, -(self._overlap_frames - 1) :]
        start_latents = torch.cat([head, tail], dim=1)
        if int(start_latents.shape[1]) != self._overlap_frames:
            raise RuntimeError(
                "CausVid start_latents has "
                f"{int(start_latents.shape[1])} latent frames, expects {self._overlap_frames}"
            )
        return start_latents

    def _materialize_resume_start(self) -> tuple[Any | None, dict[str, Any] | None]:
        """Rebuild ``start_latents`` from the adopted tail video (resume path).

        Reads the committed tail mp4, takes the last re-encode window, and
        VAE-encodes it to exactly ``overlap`` latent frames. Returns
        ``(start_latents, fallback)``: ``fallback`` is None on success and
        ``{"reason", "tail_path"}`` when the tail is missing, unreadable, or
        re-encodes to the wrong shape — the caller renders fresh (never
        crashes on a stale anchor) and records the fallback in the segment
        result so the restart is visible above worker stderr (issue 134).
        The pending anchor clears only after the outcome is recorded,
        never before validation.
        """
        tail_path = self._pending_tail_path
        overlap = self._pending_overlap
        if tail_path is None or not Path(tail_path).exists():
            print(
                f"causvid resume tail {tail_path} missing — starting fresh",
                file=sys.stderr,
            )
            self._pending_tail_path = None
            return None, {"reason": "missing", "tail_path": tail_path}
        try:
            stacked = np.stack(_imageio_v2().mimread(str(tail_path)))
            window = select_tail_window(stacked, overlap)
            tensor = self._torch.from_numpy((window.astype(np.float32)) / 255.0)
            video = tensor.permute(0, 3, 1, 2).unsqueeze(0)
            scaled = (video * 2.0 - 1.0).transpose(2, 1).to(self._torch.bfloat16)
            start_latents = _vae_encode_window(self._pipeline.vae, scaled, self._torch.bfloat16)
        except Exception as exc:
            print(
                f"causvid resume tail {tail_path} unreadable ({exc}) — starting fresh",
                file=sys.stderr,
            )
            self._pending_tail_path = None
            return None, {"reason": "unreadable", "tail_path": tail_path}
        if int(start_latents.shape[1]) != overlap:
            print(
                f"causvid resume re-encode gave {int(start_latents.shape[1])} latent "
                f"frames, expects {overlap} — starting fresh",
                file=sys.stderr,
            )
            self._pending_tail_path = None
            return None, {"reason": "shape_mismatch", "tail_path": tail_path}
        self._pending_tail_path = None
        return start_latents, None

    def _rollout_start(self, scene_cut: bool) -> tuple[Any | None, bool, dict[str, Any] | None]:
        """Continuation latents for one rollout; (start, was_fresh, fallback)."""
        if scene_cut:
            return None, True, None
        if self._start_latents is not None:
            return self._start_latents, False, None
        if self._pending_tail_path is not None:
            rebuilt, fallback = self._materialize_resume_start()
            return rebuilt, rebuilt is None, fallback
        return None, True, None

    def generate_blocks(
        self,
        prompts: list[str],
        seeds: list[int],
        scene_cuts: list[bool],
        output_path: Path,
        width: int = NATIVE_WIDTH,
        height: int = NATIVE_HEIGHT,
        fps: int = NATIVE_FPS,
        segment_id: str = "000000",
        prompt_plan_digest: str | None = None,
        requested_frames: int | None = None,
    ) -> dict[str, Any]:
        """Render one segment: a rollout per prompt, 72 novel frames each.

        Every rollout commits ``decoded - dropped`` frames (uniform —
        rollout 0 drops its tail exactly like the rest, mirroring the
        script). Counts are measured from the real decoded tensors (§4.3
        rule: never assume the pipeline returned exactly the request).
        """
        if not prompts or not (len(prompts) == len(seeds) == len(scene_cuts)):
            raise ValueError("prompts/seeds/scene_cuts must be non-empty equal-length lists")
        if fps != NATIVE_FPS:
            raise ValueError(
                f"CausVid native fps is {NATIVE_FPS} — refusing to relabel as {fps} "
                "(DESIGN §5.4: run 16 fps end-to-end or add a validated resample stage)"
            )
        validate_native_geometry(width, height, self._latent_shape)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        novel_clips: list[NDArray[np.uint8]] = []
        decoded_per_rollout: list[int] = []
        novel_per_rollout: list[int] = []
        fresh_rollouts = 0
        generated_total = 0
        resume_fallback: dict[str, Any] | None = None
        prompt_changed = self._last_prompt is not None and prompts[0] != self._last_prompt
        # One T5 shuttle for the whole segment (issue 029): pre-encode
        # every prompt up front on a single CUDA roundtrip (each embed
        # is ~4 MB on GPU) instead of shuttling the 11 GB model per
        # rollout — repeated alloc/free cycles fragment into OOMs.
        conditionals = self._encode_conditionals(prompts)
        for index, (prompt, seed, cut) in enumerate(zip(prompts, seeds, scene_cuts, strict=True)):
            conditional = conditionals[index]
            start, was_fresh, fallback = self._rollout_start(cut)
            if fallback is not None and resume_fallback is None:
                resume_fallback = fallback
            fresh_rollouts += 1 if was_fresh else 0
            rollout = self._run_rollout(prompt, seed, start, conditional)
            decoded_per_rollout.append(rollout.decoded)
            generated_total += rollout.decoded
            novel = commit_novel_frames(rollout.frames, self._overlap_frames)
            novel_per_rollout.append(int(novel.shape[0]))
            novel_clips.append(novel)
            self._start_latents = self._advance_start_latents(rollout.video, rollout.latents)
        if len(novel_clips) == 1:
            video_frames = novel_clips[0]
        else:
            video_frames = np.concatenate(novel_clips, axis=0)
        committed_frames = int(video_frames.shape[0])
        _save_mp4(video_frames, output_path, NATIVE_FPS)
        # Run-file pruning: no `video_tail.mp4` is persisted — the tape
        # records the would-be path, and resume derives it from the
        # segment video written above.
        tail_path = output_path.parent / TAIL_FILENAME
        tape = build_recovery_tape(
            source_segment_id=segment_id,
            conditioning_tail_path=str(tail_path),
            overlap_frames=self._overlap_frames,
            num_frame_per_block=self._num_frame_per_block,
            decoded_frames_per_rollout=decoded_per_rollout[0],
            novel_frames_per_rollout_count=novel_per_rollout[0],
            rollouts=len(prompts),
            latent_shape=self._latent_shape,
            prompts=prompts,
            seeds=seeds,
            width=width,
            height=height,
            fps=NATIVE_FPS,
            config_sha256=self._config_sha256,
            prompt_plan_digest=prompt_plan_digest,
        )
        tape_path = output_path.parent / TAPE_FILENAME
        video_common.write_tape_atomic(tape_path, tape)
        self._last_prompt = prompts[-1]
        dropped_total = generated_total - committed_frames
        del novel_clips, video_frames
        return {
            "frames": committed_frames,
            "returned_frames": committed_frames,
            "fps": NATIVE_FPS,
            "native_fps": NATIVE_FPS,
            "width": width,
            "height": height,
            "requested_frames": (
                requested_frames if requested_frames is not None else committed_frames
            ),
            "generated_frames": generated_total,
            "decoded_frames_per_rollout": decoded_per_rollout[0],
            "decoded_per_rollout": decoded_per_rollout,
            "conditioning_frames": dropped_total,
            "dropped_tail_frames": dropped_total,
            "dropped_tail_frames_per_rollout": dropped_tail_frames(self._overlap_frames),
            "novel_frames": committed_frames,
            "novel_frames_per_rollout": novel_per_rollout[0],
            "novel_per_rollout": novel_per_rollout,
            "committed_frames": committed_frames,
            "rollouts": len(prompts),
            "fresh_rollouts": fresh_rollouts,
            "resume_fallback": resume_fallback,
            "scene_cuts": list(scene_cuts),
            "overlap_frames": self._overlap_frames,
            "num_frame_per_block": self._num_frame_per_block,
            "latent_shape": list(self._latent_shape),
            "prompt_changed": prompt_changed,
            "start_latents_from": START_FROM_UPSTREAM,
            "seeds": list(seeds),
            "conditioning_tail_path": str(tail_path),
            "recovery_path": str(tape_path),
        }

    def resume_from_tape(self, tape: dict[str, Any]) -> dict[str, Any]:
        """Adopt the previous segment's tail video as the re-encode anchor.

        No GPU work here (mirrors ltxv): the anchor materializes lazily at
        the next ``generate_blocks`` via :meth:`_materialize_resume_start`.

        A missing tail file is derived from the sibling segment video
        (run-file pruning) and written to the recorded path, so later
        resumes hit it directly. The derived file holds the last 25
        frames, or the full re-encode window when the overlap demands
        more — the materializer takes the newest window it needs. The
        tape hash stays advisory: a derived tail re-hashes the tape in
        memory when it carries a tail hash (tapes without one are left
        alone), and an existing tail is adopted untouched — resume never
        hard-fails on a hash mismatch.
        """
        parsed = parse_recovery_tape(tape)
        tail_path = str(parsed["conditioning_tail_path"])
        overlap = int(parsed["num_overlap_frames"])
        tail_frames = max(video_common.DERIVED_TAIL_FRAMES, reencode_window_frames(overlap))
        outcome = video_common.ensure_conditioning_tail(Path(tail_path), tail_frames=tail_frames)
        if outcome.derived:
            print(
                f"causvid derived missing tail from the segment video: {tail_path}",
                file=sys.stderr,
            )
            if "conditioning_tail_sha256" in parsed:
                parsed["conditioning_tail_sha256"] = sha256_file(outcome.path)
        tape_latent = parsed.get("latent_shape")
        if (
            isinstance(tape_latent, list)
            and len(tape_latent) == 5
            and [int(dim) for dim in tape_latent] != self._latent_shape
        ):
            raise ValueError(
                f"CausVid tape latent shape {tape_latent} does not match "
                f"session shape {self._latent_shape} — re-render from seed"
            )
        self._pending_tail_path = tail_path
        self._pending_overlap = int(parsed["num_overlap_frames"])
        self._start_latents = None
        last_prompt = parsed.get("last_prompt")
        self._last_prompt = str(last_prompt) if isinstance(last_prompt, str) else None
        return {
            "resumed": True,
            "conditioning_tail_path": tail_path,
            "start_latents_from": START_FROM_RESUME,
        }

    def evict(self) -> None:
        """Unload the stack so audio can own the GPU (§40 pattern)."""
        torch = self._torch
        self._start_latents = None
        self._pending_tail_path = None
        self._last_prompt = None
        del self._pipeline
        gc.collect()
        torch.cuda.empty_cache()


_SESSION: CausvidSession | None = None
_INIT_PARAMS: dict[str, Any] = {}


def _build_session() -> CausvidSession:
    models_dir = _INIT_PARAMS["models_dir"]
    assert isinstance(models_dir, Path)
    _enter_causvid_tree(models_dir)
    latent = _INIT_PARAMS.get("latent_shape")
    assert latent is None or isinstance(latent, list)
    config = _INIT_PARAMS.get("config_path")
    assert config is None or isinstance(config, Path)
    return CausvidSession(
        models_dir,
        str(_INIT_PARAMS["device"]),
        latent_shape=latent,
        overlap_frames=int(_INIT_PARAMS.get("overlap_frames", DEFAULT_OVERLAP_FRAMES)),
        config_path=config,
    )


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    """Init: pure checks first (fail fast without torch), then build the stack."""
    global _SESSION

    checked_request(payload, models_dir=str)
    models_dir = Path(str(payload["models_dir"]))
    device = str(payload.get("device", "cuda:0"))
    if not device.startswith("cuda"):
        raise RuntimeError(f"video_causvid requires a CUDA device (got {device!r})")
    raw_shape = payload.get("latent_shape")
    latent_shape = (
        validate_latent_shape([int(dim) for dim in raw_shape])
        if isinstance(raw_shape, list)
        else list(LATENT_SHAPE)
    )
    overlap_frames = int(payload.get("overlap_frames", DEFAULT_OVERLAP_FRAMES))
    validate_overlap_frames(overlap_frames, NUM_FRAME_PER_BLOCK)
    raw_config = payload.get("config_path")
    config_path = Path(str(raw_config)) if raw_config is not None else default_config_path()
    require_weight_files(models_dir)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"missing CausVid config {config_path} "
            "(pinned repo clone should provide configs/wan_causal_dmd.yaml)"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("video_causvid requires a CUDA GPU")
    device_index = video_common.cuda_device_index(device)
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"video_causvid device {device!r} is out of range "
            f"({torch.cuda.device_count()} CUDA device(s) visible)"
        )
    started = time.monotonic()
    _INIT_PARAMS.update(
        {
            "models_dir": models_dir,
            "device": device,
            "latent_shape": latent_shape,
            "overlap_frames": overlap_frames,
            "config_path": config_path,
        }
    )
    _SESSION = _build_session()
    assert _SESSION is not None
    session = _SESSION
    name = torch.cuda.get_device_name(device)
    free_gib, total_gib = torch.cuda.mem_get_info(device)
    return {
        "status": "READY",
        "backend": RECOVERY_PROFILE,
        "gpu": name,
        "load_seconds": round(time.monotonic() - started, 1),
        "vram_free_gib": round(free_gib / 1024**3, 1),
        "vram_total_gib": round(total_gib / 1024**3, 1),
        "latent_shape": latent_shape,
        "overlap_frames": overlap_frames,
        "num_frame_per_block": session._num_frame_per_block,
        "native_fps": NATIVE_FPS,
        "config_sha256": session._config_sha256,
    }


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    del payload
    import torch

    ready = _SESSION is not None
    info: dict[str, Any] = {"status": "READY" if ready else "IDLE"}
    if torch.cuda.is_available():
        device = str(_INIT_PARAMS.get("device", "cuda:0"))
        free_gib, total_gib = torch.cuda.mem_get_info(device)
        info["vram_free_gib"] = round(free_gib / 1024**3, 1)
        info["vram_total_gib"] = round(total_gib / 1024**3, 1)
    return info


def handle_generate_blocks(payload: dict[str, Any]) -> dict[str, Any]:
    if _SESSION is None:
        raise RuntimeError("video_causvid not initialized — send `init` first")
    # One validated struct (issue 045): payload forms + shape checks live
    # in GenerateBlocksRequest.from_payload — no inline asserts.
    request = video_common.GenerateBlocksRequest.from_payload(
        payload, width_default=NATIVE_WIDTH, height_default=NATIVE_HEIGHT
    )
    output = request.output_path
    if request.width is None or request.height is None:
        raise ValueError("video_causvid requires width/height geometry (got native defaults)")
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
    artifacts = [str(output), str(result["conditioning_tail_path"]), str(result["recovery_path"])]
    return {
        "blocks_generated": len(request.prompts),
        "artifacts": artifacts,
        "video": result,
    }


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured fresh-rollout probes with VRAM peaks (§104).

    Probes render with ``start_latents=None`` (scene-cut-equivalent), so —
    like ltxv — this does NOT advance any stream; session continuation state
    is saved and restored around the probes (still prefer scratch).
    Reports generated (decoded) vs committed (novel) frames per rollout.
    """
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    validate_benchmark_counts(warmup, measured)
    if _SESSION is None:
        raise RuntimeError("video_causvid not initialized — send `init` first")
    import torch

    session = _SESSION
    saved_start = session._start_latents
    saved_pending = session._pending_tail_path
    saved_prompt = session._last_prompt
    session._start_latents = None
    session._pending_tail_path = None
    session._last_prompt = None
    benchmark_prompt = str(payload.get("prompt", "benchmark probe"))
    benchmark_seed = int(payload.get("seed", 0))
    generated = 0
    committed = 0

    def probe(output_path: Path, measured: bool) -> None:
        nonlocal generated, committed
        rollout = session._run_rollout(
            benchmark_prompt,
            benchmark_seed,
            None,
            session._encode_conditional(benchmark_prompt),
        )
        novel = commit_novel_frames(rollout.frames, session._overlap_frames)
        _save_mp4(novel, output_path, NATIVE_FPS)
        if measured:
            generated = rollout.decoded
            committed = int(novel.shape[0])

    benchmark_device = video_common.cuda_device_index(str(_INIT_PARAMS.get("device", "cuda:0")))
    device_arg = video_common.torch_device_arg(benchmark_device)
    try:
        outcome = video_common.run_benchmark_harness(
            warmup,
            measured,
            "voyage-causvid-bench-",
            probe,
            reset_peak_memory=lambda: torch.cuda.reset_peak_memory_stats(*device_arg),
            read_peak_gib=lambda: torch.cuda.max_memory_allocated(*device_arg) / 1024**3,
        )
    finally:
        session._start_latents = saved_start
        session._pending_tail_path = saved_pending
        session._last_prompt = saved_prompt
    walls = outcome.wall_seconds
    peaks = outcome.peak_gib
    mean = sum(walls) / len(walls)
    return {
        "backend": RECOVERY_PROFILE,
        "warmup_blocks": warmup,
        "measured_blocks": measured,
        "generated_frames_per_rollout": generated,
        "committed_frames_per_rollout": committed,
        "dropped_tail_frames_per_rollout": dropped_tail_frames(session._overlap_frames),
        "rollout_wall_seconds": [round(wall, 3) for wall in walls],
        "rollouts_per_second": round(1 / mean, 3),
        "novel_fps_equivalent": round(committed / mean, 1),
        "vram_peak_gib": round(max(peaks), 2),
        "vram_avg_gib": round(sum(peaks) / len(peaks), 2),
    }


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    """Adopt the latest committed tail video after a restart (§27.1)."""
    if _SESSION is None:
        raise RuntimeError("video_causvid not initialized — send `init` first")

    checked_request(payload, recovery_path=str)
    tape = _load_tape_json(str(payload["recovery_path"]))
    return {"resumed": True, **_SESSION.resume_from_tape(tape)}


def handle_evict_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    """Unload the video stack so audio can own the GPU (§40 pattern)."""
    del payload
    global _SESSION
    if _SESSION is not None:
        _SESSION.evict()
        _SESSION = None
    return {"evicted": True}


def handle_rebuild(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the session after an eviction and adopt the tape (§40)."""
    global _SESSION

    checked_request(payload, recovery_path=str)
    if not _INIT_PARAMS:
        raise RuntimeError("video_causvid rebuilt before init")
    if _SESSION is not None:
        _SESSION.evict()
        _SESSION = None
    _SESSION = _build_session()
    tape = _load_tape_json(str(payload["recovery_path"]))
    return {"rebuilt": True, **_SESSION.resume_from_tape(tape)}


def main() -> None:
    serve(
        video_common.standard_serve_map(
            "causvid",
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
