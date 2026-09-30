"""MMAudio SFX compat layer (SFX slice 2, DESIGN three-caption doctrine).

Resident-stack wrapper over upstream hkchengrex/MMAudio (code pin in
model_registry.py): video + director-caption conditioned effects
windows at 44.1 kHz. Mirrors the ACE-Step compat shape (initialize /
render / evict behind lazy imports) so the worker stays a thin
transport and the GPU ban (§12) holds — `torch`/`mmaudio`/`open_clip`
only load inside functions, never at module scope.

Window geometry follows the upstream demo via kijai's loader math:
CLIP branch 8 fps @ 384 px, sync branch 25 fps @ 224 px normalized to
[-1, 1]. The sync branch needs >= 16 frames (16-frame segments, stride
8) or `encode_video_with_sync` stacks an empty list — windows shorter
than ~0.7 s are padded by whole-batch tiling, same discipline as the
music path's pad-to-17.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

SFX_MODEL_SIZES = ("small_44k", "medium_44k", "large_44k_v2")
"""Ladder vocabulary: every renderable MMAudio 44 kHz variant.

small_44k (157 M params, 601 MB) is the 2060 candidate; large_44k_v2
(1.03 B, 3.9 GB, upstream-recommended) is the 4060 default. 16 kHz
variants are excluded — the pipeline is 44.1/48 kHz end to end.
"""

SfxModelSize = Literal["small_44k", "medium_44k", "large_44k_v2"]
"""Type-level ladder vocabulary (typo fails at typecheck, not on GPU)."""

SFX_WEIGHT_FILES: dict[str, str] = {
    "small_44k": "mmaudio_small_44k.pth",
    "medium_44k": "mmaudio_medium_44k.pth",
    "large_44k_v2": "mmaudio_large_44k_v2.pth",
}
"""Variant → weights filename under <models>/mmaudio/weights/."""

SFX_VAE_FILE = "v1-44.pth"
SFX_SYNCHFORMER_FILE = "synchformer_state_dict.pth"
"""Shared 44 kHz VAE (1.2 GB) + synchformer (907 MB) under ext_weights/."""

SFX_CLIP_CHECKPOINT = "open_clip_pytorch_model.bin"
"""DFN5B CLIP weights under <models>/mmaudio/clip/ (arch config comes
from open_clip's builtin `ViT-H-14-378-quickgelu` entry — no download)."""

SFX_VOCODER_DIRNAME = "bigvgan_v2_44khz_128band_512x"
"""nvidia 44 kHz vocoder snapshot dir under <models>/mmaudio/vocoder/."""

CLIP_FPS = 8.0
CLIP_SIZE = 384
SYNC_FPS = 25.0
SYNC_SIZE = 224
"""Conditioning geometry (upstream demo via the ComfyUI loader math)."""

MAX_WINDOW_SECONDS = 60.0
"""Render ceiling: windows are 8 s native; the bound only rejects
caller bugs (a 5-minute single render) before any GPU side effect."""

FLOW_STEPS = 25
FLOW_CFG_STRENGTH = 4.5
"""Euler flow-matching schedule (upstream demo default, ComfyUI parity)."""


def validate_model_size(model_size: str) -> None:
    """Reject unknown MMAudio variants before any load attempt."""
    if model_size not in SFX_MODEL_SIZES:
        known = ", ".join(SFX_MODEL_SIZES)
        raise ValueError(f"model_size must be one of {known} (got {model_size!r})")


def validate_duration_seconds(duration_seconds: float) -> None:
    """Reject non-positive/non-finite/absurd window lengths pre-render."""
    if (
        not math.isfinite(duration_seconds)
        or duration_seconds <= 0.0
        or duration_seconds > MAX_WINDOW_SECONDS
    ):
        raise ValueError(
            f"duration_seconds must be finite within (0, {MAX_WINDOW_SECONDS}] "
            f"(got {duration_seconds})"
        )


T = TypeVar("T")
"""Element type for the window-list helpers (frame tensors, names in tests)."""

MIN_SYNC_FRAMES = 16
"""Synchformer floor: fewer sync frames segments an empty list (issue 045).

Windows shorter than ~0.64 s yield fewer; those pad by whole-batch tiling
below instead of failing loud — only a zero-frame window is unservable.
"""

MAX_STACKED_BYTES = 2 * 1024**3
"""Ceiling on one window's float32 conditioning tensors (issue 045).

`MAX_WINDOW_SECONDS` bounds seconds, but seconds stop binding bytes the
moment geometry changes — this bounds what `torch.stack` actually
allocates (sync float32 @224² + clip float32 @384²). 2 GiB clears the
60 s worst case (~1.7 GiB) with headroom while failing fast on absurd
geometries before any GPU side effect.
"""


def window_frame_counts(duration_seconds: float) -> tuple[int, int]:
    """Exact (clip, sync) conditioning counts for a window (issue 045).

    Pure int math over the branch geometry — the single place the
    `int(rate × duration)` formula lives, so the worker extract and the
    render truncation can never disagree.
    """
    validate_duration_seconds(duration_seconds)
    return (int(CLIP_FPS * duration_seconds), int(SYNC_FPS * duration_seconds))


def effective_stacked_bytes(duration_seconds: float) -> int:
    """Float32 bytes one window's conditioning stacks occupy (issue 045)."""
    clip_count, sync_count = window_frame_counts(duration_seconds)
    return sync_count * 3 * SYNC_SIZE * SYNC_SIZE * 4 + clip_count * 3 * CLIP_SIZE * CLIP_SIZE * 4


def check_stacked_bytes(duration_seconds: float) -> int:
    """Fail fast when a window's conditioning would exceed the byte budget."""
    total = effective_stacked_bytes(duration_seconds)
    if total > MAX_STACKED_BYTES:
        raise ValueError(
            f"sfx window stacked bytes {total} exceed budget {MAX_STACKED_BYTES} "
            f"for {duration_seconds:.2f}s "
            f"(sync {SYNC_SIZE}px@{SYNC_FPS:g}fps + clip {CLIP_SIZE}px@{CLIP_FPS:g}fps)"
        )
    return total


def tile_to_length(items: list[T], length: int) -> list[T]:
    """Repeat a batch whole until `length`, then truncate (issue 045).

    Whole-batch tiling preserves temporal order (unlike per-frame
    repeat); an empty source tiles nothing, so padding from zero frames
    fails loud instead of rendering silence.
    """
    if length < 0:
        raise ValueError(f"tile length must be >= 0 (got {length})")
    if not items:
        if length == 0:
            return []
        raise ValueError(f"cannot tile an empty window to {length} frames")
    tiled: list[T] = []
    while len(tiled) < length:
        tiled.extend(items)
    return tiled[:length]


def pad_to_sync_floor(
    clip_frames: list[T], sync_frames: list[T], duration_seconds: float
) -> tuple[list[T], list[T], float]:
    """Pad a short-tail window to the 16-frame sync floor (issue 045).

    Returns `(clip, sync, effective_duration)`: healthy windows pass
    through untouched; a servable tail (≥1 sync frame below the floor)
    tiles both branches to a duration-consistent 0.64 s window
    (16 sync frames, 5 clip frames — the counts `window_frame_counts`
    implies for 16/25 s), so the sequence lengths the model asserts
    stay exact and the caller trims the rendered take to its window.
    Zero sync frames is genuinely unservable and fails loud.
    """
    validate_duration_seconds(duration_seconds)
    if not sync_frames or not clip_frames:
        raise ValueError(
            f"sfx window unservable: need ≥1 clip + ≥1 sync frame "
            f"(got {len(clip_frames)} + {len(sync_frames)})"
        )
    expected_clip, expected_sync = window_frame_counts(duration_seconds)
    healthy_clip = max(expected_clip, int(MIN_SYNC_FRAMES * CLIP_FPS / SYNC_FPS))
    if len(sync_frames) >= max(expected_sync, MIN_SYNC_FRAMES) and len(clip_frames) >= healthy_clip:
        return (clip_frames, sync_frames, duration_seconds)
    if len(sync_frames) >= MIN_SYNC_FRAMES:
        # At/above the synchformer floor but short of the request: servable
        # as-is — the caller (render truncation vs fail-loud) decides.
        return (clip_frames, sync_frames, duration_seconds)
    effective = MIN_SYNC_FRAMES / SYNC_FPS
    padded_clip_count, padded_sync_count = window_frame_counts(effective)
    return (
        tile_to_length(clip_frames, padded_clip_count),
        tile_to_length(sync_frames, padded_sync_count),
        effective,
    )


@dataclass
class SfxStack:
    """Resident MMAudio stack: transformer + feature utils + scheduler."""

    model: Any
    feature_utils: Any
    flow_matching: Any
    model_size: str
    device: str


def _stack_paths(models_dir: Path, model_size: str) -> dict[str, Path]:
    """Resolve every weight file; fail closed with a named missing file."""
    base = Path(models_dir) / "mmaudio"
    paths = {
        "weights": base / "weights" / SFX_WEIGHT_FILES[model_size],
        "vae": base / "ext_weights" / SFX_VAE_FILE,
        "synchformer": base / "ext_weights" / SFX_SYNCHFORMER_FILE,
        "clip": base / "clip" / SFX_CLIP_CHECKPOINT,
        "vocoder": base / "vocoder" / SFX_VOCODER_DIRNAME,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"MMAudio {model_size} stack incomplete under {base} "
            f"(missing: {', '.join(missing)}; run `voyage models download sfx-mmaudio`)"
        )
    return paths


def _load_feature_utils(
    vae_ckpt: str,
    synchformer_ckpt: str,
    clip_checkpoint: str,
    vocoder_dir: Path,
    dtype: Any,
) -> Any:
    """Conditioning stack from registry-pinned files (no hub round-trip).

    The pinned constructors hardcode two hub ids — the nvidia 44 kHz
    vocoder (`AutoEncoderModule`) and the DFN5B CLIP tower
    (`FeaturesUtils`) — so both call sites are briefly redirected at
    the pinned snapshot files (HubMixin `from_pretrained` and
    open_clip's `create_model` both accept local paths; restored in
    `finally`). The transformer itself loads one level up with explicit
    fp16 placement; everything here constructs fp32 CPU and casts once.
    """
    import mmaudio.ext.autoencoder.autoencoder as autoencoder_module
    import mmaudio.model.utils.features_utils as features_module

    vocoder_cls = autoencoder_module.BigVGANv2
    original_from_pretrained = vocoder_cls.from_pretrained

    @classmethod  # type: ignore[misc]
    def _pinned_vocoder(cls: Any, name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(name_or_path) == "nvidia/bigvgan_v2_44khz_128band_512x":
            name_or_path = str(vocoder_dir)
        return original_from_pretrained(name_or_path, *args, **kwargs)

    original_create = features_module.create_model_from_pretrained

    def _pinned_clip(name: Any, *args: Any, **kwargs: Any) -> Any:
        if str(name).startswith("hf-hub:apple/DFN5B"):
            import open_clip

            return open_clip.create_model("ViT-H-14-378-quickgelu", pretrained=clip_checkpoint)
        return original_create(name, *args, **kwargs)

    vocoder_cls.from_pretrained = _pinned_vocoder
    features_module.create_model_from_pretrained = _pinned_clip
    try:
        feature_utils = features_module.FeaturesUtils(
            tod_vae_ckpt=vae_ckpt,
            synchformer_ckpt=synchformer_ckpt,
            enable_conditions=True,
            mode="44k",
        ).eval()
    finally:
        vocoder_cls.from_pretrained = original_from_pretrained
        features_module.create_model_from_pretrained = original_create
    return feature_utils.to(dtype=dtype)


def initialize(models_dir: str, device: str, model_size: str) -> SfxStack:
    """Load the resident stack (plain torch.load — no meta-device tricks).

    fp32 on CPU-able construction, then `.to(device, fp16)`: fp16 halves
    the 4060 residency (large ≈ 6 GB per the upstream note) and matches
    the ComfyUI loader precision. Weights stay untouched on disk.
    """
    validate_model_size(model_size)
    import mmaudio.model.networks as networks
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.sequence_config import CONFIG_44K

    resolved = _stack_paths(Path(models_dir), model_size)
    dtype = torch.float16
    factory = {
        "small_44k": networks.small_44k,
        "medium_44k": networks.medium_44k,
        "large_44k_v2": networks.large_44k_v2,
    }[model_size]
    weights = torch.load(str(resolved["weights"]), map_location="cpu", weights_only=True)
    with init_empty_weights():
        model = factory()
    for name, _ in model.named_parameters():
        set_module_tensor_to_device(model, name, device="cpu", dtype=dtype, value=weights[name])
    del weights
    model = model.eval()
    model.seq_cfg = CONFIG_44K
    feature_utils = _load_feature_utils(
        vae_ckpt=str(resolved["vae"]),
        synchformer_ckpt=str(resolved["synchformer"]),
        clip_checkpoint=str(resolved["clip"]),
        vocoder_dir=resolved["vocoder"],
        dtype=dtype,
    )
    flow_matching = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=FLOW_STEPS)
    return SfxStack(
        model=model,
        feature_utils=feature_utils,
        flow_matching=flow_matching,
        model_size=model_size,
        device=device,
    )


def render_window(
    stack: SfxStack,
    caption: str,
    negative_caption: str,
    clip_frames: Any,
    sync_frames: Any,
    duration_seconds: float,
    seed: int,
    save_path: Path,
) -> Path:
    """Render one caption-conditioned effects window to 44.1 kHz FLAC.

    `clip_frames` (T,3,384,384 float32 0..1) and `sync_frames`
    (T,3,224,224 normalized) come from the caller (worker: ffmpeg
    extracts; tests: synthetic tensors). Duration resolves from the
    sync count so the file matches reality; the model sequence lengths
    update per window (same call the ComfyUI sampler makes).

    Short tails pad to the 16-frame sync floor by whole-batch tiling
    (issue 045) instead of failing loud — the padded window renders at
    its effective 0.64 s duration and the caller trims to its own
    window. Only a zero-frame window is unservable and fails loud.
    """
    validate_duration_seconds(duration_seconds)
    check_stacked_bytes(duration_seconds)
    import torch
    from mmaudio.eval_utils import generate

    torch_device = torch.device(stack.device)
    # Exact counts (canonical loader behavior): the duration-derived
    # sequence lengths assert equality downstream (synchformer emits
    # S segments × 8, CLIP passes frames through). Below the sync
    # floor, tile to a duration-consistent 0.64 s window instead of
    # truncating into the empty-stack crash; anything else short of
    # the request fails loud.
    expected_clip, expected_sync = window_frame_counts(duration_seconds)
    clip_list = list(clip_frames)[:expected_clip]
    sync_list = list(sync_frames)[:expected_sync]
    if len(clip_list) < expected_clip or len(sync_list) < expected_sync:
        if 0 < len(sync_list) < MIN_SYNC_FRAMES and clip_list:
            clip_list, sync_list, duration_seconds = pad_to_sync_floor(
                clip_list, sync_list, duration_seconds
            )
        else:
            raise ValueError(
                f"sfx window starved: need {expected_clip} clip + {expected_sync} sync frames "
                f"for {duration_seconds:.2f}s, got {len(clip_list)} + {len(sync_list)}"
            )
    sync_batch = torch.stack(sync_list).unsqueeze(0)
    clip_batch = torch.stack(clip_list).unsqueeze(0)
    stack.model.seq_cfg.duration = duration_seconds
    stack.model.update_seq_lengths(
        stack.model.seq_cfg.latent_seq_len,
        stack.model.seq_cfg.clip_seq_len,
        stack.model.seq_cfg.sync_seq_len,
    )
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    stack.feature_utils.to(torch_device)
    stack.model.to(torch_device)
    with torch.no_grad():
        audios = generate(
            clip_batch.to(torch_device),
            sync_batch.to(torch_device),
            [caption or "ambient sound effects"],
            negative_text=[negative_caption or ""],
            feature_utils=stack.feature_utils,
            net=stack.model,
            fm=stack.flow_matching,
            rng=generator,
            cfg_strength=FLOW_CFG_STRENGTH,
        )
    import soundfile

    save_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = audios.float().cpu()
    soundfile.write(str(save_path), waveform[0].T.numpy(), 44100)
    return save_path


def evict(stack: SfxStack) -> None:
    """Drop the resident stack so music/video can reclaim the GPU.

    Issue 045: no `.to("cpu")` round-trip first — that copies the whole
    multi-GB stack device-to-host just to free it. Dropping the refs
    plus `gc.collect()` releases the Python side; `empty_cache` returns
    the caching-allocator blocks (the `del`-alone-frees-nothing lesson
    is about reference cycles, which the collect covers).
    """
    import gc

    import torch

    del stack.model
    del stack.feature_utils
    del stack.flow_matching
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
