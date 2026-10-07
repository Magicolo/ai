"""Video worker: LTXV 2B-distilled alternative backend (Phase 7, LTXV-only).

Native ltx-video library path (NOT diffusers — the voyage-video image has
no LTX pipelines): LTXVInference building blocks called directly so the
T5 text encoder stays on CPU in bf16. The stock infer() path OOMs moving
the fp32 T5-XXL to GPU, and the pipeline moves a resident text encoder to
GPU unconditionally even with precomputed embeds — so this worker passes
text_encoder=None with CPU-precomputed bf16 embeds (Slice 1 probe).

Continuity model (Stream A, DESIGN §5.3): stateless segment extension, not
a persistent KV stream. Every segment renders one 121-frame clip; when a
25-frame conditioning tail exists (and the block is not a scene cut) the
clip is conditioned on that tail video at start frame 0, the 25-frame
prefix is discarded, and only the ~96 novel frames are committed. The tail
(`video_tail.mp4`, last 25 committed frames) is derived on demand at
resume from the sibling segment video — `generate_blocks` no longer
persists it — so the supervisor's resume/rebuild flow works unchanged.
Recovery tape ``recovery.pt`` carries the §5.3 JSON record (kept
filename for supervisor discovery; JSON content — old torch-pickle tapes
are unresumable by design).

Precision: bf16 first; on CUDA OOM the DiT is quantized in place with
torchao dynamic fp8 (W8-only eager dequant OOMs on sm89, dynamic W8A8
does not) and the block retried once.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from voyage.hashing import sha256_file as shared_sha256_file
from voyage.model_registry import LTXV_COMMIT, LTXV_HF_REVISION, LTXV_TE_REPO, LTXV_TE_REVISION
from voyage.workers import video_common
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts
from voyage.workers.video_common import TAIL_FILENAME as TAIL_FILENAME
from voyage.workers.video_common import TAPE_FILENAME as TAPE_FILENAME
from voyage.workers.video_ltxv_validators import SPATIAL_GRANULARITY as SPATIAL_GRANULARITY
from voyage.workers.video_ltxv_validators import padded_size as padded_size
from voyage.workers.video_ltxv_validators import (
    validate_conditioning_start as validate_conditioning_start,
)
from voyage.workers.video_ltxv_validators import validate_fps as validate_fps
from voyage.workers.video_ltxv_validators import validate_frame_count as validate_frame_count
from voyage.workers.video_ltxv_validators import validate_spatial_size as validate_spatial_size

DIT_FILENAME = "ltxv-2b-0.9.8-distilled.safetensors"
UPSC_FILENAME = "ltxv-spatial-upscaler-0.9.8.safetensors"
LTXV_SUBDIR = "ltxv-2b"
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
RECOVERY_PROFILE = "ltxv"

# Stream A segment accounting (DESIGN §5.3): 121 = 8*15+1 and 25 = 8*3+1
# satisfy the upstream (F-1)%8==0 contract; 121-25 = 96 novel frames (4.0 s
# at 24 fps). Baked constants, not tunables — the 81/97/121 benchmark matrix
# is deferred follow-up (TASK §30.2).
SEGMENT_TARGET_FRAMES = 121
CONDITIONING_TAIL_FRAMES = 25
COMMITTED_NOVEL_FRAMES = SEGMENT_TARGET_FRAMES - CONDITIONING_TAIL_FRAMES
# Upstream spatial granularity: width/height divisible by 32 (one-stage);
# two-stage multiscale wants 64. 768x512 passes both; 768x432 passes neither
# (pads to 768x448) — hence the native 768x512 preset (§5.3 verdict).
# SPATIAL_GRANULARITY + padded_size/validate_* moved to video_ltxv_validators
# (issue 036); facades above keep `video_ltxv.<name>` importers working.
SPATIAL_GRANULARITY_TWO_STAGE = 64
STATE_MODE = "reconstructable_prefix"

# Probe-verified distilled schedules (Slice 1): cfg 1, STG off, layer 42
# skipped via attention-values strategy. Baked constants, not tunables.
FIRST_PASS: dict[str, Any] = {
    "timesteps": [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725],
    "guidance_scale": 1,
    "stg_scale": 0,
    "rescaling_scale": 1,
    "skip_block_list": [42],
}
SECOND_PASS: dict[str, Any] = {
    "timesteps": [0.9094, 0.725, 0.4219],
    "guidance_scale": 1,
    "stg_scale": 0,
    "rescaling_scale": 1,
    "skip_block_list": [42],
}
DOWNSCALE_FACTOR = 0.6666666
IMAGE_COND_NOISE_SCALE = 0.15

MILLISECONDS_PER_SECOND = 1000.0
"""`perf_counter` seconds → wall milliseconds for `stage_ms` telemetry."""

STAGE_MILLISECONDS_KEYS: tuple[str, ...] = ("encode_ms", "denoise_ms", "save_ms", "tape_ms")
"""Keys of the `stage_ms` mapping in every `generate_blocks` result (Stage A)."""


# Moved to voyage.workers.video_ltxv_validators (issue 036):
# padded_size, validate_spatial_size, validate_frame_count, validate_fps,
# validate_conditioning_start — facades above re-export them verbatim.


def is_oom(failure: BaseException) -> bool:
    """True when `failure` is a CUDA out-of-memory (issue 049).

    Torch-free (no torch import — the slim gates image has none): matches
    the OOM exception class by name (`torch.cuda.OutOfMemoryError` and the
    `torch.OutOfMemoryError` alias both end there) plus the allocator's
    `out of memory` message text, which is how OOMs surface when upstream
    code re-raises them as plain RuntimeError (cuBLAS/cuDNN alloc sites).
    Mirrors `voyage.audio.acestep.is_oom` and the `augment_worker`
    string-match idiom; kept local so this fix touches only this worker
    (a shared `video_common` predicate is a future merge, cf. issue 052).
    """
    if type(failure).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(failure).lower()


def split_prefix_novel(generated_frames: int, conditioning_frames: int) -> tuple[int, int]:
    """Return (prefix_discarded, novel_committed) for one extension clip.

    Fresh clips (conditioning 0) commit everything; conditioned clips drop
    the prefix and commit the remainder. Pure accounting — the caller must
    measure the real tensor shape and never assume it matches.
    """
    if conditioning_frames < 0 or conditioning_frames > generated_frames:
        raise ValueError(f"conditioning {conditioning_frames} out of range [0, {generated_frames}]")
    return (conditioning_frames, generated_frames - conditioning_frames)


def validate_tail_length(actual_frames: int, expected_frames: int) -> int:
    """Fail loud when a chained tail is short (issue 064, slim-testable).

    Extracted from the inline tail assert in `generate_blocks` so the
    negative-slice guard is unit-covered without a GPU session: a short
    `novel` used to slice short silently and degrade the next block's
    anchor. Returns `actual_frames` for call-site fluency.
    """
    if actual_frames != expected_frames:
        raise ValueError(
            f"LTXV tail has {actual_frames} frames, expects "
            f"{expected_frames} — refusing a short anchor"
        )
    return actual_frames


def tail_frames_for_conditioning(frame_array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Shape-gate the in-memory tensor handoff (issue 028, slim-testable).

    Upstream `prepare_conditioning` takes the tail as a (T, H, W, C) uint8
    array as well as a media path; conditioning chained blocks on these
    frames directly skips the lossy mp4 encode/decode roundtrip on the
    continuity-critical tail. Duck-typed on the array (worker image passes
    torch-derived numpy, slim tests pass numpy directly) — only shape and
    dtype are asserted, never the tensor type.
    """
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(
            f"LTXV conditioning frames must be (T, H, W, 3) (got shape {frame_array.shape})"
        )
    if frame_array.shape[0] < 1:
        raise ValueError("LTXV conditioning frames must hold at least one frame")
    if frame_array.dtype != np.uint8:
        raise ValueError(f"LTXV conditioning frames must be uint8 (got {frame_array.dtype})")
    return frame_array


def _tail_clip_to_handoff_frames(tail_clip: Any) -> NDArray[np.uint8] | None:
    """Bottle a block tail for the next block's in-memory handoff (issue 028).

    Layout mirrors `_save_mp4` (channel-first tail → (T, H, W, C) + the
    issue-065 clip + alpha drop), then the (T, H, W, C) uint8 shape gate.
    The result is a CPU numpy copy, so the caller can free the GPU tail
    while the handoff survives. Returns None when the tail cannot be
    bottled (unexpected tensor API — e.g. stubbed sessions in CPU tests):
    the next block then falls back to the chain mp4 path, which is today's
    behavior, and the miss is noted on stderr so a layout drift degrades
    to the status quo instead of silently corrupting the anchor.
    """
    try:
        tail_video = tail_clip[0]
        if tail_video.dim() == 4 and tail_video.shape[0] <= 4:
            tail_numpy = tail_video.permute(1, 2, 3, 0).float().cpu().numpy()
        else:
            tail_numpy = tail_video.permute(0, 2, 3, 1).float().cpu().numpy()
        tail_clipped = video_common.clip_array_to_uint8(tail_numpy)
        if tail_clipped.shape[-1] == 4:
            tail_clipped = tail_clipped[..., :3]
        return tail_frames_for_conditioning(tail_clipped)
    except Exception as exc:  # noqa: BLE001 — fallback is today's mp4 path
        print(f"ltxv tensor handoff unavailable ({exc}); using chain mp4", file=sys.stderr)
        return None


def prompt_plan_hash(prompts: list[str]) -> str:
    """Deterministic hash of the segment's prompt sequence (recovery tape)."""
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def generation_profile_hash(
    width: int, height: int, fps: int, segment_target: int, conditioning_tail: int
) -> str:
    """Hash the reproducible generation profile (recovery tape)."""
    profile_text = (
        f"ltxv|{width}x{height}@{fps}|target={segment_target}"
        f"|tail={conditioning_tail}|dit={DIT_FILENAME}"
        f"|first={FIRST_PASS['timesteps']}|second={SECOND_PASS['timesteps']}"
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
    session keep reporting prompt changes truthfully.

    The tail hash rides along only when the caller hashed a file for it:
    run-file pruning records the would-be tail path with no hash (nothing
    is persisted to hash), and resume fills it in after deriving.
    """
    tape: dict[str, Any] = {
        "backend": RECOVERY_PROFILE,
        "state_mode": STATE_MODE,
        "source_segment_id": source_segment_id,
        "conditioning_tail_path": conditioning_tail_path,
        "prompt_plan_hash": prompt_plan_digest or prompt_plan_hash(prompts),
        "seed": seeds[0] if seeds else 0,
        "seeds": list(seeds),
        "last_prompt": prompts[-1] if prompts else "",
        "model_revision": LTXV_HF_REVISION,
        "pipeline_revision": LTXV_COMMIT,
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
    """Validate a §5.3 JSON tape; reject old torch-pickle tapes loudly.

    Clean break (Stream A): tapes without ``backend``/``state_mode`` are the
    pre-§5.3 ``{"profile": "ltxv", "tail_png": ...}`` format and are
    unresumable — the caller must re-render from seed.

    A missing tail *file* is not a parse error (run-file pruning): the
    tape records the would-be path, and resume derives it from the
    sibling segment video. Only a missing tail *path* fails here.
    """
    if not isinstance(tape, dict):
        raise ValueError("LTXV recovery tape must be a JSON object")
    if tape.get("backend") != RECOVERY_PROFILE or tape.get("state_mode") != STATE_MODE:
        raise ValueError(
            "LTXV recovery tape is the pre-Stream-A torch format "
            "(profile/tail_png) — unresumable by design; re-render from seed"
        )
    tail_path = tape.get("conditioning_tail_path")
    if not isinstance(tail_path, str) or not tail_path:
        raise ValueError("LTXV recovery tape has no conditioning tail path")
    return tape


def _resolve_te_source(models_dir: Path) -> str:
    """Map the PixArt TE repo to its single /models snapshot, fetching when absent.

    Mirrors ``workers/director._resolve_model_source`` (073): a present
    snapshot resolves to ``<models_dir>/<LTXV_TE_SUBDIR>``; a missing one is
    fetched into /models (the HF_HUB_OFFLINE guard is lifted for that fetch —
    it protects the ephemeral cache, not the persistent volume) and
    re-checked. Download failure raises so init fails loudly instead of
    serving hub-drifted weights. An id outside the registry keeps hub
    behavior (unreachable today — the TE repo is registered).
    """
    from voyage import model_registry  # lazy: attribute access stays monkeypatchable (§12)

    ref = model_registry.resolve_snapshot(LTXV_TE_REPO)
    if ref is None:
        return LTXV_TE_REPO
    if model_registry.snapshot_present(models_dir, ref):
        return str(models_dir / ref.relative_dir)
    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        model_registry.download_model(models_dir, ref.spec_name)
    finally:
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline
    if not model_registry.snapshot_present(models_dir, ref):
        raise RuntimeError(f"snapshot {ref.relative_dir} still incomplete after download")
    return str(models_dir / ref.relative_dir)


def _verify_stack_manifest(models_dir: Path) -> None:
    """Fail-closed load-time sha gate for the LTXV stack (issue 209).

    Verifies the DiT + upscaler against the download manifest before the
    transformer ever loads them — the per-file-dict sibling of the causvid
    single-sha call site. Runs first in ``__init__`` (before the torch /
    ltx-video imports) so a swapped file fails fast with stdlib only. The
    PixArt TE snapshot carries no per-file pins, so it stays
    presence-gated via ``_resolve_te_source`` like today.
    """
    from voyage.registry_ltx25 import verify_recorded_shas

    stack_dir = models_dir / LTXV_SUBDIR
    verify_recorded_shas(
        models_dir,
        "ltxv",
        {
            f"{LTXV_SUBDIR}/{DIT_FILENAME}": stack_dir / DIT_FILENAME,
            f"{LTXV_SUBDIR}/{UPSC_FILENAME}": stack_dir / UPSC_FILENAME,
        },
    )


class LTXVSession:
    """Resident LTXV stack: bf16 DiT + VAE on CUDA, T5 on CPU, embed cache."""

    def __init__(self, models_dir: Path, device: str) -> None:
        _verify_stack_manifest(models_dir)
        import torch
        from ltx_video.inference import (  # type: ignore[import-not-found]
            create_latent_upsampler,
            create_transformer,
        )
        from ltx_video.models.autoencoders.causal_video_autoencoder import (  # type: ignore[import-not-found]
            CausalVideoAutoencoder,
        )
        from ltx_video.models.transformers.symmetric_patchifier import (  # type: ignore[import-not-found]
            SymmetricPatchifier,
        )
        from ltx_video.pipelines.pipeline_ltx_video import (  # type: ignore[import-not-found]
            LTXMultiScalePipeline,
            LTXVideoPipeline,
        )
        from ltx_video.schedulers.rf import RectifiedFlowScheduler  # type: ignore[import-not-found]
        from transformers import T5EncoderModel, T5Tokenizer

        dit_path = models_dir / LTXV_SUBDIR / DIT_FILENAME
        upsc_path = models_dir / LTXV_SUBDIR / UPSC_FILENAME
        for needed in (dit_path, upsc_path):
            if not needed.exists():
                raise FileNotFoundError(
                    f"missing LTXV weight file {needed} — run `voyage models download` first"
                )
        torch.set_grad_enabled(False)
        print("loading LTXV transformer (bf16) ...", file=sys.stderr)
        transformer = create_transformer(str(dit_path), "bfloat16").to(device)
        print("loading LTXV VAE (bf16) ...", file=sys.stderr)
        vae = CausalVideoAutoencoder.from_pretrained(str(dit_path)).to(device, dtype=torch.bfloat16)
        scheduler = RectifiedFlowScheduler.from_pretrained(str(dit_path))
        print("loading T5 text encoder (CPU, bf16) ...", file=sys.stderr)
        # Pinned-snapshot load (073): the tokenizer/encoder come from the
        # registry's PixArt snapshot at the pinned revision, never from a
        # bare hub id — `local_files_only` keeps every session init
        # offline-first, and `revision` (supported by the pinned
        # transformers 4.57.6, probe-verified) documents the pin for any
        # hub-shaped input.
        te_source = _resolve_te_source(models_dir)
        tokenizer = T5Tokenizer.from_pretrained(
            te_source,
            subfolder="tokenizer",
            local_files_only=True,
            revision=LTXV_TE_REVISION,
        )
        text_encoder = T5EncoderModel.from_pretrained(
            te_source,
            subfolder="text_encoder",
            local_files_only=True,
            revision=LTXV_TE_REVISION,
        ).to(torch.bfloat16)
        # text_encoder=None: embeds are precomputed on CPU (see module
        # docstring). Positional order mirrors the probe — the pipeline
        # takes (tokenizer, text_encoder, vae, transformer, scheduler,
        # patchifier, ...).
        pipeline = LTXVideoPipeline(
            tokenizer,
            None,
            vae,
            transformer,
            scheduler,
            SymmetricPatchifier(patch_size=1),
            None,
            None,
            None,
            None,
        )
        upsampler = create_latent_upsampler(str(upsc_path), device)
        self._torch = torch
        self._device = device
        self._transformer = transformer
        self._pipeline = pipeline
        self._multiscale = LTXMultiScalePipeline(pipeline, latent_upsampler=upsampler)
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder
        # Issue 030: bounded LRU (was an unbounded dict pinning
        # device-side masks forever); entries are stored CPU-side and
        # moved to `self._device` on use (never a hardcoded "cuda").
        self._embed_cache = video_common.EmbedCache()
        self._negative: tuple[Any, Any] | None = None
        self._conditioning_tail_path: str | None = None
        self._last_prompt: str | None = None
        self._fp8_fallback = False
        # Stage A telemetry (DESIGN §22.5): per-block encode/denoise split
        # from the latest `_generate_block` call; `generate_blocks` sums
        # them across blocks into the result's `stage_ms` mapping.
        self._last_block_encode_ms = 0.0
        self._last_block_denoise_ms = 0.0

    def _encode(self, text: str) -> tuple[Any, Any]:
        """CPU T5 encode (~25 s, cached per prompt); mask moved to the device on use.

        Issue 030: the entry is stored CPU-side (host RAM, not resident
        VRAM) under a bounded LRU, and both tensors move to
        `self._device` on every use — including hits, so a cached mask
        never stays pinned to a stale device.
        """
        cached = self._embed_cache.get(text)
        if cached is not None:
            embeds, mask = cached
            return (embeds.to(self._device), mask.to(self._device))
        torch = self._torch
        inputs = self._tokenizer(
            text,
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            embeds = self._text_encoder(inputs.input_ids)[0].to(torch.bfloat16)
        # Masks ride into device cross-attention; the pipeline only moves
        # embeds, so the mask must be moved by the caller (probe lesson).
        # Issue 074: the session device, not a hardcoded "cuda" (breaks cuda:1).
        self._embed_cache.put(text, (embeds, inputs.attention_mask))
        return (embeds.to(self._device), inputs.attention_mask.to(self._device))

    def _quantize_fp8_fallback(self) -> None:
        """In-place torchao dynamic-fp8 DiT quant (OOM fallback, once)."""
        from torchao.quantization.quant_api import (  # type: ignore[import-not-found]
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )

        print("quantizing LTXV transformer to dynamic FP8 ...", file=sys.stderr)
        quantize_(self._transformer, Float8DynamicActivationFloat8WeightConfig())
        self._fp8_fallback = True

    def _generate_block(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        frames: int,
        fps: int,
        conditioning_media: str | NDArray[np.uint8] | None,
    ) -> Any:
        """One 121-frame extension clip; returns the (B,C,T,H,W) tensor.

        ``conditioning_media`` is the 25-frame tail — either the tail video
        path or the same tail as an in-memory (T, H, W, C) uint8 array
        (issue 028 tensor handoff: skips the lossy mp4 roundtrip on the
        continuity-critical tail), or None for a fresh text-to-video start.
        It is conditioned at start frame 0 per the upstream extension
        contract. The caller discards the prefix. The union keeps one
        positional slot, so stubbed sessions keep working unchanged.
        """
        from ltx_video.inference import calculate_padding, prepare_conditioning

        conditioning_source: str | NDArray[np.uint8] | None = conditioning_media
        if conditioning_source is not None and not isinstance(conditioning_source, str):
            conditioning_source = tail_frames_for_conditioning(conditioning_source)
        torch = self._torch
        # Stage A telemetry: TE time vs denoise time, reported per call on
        # `self` so `generate_blocks` can sum the split across blocks.
        encode_elapsed_ms = 0.0
        denoise_elapsed_ms = 0.0
        if self._negative is None:
            encode_started = time.perf_counter()
            self._negative = self._encode(NEGATIVE_PROMPT)
            encode_elapsed_ms += (time.perf_counter() - encode_started) * MILLISECONDS_PER_SECOND
        encode_started = time.perf_counter()
        pos_embeds, pos_mask = self._encode(prompt)
        encode_elapsed_ms += (time.perf_counter() - encode_started) * MILLISECONDS_PER_SECOND
        neg_embeds, neg_mask = self._negative
        height_p = padded_size(height)
        width_p = padded_size(width)
        frames_p = padded_size(frames - 1, 8) + 1
        padding = calculate_padding(height, width, height_p, width_p)
        conditioning = (
            prepare_conditioning(
                [conditioning_source],
                [1.0],
                [0],
                height,
                width,
                frames_p,
                padding,
                self._pipeline,
            )
            if conditioning_source is not None
            else None
        )
        generator = torch.Generator(device=self._device).manual_seed(seed)
        try:
            denoise_started = time.perf_counter()
            block_images = self._run_multiscale(
                pos_embeds,
                pos_mask,
                neg_embeds,
                neg_mask,
                conditioning,
                generator,
                height_p,
                width_p,
                frames_p,
                fps,
                frames,
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            # Broad catch + string predicate (issue 049): CUDA OOMs often
            # surface as plain RuntimeError ("CUDA out of memory ...") from
            # cuBLAS/cuDNN alloc sites, which the old torch-only except let
            # straight through to a full session rebuild. Non-OOM
            # RuntimeErrors re-raise untouched.
            if not is_oom(exc):
                raise
            if self._fp8_fallback:
                raise
            # gc before empty_cache (documented order):
            # without it the cache release frees ~0 bytes under cycles.
            gc.collect()
            torch.cuda.empty_cache()
            detail = "cuda unavailable"
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated_gib = torch.cuda.memory_allocated() / 1024**3
                reserved_gib = torch.cuda.memory_reserved() / 1024**3
                detail = f"allocated={allocated_gib:.2f} GiB reserved={reserved_gib:.2f} GiB"
            print(
                f"LTXV OOM ({exc}); cleared cache ({detail}), "
                "quantizing to dynamic fp8 and retrying once ...",
                file=sys.stderr,
            )
            self._quantize_fp8_fallback()
            generator.manual_seed(seed)
            denoise_started = time.perf_counter()
            retried_images = self._run_multiscale(
                pos_embeds,
                pos_mask,
                neg_embeds,
                neg_mask,
                conditioning,
                generator,
                height_p,
                width_p,
                frames_p,
                fps,
                frames,
            )
            denoise_elapsed_ms += (time.perf_counter() - denoise_started) * MILLISECONDS_PER_SECOND
            self._last_block_encode_ms = encode_elapsed_ms
            self._last_block_denoise_ms = denoise_elapsed_ms
            return retried_images
        else:
            denoise_elapsed_ms += (time.perf_counter() - denoise_started) * MILLISECONDS_PER_SECOND
            self._last_block_encode_ms = encode_elapsed_ms
            self._last_block_denoise_ms = denoise_elapsed_ms
            return block_images

    def _run_multiscale(
        self,
        pos_embeds: Any,
        pos_mask: Any,
        neg_embeds: Any,
        neg_mask: Any,
        conditioning: Any,
        generator: Any,
        height_p: int,
        width_p: int,
        frames_p: int,
        fps: int,
        frames: int,
    ) -> Any:
        from ltx_video.inference import SkipLayerStrategy

        images = self._multiscale(
            downscale_factor=DOWNSCALE_FACTOR,
            first_pass=FIRST_PASS,
            second_pass=SECOND_PASS,
            skip_layer_strategy=SkipLayerStrategy.AttentionValues,
            generator=generator,
            output_type="pt",
            height=height_p,
            width=width_p,
            num_frames=frames_p,
            frame_rate=fps,
            prompt_embeds=pos_embeds,
            prompt_attention_mask=pos_mask,
            negative_prompt=None,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_attention_mask=neg_mask,
            conditioning_items=conditioning,
            is_video=True,
            vae_per_channel_normalize=True,
            image_cond_noise_scale=IMAGE_COND_NOISE_SCALE,
            mixed_precision=False,
            # No CPU offload: upstream overrides our `device` with
            # `self._execution_device` inside __call__, so once the
            # transformer is bounced to CPU between passes the second
            # pass prepares latents on CPU while holding our CUDA
            # generator -> flaky "Cannot generate a cpu tensor from a
            # generator of type cuda" (~40% of blocks, E2E log). The
            # resident stack fits in 16 GB without offload (benchmark
            # peak ~7 GB with it), so keep everything on CUDA.
            offload_to_cpu=False,
            device=self._device,
            enhance_prompt=False,
            decode_timestep=0.05,
            decode_noise_scale=0.025,
        ).images
        return images[:, :, :frames]

    def generate_blocks(
        self,
        prompts: list[str],
        seeds: list[int],
        scene_cuts: list[bool],
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        segment_id: str = "000000",
        prompt_plan_digest: str | None = None,
        requested_frames: int | None = None,
        segment_target_frames: int = SEGMENT_TARGET_FRAMES,
        conditioning_tail_frames: int = CONDITIONING_TAIL_FRAMES,
    ) -> dict[str, Any]:
        """Render one §5.3 segment: 121-frame clip(s), 25-frame prefix drop.

        Block 0 conditions on the resident tail video (previous segment)
        unless this is a fresh session, the tail file is gone, or the block
        is a scene cut; every later block conditions on the block before it
        via a temporary tail video. Fresh blocks commit all 121 frames;
        conditioned blocks discard the 25-frame prefix and commit 96 novel
        frames. Counts below are measured from the real tensors (§4.3 rule:
        never assume the pipeline returned exactly the request).

        Stage A telemetry (DESIGN §22.5): the result always carries
        `stage_ms` (`encode_ms` = TE `_encode` total, `denoise_ms` =
        `_run_multiscale` total, `save_ms` = `_save_mp4` total, `tape_ms` =
        recovery-tape write) — wall milliseconds via `perf_counter` only,
        so no gating flag is needed. Doubles that replace
        `_generate_block` without reporting the split contribute 0.0 to
        the encode/denoise totals (read via `getattr` defaults).
        """
        if not prompts or not (len(prompts) == len(seeds) == len(scene_cuts)):
            raise ValueError("prompts/seeds/scene_cuts must be non-empty equal-length lists")
        validate_spatial_size(width, height)
        validate_frame_count(segment_target_frames)
        validate_frame_count(conditioning_tail_frames)
        validate_conditioning_start(0, segment_target_frames)
        validate_fps(fps)

        torch = self._torch
        output_path.parent.mkdir(parents=True, exist_ok=True)
        novel_clips: list[Any] = []
        chain_tails: list[Path] = []
        pending_tail: Path | None = None
        pending_tail_frames: NDArray[np.uint8] | None = None
        generated_total = 0
        conditioning_total = 0
        encode_total_ms = 0.0
        denoise_total_ms = 0.0
        save_total_ms = 0.0
        resident_tail = self._conditioning_tail_path
        prompt_changed = self._last_prompt is not None and prompts[0] != self._last_prompt
        fresh_blocks = 0
        resume_fallback: dict[str, Any] | None = None
        try:
            for index, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
                conditioning_source: str | NDArray[np.uint8] | None
                if index == 0:
                    tail_candidate = resident_tail
                    if tail_candidate is None:
                        conditioning_source = None
                        fresh_blocks += 1
                        resume_fallback = {"reason": "fresh_session", "tail_path": None}
                    elif scene_cuts[0]:
                        conditioning_source = None
                        fresh_blocks += 1
                        resume_fallback = {"reason": "scene_cut", "tail_path": tail_candidate}
                    elif not Path(tail_candidate).exists():
                        conditioning_source = None
                        fresh_blocks += 1
                        resume_fallback = {
                            "reason": "missing_tail",
                            "tail_path": tail_candidate,
                        }
                        print(
                            f"ltxv resume tail {tail_candidate} missing — starting fresh",
                            file=sys.stderr,
                        )
                    else:
                        conditioning_source = tail_candidate
                elif pending_tail_frames is not None:
                    # Issue 028 tensor handoff: chain onto the previous
                    # block's in-memory tail frames — NOT the stale resident
                    # tail (which still points at the previous segment until
                    # this call commits below), and without the lossy mp4
                    # roundtrip. The chain mp4 below stays as the
                    # crash-recovery/fallback artifact.
                    conditioning_source = pending_tail_frames
                else:
                    if not chain_tails:
                        raise RuntimeError("LTXV block chain lost its tail video")
                    # Chain onto the previous block's freshly rendered tail video
                    # — NOT the stale resident tail (which still points at the
                    # previous segment until this call commits below).
                    conditioning_source = str(chain_tails[-1])
                block = self._generate_block(
                    prompt, seed, width, height, segment_target_frames, fps, conditioning_source
                )
                encode_total_ms += float(getattr(self, "_last_block_encode_ms", 0.0))
                denoise_total_ms += float(getattr(self, "_last_block_denoise_ms", 0.0))
                generated_frames = int(block.shape[2])
                generated_total += generated_frames
                if conditioning_source is None:
                    novel = block
                else:
                    conditioning_total += conditioning_tail_frames
                    _, novel_count = split_prefix_novel(generated_frames, conditioning_tail_frames)
                    novel = block[:, :, generated_frames - novel_count :, :, :]
                novel_clips.append(novel)
                # Temporary tail video for the next block in this call: last 25
                # committed frames. Chain files are deleted after the commit
                # (run-file pruning) — the tape records the would-be tail
                # path, and resume derives it from the segment video.
                tail_clip = novel[:, :, -conditioning_tail_frames:, :, :]
                # Issue 064: a short `novel` slices short (negative-slice
                # semantics) — committing that as a full 25-frame anchor
                # would silently degrade the next block, so fail loudly.
                validate_tail_length(int(tail_clip.shape[2]), conditioning_tail_frames)
                pending_tail = output_path.parent / f"{output_path.stem}_chain{index:02d}.mp4"
                save_started = time.perf_counter()
                _save_mp4(tail_clip, pending_tail, fps)
                save_total_ms += (time.perf_counter() - save_started) * MILLISECONDS_PER_SECOND
                chain_tails.append(pending_tail)
                pending_tail = None
                # Issue 028: bottle the tail for the next block's in-memory
                # handoff (None when the tail cannot be bottled — the next
                # block then falls back to the chain mp4, today's behavior).
                pending_tail_frames = _tail_clip_to_handoff_frames(tail_clip)
                del tail_clip
        except Exception:
            # Issue 064: the OOM-retry path must not leave `_chain*.mp4`
            # orphans beside the segment (they confuse recovery/tape
            # accounting) — unlink committed tails plus the in-flight one.
            if pending_tail is not None:
                pending_tail.unlink(missing_ok=True)
            for stale_tail in chain_tails:
                stale_tail.unlink(missing_ok=True)
            raise
        video = novel_clips[0] if len(novel_clips) == 1 else torch.cat(novel_clips, dim=2)
        committed_frames = int(video.shape[2])
        save_started = time.perf_counter()
        _save_mp4(video, output_path, fps)
        save_total_ms += (time.perf_counter() - save_started) * MILLISECONDS_PER_SECOND
        tail_path = output_path.parent / TAIL_FILENAME
        for stale in chain_tails:
            stale.unlink(missing_ok=True)
        tape = build_recovery_tape(
            source_segment_id=segment_id,
            conditioning_tail_path=str(tail_path),
            prompts=prompts,
            seeds=seeds,
            width=width,
            height=height,
            fps=fps,
            segment_target_frames=segment_target_frames,
            conditioning_tail_frames=conditioning_tail_frames,
            prompt_plan_digest=prompt_plan_digest,
        )
        tape_path = output_path.parent / TAPE_FILENAME
        tape_started = time.perf_counter()
        video_common.write_tape_atomic(tape_path, tape)
        tape_total_ms = (time.perf_counter() - tape_started) * MILLISECONDS_PER_SECOND
        self._conditioning_tail_path = str(tail_path)
        self._last_prompt = prompts[-1]
        prefix_discarded = generated_total - committed_frames
        del novel_clips, video
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
            "conditioning_tail_frames": conditioning_tail_frames,
            "conditioning_start_frame": 0,
            "native_fps": fps,
            "prompt_changed": prompt_changed,
            "fresh_blocks": fresh_blocks,
            "resume_fallback": resume_fallback,
            "conditioning_tail_path": str(tail_path),
            "recovery_path": str(tape_path),
            "fp8_fallback": self._fp8_fallback,
            "stage_ms": {
                "encode_ms": encode_total_ms,
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
            print(f"ltxv derived missing tail from the segment video: {tail_path}", file=sys.stderr)
            if "conditioning_tail_sha256" in parsed:
                parsed["conditioning_tail_sha256"] = sha256_file(outcome.path)
        self._conditioning_tail_path = tail_path
        last_prompt = parsed.get("last_prompt")
        self._last_prompt = str(last_prompt) if isinstance(last_prompt, str) else None
        return {"resumed": True, "conditioning_tail_path": tail_path}

    def evict(self) -> None:
        """Unload the stack so audio can own the GPU (§40 pattern)."""
        torch = self._torch
        self._embed_cache.clear()
        self._negative = None
        self._conditioning_tail_path = None
        self._last_prompt = None
        del self._multiscale, self._pipeline, self._transformer
        del self._tokenizer, self._text_encoder
        gc.collect()
        torch.cuda.empty_cache()


def _save_mp4(images: Any, path: Path, fps: int) -> None:
    """Write a (B,C,T,H,W) tensor as h264 (probe-verified layout).

    Tensor-to-frames conversion stays here (layout is backend-specific);
    the clip + mimsave mechanics live in :mod:`video_common` (issue 019).
    """
    video = images[0]
    if video.dim() == 4 and video.shape[0] <= 4:
        frames = video.permute(1, 2, 3, 0).float().cpu().numpy()
    else:
        frames = video.permute(0, 2, 3, 1).float().cpu().numpy()
    # Issue 065: clip before uint8 — VAE overshoot outside [0, 1] wraps
    # modulo 256 without it (1.01 becomes near-black; cf. causvid).
    clipped = video_common.clip_array_to_uint8(frames)
    if clipped.shape[-1] == 4:
        clipped = clipped[..., :3]
    video_common.save_mp4([clipped[index] for index in range(clipped.shape[0])], path, fps)


_SESSION: LTXVSession | None = None
_INIT_PARAMS: dict[str, Any] = {}


def _build_session() -> LTXVSession:
    models_dir = _INIT_PARAMS["models_dir"]
    assert isinstance(models_dir, Path)
    return LTXVSession(models_dir, str(_INIT_PARAMS["device"]))


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    global _SESSION
    import torch

    checked_request(payload, models_dir=str)
    models_dir = Path(str(payload["models_dir"]))
    device = str(payload.get("device", "cuda:0"))
    if not device.startswith("cuda"):
        raise RuntimeError(f"video_ltxv requires a CUDA device (got {device!r})")
    if not torch.cuda.is_available():
        raise RuntimeError("video_ltxv requires a CUDA GPU")
    started = time.monotonic()
    _INIT_PARAMS.update({"models_dir": models_dir, "device": device})
    _SESSION = _build_session()
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
    # Issue 064: reject a bad frame rate before the session check so the
    # boundary fails fast (and stays CPU-testable without a GPU session).
    validate_fps(int(payload["fps"]))
    if _SESSION is None:
        raise RuntimeError("video_ltxv not initialized — send `init` first")
    # One validated struct (issue 045): payload forms + shape checks live
    # in GenerateBlocksRequest.from_payload — no inline asserts.
    request = video_common.GenerateBlocksRequest.from_payload(
        payload, width_default=768, height_default=512
    )
    output = request.output_path
    if request.width is None or request.height is None:
        raise ValueError("video_ltxv requires width/height geometry (got native defaults)")
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
    """Time warmup + measured single-segment probes with VRAM peaks (§104).

    Probes are fresh text-to-video renders (scene cut, no resident tail), so
    this does NOT advance any stream — safe to run on a live
    session between segments (still prefer scratch). Requires `init` first.
    Each probe commits a full 121-frame fresh segment (no prefix to drop).
    """
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    validate_benchmark_counts(warmup, measured)
    if _SESSION is None:
        raise RuntimeError("video_ltxv not initialized — send `init` first")
    import torch

    session = _SESSION
    saved_tail = session._conditioning_tail_path
    saved_prompt = session._last_prompt
    session._conditioning_tail_path = None
    session._last_prompt = None
    benchmark_prompt = str(payload.get("prompt", "benchmark probe"))
    benchmark_seed = int(payload.get("seed", 0))
    benchmark_width = int(payload.get("width", 768))
    benchmark_height = int(payload.get("height", 512))
    benchmark_fps = int(payload.get("fps", 24))
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
            "voyage-ltxv-bench-",
            probe,
            reset_peak_memory=torch.cuda.reset_peak_memory_stats,
            read_peak_gib=lambda: torch.cuda.max_memory_allocated() / 1024**3,
        )
    finally:
        session._conditioning_tail_path = saved_tail
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
    """Read a §5.3 JSON tape; old torch tapes fail with a clean-break error."""
    try:
        with open(recovery_path, encoding="utf-8") as handle:
            loaded: Any = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "LTXV recovery tape is the pre-Stream-A torch format "
            "(profile/tail_png) — unresumable by design; re-render from seed"
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError("LTXV recovery tape must be a JSON object")
    return loaded


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    """Adopt the latest committed tail video after a restart (§27.1)."""
    if _SESSION is None:
        raise RuntimeError("video_ltxv not initialized — send `init` first")

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
        raise RuntimeError("video_ltxv rebuilt before init")
    if _SESSION is not None:
        _SESSION.evict()
        _SESSION = None
    _SESSION = _build_session()
    tape = _load_tape_json(str(payload["recovery_path"]))
    return {"rebuilt": True, **_SESSION.resume_from_tape(tape)}


def main() -> None:
    serve(
        video_common.standard_serve_map(
            "ltxv",
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
