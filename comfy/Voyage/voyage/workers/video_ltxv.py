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
file ``video_tail.mp4`` (last 25 committed frames) beside the segment video
is the crash-recovery anchor, so the supervisor's resume/rebuild flow works
unchanged. Recovery tape ``recovery.pt`` carries the §5.3 JSON record (kept
filename for supervisor discovery; JSON content — old torch-pickle tapes
are unresumable by design).

Precision: bf16 first; on CUDA OOM the DiT is quantized in place with
torchao dynamic fp8 (the slice-4-verified longlive recipe — W8-only eager
dequant OOMs on sm89, dynamic W8A8 does not) and the block retried once.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from voyage.model_registry import LTXV_COMMIT, LTXV_HF_REVISION
from voyage.workers.loop import checked_request, serve

DIT_FILENAME = "ltxv-2b-0.9.8-distilled.safetensors"
UPSC_FILENAME = "ltxv-spatial-upscaler-0.9.8.safetensors"
LTXV_SUBDIR = "ltxv-2b"
TE_REPO_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
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
SPATIAL_GRANULARITY = 32
SPATIAL_GRANULARITY_TWO_STAGE = 64
STATE_MODE = "reconstructable_prefix"
TAIL_FILENAME = "video_tail.mp4"
TAPE_FILENAME = "recovery.pt"

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


def validate_conditioning_start(start_frame: int, target_frames: int) -> None:
    """Enforce the upstream multiple-of-8 target-frame rule for extensions."""
    if start_frame < 0 or start_frame >= target_frames:
        raise ValueError(
            f"LTXV conditioning start {start_frame} out of range [0, {target_frames - 1}]"
        )
    if start_frame % 8 != 0:
        raise ValueError(f"LTXV conditioning start {start_frame} must be a multiple of 8")


def split_prefix_novel(generated_frames: int, conditioning_frames: int) -> tuple[int, int]:
    """Return (prefix_discarded, novel_committed) for one extension clip.

    Fresh clips (conditioning 0) commit everything; conditioned clips drop
    the prefix and commit the remainder. Pure accounting — the caller must
    measure the real tensor shape and never assume it matches.
    """
    if conditioning_frames < 0 or conditioning_frames > generated_frames:
        raise ValueError(f"conditioning {conditioning_frames} out of range [0, {generated_frames}]")
    return (conditioning_frames, generated_frames - conditioning_frames)


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
    """Chunked SHA-256 (constant memory — tails are small, videos are not)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_recovery_tape(
    *,
    source_segment_id: str,
    conditioning_tail_path: str,
    conditioning_tail_sha256: str,
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
    """
    return {
        "backend": RECOVERY_PROFILE,
        "state_mode": STATE_MODE,
        "source_segment_id": source_segment_id,
        "conditioning_tail_path": conditioning_tail_path,
        "conditioning_tail_sha256": conditioning_tail_sha256,
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


def parse_recovery_tape(tape: dict[str, Any]) -> dict[str, Any]:
    """Validate a §5.3 JSON tape; reject old torch-pickle tapes loudly.

    Clean break (Stream A): tapes without ``backend``/``state_mode`` are the
    pre-§5.3 ``{"profile": "ltxv", "tail_png": ...}`` format and are
    unresumable — the caller must re-render from seed.
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
    if not Path(tail_path).exists():
        raise ValueError(f"LTXV conditioning tail missing: {tail_path}")
    return tape


class LTXVSession:
    """Resident LTXV stack: bf16 DiT + VAE on CUDA, T5 on CPU, embed cache."""

    def __init__(self, models_dir: Path, device: str) -> None:
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
        tokenizer = T5Tokenizer.from_pretrained(TE_REPO_ID, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(TE_REPO_ID, subfolder="text_encoder").to(
            torch.bfloat16
        )
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
        self._embed_cache: dict[str, tuple[Any, Any]] = {}
        self._negative: tuple[Any, Any] | None = None
        self._conditioning_tail_path: str | None = None
        self._last_prompt: str | None = None
        self._fp8_fallback = False

    def _encode(self, text: str) -> tuple[Any, Any]:
        """CPU T5 encode (~25 s, cached per prompt); mask moved to CUDA."""
        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached
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
        # Masks ride into CUDA cross-attention; the pipeline only moves
        # embeds, so the mask must be moved here (probe lesson).
        result = (embeds, inputs.attention_mask.to("cuda"))
        self._embed_cache[text] = result
        return result

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
        conditioning_media_path: str | None,
    ) -> Any:
        """One 121-frame extension clip; returns the (B,C,T,H,W) tensor.

        ``conditioning_media_path`` is the 25-frame tail video (or None for a
        fresh text-to-video start); it is conditioned at start frame 0 per
        the upstream extension contract. The caller discards the prefix.
        """
        from ltx_video.inference import calculate_padding, prepare_conditioning

        torch = self._torch
        if self._negative is None:
            self._negative = self._encode(NEGATIVE_PROMPT)
        pos_embeds, pos_mask = self._encode(prompt)
        neg_embeds, neg_mask = self._negative
        height_p = padded_size(height)
        width_p = padded_size(width)
        frames_p = padded_size(frames - 1, 8) + 1
        padding = calculate_padding(height, width, height_p, width_p)
        conditioning = (
            prepare_conditioning(
                [conditioning_media_path],
                [1.0],
                [0],
                height,
                width,
                frames_p,
                padding,
                self._pipeline,
            )
            if conditioning_media_path is not None
            else None
        )
        generator = torch.Generator(device=self._device).manual_seed(seed)
        try:
            return self._run_multiscale(
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
        except torch.OutOfMemoryError:
            if self._fp8_fallback:
                raise
            torch.cuda.empty_cache()
            self._quantize_fp8_fallback()
            generator.manual_seed(seed)
            return self._run_multiscale(
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
        """
        if not prompts or not (len(prompts) == len(seeds) == len(scene_cuts)):
            raise ValueError("prompts/seeds/scene_cuts must be non-empty equal-length lists")
        validate_spatial_size(width, height)
        validate_frame_count(segment_target_frames)
        validate_frame_count(conditioning_tail_frames)
        validate_conditioning_start(0, segment_target_frames)

        torch = self._torch
        output_path.parent.mkdir(parents=True, exist_ok=True)
        novel_clips: list[Any] = []
        chain_tails: list[Path] = []
        generated_total = 0
        conditioning_total = 0
        resident_tail = self._conditioning_tail_path
        prompt_changed = self._last_prompt is not None and prompts[0] != self._last_prompt
        for index, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
            if index == 0:
                tail_candidate = resident_tail
                if tail_candidate is None or scene_cuts[0] or not Path(tail_candidate).exists():
                    conditioning: str | None = None
                else:
                    conditioning = tail_candidate
            else:
                if not chain_tails:
                    raise RuntimeError("LTXV block chain lost its tail video")
                # Chain onto the previous block's freshly rendered tail video
                # — NOT the stale resident tail (which still points at the
                # previous segment until this call commits below).
                conditioning = str(chain_tails[-1])
            block = self._generate_block(
                prompt, seed, width, height, segment_target_frames, fps, conditioning
            )
            generated_frames = int(block.shape[2])
            generated_total += generated_frames
            if conditioning is None:
                novel = block
            else:
                conditioning_total += conditioning_tail_frames
                _, novel_count = split_prefix_novel(generated_frames, conditioning_tail_frames)
                novel = block[:, :, generated_frames - novel_count :, :, :]
            novel_clips.append(novel)
            # Temporary tail video for the next block in this call: last 25
            # committed frames. The final block's tail becomes video_tail.mp4.
            tail_clip = novel[:, :, -conditioning_tail_frames:, :, :]
            chain_tail = output_path.parent / f"{output_path.stem}_chain{index:02d}.mp4"
            _save_mp4(tail_clip, chain_tail, fps)
            chain_tails.append(chain_tail)
            del tail_clip
        video = novel_clips[0] if len(novel_clips) == 1 else torch.cat(novel_clips, dim=2)
        committed_frames = int(video.shape[2])
        _save_mp4(video, output_path, fps)
        tail_path = output_path.parent / TAIL_FILENAME
        if len(chain_tails) == 1:
            chain_tails[0].replace(tail_path)
        else:
            chain_tails[-1].replace(tail_path)
            for stale in chain_tails[:-1]:
                stale.unlink(missing_ok=True)
        tail_checksum = sha256_file(tail_path)
        tape = build_recovery_tape(
            source_segment_id=segment_id,
            conditioning_tail_path=str(tail_path),
            conditioning_tail_sha256=tail_checksum,
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
        tape_text = json.dumps(tape, indent=2, sort_keys=True) + "\n"
        tape_tmp = tape_path.with_suffix(".tmp")
        tape_tmp.write_text(tape_text, encoding="utf-8")
        tape_tmp.replace(tape_path)
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
            "conditioning_tail_path": str(tail_path),
            "conditioning_tail_sha256": tail_checksum,
            "recovery_path": str(tape_path),
            "fp8_fallback": self._fp8_fallback,
        }

    def resume_from_tape(self, tape: dict[str, Any]) -> dict[str, Any]:
        """Adopt the previous segment's tail video as the conditioning anchor."""
        parsed = parse_recovery_tape(tape)
        tail_path = str(parsed["conditioning_tail_path"])
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
    """Write a (B,C,T,H,W) tensor as h264 (probe-verified layout)."""
    import imageio.v2 as imageio  # type: ignore[import-not-found]

    video = images[0]
    if video.dim() == 4 and video.shape[0] <= 4:
        frames = video.permute(1, 2, 3, 0).float().cpu().numpy()
    else:
        frames = video.permute(0, 2, 3, 1).float().cpu().numpy()
    frames = (frames * 255).astype("uint8")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    imageio.mimsave(str(path), list(frames), fps=fps, codec="libx264")


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
    if _SESSION is None:
        raise RuntimeError("video_ltxv not initialized — send `init` first")
    if "prompts" in payload or "seeds" in payload:
        checked_request(payload, segment_id=str, output_path=str, fps=int)
        raw_prompts = payload["prompts"]
        raw_seeds = payload["seeds"]
        assert isinstance(raw_prompts, list) and isinstance(raw_seeds, list)
        prompts = [str(item) for item in raw_prompts]
        seeds = [int(item) for item in raw_seeds]
        raw_cuts = payload.get("scene_cuts", [False] * len(prompts))
        assert isinstance(raw_cuts, list) and len(raw_cuts) == len(prompts)
        scene_cuts = [bool(item) for item in raw_cuts]
    else:
        checked_request(payload, segment_id=str, prompt=str, seed=int, output_path=str, fps=int)
        prompts = [str(payload["prompt"])]
        seeds = [int(payload["seed"])]
        scene_cuts = [bool(payload.get("scene_cut", False))]
    output = Path(str(payload["output_path"]))
    segment_id = str(payload["segment_id"])
    requested = payload.get("frames")
    prompt_digest_raw = payload.get("prompt_plan_hash")
    prompt_digest = str(prompt_digest_raw) if isinstance(prompt_digest_raw, str) else None
    result = _SESSION.generate_blocks(
        prompts=prompts,
        seeds=seeds,
        scene_cuts=scene_cuts,
        output_path=output,
        width=int(payload.get("width", 768)),
        height=int(payload.get("height", 512)),
        fps=int(payload["fps"]),
        segment_id=segment_id,
        prompt_plan_digest=prompt_digest,
        requested_frames=int(requested) if isinstance(requested, int) else None,
    )
    artifacts = [str(output), str(result["conditioning_tail_path"]), str(result["recovery_path"])]
    return {
        "blocks_generated": len(prompts),
        "artifacts": artifacts,
        "video": result,
    }


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured single-segment probes with VRAM peaks (§104).

    Probes are fresh text-to-video renders (scene cut, no resident tail), so
    unlike longlive this does NOT advance any stream — safe to run on a live
    session between segments (still prefer scratch). Requires `init` first.
    Each probe commits a full 121-frame fresh segment (no prefix to drop).
    """
    if _SESSION is None:
        raise RuntimeError("video_ltxv not initialized — send `init` first")
    import tempfile

    import torch

    session = _SESSION
    saved_tail = session._conditioning_tail_path
    saved_prompt = session._last_prompt
    session._conditioning_tail_path = None
    session._last_prompt = None
    try:
        warmup = int(payload.get("warmup", 1))
        measured = int(payload.get("measured", 3))
        walls: list[float] = []
        peaks: list[float] = []
        committed = 0
        generated = 0
        with tempfile.TemporaryDirectory(prefix="voyage-ltxv-bench-") as tmp:
            for index in range(warmup + measured):
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                result = session.generate_blocks(
                    prompts=[str(payload.get("prompt", "benchmark probe"))],
                    seeds=[int(payload.get("seed", 0))],
                    scene_cuts=[True],
                    output_path=Path(tmp) / f"b{index}.mp4",
                    width=int(payload.get("width", 768)),
                    height=int(payload.get("height", 512)),
                    fps=int(payload.get("fps", 24)),
                    segment_id="benchmark",
                )
                elapsed = time.monotonic() - started
                if index >= warmup:
                    walls.append(elapsed)
                    peaks.append(torch.cuda.max_memory_allocated() / 1024**3)
                    committed = int(result["committed_frames"])
                    generated = int(result["generated_frames"])
    finally:
        session._conditioning_tail_path = saved_tail
        session._last_prompt = saved_prompt
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
        {
            "init": handle_init,
            "health": handle_health,
            "generate_blocks": handle_generate_blocks,
            "benchmark": handle_benchmark,
            "evict_gpu": handle_evict_gpu,
            "rebuild": handle_rebuild,
            "checkpoint": lambda payload: {
                "checkpoint_id": f"ltxv-{payload.get('segment_id', 'none')}"
            },
            "resume": handle_resume,
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()
