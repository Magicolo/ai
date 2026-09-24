"""Video worker: LTXV 2B-distilled alternative backend (Phase 7, LTXV-only).

Native ltx-video library path (NOT diffusers — the voyage-video image has
no LTX pipelines): LTXVInference building blocks called directly so the
T5 text encoder stays on CPU in bf16. The stock infer() path OOMs moving
the fp32 T5-XXL to GPU, and the pipeline moves a resident text encoder to
GPU unconditionally even with precomputed embeds — so this worker passes
text_encoder=None with CPU-precomputed bf16 embeds (Slice 1 probe).

Continuity model: LTXV has no persistent KV stream, so cross-block and
cross-segment continuity comes from tail-frame conditioning — every block
after the first is conditioned on the previous block's tail frame PNG at
output frame 0 (scene_cut forces a fresh text-to-video start instead).
The tail PNG doubles as the crash-recovery tape beside the segment video,
so the supervisor's resume/rebuild flow works unchanged (recovery.pt
carrying {"profile": "ltxv", "tail_png": ...}).

Precision: bf16 first; on CUDA OOM the DiT is quantized in place with
torchao dynamic fp8 (the slice-4-verified longlive recipe — W8-only eager
dequant OOMs on sm89, dynamic W8A8 does not) and the block retried once.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any

from voyage.workers.loop import checked_request, serve

DIT_FILENAME = "ltxv-2b-0.9.8-distilled.safetensors"
UPSC_FILENAME = "ltxv-spatial-upscaler-0.9.8.safetensors"
LTXV_SUBDIR = "ltxv-2b"
TE_REPO_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
RECOVERY_PROFILE = "ltxv"

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
NATIVE_BLOCK_FRAMES = 25


def padded_size(value: int, multiple: int = 32) -> int:
    """Round `value` up to a multiple (LTXV pads latents to /32)."""
    return ((value - 1) // multiple + 1) * multiple


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
        self._tail_png: str | None = None
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
        conditioning_png: str | None,
    ) -> Any:
        """One native block; returns the (B,C,T,H,W) frame tensor trimmed."""
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
                [conditioning_png],
                [1.0],
                [0],
                height,
                width,
                frames_p,
                padding,
                self._pipeline,
            )
            if conditioning_png is not None
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
        frames: int = NATIVE_BLOCK_FRAMES,
    ) -> dict[str, Any]:
        """Render blocks, chaining each onto the previous tail frame.

        Block 0 conditions on the resident tail (previous segment) unless
        this is a fresh session or the block is a scene cut; every later
        block conditions on the block before it. Frame 0 of blocks > 0
        duplicates the tail it was conditioned on, so it is dropped before
        concatenation. The new tail PNG + recovery.pt land beside the video.
        """
        import imageio.v2 as imageio  # type: ignore[import-not-found]

        torch = self._torch
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered: list[Any] = []
        chain_pngs: list[Path] = []
        conditioning = None if (self._tail_png is None or scene_cuts[0]) else self._tail_png
        for index, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
            if index > 0:
                if not chain_pngs:
                    raise RuntimeError("LTXV block chain lost its tail frame")
                # Chain onto the previous block's freshly rendered tail —
                # NOT the stale resident tail (which still points at the
                # previous segment until this call commits below). Without
                # this every block after the first re-extends the old tail
                # and the block boundary jumps ~6x the baseline (measured
                # 0.114 vs 0.02 on a blocks=2 E2E segment).
                conditioning = str(chain_pngs[-1])
            block = self._generate_block(prompt, seed, width, height, frames, fps, conditioning)
            tail_frame = block[0, :, -1, :, :].permute(1, 2, 0).float().cpu().numpy() * 255
            chain_png = output_path.parent / f"{output_path.stem}_chain{index:02d}.png"
            imageio.imwrite(str(chain_png), tail_frame.astype("uint8"))
            chain_pngs.append(chain_png)
            del tail_frame
            rendered.append(block if index == 0 else block[:, :, 1:, :, :])
        video = rendered[0] if len(rendered) == 1 else torch.cat(rendered, dim=2)
        total_frames = int(video.shape[2])
        _save_mp4(video, output_path, fps)
        tail_path = output_path.parent / f"{output_path.stem}_tail.png"
        chain_pngs[-1].replace(tail_path)
        for stale in chain_pngs[:-1]:
            stale.unlink(missing_ok=True)
        self._tail_png = str(tail_path)
        tape_path = output_path.parent / "recovery.pt"
        torch.save({"profile": RECOVERY_PROFILE, "tail_png": str(tail_path)}, tape_path)
        del rendered, video
        return {
            "frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "recovery_path": str(tape_path),
            "fp8_fallback": self._fp8_fallback,
        }

    def resume_from_tape(self, tape: dict[str, Any]) -> dict[str, Any]:
        """Adopt a previous tail PNG as the conditioning anchor."""
        tail_png = tape.get("tail_png")
        if not isinstance(tail_png, str) or not Path(tail_png).exists():
            raise ValueError("LTXV recovery tape has no usable tail frame")
        self._tail_png = tail_png
        return {"resumed": True, "tail_png": tail_png}

    def evict(self) -> None:
        """Unload the stack so audio can own the GPU (§40 pattern)."""
        torch = self._torch
        self._embed_cache.clear()
        self._negative = None
        self._tail_png = None
        del self._multiscale, self._pipeline, self._transformer
        del self._tokenizer, self._text_encoder
        gc.collect()
        torch.cuda.empty_cache()


def _save_mp4(images: Any, path: Path, fps: int) -> None:
    """Write a (B,C,T,H,W) tensor as h264 (probe-verified layout)."""
    import imageio.v2 as imageio

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
    result = _SESSION.generate_blocks(
        prompts=prompts,
        seeds=seeds,
        scene_cuts=scene_cuts,
        output_path=output,
        width=int(payload.get("width", 768)),
        height=int(payload.get("height", 512)),
        fps=int(payload["fps"]),
    )
    artifacts = [str(output), str(result["recovery_path"])]
    return {
        "blocks_generated": len(prompts),
        "artifacts": artifacts,
        "video": result,
    }


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured single-block probes with VRAM peaks (§104).

    Probes are fresh text-to-video renders that never touch the resident
    tail, so unlike longlive this does NOT advance any stream — safe to
    run on a live session between segments (still prefer scratch).
    Requires `init` first.
    """
    if _SESSION is None:
        raise RuntimeError("video_ltxv not initialized — send `init` first")
    import tempfile

    import torch

    session = _SESSION
    saved_tail = session._tail_png
    session._tail_png = None
    try:
        warmup = int(payload.get("warmup", 1))
        measured = int(payload.get("measured", 3))
        walls: list[float] = []
        peaks: list[float] = []
        frames = 0
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
                )
                elapsed = time.monotonic() - started
                if index >= warmup:
                    walls.append(elapsed)
                    peaks.append(torch.cuda.max_memory_allocated() / 1024**3)
                    frames = int(result["frames"])
    finally:
        session._tail_png = saved_tail
    mean = sum(walls) / len(walls)
    return {
        "backend": RECOVERY_PROFILE,
        "warmup_blocks": warmup,
        "measured_blocks": measured,
        "frames_per_block": frames,
        "block_wall_seconds": [round(wall, 3) for wall in walls],
        "blocks_per_second": round(1 / mean, 3),
        "fps_equivalent": round(frames / mean, 1),
        "vram_peak_gib": round(max(peaks), 2),
        "vram_avg_gib": round(sum(peaks) / len(peaks), 2),
    }


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    """Adopt the latest committed tail frame after a restart (§27.1)."""
    if _SESSION is None:
        raise RuntimeError("video_ltxv not initialized — send `init` first")
    import torch

    checked_request(payload, recovery_path=str)
    with open(str(payload["recovery_path"]), "rb") as handle:
        tape = torch.load(handle, map_location="cpu", weights_only=False)
    if not isinstance(tape, dict) or tape.get("profile") != RECOVERY_PROFILE:
        raise ValueError("recovery tape profile mismatch")
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
    import torch

    checked_request(payload, recovery_path=str)
    if not _INIT_PARAMS:
        raise RuntimeError("video_ltxv rebuilt before init")
    _SESSION = _build_session()
    with open(str(payload["recovery_path"]), "rb") as handle:
        tape = torch.load(handle, map_location="cpu", weights_only=False)
    if not isinstance(tape, dict) or tape.get("profile") != RECOVERY_PROFILE:
        raise ValueError("recovery tape profile mismatch")
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
