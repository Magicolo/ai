"""LongLive 2.0 video worker: `python -m voyage.workers.video_longlive`.

Runs ONLY in the CUDA worker image (`worker/Dockerfile.video`): Python 3.10,
torch 2.8/cu128, LongLive@6b36d20, flash-attn 2 (mandatory — upstream
attention asserts it). Phase 1 scope: finite single-block segments, eager
mode, TorchAO FP8 PTQ; KV persistence across segments is Phase 2.

16 GB VRAM design (RTX 4060 Ti):
- Generator BF16 (~10 GB) is quantized in place to FP8 W8A8 before inference.
- The UMT5-XXL text encoder is replaced with a CPU/bf16 twin injected via
  the pipeline's `text_encoder=` seam (upstream's wrapper builds fp32 +
  auto-CUDA, which alone exceeds 16 GB). Only the 4 MB embeds move to GPU.
- VAE stays on GPU; its feat cache is cleared after every segment.

Typing note: GPU-only imports carry `# type: ignore[import-not-found]` on
their FIRST occurrence per module only — this mypy emits one missing-module
error per file, so later bare imports of the same module need no ignore
(and an extra ignore would trip `warn_unused_ignores`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

VOYAGE_LONGLIVE_DIR = Path(os.environ.get("VOYAGE_LONGLIVE_DIR", "/opt/longlive"))
VOYAGE_MODELS_DIR = Path(os.environ.get("VOYAGE_MODELS_DIR", "/models"))

if str(VOYAGE_LONGLIVE_DIR) not in sys.path:
    sys.path.insert(0, str(VOYAGE_LONGLIVE_DIR))

from voyage.workers.loop import checked_request, serve  # noqa: E402


class CpuUmt5Encoder:
    """Call-compatible twin of upstream WanTextEncoder, CPU/bf16 resident.

    Accepts the same `text_prompts=[...]` keyword call and returns
    `{"prompt_embeds": cuda_tensor}` so the pipeline never knows the
    difference. A ~4.4B-param encoder in fp32 on GPU is incompatible with
    16 GB cards; CPU/bf16 encode costs minutes per segment (Phase 1 smoke
    accepts this — GPU sequencing is later-phase optimization).
    """

    def __init__(self, wan_dir: Path, device: str) -> None:
        import torch  # type: ignore[import-not-found]
        from wan_5b.modules.t5 import umt5_xxl  # type: ignore[import-not-found]
        from wan_5b.modules.tokenizers import HuggingfaceTokenizer  # type: ignore[import-not-found]

        self._torch = torch
        self._device = torch.device(device)
        self._model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        ).eval()
        state = torch.load(
            wan_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            map_location="cpu",
            weights_only=False,
        )
        self._model.load_state_dict(state)
        self._tokenizer = HuggingfaceTokenizer(
            name=str(wan_dir / "google" / "umt5-xxl"),
            seq_len=512,
            clean="whitespace",
        )

    def __call__(self, text_prompts: list[str]) -> dict[str, Any]:
        torch = self._torch
        ids, mask = self._tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        seq_lens = mask.gt(0).sum(dim=1).long()
        with torch.inference_mode():
            context = self._model(ids, mask)
        for row, length in zip(context, seq_lens, strict=True):
            row[length:] = 0.0  # match upstream padding semantics
        return {"prompt_embeds": context.to(self._device)}


def _enter_longlive_tree(models_dir: Path) -> None:
    """Reproduce upstream's CWD contract: relative `wan_models/` lookups.

    Upstream resolves `wan_models/Wan2.2-TI2V-5B/` against the process CWD
    (repo root). The supervisor spawns workers with CWD=run_dir, so chdir
    into the LongLive tree and link its `wan_models/` at our models volume.
    All voyage paths are absolute, so the chdir is side-effect free.
    """
    link = VOYAGE_LONGLIVE_DIR / "wan_models"
    target = models_dir / "wan_models"
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        link.symlink_to(target)
    os.chdir(VOYAGE_LONGLIVE_DIR)


def build_longlive_config(generator_ckpt: Path, latent_shape: list[int]) -> Any:
    """Mirror configs/fp8/inference_fp8.yaml as an OmegaConf object.

    Factored out so memory probes and tests exercise the exact config the
    session builds. OmegaConf import stays function-level (worker image only).

    16 GB VRAM sizing (RTX 4060 Ti, DESIGN §140): the KV cache is
    `local_attn_size × frame_seq_length` bf16 tokens per layer, allocated for
    BOTH the conditional and (unused at guidance 1.0) unconditional branch —
    upstream never gates the neg half. At the upstream default
    local_attn_size=32 and 80×44 latents that cache alone is ~20 GB. Dropping
    the window to 8 brings it to ~5.2 GB (narrower temporal context, valid
    Phase 1 smoke tradeoff; revisit with KV eviction/quant in later phases).
    """
    from omegaconf import OmegaConf  # type: ignore[import-not-found]
    from utils.config import normalize_config  # type: ignore[import-not-found]

    raw_config = {
        "model_kwargs": {
            "model_name": "Wan2.2-TI2V-5B",
            "timestep_shift": 5.0,
            "num_frame_per_block": 8,
            "local_attn_size": 8,
        },
        "use_ema": False,
        "num_samples": 1,
        "num_output_frames": latent_shape[1],
        "image_or_video_shape": list(latent_shape),
        "sampling_steps": 4,
        "guidance_scale": 1.0,
        "inference": {
            "sampling_steps": 4,
            "sink_size": 8,
            "guidance_scale": 1.0,
            "multi_shot_sink": True,
            "multi_shot_rope_offset": 8,
            # NOTE: streaming_vae=true is unusable here — the pipeline's
            # streaming path needs VAE.cached_decode, which the bundled
            # WanVAE_ build lacks. Chunked decode happens in generate()
            # via decode_to_pixel_chunk instead (public wrapper API).
            "streaming_vae": False,
            "async_vae": False,
            "vae_type": "wan",
        },
        "checkpoints": {"generator_ckpt": str(generator_ckpt)},
        "fp8_quant": True,
    }
    return normalize_config(OmegaConf.create(raw_config))


def _install_pos_only_caches(pipe: Any) -> bool:
    """Replace the KV/crossattn initializers with pos-branch-only twins.

    Upstream `_initialize_kv_cache` / `_initialize_crossattn_cache`
    (pipeline/causal_diffusion_inference.py @6b36d20) ALWAYS allocate the
    negative (unconditional) branch — even at guidance_scale=1.0, where
    inference() sets unconditional_dict=None and never touches it. On a
    16 GB card that wasted half (~2.6 GB at local_attn 8) is the difference
    between OOM and a finished segment.

    This replicates ONLY the non-quantized path (we never set quantize_kv;
    NVFP4 is Blackwell-only) minus the neg lists, and applies ONLY when
    guidance is exactly 1.0 — otherwise the upstream methods stay in place.
    Returns True when installed. Revisit if the upstream pin moves.
    """
    import torch
    from utils.config import wan_default_config

    if float(getattr(pipe, "guidance_scale", 1.0)) != 1.0:
        return False
    if bool(getattr(pipe, "quantize_kv", False)):
        return False
    num_heads = wan_default_config[pipe.model_name]["num_heads"]
    head_dim = wan_default_config[pipe.model_name]["head_dim"]

    def _kv_size() -> tuple[int, int]:
        if pipe.local_attn_size != -1:
            size = pipe.local_attn_size * pipe.frame_seq_length
        else:
            size = 3 * pipe.num_frame_per_block * pipe.frame_seq_length
        block = pipe.num_frame_per_block * pipe.frame_seq_length
        return size, block

    def init_kv(bound_self: Any, batch_size: int, dtype: Any, device: Any) -> None:
        kv_size, block_size = _kv_size()
        pos = []
        for _ in range(bound_self.num_transformer_blocks):
            pos.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_size, num_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_size, num_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "quantized": False,
                    "block_token_size": block_size,
                    "max_blocks": kv_size // block_size,
                    "num_heads": num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        pipe.kv_cache_pos = pos
        pipe.kv_cache_neg = []  # never read at guidance 1.0 (unconditional None)

    def init_cross(bound_self: Any, batch_size: int, dtype: Any, device: Any) -> None:
        pos = []
        neg = []
        for _ in range(bound_self.num_transformer_blocks):
            pos.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, num_heads, head_dim], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, num_heads, head_dim], dtype=dtype, device=device
                    ),
                    "is_init": False,
                }
            )
            # Tensor-free placeholder: the per-chunk reset loop writes
            # crossattn_cache_neg[i]["is_init"] UNGUARDED (upstream line ~543),
            # but every READ is use_cfg-guarded — so this costs zero VRAM.
            neg.append({"is_init": False})
        pipe.crossattn_cache_pos = pos
        pipe.crossattn_cache_neg = neg

    pipe._initialize_kv_cache = init_kv.__get__(pipe)
    pipe._initialize_crossattn_cache = init_cross.__get__(pipe)
    return True


class LongLiveSession:
    """Resident pipeline: built once at `init`, reused per segment."""

    def __init__(self, models_dir: Path, device: str, latent_shape: list[int]) -> None:
        import torch
        from pipeline import CausalDiffusionInferencePipeline  # type: ignore[import-not-found]
        from utils.fp8 import quantize_model_fp8  # type: ignore[import-not-found]

        wan_dir = models_dir / "wan_models" / "Wan2.2-TI2V-5B"
        generator_ckpt = models_dir / "longlive2" / "model_bf16.pt"
        for needed in (
            wan_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            wan_dir / "Wan2.2_VAE.pth",
            generator_ckpt,
        ):
            if not needed.exists():
                raise FileNotFoundError(
                    f"missing weight file {needed} — run `voyage models download` first"
                )

        self._torch = torch
        self._device = torch.device(device)
        self._latent_shape = list(latent_shape)
        torch.set_grad_enabled(False)

        config = build_longlive_config(generator_ckpt, list(latent_shape))
        pipeline = CausalDiffusionInferencePipeline(
            config,
            device=self._device,
            text_encoder=CpuUmt5Encoder(wan_dir, device),
        )
        pos_only = _install_pos_only_caches(pipeline)
        print(f"pos-only KV caches: {pos_only}", file=sys.stderr)
        # Mirror inference.py: unwrap the checkpoint container (keys:
        # generator + export metadata), strict-load, bf16, in-place FP8.
        import utils.nvfp4_checkpoint as nvfp4_ckpt  # type: ignore[import-not-found]

        print("loading generator checkpoint ...", file=sys.stderr)
        generator_container = torch.load(str(generator_ckpt), map_location="cpu")
        generator_state = nvfp4_ckpt.unwrap_generator_state_dict(generator_container, use_ema=False)
        pipeline.generator.load_state_dict(generator_state, strict=True)
        del generator_container, generator_state
        pipeline = pipeline.to(dtype=torch.bfloat16)
        pipeline.generator.to(device=self._device)
        print("quantizing generator to FP8 ...", file=sys.stderr)
        quantize_model_fp8(pipeline.generator.model, verbose=True)
        pipeline.generator.model.eval().requires_grad_(False)
        # Eager mode for Phase 1 (torch_compile off — no warmup samples yet).
        pipeline.vae.to(device=self._device)
        self._pipeline = pipeline
        self._config = config

    def generate(self, prompt: str, seed: int, output_path: Path, fps: int) -> dict[str, Any]:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
        from einops import rearrange  # type: ignore[import-not-found]
        from utils.misc import set_seed  # type: ignore[import-not-found]

        torch = self._torch
        set_seed(seed)
        shape = self._latent_shape
        noise = torch.randn(
            [1, shape[1], shape[2], shape[3], shape[4]],
            device=self._device,
            dtype=torch.bfloat16,
        )
        with torch.inference_mode():
            latents = self._pipeline.inference(
                noise=noise, text_prompts=[[prompt]], return_latents=True
            )
        # Chunked VAE decode (public wrapper API): whole-segment
        # decode_to_pixel OOMs at 1280x704x29f on 16 GB, and even 2-latent
        # chunks exceed budget (~14.6 GB resident at decode). One latent
        # frame per chunk peaks at 8.7 GB end-to-end (measured). Each chunk
        # restarts the causal history, so a chunk carries no temporal
        # expansion (8 latents -> 8 frames, not 29) — verify visually; the
        # fallback is CPU decode (slow, full 29f causal).
        generated = self._pipeline.vae.decode_to_pixel_chunk(latents, use_cache=False, chunk_size=1)
        video = (255.0 * rearrange(generated, "b t c h w -> b t h w c").cpu()).to(torch.uint8)
        self._pipeline.vae.model.clear_cache()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames = [video[0, index].numpy() for index in range(video.shape[1])]
        with imageio.get_writer(
            str(output_path), fps=fps, codec="libx264", macro_block_size=None
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        height, width = int(video.shape[3]), int(video.shape[4])
        return {
            "frames": len(frames),
            "fps": fps,
            "width": width,
            "height": height,
        }


_SESSION: LongLiveSession | None = None


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    global _SESSION
    import torch

    checked_request(payload, models_dir=str, device=str, latent_shape=list)
    models_dir = Path(str(payload["models_dir"]))
    device = str(payload.get("device", "cuda:0"))
    if not torch.cuda.is_available():
        raise RuntimeError("video_longlive requires a CUDA GPU")
    raw_shape = payload["latent_shape"]
    assert isinstance(raw_shape, list)
    latent_shape = [int(v) for v in raw_shape]
    _enter_longlive_tree(models_dir)
    _SESSION = LongLiveSession(models_dir, device, latent_shape)
    name = torch.cuda.get_device_name(0)
    free_gib, total_gib = torch.cuda.mem_get_info()
    return {
        "status": "READY",
        "backend": "longlive2-bf16-fp8",
        "gpu": name,
        "vram_free_gib": round(free_gib / 1024**3, 1),
        "vram_total_gib": round(total_gib / 1024**3, 1),
    }


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    ready = _SESSION is not None
    info: dict[str, Any] = {"status": "READY" if ready else "IDLE"}
    if torch.cuda.is_available():
        free_gib, total_gib = torch.cuda.mem_get_info()
        info["vram_free_gib"] = round(free_gib / 1024**3, 1)
        info["vram_total_gib"] = round(total_gib / 1024**3, 1)
    return info


def handle_generate_blocks(payload: dict[str, Any]) -> dict[str, Any]:
    checked_request(payload, segment_id=str, prompt=str, seed=int, output_path=str, fps=int)
    if _SESSION is None:
        raise RuntimeError("video_longlive not initialized — send `init` first")
    output = Path(str(payload["output_path"]))
    result = _SESSION.generate(
        prompt=str(payload["prompt"]),
        seed=int(payload["seed"]),
        output_path=output,
        fps=int(payload["fps"]),
    )
    return {
        "blocks_generated": 1,
        "artifacts": [str(output)],
        "video": result,
    }


def main() -> None:
    serve(
        {
            "init": handle_init,
            "health": handle_health,
            "generate_blocks": handle_generate_blocks,
            "checkpoint": lambda payload: {
                "checkpoint_id": f"longlive-{payload.get('segment_id', 'none')}"
            },
            "resume": lambda payload: {
                "resumed": True,
                "checkpoint_id": payload.get("checkpoint_id"),
            },
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()
