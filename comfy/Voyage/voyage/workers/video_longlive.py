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

# Upstream scene-cut signal: prompt prefix detected by _is_scene_cut when
# multi_shot_sink is on (ours: true). Embeddings always use the bare prompt;
# only raw_prompts carry the prefix (conditioning vs boundary signal split).
SCENE_CUT_PREFIX = "The scene transitions. "


def apply_scene_cut_prefix(prompt: str, scene_cut: bool) -> str:
    """Prepend the cut prefix for boundary blocks (pure helper, slim-testable)."""
    if scene_cut and not prompt.startswith(SCENE_CUT_PREFIX):
        return SCENE_CUT_PREFIX + prompt
    return prompt


class CpuUmt5Encoder:
    """Call-compatible twin of upstream WanTextEncoder, CPU/bf16 resident.

    Accepts the same `text_prompts=[...]` keyword call and returns
    `{"prompt_embeds": cuda_tensor}` so the pipeline never knows the
    difference. A ~4.4B-param encoder in fp32 on GPU is incompatible with
    16 GB cards; CPU/bf16 encode costs minutes per segment (Phase 1 smoke
    accepts this — GPU sequencing is later-phase optimization).
    """

    def __init__(self, wan_dir: Path, device: str) -> None:
        import torch
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
        # Self-contained inference mode: callers (probes, resume paths) may
        # invoke outside torch.inference_mode, where the padding zeroing
        # below would raise (inplace update on an inference tensor).
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
        # Relative RoPE (DESIGN Phase 2 remainder): top-level config attr,
        # default False upstream. Applied to the dit model in LongLiveSession
        # (mirrors inference() per-call setup, which our direct
        # _inference_inner path bypasses). Compute-only — no VRAM impact.
        "use_relative_rope": True,
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


class LongLiveStreamSession:
    """Persistent causal stream (DESIGN §22): caches survive across blocks.

    Upstream `inference()` either allocates caches (first call) or RESETS
    their positions to zero (later calls) — repeated calls are separate
    streams, not one voyage. This session replicates the preamble
    (prompt encode + output buffer) once per block and calls the
    lower-level `_inference_inner` directly, so `global/local_end_index`
    advance monotonically and the rolling window (§24) evicts old entries:
    memory is flat by construction no matter how many blocks append.

    One block = `num_frame_per_block` latents (8). Same-prompt text
    embeddings are cached (§22.4 step 3) — a CPU T5-XXL forward costs
    minutes, so repeat encodes must not rerun per block.

    Noise is stream-level (§22.5): upstream draws ONE noise tensor per
    sequence and slices it per chunk, so a single generator drawn
    sequentially per block reproduces that exact trajectory. A fresh
    generator per block breaks it at every boundary (measured 6-9x frame
    diff jumps) — hence the persistent `self._noise_rng`, seeded by the
    first block's seed and carried across evict/rebuild via the
    recovery.pt tape (`noise_rng_state`).

    Buffers are sequence-level too: `_inference_inner` indexes its `noise`
    and `output` arguments by ABSOLUTE frame (`noise_start_frame =
    cache_start_frame - num_input_frames`, `output[:, cache_start: ...]`),
    so per-block-sized buffers read/write out of bounds past block 0
    (empty-slice crash on noise, silent zero latents on output).
    `begin_sequence` therefore draws the full-sequence noise and allocates
    the full output once per `generate_blocks` call; `append_block` works
    on the slice for its chunk. This mirrors one upstream `inference()`
    call over the same frames exactly.
    """

    def __init__(self, pipeline: Any, latent_shape: list[int], device: Any) -> None:
        self._pipeline = pipeline
        self._latent_shape = list(latent_shape)
        self._device = device
        self._next_start_frame = 0  # latent frames committed to the stream
        self._blocks_appended = 0
        self._embed_cache: dict[str, tuple[Any, Any]] = {}
        self._noise_rng: Any = None  # stream noise generator (seeded on first block)
        self._seq_noise: Any = None  # full-sequence noise for the in-flight call
        self._seq_output: Any = None  # full-sequence output buffer, same frames
        self._seq_offset = 0  # chunk offset (in blocks) into the in-flight buffers

    @property
    def blocks_appended(self) -> int:
        return self._blocks_appended

    @property
    def next_start_frame(self) -> int:
        return self._next_start_frame

    def reset(self) -> None:
        """Drop caches + position (fresh stream; full recover() lands later)."""
        pipe = self._pipeline
        pipe.kv_cache_pos = None
        pipe.kv_cache_neg = None
        pipe.crossattn_cache_pos = None
        pipe.crossattn_cache_neg = None
        self._next_start_frame = 0
        self._blocks_appended = 0
        self._noise_rng = None
        self._seq_noise = None
        self._seq_output = None
        self._seq_offset = 0

    def offload_caches(self) -> None:
        """Move KV/crossattn tensors to CPU (frees VRAM for VAE decode)."""
        pipe = self._pipeline
        for cache in (pipe.kv_cache_pos, pipe.crossattn_cache_pos):
            if not cache:
                continue
            for entry in cache:
                for key in ("k", "v"):
                    value = entry.get(key)
                    if value is not None and hasattr(value, "device"):
                        entry[key] = value.cpu()

    def restore_caches(self) -> None:
        """Move KV/crossattn tensors back to the worker device."""
        import torch

        pipe = self._pipeline
        for cache in (pipe.kv_cache_pos, pipe.crossattn_cache_pos):
            if not cache:
                continue
            for entry in cache:
                for key in ("k", "v"):
                    value = entry.get(key)
                    if value is not None and hasattr(value, "device"):
                        entry[key] = value.to(self._device)
        torch.cuda.empty_cache()

    def resume_from_tape(self, tape: dict[str, Any]) -> dict[str, int]:
        """Rebuild causal context after restart (DESIGN §27.1).

        Loads the tail latents + embeds, allocates empty caches, replays one
        generator forward at clean timestep=0 (mirrors the upstream recache
        pattern). Never serializes the whole KV cache.

        POSITION CONTRACT (measured): the forward's gather math derives read
        windows from the cache counters, so a fresh (empty) cache MUST start
        at current_start=0 — replaying at the taped absolute position reads
        unwritten ring slots (empty gather → expand crash, observed twice).
        The stream clock is therefore REWOUND to the tail length (one
        window). This loses nothing observable: the rolling window holds
        exactly one tail worth of content, so post-resume state is
        structurally identical to a fresh stream that generated the tail
        (counters global=tail, local=ring — verified by probe). Absolute
        video timeline is supervisor-owned and stays monotonic regardless.
        """
        import torch

        pipe = self._pipeline
        tail = tape["tail_latents"].to(self._device)
        tail_frames = int(tail.shape[1])
        embeds = tape["prompt_embeds"].to(self._device)
        if pipe.kv_cache_pos is None:
            pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=self._device)
            pipe._initialize_crossattn_cache(
                batch_size=1, dtype=torch.bfloat16, device=self._device
            )
        timestep = torch.zeros([1, 1], device=self._device, dtype=torch.int64)
        with torch.inference_mode():
            pipe.generator(
                noisy_image_or_video=tail,
                conditional_dict={"prompt_embeds": embeds},
                timestep=timestep,
                kv_cache=pipe.kv_cache_pos,
                crossattn_cache=pipe.crossattn_cache_pos,
                current_start=0,
                cache_start=0,
            )
        self._next_start_frame = tail_frames
        self._blocks_appended = 1
        # Continue the stream noise trajectory: the tape carries the RNG
        # state from the end of the taped segment, so the next appended
        # block draws exactly where the stream left off — including across
        # evict/rebuild cycles. Tapes predate this key only across code
        # versions (never resumed), so direct access is correct.
        self._noise_rng = torch.Generator(device=self._device)
        self._noise_rng.set_state(torch.as_tensor(tape["noise_rng_state"]).cpu())
        return {
            "next_start_frame": self._next_start_frame,
            "blocks_appended": self._blocks_appended,
        }

    def noise_rng_state(self) -> Any:
        """CPU snapshot of the stream noise RNG (taped per segment, §27)."""
        if self._noise_rng is None:
            raise RuntimeError("stream noise RNG read before the first block")
        return self._noise_rng.get_state().cpu()

    def _encode(self, prompt: str) -> tuple[Any, Any]:
        cached = self._embed_cache.get(prompt)
        if cached is not None:
            return cached
        from utils.prompt_conditioning import encode_prompt_blocks  # type: ignore[import-not-found]

        cond, cond_list = encode_prompt_blocks(self._pipeline.text_encoder, [[prompt]], 1)
        self._embed_cache[prompt] = (cond, cond_list)
        return cond, cond_list

    def begin_sequence(self, total_blocks: int, seed: int) -> None:
        """Open one sequence: full noise draw + full output buffer.

        Called once per `generate_blocks` payload. The noise draw continues
        the stream RNG (seeded once by the first block's seed ever seen),
        so the trajectory is continuous across blocks AND segments; a
        rebuild-then-retry redraws the identical prefix because the tape
        restores the RNG state. Only `seeds[0]` of the payload is used —
        the rest ride the supervisor protocol unused.
        """
        import torch

        if total_blocks < 1:
            raise ValueError("begin_sequence needs at least one block")
        if self._noise_rng is None:
            self._noise_rng = torch.Generator(device=self._device).manual_seed(seed)
        shape = self._latent_shape
        block_frames = int(self._pipeline.num_frame_per_block)
        total_frames = total_blocks * block_frames
        self._seq_noise = torch.randn(
            [1, total_frames, shape[2], shape[3], shape[4]],
            device=self._device,
            dtype=torch.bfloat16,
            generator=self._noise_rng,
        )
        self._seq_output = torch.zeros(
            [1, total_frames, shape[2], shape[3], shape[4]],
            device=self._device,
            dtype=torch.bfloat16,
        )
        self._seq_offset = 0

    def append_block(self, prompt: str, scene_cut: bool = False) -> Any:
        """Denoise one block into the persistent stream; return its latents.

        Works on this block's slice of the sequence buffers opened by
        `begin_sequence`, at the stream-absolute position — the equivalent
        of one chunk inside a single upstream `inference()` call.
        scene_cut prepends the upstream cut prefix to raw_prompts only
        (zero-KV + sink re-pin fire inside _inference_inner); the text
        embedding still encodes the bare prompt.
        """
        if self._seq_noise is None or self._seq_output is None:
            raise RuntimeError("append_block called without begin_sequence")

        pipe = self._pipeline
        shape = self._latent_shape
        block_frames = int(pipe.num_frame_per_block)
        start = self._seq_offset * block_frames
        end = start + block_frames
        cond, cond_list = self._encode(prompt)
        if pipe.kv_cache_pos is None:
            import torch

            pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=self._device)
            pipe._initialize_crossattn_cache(
                batch_size=1, dtype=torch.bfloat16, device=self._device
            )
        # NOTE: no position reset — that is the whole point. Positions
        # persist in the cache dicts; only the per-call window advances.
        # current/cache_start stay absolute and in lockstep (T2V: upstream
        # advances both per chunk from the same start).
        pipe._inference_inner(
            noise=self._seq_noise,
            batch_size=1,
            num_frames=block_frames,
            num_channels=shape[2],
            height=shape[3],
            width=shape[4],
            num_blocks=1,
            num_input_frames=0,
            num_output_frames=block_frames,
            output=self._seq_output,
            conditional_dict=cond,
            conditional_dict_list=cond_list,
            unconditional_dict=None,
            use_cfg=False,
            initial_latent=None,
            clamp_i2v_first_chunk=False,
            return_latents=True,
            current_start_frame=self._next_start_frame,
            cache_start_frame=self._next_start_frame,
            raw_prompts=[[apply_scene_cut_prefix(prompt, scene_cut)]],
        )
        chunk = self._seq_output[:, start:end]
        self._seq_offset += 1
        self._next_start_frame += block_frames
        self._blocks_appended += 1
        return chunk


_PROFILE_BY_QUANTIZATION = {"fp8": "longlive2-bf16-fp8", "bf16": "longlive2-bf16"}


def profile_for_quantization(quantization: str) -> str:
    """Recovery profile for a DiT precision (tapes never resume across numerics)."""
    try:
        return _PROFILE_BY_QUANTIZATION[quantization]
    except KeyError:
        raise ValueError(
            f"unknown quantization {quantization!r} (known: {sorted(_PROFILE_BY_QUANTIZATION)})"
        ) from None


class LongLiveSession:
    """Resident pipeline: built once at `init`, reused per segment."""

    def __init__(
        self, models_dir: Path, device: str, latent_shape: list[int], quantization: str = "fp8"
    ) -> None:
        import torch
        from pipeline import CausalDiffusionInferencePipeline  # type: ignore[import-not-found]
        from utils.fp8 import quantize_model_fp8  # type: ignore[import-not-found]

        # Fail fast on unknown precision before the expensive load.
        self._profile = profile_for_quantization(quantization)

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
        # RoPE setup mirrors inference() per-call preamble (which the direct
        # _inference_inner path bypasses): relative RoPE on, temporal offset
        # zeroed; per-shot offsets then evolve inside _inference_inner.
        dit = pipeline._dit_model
        dit.use_relative_rope = True
        dit.rope_temporal_offset = 0.0
        print("use_relative_rope: True", file=sys.stderr)
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
        if quantization == "fp8":
            print("quantizing generator to FP8 ...", file=sys.stderr)
            quantize_model_fp8(pipeline.generator.model, verbose=True)
        else:
            print("keeping generator in BF16 (no FP8 quantization) ...", file=sys.stderr)
        pipeline.generator.model.eval().requires_grad_(False)
        # Eager mode for Phase 1 (torch_compile off — no warmup samples yet).
        pipeline.vae.to(device=self._device)
        self._pipeline = pipeline
        self._config = config
        self._stream = LongLiveStreamSession(pipeline, list(latent_shape), self._device)

    @property
    def stream(self) -> LongLiveStreamSession:
        return self._stream

    def evict(self) -> None:
        """Drop the resident pipeline so audio can own the GPU (§40).

        The stream position survives on disk via the segment recovery.pt
        tapes; `rebuild` constructs a fresh session and resumes from the
        latest tape, so this object is terminal after evict.
        """
        del self._stream
        del self._pipeline
        # gc first: reference cycles (caches, hooks, closures) keep GPU
        # tensors alive past `del`; without a collection empty_cache frees
        # nothing (measured: 0 bytes freed vs ~9.4GB with gc).
        import gc

        gc.collect()
        self._torch.cuda.empty_cache()

    def generate_blocks(
        self,
        prompts: list[str],
        seeds: list[int],
        scene_cuts: list[bool],
        output_path: Path,
        fps: int,
    ) -> dict[str, Any]:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
        from einops import rearrange  # type: ignore[import-not-found]

        if not prompts or not (len(prompts) == len(seeds) == len(scene_cuts)):
            raise ValueError("prompts/seeds/scene_cuts must be non-empty equal-length lists")
        torch = self._torch
        block_latents = []
        # One sequence per payload: full noise + output buffers, sliced per
        # block at absolute stream positions (mirrors one upstream
        # inference() call). Only seeds[0] seeds the stream RNG.
        self._stream.begin_sequence(len(prompts), seeds[0])
        with torch.inference_mode():
            for prompt, cut in zip(prompts, scene_cuts, strict=True):
                block_latents.append(self._stream.append_block(prompt, cut))
        latents = torch.cat(block_latents, dim=1)
        # Recovery tail (DESIGN §27): last block's clean latents + embeds so
        # a restarted worker rebuilds causal context without re-encoding
        # (CPU T5 costs minutes). Written beside the segment video.
        tail_prompt = prompts[-1]
        _cond, _cond_list = self._stream._encode(tail_prompt)
        tape = {
            "tail_latents": block_latents[-1].detach().cpu(),
            "prompt_embeds": _cond["prompt_embeds"].detach().cpu(),
            "next_start_frame": self._stream.next_start_frame,
            "blocks_appended": self._stream.blocks_appended,
            "noise_rng_state": self._stream.noise_rng_state(),
            "profile": self._profile,
            "dtype": "bfloat16",
            "latent_shape": list(self._latent_shape),
        }
        recovery_path = output_path.with_name("recovery.pt")
        torch.save(tape, str(recovery_path))
        # Full causal decode (93f per 3-block segment): the VAE transient at
        # 1280x704 is ~10 GB regardless of chunk size (full-frame spatial
        # intermediates), so chunking alone cannot fit it alongside the
        # resident stack. Offload generator + caches to CPU (measured:
        # 1.34 GB resident, decode adds ~nothing), decode the whole
        # segment causally in one call, then restore. PCIe roundtrip costs
        # tens of seconds; the stream (caches) survives intact.
        pipe = self._pipeline
        pipe.generator.to("cpu")
        self._stream.offload_caches()
        torch.cuda.empty_cache()
        try:
            with torch.inference_mode():
                generated = pipe.vae.decode_to_pixel_chunk(
                    latents, use_cache=False, chunk_size=int(latents.shape[1])
                )
        finally:
            pipe.generator.to(self._device)
            self._stream.restore_caches()
        video = (255.0 * rearrange(generated, "b t c h w -> b t h w c").cpu()).to(torch.uint8)
        pipe.vae.model.clear_cache()
        del latents, generated, block_latents
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames = [video[0, index].numpy() for index in range(video.shape[1])]
        with imageio.get_writer(
            str(output_path), fps=fps, codec="libx264", macro_block_size=None
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        height, width = int(video.shape[2]), int(video.shape[3])
        return {
            "frames": len(frames),
            "fps": fps,
            "width": width,
            "height": height,
            "blocks": len(prompts),
            "stream_start_frame": self._stream.next_start_frame - 8 * len(prompts),
            "recovery_path": str(recovery_path),
        }


_SESSION: LongLiveSession | None = None
_INIT_PARAMS: dict[str, Any] = {}


def _build_session() -> LongLiveSession:
    """Construct the resident session from the stored init params."""
    models_dir = _INIT_PARAMS["models_dir"]
    assert isinstance(models_dir, Path)
    _enter_longlive_tree(models_dir)
    return LongLiveSession(
        models_dir,
        str(_INIT_PARAMS["device"]),
        list(_INIT_PARAMS["latent_shape"]),
        str(_INIT_PARAMS["quantization"]),
    )


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    global _SESSION
    import torch

    checked_request(payload, models_dir=str, device=str, latent_shape=list)
    models_dir = Path(str(payload["models_dir"]))
    device = str(payload.get("device", "cuda:0"))
    if not device.startswith("cuda"):
        raise RuntimeError(f"video_longlive requires a CUDA device (got {device!r})")
    if not torch.cuda.is_available():
        raise RuntimeError("video_longlive requires a CUDA GPU")
    raw_shape = payload["latent_shape"]
    assert isinstance(raw_shape, list)
    latent_shape = [int(v) for v in raw_shape]
    quantization = str(payload.get("quantization", "fp8"))
    profile = profile_for_quantization(quantization)
    _INIT_PARAMS.update(
        {
            "models_dir": models_dir,
            "device": device,
            "latent_shape": latent_shape,
            "quantization": quantization,
        }
    )
    _SESSION = _build_session()
    name = torch.cuda.get_device_name(0)
    free_gib, total_gib = torch.cuda.mem_get_info()
    return {
        "status": "READY",
        "backend": profile,
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
    if _SESSION is None:
        raise RuntimeError("video_longlive not initialized — send `init` first")
    # Multi-block form (Phase 2): prompts/seeds lists, one entry per block.
    # Single-block form (Phase 1): bare prompt/seed.
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
        fps=int(payload["fps"]),
    )
    artifacts = [str(output), str(result["recovery_path"])]
    return {
        "blocks_generated": len(prompts),
        "artifacts": artifacts,
        "video": result,
    }


def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild causal context from a recovery.pt tape (DESIGN §27.1)."""
    if _SESSION is None:
        raise RuntimeError("video_longlive not initialized — send `init` first")
    import torch

    checked_request(payload, recovery_path=str)
    with open(str(payload["recovery_path"]), "rb") as handle:
        tape = torch.load(handle, map_location="cpu", weights_only=False)
    expected = profile_for_quantization(str(_INIT_PARAMS["quantization"]))
    if not isinstance(tape, dict) or tape.get("profile") != expected:
        raise ValueError("recovery tape profile mismatch")
    position = _SESSION.stream.resume_from_tape(tape)
    return {"resumed": True, **position}


def handle_evict_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    """Unload the video stack so audio can own the GPU (§40)."""
    del payload
    global _SESSION
    if _SESSION is not None:
        _SESSION.evict()
        _SESSION = None
    return {"evicted": True}


def handle_rebuild(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the session after an eviction and resume from tape (§40).

    Payload carries `recovery_path` (latest committed tape): the fresh
    session continues the stream instead of starting a new one. Without
    init params (never initialized) this is an error, not a silent fresh
    start — the supervisor always inits before first use.
    """
    global _SESSION
    import torch

    checked_request(payload, recovery_path=str)
    if not _INIT_PARAMS:
        raise RuntimeError("video_longlive rebuilt before init")
    _SESSION = _build_session()
    with open(str(payload["recovery_path"]), "rb") as handle:
        tape = torch.load(handle, map_location="cpu", weights_only=False)
    expected = profile_for_quantization(str(_INIT_PARAMS["quantization"]))
    if not isinstance(tape, dict) or tape.get("profile") != expected:
        raise ValueError("recovery tape profile mismatch")
    position = _SESSION.stream.resume_from_tape(tape)
    return {"rebuilt": True, **position}


def main() -> None:
    serve(
        {
            "init": handle_init,
            "health": handle_health,
            "generate_blocks": handle_generate_blocks,
            "evict_gpu": handle_evict_gpu,
            "rebuild": handle_rebuild,
            "checkpoint": lambda payload: {
                "checkpoint_id": f"longlive-{payload.get('segment_id', 'none')}"
            },
            "resume": handle_resume,
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()
def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured `generate_blocks` probes with VRAM peaks (§104).

    Reuses the live stream session, so measured blocks ADVANCE the stream —
    run this on a scratch session (or before a run), never mid-voyage.
    Requires `init` first; without a session this is an error, not a
    silent fake measurement.
    """
    if _SESSION is None:
        raise RuntimeError("video_longlive not initialized — send `init` first")
    import tempfile
    import time

    import torch

    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    probe: dict[str, Any] = {
        "segment_id": str(payload.get("segment_id", "benchmark")),
        "prompts": [str(payload.get("prompt", "benchmark probe"))],
        "seeds": [int(payload.get("seed", 0))],
        "scene_cuts": [False],
        "fps": int(payload.get("fps", 24)),
    }
    walls: list[float] = []
    peaks: list[float] = []
    frames: object = "unknown"
    with tempfile.TemporaryDirectory(prefix="voyage-bench-") as tmp:
        for index in range(warmup + measured):
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            result = handle_generate_blocks(
                {**probe, "output_path": str(Path(tmp) / f"b{index}.mp4")}
            )
            elapsed = time.monotonic() - started
            peak_gib = torch.cuda.max_memory_allocated() / 1024**3
            if index >= warmup:
                walls.append(elapsed)
                peaks.append(peak_gib)
                video = result.get("video")
                if isinstance(video, dict) and isinstance(video.get("frames"), int):
                    frames = video["frames"]
    mean = sum(walls) / len(walls)
    blocks = len(probe["prompts"])
    return {
        "backend": "longlive2",
        "warmup_blocks": warmup * blocks,
        "measured_blocks": measured * blocks,
        "frames_per_block": frames,
        "block_wall_seconds": [round(wall, 3) for wall in walls],
        "blocks_per_second": round(blocks / mean, 3),
        "vram_peak_gib": round(max(peaks), 2),
        "vram_avg_gib": round(sum(peaks) / len(peaks), 2),
    }


            "benchmark": handle_benchmark,
