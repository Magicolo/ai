# UPSTREAM_LONG_LIVE_PATCHES — LongLive runtime modifications

Upstream: [NVlabs/LongLive](https://github.com/NVlabs/LongLive),
pinned at `6b36d20ec6f7958d29d11a704dfa64611a9f2572`
(`worker/Dockerfile.video`). We never fork upstream: all changes are
runtime method replacements in `voyage/workers/video_longlive.py`,
guarded to no-op when preconditions don't hold. Revisit every patch if
the pin moves.

## 1. Pos-only KV / cross-attention caches (`_install_pos_only_caches`)

- **Upstream file**: `pipeline/causal_diffusion_inference.py` —
  `_initialize_kv_cache` / `_initialize_crossattn_cache` @6b36d20.
- **Reason**: upstream always allocates the negative (unconditional)
  branch, even at `guidance_scale=1.0` where `inference()` sets
  `unconditional_dict=None` and never touches it. On 16 GB that wasted
  half (~2.6 GB at local attention 8) is OOM vs finished segment.
- **Behavioral difference**: at guidance exactly 1.0 (and only then —
  otherwise upstream methods stay in place), only the positive branch
  allocates; negatives are tensor-free `{"is_init": False}` placeholders
  (every read is `use_cfg`-guarded; the per-chunk reset loop writes one
  flag unguarded, which the placeholder absorbs). Non-quantized path
  only (we never set `quantize_kv`; NVFP4 is Blackwell-only).
- **Test**: `tests/test_phase2.py` (continuity: seg-boundary diff ==
  block-boundary diff) + kill-test → VALID 186f; soak VALID 372f flat.
- **Upstreamable**: yes — a `guidance_scale==1.0` fast path; not yet
  proposed.

## 2. Direct `_inference_inner` continuation (`LongLiveStreamSession`)

- **Upstream file**: same module's `inference()` — allocates caches on
  first call, **resets positions to zero** on later calls.
- **Reason**: repeated `inference()` calls are separate streams, not one
  voyage; positions must advance monotonically across blocks *and*
  segments for the rolling window to evict (flat memory by construction).
- **Behavioral difference**: the session replicates the preamble (prompt
  encode + output buffer) once per block and calls the lower-level
  `_inference_inner` with tracked start frame; noise is stream-level
  (one generator drawn sequentially per block — a fresh generator per
  block caused 6–9x frame-diff jumps at boundaries).
- **Test**: `tests/test_phase2.py` + `tests/test_precision.py`.
- **Upstreamable**: partially — the reset-on-recall is arguably a bug
  for streaming use; our session wrapper is voyage-specific.

## 3. Config deviations from upstream defaults

| Setting | Upstream | Ours | Why |
|---------|----------|------|-----|
| `use_relative_rope` | `False` | `True` | continuity slice; zero VRAM delta, recorded per segment |
| `local_attn_size` | 32 | 8 | KV cache is `local_attn × frame_seq` bf16 per layer ×2 branches — 32 OOMs 16 GB |

## 4. Adapter shims (not patches)

- **CPU/bf16 text-encoder twin** via the pipeline's `text_encoder=` seam
  (call-compatible with upstream `WanTextEncoder`, CPU-resident).
- **CWD contract shim**: upstream resolves `wan_models/Wan2.2-TI2V-5B/`
  against the process CWD — the worker reproduces that layout.
- **Chunked VAE decode** via the public `decode_to_pixel_chunk(1)` API
  (full decode OOMs; 8.7 GB peak chunked).
