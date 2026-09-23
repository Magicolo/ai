# BACKENDS — worker adapter interface

## The adapter contract

Each backend is a **worker subprocess** speaking typed JSONL-RPC
(`voyage/rpc.py`, ops in `voyage.workers.loop`): requests on stdin,
responses on stdout (reserved — workers log to stderr only), one JSON
object per line with `{id, op, payload}` → `{id, ok, result|error}`.

To add a backend, write a `voyage/workers/<name>.py` with `handle_*`
functions and a `serve({...})` op map, then register it in
`voyage/supervisor.py` (`VIDEO_WORKER_MODULES` / audio / director
selection). Ops: `init` (payload carries models_dir/device/shape),
`health` (must report backend + VRAM when on CUDA), `generate_blocks` /
`generate_audio` / `decide`, `checkpoint` / `resume`, `evict_gpu` /
`rebuild` (GPU time-sharing), `benchmark` (see BENCHMARKING),
`shutdown`.

Error semantics: handler exceptions become `WORKER_ERROR`
(retryable → supervisor restarts the worker, budget-limited);
`UNKNOWN_OP` / `NOT_IMPLEMENTED` are non-retryable. A worker that needs
models must fail `init` loudly, never render silently wrong media.

## Built-in backends (no weights)

| Role | Backend | What it renders |
|------|---------|-----------------|
| video | `fake` | deterministic `testsrc` 320×180-class H.264 |
| audio | `fake` | `sine` tone, WAV slices / FLAC takes |
| director | `deterministic` | rule-based decisions, no embeddings |
| inspector | `skipped` | visual feedback off |

Fake media are real codecs in real containers, so `validate`, the
finalizer concat path, and checksums run exactly as in production.

## GPU backends

| Role | Backend | Image | Notes |
|------|---------|-------|-------|
| video | `longlive2` | `voyage-video:latest` | resident BF16+FP8 stream, `fp8`\|`bf16` quantization |
| audio | `acestep` | `voyage-video:latest` | turbo config, 0.6 B planner offloaded to CPU |
| director | `qwen` | `voyage-director:latest` | Qwen3-8B, non-thinking, temp 0.7 |

Select in TOML (`config.video.backend`, `config.audio.backend`,
`config.director.backend`) or per-invocation for runs
(`voyage run --director … --quantization …`).

## Experimental backends

- **Visual inspector** (`[experimental] visual_inspector`, default off):
  Qwen3.5-9B reads the previous segment's middle frame; measured metrics
  feed the director context, amendments apply post-validation with a
  provisional `style_similarity_min = 0.60`. Advisory only — retry→skip,
  never blocks a commit. Needs `models download inspector-qwen35`.
- **`longlive2-bf16` recovery profile**: tapes never resume across
  numerics — a `fp8` tape will not load under `bf16` and vice versa.
