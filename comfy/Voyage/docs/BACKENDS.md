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
| video | `ltxv` | `voyage-video:latest` | 2B-distilled T2V + tail-conditioned extensions, bf16-first (fp8 fallback) |
| audio | `acestep` | `voyage-video:latest` | turbo config, 0.6 B planner offloaded to CPU |
| director | `qwen` | `voyage-director:latest` | Qwen3-8B, non-thinking, temp 0.7 |

Select in TOML (`config.video.backend`, `config.audio.backend`,
`config.director.backend`) or per-invocation for runs
(`voyage run --director … --quantization …`).

## LTXV chaining model (`ltxv`, Phase 7 alternative)

Resident session like `longlive2` (multi-block payload, resume hook,
acestep GPU swap — see `STREAMING_VIDEO_BACKENDS` in
`voyage/supervisor.py`), but chaining is explicit, not KV-cache: block 0
renders text-to-video from cached CPU T5 bf16 embeds
(`text_encoder=None` — the pipeline would otherwise move the 18.8 GB
fp32 encoder to GPU); blocks 1+ render tail-conditioned extensions from
the previous block's tail frame, frame 0 deduped. Each block writes a
chain PNG; the last becomes the segment tail, the rest are deleted. The
tail PNG doubles as the crash-recovery tape beside the segment video
(`recovery.pt` carries profile `ltxv` — tapes never resume across
backends); `scene_cut` forces a fresh start. Native 768×512; draft
640×352 verified. Needs `models download ltxv-2b`. The `benchmark` op
saves/restores tail state around its probes, so unlike longlive it does
not advance any stream — safe to run mid-sequence.

## Experimental backends

- **Visual inspector** (`[experimental] visual_inspector`, default off):
  Qwen3.5-9B reads the previous segment's middle frame; measured metrics
  feed the director context, amendments apply post-validation with a
  provisional `style_similarity_min = 0.60`. Advisory only — retry→skip,
  never blocks a commit. Needs `models download inspector-qwen35`.
- **`longlive2-bf16` recovery profile**: tapes never resume across
  numerics — a `fp8` tape will not load under `bf16` and vice versa.
