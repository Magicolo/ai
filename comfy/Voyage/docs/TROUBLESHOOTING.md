# TROUBLESHOOTING

## OOM (CUDA out of memory)

- First suspect: co-resident stacks. The run loop evicts between video
  and audio automatically; if you drive workers manually, evict (`evict_gpu`)
  before switching. `del` alone frees nothing — eviction is `del` +
  `gc.collect()` + `empty_cache()`.
- `torch.compile` is deferred (383 s warmup, zero VRAM delta) — don't enable it.
- Full-res VAE decode OOMs: the worker decodes chunked (`decode_to_pixel_chunk(1)`,
  ~8.7 GB peak). If you changed latent shape, re-check the 16 GB budget.
- Longest soak held ~350 MB headroom at 13.17 GiB — tight but flat.

## CUDA failures

- `voyage doctor` first: driver, `nvidia-smi`, CUDA visibility,
  torch-CUDA match, disk free, models-dir presence with a per-backend
  `verify_*` summary. It does NOT yet check compute capability, CUDA
  runtime version, FlashAttention/Triton, checkpoint compat, fs
  permissions, worker interpreters, or ACE-Step runtime availability —
  and `models verify` remains the
  authority for weight files (doctor only summarizes presence + size).
- transformers must be 4.57.6 in the video image (5.x breaks LongLive).
- `quantization`: `bf16` fits and kills the fp8 highlight blowout;
  saturated extremes can still blow out — prompt care at peak brights.

## Model mismatch

- `models verify` — every weight file, presence + size sanity.
- Checkpoints must match upstream `MAIN_MODEL_COMPONENTS` layout under
  `<models>/acestep/checkpoints/` (includes the gate-only 1.7 B LM).
- Tapes never resume across numerics: switching `fp8|bf16` starts the
  stream fresh from the next segment. `use_relative_rope` is recorded per
  segment — never change it silently across resume.

## ffmpeg errors

- Fake backends shell out to container ffmpeg; a missing binary means a
  wrong image. Take-file extensions must match codecs (`.wav` ↔
  `pcm_s16le`, `.flac` ↔ flac).

## Audio worker failure

- The 0.6 B planner must `initialize(offload_to_cpu=True)` or the DiT
  preflight fails. 2060-class cards cannot hold the ACE DiT — render
  audio on the 16 GB card via the standard evict/render/rebuild cycle.

## Disk full

- The run rests at `PAUSED_DISK_FULL` (never FAILED): free space, `run`
  again. Default reserve is 5.0 GiB in dev TOML (spec default 20.0) —
  raise for production. `finalize` preflights the same reserve.
- Containers run as the invoking host user (`voyager`, UID/GID baked at
  build time from `id` and pinned with `--user` at runtime), so renders
  leave host-owned files on bind-mounts (`output/`, `/tmp`, `~/.cache/
  voyage-models`) and plain `rm -rf` works. All images use `WORKDIR /app`
  (video image fixed, issue 053), so bare `docker run` with relative paths
  lands in `/app`, not ephemeral layers; `run.sh` keeps `-w /app` as
  belt-and-braces. If you see root-owned files, the image predates the
  user rollout: rebuild (`scripts/build*.sh`) and one-time
  `sudo chown -R $(id -u):$(id -g)` the affected dirs.

## Corrupt checkpoint / segment

- `validate` names the exact problem (checksums recomputed, numbering,
  orphans, tapes). Orphan `*.partial` = interrupted commit: inspect,
  remove, re-run. `finalize --skip-bad` routes around corrupt segments.

## Worker environment problems

- `SubprocessWorker.stop()` tolerates dead pipes; a SIGKILLed worker
  always surfaces as restart-class, never a hang (RPC timeout 600 s
  default, `rpc_timeout_seconds`). One restart budget per worker per run
  (3); exhaustion opens the circuit breaker → FAILED with the reason in
  `last_error` and `logs/metrics.jsonl`.
