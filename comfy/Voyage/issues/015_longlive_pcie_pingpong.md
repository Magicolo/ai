# 015 — LongLive per-segment double PCIe ping-pong (VAE-offload + generator-offload)

- Status: open
- Severity: major (~11.5% steady wall, measured)
- Area: performance — `generate_blocks` offload choreography
- Rank rationale: measured, on every segment, with a known upstream fix direction
  (tiling/streaming VAE) that is currently disabled.

## Technical description

- `voyage/workers/video_longlive.py:742-754` VAE→CPU before denoise.
- `voyage/workers/video_longlive.py:797-820` generator→CPU + `offload_caches()`
  before `decode_to_pixel_chunk`, restore after.
- `Voyage/reports/longlive-audit.md:10-21` stage table closes at ~100%:
  `denoise ~16.9s (44%) + vae_decode ~16.3s (43%) + offload_for_decode ~2.7s +
  restore ~1.25s + offload_vae ~0.37s ≈ 4.4s (11.5%) + media 0.7s`.

Every 1-block/29-frame segment moves VAE (1.31 GiB) to CPU and back **and**
generator + KV caches to CPU and back. `torch.cuda.empty_cache()` is called 3×
per segment without `gc.collect()` except in `evict()` — mid-segment
fragmentation unaddressed. `video_longlive.py:791-796` admits `~10GB VAE
transient regardless of chunk size`.

## Why this is an issue

Roughly 4.4 s of every segment — about 11.5% of steady wall clock, measured —
goes to moving the 1.31 GiB VAE to CPU and back plus generator and KV caches
to CPU and back, with three `empty_cache()` calls and no collection to fight
fragmentation. The cost fires per segment regardless of content, so it is pure
overhead rather than model work, and it scales linearly with voyage length into
days of PCIe time. The known fix direction (tiling or streaming VAE decode)
was never attempted, leaving a measured, permanent tax on every frame the
system will ever render. Every segment's latency budget carries it.

## Evidence

- `longlive-audit.md:51-55` verdict table; `rg -n
  "offload_vae_to_cpu_ms|offload_for_decode_ms|restore_after_decode"
  Voyage/voyage/workers/video_longlive.py`.
- `rg enable_tiling/slicing` returns zero hits — tiling never attempted.

Re-verified 2026-09-25:

```
$ rg -n "offload_vae_to_cpu_ms|offload_for_decode_ms|restore_after_decode_ms|enable_tiling|enable_slicing" voyage/workers/video_longlive.py
50:    "offload_vae_to_cpu_ms",
53:    "offload_for_decode_ms",
55:    "restore_after_decode_ms",
750:            timer.start("offload_vae_to_cpu_ms")
754:            timer.stop("offload_vae_to_cpu_ms")
799:            timer.start("offload_for_decode_ms")
804:            timer.stop("offload_for_decode_ms")
816:                timer.start("restore_after_decode_ms")
820:                timer.stop("restore_after_decode_ms")
```

No `enable_tiling`/`enable_slicing` hits — tiling still never attempted.

## Reproduction

Read `generate_blocks:718-847`; per-segment stage timings show the ~4.4 s
offload wedge on every segment.

## Source references

- `voyage/workers/video_longlive.py:718-847`; `Voyage/reports/longlive-audit.md`.

## Resolution candidates

1. Fuse moves: keep VAE on CPU across `generate_blocks`, decode first, then bring
   generator back once (4 moves → 3; saves ~0.4 s/segment, ~1%).
2. Real fix: VAE tiling/slicing so decode fits alongside DiT (eliminates ~4.4 s).
   Upstream `streaming_vae` is disabled (`build_longlive_config:204-208` —
   `VAE.cached_decode` missing). Port cached/streaming decode or enable
   `vae.enable_tiling()`.

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep from the committed audit numbers.
- Open: (1) is mechanical; (2) needs upstream port + idle-GPU validation.
- 2026-09-25 (repair pass): added `## Why this is an issue`; Evidence enriched
  with live offload-timer output; refs verified current.
