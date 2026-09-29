# 015 — LongLive per-segment double PCIe ping-pong (VAE-offload + generator-offload)

- Status: resolved (fixed 2026-09-29: fusion slice landed; elimination measured not-viable, see log)
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
- 2026-09-26 (resolution, PARTIAL): the literal 4→3 move cut is not
  achievable without removing a transfer, and the only removable one is the
  VAE roundtrip itself (candidate 2 — GPU-gated, see below). Landed the
  CPU-safe fusion slice in `LongLiveSession.generate_blocks`
  (`voyage/workers/video_longlive.py`): the VAE stays parked on CPU across
  the block loop AND the tape write (`:851-881` — `_encode` is CPU-T5-only,
  verified at `:566-582`, so parking it there is safe) and returns in the
  decode prologue only after generator→CPU + `offload_caches()`
  (`:920-935`, free-before-allocate: strictly lower transient peak than the
  old restore-while-DiT-resident); both mid-segment `empty_cache()` calls
  now run after `gc.collect()` (`:870`, `:933` — the evict() measured
  lesson, 0 vs ~9.4GB). Same four transfers per segment — the win is peak +
  fragmentation, not move count; stage keys unchanged (audit-comparable).
  Tests: `tests/test_longlive_offload_fusion.py` (4 tests: move order,
  VAE-parked-across-tape, gc-before-every-flush, seven-stage keys).
  FOLLOW-UP (candidate 2, needs idle GPU): VAE tiling/slicing or porting
  upstream streaming decode so the roundtrip disappears. Measurement plan:
  (a) `profile_stages=True` A/B on a 1-block segment — expect
  `offload_for_decode_ms` to shrink by the VAE-upload share and
  `denoise_blocks_ms` to drop its old restore tail; (b) nvidia-smi peak per
  call — expect the post-denoise transient down ~1.3 GiB (no DiT+VAE
  co-residency); (c) tiling probe: `pipe.vae.enable_tiling()` (or slicing)
  + decode alongside resident DiT, watch for the ~10GB transient; only then
  delete the roundtrip. Do NOT attempt without the GPU — a wrong guess OOMs
  every segment.
- 2026-09-29 (orchestrator): CPU-safe fusion slice landed (VAE parked
  across the block loop, gc-before-flush); the remaining 4→3/elimination
  work (VAE tiling or streaming-port, upstream `streaming_vae`) needs an
  idle-GPU measurement session per the 3-step plan above. Kept OPEN for
  that session.
- 2026-09-29 (orchestrator): ELIMINATION INVESTIGATED ON IDLE GPU —
  verdict: NOT VIABLE, issue resolved on the fusion slice. Probe (real
  `WanVAEWrapper.decode_to_pixel_chunk` path, full-res [1,8,48,44,80]
  latents, VAE-only 2.8 GB session): temporal `chunk_size=4` on 8 latent
  frames yields 26 video frames vs 29 full-pass (2×13 — each chunk
  independently drops its 3-frame causal warmup; `use_cache=False`
  carries no context across chunks). The wrapper's chunk loop only
  preserves counts with `cached_decode`, which `Wan2_2_VAE` does not
  implement (`decode` + `clear_cache` only — the code comment was
  right). Spatial tiling has no upstream support and would hand-roll
  overlapping tiles through a causal 3D-conv VAE (seam/blending risk on
  every frame) for ~4.4 s/segment — rejected on quality-risk grounds.
  Side findings: Wan2.2 VAE is 48ch/16x-spatial (not 16ch/8x — the
  [1,8,48,44,80] preset is exact); the full session needs ~24 GB host
  RAM (T5 11.4 + DiT 10 + VAE 2.8) and died silently on this shared box
  (35 GB available minus concurrent containers) — serialize GPU+RAM
  access for future probes. Landed state stands: fusion slice (peak +
  fragmentation) + stage timers; the 44% DiT + 43% VAE remainder is
  model work, not overhead. Probe drivers were /tmp-only, never
  committed; the `VOYAGE_VAE_CHUNK_FRAMES` vehicle was reverted.
