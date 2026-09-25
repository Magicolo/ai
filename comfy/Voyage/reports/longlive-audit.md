# LongLive root-cause audit (TASK §16, DESIGN §137D phase 1)

Measured 2026-09-24 on the idle RTX 4060 Ti (cc 8.9, 15.57 GiB) via
`benchmark video`-equivalent single-block probes with the new
`profile_stages` CUDA-event timers (`voyage/workers/video_longlive.py`,
`tests/test_longlive_stages.py`). Native geometry 1280x704 @ 24 fps,
local_attn 16 / sink 8, 8 latents/block, 4 sampling steps. Full-res,
production settings — no reduced probe.

## Stage splits (ms per 1-block / 29-frame segment)

| Stage | Block 0 (first call) | Block 1 | Block 2 | Steady share |
|---|---|---|---|---|
| `offload_vae_to_cpu_ms` | 385 | 375 | 366 | ~1% |
| `denoise_blocks_ms` (DiT forward + KV) | 72,000 | 16,820 | 16,908 | ~44% |
| `tape_encode_ms` | 0.01 | 0.01 | 0.01 | ~0% (embed cache hit) |
| `offload_for_decode_ms` (generator+cache → CPU) | 2,515 | 2,760 | 2,787 | ~7% |
| `vae_decode_ms` (29f @ 1280x704) | 16,441 | 16,299 | 16,384 | ~43% |
| `restore_after_decode_ms` | 1,242 | 1,250 | 1,259 | ~3% |
| `media_write_ms` (GPU→CPU + x264) | 774 | 737 | 725 | ~2% |
| **Sum vs block wall** | **93.4s vs 93.4s** | **38.2s vs 38.3s** | **38.4s vs 38.5s** | **~100% closed** |

Steady state: **~38.3 s wall per 29 frames (1.21 s video) = ~32:1**.
The original ~100:1 observation was cold-start-dominated (first segment
~98 s incl. model load + first-call warmup ≈ 81:1); it is not the
steady-state throughput.

## Quantization paths

- **fp8 (default)**: fits, 14.28 GiB peak (`benchmark video`, 3 blocks).
  Worker log proves the quantized path engages:
  `[FP8] TorchAO W8A8 quantized 300 linear layers with row-wise scaling;
  kept 6 layers in BF16` — no Python/reference fallback (§B negative).
- **bf16**: `init` succeeds (11.1 GiB resident, 4.3 GiB free) but the
  first DiT forward OOMs deterministically — 14.93 GiB in use, 94 MiB
  free, +42 MiB requested. The unquantized full-res path does not fit
  16 GiB. bf16 remains valid only as a recovery-profile label, not a
  runnable 4060 Ti configuration (§G data point).

## First-call excess (~55 s, once per session)

Block 0 denoise costs 72 s vs 16.9 s steady. Excluded from the steady
budget per TASK §4.2 (first-call/warm-up separation). Likely composition:
CPU-resident T5-XXL (11 GB) text encode on the cold path + kernel
autotuning on first forward (eager mode, no compile). Reducible in
principle (warmed session, cached embeds) but paid once per worker
 lifetime — not a steady-state concern.

## Verdict: category I (mixed causes), quantified

| Contributor | Share of steady wall | Category |
|---|---|---|
| DiT denoise, 4 steps × 8 latents (genuine model throughput) | ~44% | genuine throughput (G-adjacent, not a defect) |
| VAE full-causal decode, 29f @ 1280x704 (~10 GB transient) | ~43% | D |
| PCIe offload roundtrips (VAE aside + generator/cache park + restore) | ~11.5% | C |
| Media write + tape | ~2% | — |

Excluded with evidence: **A** (budget closes at ~100% — no hidden
implementation sink), **B** (torchao W8A8 provably engaged), **E**
(eager, compile deferred by decision), **F** (single-block probes;
KV window flat by the Phase-2 design), **H** (4 steps / 1280x704 is
the upstream recipe at native geometry, not a mismatch).

Do not conclude "LongLive is simply slow": the transformer accounts
for ~44%, not a majority — the VAE decode (~43%) and the offload
roundtrips (~11.5%) are first-class citizens. What would actually move
steady-state throughput: fewer decoded pixels per segment (lower
render resolution), fewer DiT steps (upstream recipe change), or a VAE
that fits alongside the generator (eliminating the ~4.4 s/segment
PCIe roundtrip). Plumbing fixes cannot beat ~44% DiT + ~43% VAE.

Raw probe logs: `/tmp/auditprobe/` (ephemeral, host-only) —
`audit-fp8.responses.jsonl` (stage splits), `audit-fp8.stderr.log`
(worker log incl. the FP8-quant line), `audit-bf16.*` (init-OK +
forward-OOM evidence).
