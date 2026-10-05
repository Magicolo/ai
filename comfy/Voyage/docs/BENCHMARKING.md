# BENCHMARKING — protocol, commands, reports

Every benchmark follows DESIGN §104: exact model revision, checkpoint,
resolution, block size, attention window, step count, quantization, VAE
mode, GPU model, driver, CUDA, PyTorch, warm-up count, measured blocks.
First-run load/compile is always excluded from steady-state numbers.

## Commands

```bash
# Block-level: warmup + measured probe through the real worker RPC:
./scripts/run.sh benchmark video --run <dir> [--warmup 1 --measured 3]
./scripts/run.sh benchmark audio --run <dir> [--warmup 1 --measured 3]
# Segment-level: throwaway temp run, N segments, stage means + deltas:
./scripts/run.sh benchmark end-to-end [--segments 2]
# Stability: real run, N segments, resource-trend report (exit 1 on errors):
./scripts/run.sh soak --run <dir> --segments N
```

`benchmark video|audio` needs `--run` (workers start from that run's
config). `end-to-end` never touches your run — it builds a temp dir.
`soak` commits real segments, then reports.

## Report fields

```text
blocks/sec
video-fps-equivalent          (blocks/sec × frames per block)
seconds-generated / wall-second
peak-VRAM                     (torch.cuda max; "unknown" on fake backends)
average-VRAM
VAE percentage of wall time
text-encoding percentage
```

Fake backends honestly report `unknown` for GPU-only fields; their
numbers measure harness overhead, not model speed — compare GPU runs to
GPU runs. Worker `benchmark` ops live in each `voyage/workers/*.py`
(`handle_benchmark`); report formatting in `voyage/bench.py`.

## What to measure

- **Warm-up handling**: `--warmup` iterations run before timing starts;
  default 1 (GPU first-block effects are larger — raise to 2+).
- **Block throughput**: `blocks/sec` from `generate_blocks` probes.
- **Segment throughput**: `end-to-end` stage means (inspect/director/
  video/audio/validate/commit) — audio dominates until takes cover ahead.
- **Finalize quality**: e2e numbers are comparable only at the same
  multiplier settings (`--upscale`/`--interpolate`, `docs/AUGMENT.md` —
  defaults 1/1 ship native geometry). Raising the multipliers lifts
  pixels × fps (model pass + re-encode), so finalize dominates
  those e2e wall times. Compare GPU runs to GPU runs at the same
  settings; keep 1/1 for native-geometry timings.
- **Peak VRAM**: per-probe CUDA peak; the soak trend shows growth.
- **CPU usage**: supervisor RSS peak ships in every `resource_gauges`
  event (`ru_maxrss`); worker CPU is not sampled — use container stats.
- **VAE time / text-encoding time**: reported by GPU worker probes as a
  percentage of block wall time (chunked decode and cached embeddings
  keep both flat across a run).
- **Audio generation time**: take render seconds per take in the soak
  stage table; steady state renders ~1 take per 45 s of timeline.

## Soak acceptance

Flat RSS/VRAM peaks (±100 MB class), stage means stable, `validate`
clean at the end, no `circuit_breaker_open` events. The endurance pytest
marker (`pytest -m endurance`) runs the multi-segment flatness test.

## Backend qualification

Before calling any backend profile production-ready, run the DESIGN
§137A qualification (smoke → resolution/FPS → VRAM/RAM → steady state →
crash recovery → 3-segment visual review) via the backend-agnostic
driver:

```bash
./scripts/qualify.sh [--backend ltxv|causvid] [--segments N] <absolute-run-dir>
```

`<run-dir>` must be absolute (workers spawn with CWD=run_dir, so a
relative dir doubles up inside payload paths). The driver benchmarks,
runs, validates, and tees the JSON summary to `reports/`; crash
recovery (kill -9 the video worker mid-segment, then resume) and the
eyeball visual review stay manual. Past qualification evidence lives in
`reports/video-backends.md`. Never run
under contention: the driver aborts when >2 GiB on GPU 0 is held by
another process.
