# BENCHMARKING — protocol, commands, reports

Every benchmark follows DESIGN §104: exact model revision, checkpoint,
resolution, block size, attention window, step count, quantization, VAE
mode, GPU model, driver, CUDA, PyTorch, warm-up count, measured blocks.
First-run load/compile is always excluded from steady-state numbers.

## Commands

There is no `benchmark`/`soak` CLI verb (two-verb CLI: `configure` +
`generate` only). Benchmarks run through the worker `benchmark` ops and
the pytest/qualification harness instead:

```bash
# Block-level: warmup + measured probes through the real worker RPC
# (fake backends on CPU; same ops drive GPU workers — see voyage/bench.py):
python -m pytest tests/test_benchmark.py
# Continuity math on a committed run (boundary vs within-segment diffs):
python -m voyage.boundary_metrics --run /app/output/<name>
# Stability: multi-segment flatness (RSS/VRAM trend, exit 1 on errors):
python -m pytest -m endurance
```

The worker `benchmark` ops still need a configured run (workers start
from that run's config). End-to-end harness runs never touch your run —
they build a temp dir. Soak-style runs commit real segments, then report.

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

`<run-dir>` must be absolute AND equal to `$PWD/output/<basename>`
(enforced, exit 2 — otherwise the manifest check and the generate call
address different runs and the artifact attests the wrong run).
`--segments N` is additive: when passed it forwards to
`generate <name> --segments N`, extending the stored manifest plan by
N (it never sets the total); omitted, the stored plan generates as-is.
To qualify an exact total, configure the plan first
(`configure <name> --segments <N>`) and omit the flag. The JSON
summary records the flag (or null), the stored plan before generate,
and the rendered segment count. The driver benchmarks,
runs, validates, and tees the JSON summary to `reports/`; crash
recovery (kill -9 the video worker mid-segment, then resume) and the
eyeball visual review stay manual. Past qualification evidence lives in
`reports/video-backends.md`. Never run
under contention: the driver aborts when >2 GiB on GPU 0 is held by
another process.
