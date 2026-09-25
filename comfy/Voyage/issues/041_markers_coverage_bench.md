# 041 — Markers registered but dead; no coverage gate; benchmark evidence gaps

- Status: open
- Severity: medium (GPU-gated tests don't exist as a category; coverage unknown;
  perf claims lack artifacts)
- Area: tests/benchmarks — `pyproject.toml:64-70`, `bench.py`, reports
- Rank rationale: `gpu` marker has 0 uses; `endurance` 1 use; no `--cov`, so the
  gaps in 038 stay invisible to gates; several headline numbers have no in-tree
  artifact.

## Technical description

- `markers=[gpu, endurance]` (`pyproject.toml:64-70`,
  `addopts=-p no:cacheprovider --strict-markers`); `rg -n "pytest.mark.gpu"
  Voyage/tests` → 0 hits (verified live 2026-09-25 — only 6 `parametrize` + 1
  `endurance` at `test_benchmark.py:161`). GPU legs live in `qualify.sh`/manual
  runs (`reports/video-backends.md`).
- No `[tool.coverage]`, no `--cov`, no `fail_under` (Zoomy: `fail_under=67`
  ratchet, data file to `/tmp`).
- `voyage/bench.py:12-19` reports only count/mean/min/max — no p50/p90, no
  VRAM-over-time, no PCIe/offload split, no T5-encode vs denoise vs decode split
  (LongLive `stage_ms` has it per-call but `benchmark` aggregates wall+peak only:
  `video_longlive.py:1013-1025`).
- `tests/test_benchmark.py:53-92` asserts fake math with `vram_peak_gib ==
  "unknown"` — GPU VRAM assertions exist only in worker `handle_benchmark` paths
  never exercised in CI. `test_precision.py` asserts config/profile mapping only;
  the `[FP8] TorchAO W8A8 quantized 300 layers` log line is asserted nowhere.
- `reports/video-backends.md:119-122` LTXV leg **empty**; CausVid zero GPU numbers
  (pins only). `worker/Dockerfile.video:46` quotes "30s take 4.3s peak 12.84GB"
  as a comment — no benchmark artifact reproduces it. `reports/longlive-audit.md:
  73-75` raw logs lived in `/tmp/auditprobe/` (ephemeral) — the 38.3 s steady /
  14.28 GiB peak / bf16-OOM claims have no in-tree artifact.

## Why this is an issue

A `gpu` marker with zero uses means GPU-gated tests do not exist as a category — GPU legs live in shell scripts and manual runs, invisible to `pytest --collect-only` and to anyone auditing what CI actually covers. With no coverage gate, the gaps catalogued in 038 stay invisible to the gates themselves, so coverage can only decay. And headline numbers (38.3 s steady, 14.28 GiB peak, bf16-OOM) with no in-tree artifact are folklore: unverifiable, unrepeatable, and citable in design arguments they cannot defend.

## Evidence

`rg -n "pytest.mark" Voyage/tests` (verified live 2026-09-25); `rg -n
"coverage|cov" Voyage/pyproject.toml Voyage/scripts/*.sh` → only the "Any in
every image" comment.

## Reproduction

`pytest --collect-only -m gpu` → no tests collected.

## Source references

- Files/lines above; `voyage/bench.py`; `voyage/workers/video_longlive.py:
  1013-1025`; `Voyage/reports/video-backends.md`.

## Resolution candidates

1. Mark GPU tests (`longlive/ltxv/causvid/precision` live probes) `gpu`; deselect
   in `test.sh` default + explicit `--run-gpu` path.
2. Add coverage with `/tmp` data file + modest `fail_under` ratchet.
3. Cheapest measurements (no code changes): `benchmark video --profile_stages` per
   backend on idle GPU, commit the JSON; `audio benchmark` 45 s-take numbers into
   `video-backends.md`; extend `test_longlive_stages.py` key-closure to
   LTXV/CausVid; aggregate `director_prefetch_hit/miss` in `soak`.

## Investigation / progress / resolution log

- 2026-09-25: found by standards + perf sweeps (convergent).
- 2026-09-25: repair pass — added `## Why this is an issue`; marker state
  re-verified live (still 0 `pytest.mark.gpu` hits; `parametrize` ×6 +
  `endurance` ×1 at `test_benchmark.py:161` — current).
- Open: markers + coverage first (cheap), GPU benchmark artifacts need idle GPU.
