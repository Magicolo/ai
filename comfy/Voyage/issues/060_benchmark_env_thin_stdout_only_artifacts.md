# 060 — `_benchmark_env()` omits most DESIGN §104 setup fields; benchmark/soak/qualify artifacts are stdout-only, never persisted

**Severity:** MEDIUM

**File:line:** `voyage/cli.py:1424-1439` (`_benchmark_env`); report sites `voyage/cli.py` (`print(format_report(...))` ×3); `scripts/qualify.sh:34-36`; `docs/BENCHMARKING.md:1-6`

**Overlaps with:** 163 (benchmark/soak missing SFX axes — sibling benchmark-coverage gap; not a duplicate)
- **Description:** `BENCHMARKING.md` requires "exact model revision, checkpoint, resolution, block size, attention window, step count, quantization, VAE mode, GPU model, driver, CUDA, PyTorch, warm-up count, measured blocks." `_benchmark_env()` returns exactly `{gpu (first nvidia-smi line or "unknown"), torch, cuda_available}` — no driver, no CUDA runtime, no model revisions, no geometry/blocks/steps/quantization/VAE/attention. Every report path then `print(format_report(...))` and exits; `qualify.sh` prints `summarize_run` JSON to stdout with no redirect. Rerun and the numbers are gone; `reports/` holds hand-written files with no machine artifact behind them.
- **Rationale:** Unreproducible benchmarks are anecdotes. The repo already learned this (report tables cite pins "quoted from model_registry.py, not measured" and image IDs) but the tooling doesn't capture them automatically, so each report is hand-assembled and drifts.
- **Evidence (re-verified live 2026-09-30):**
```python
# voyage/cli.py:1424-1439 (live; sweep cited 1371-1384, drifted +~50 by concurrent growth)
def _benchmark_env() -> dict[str, object]:
    facts = doctor_probe()
    gpus = facts.get("gpus")
    gpu = gpus[0] if isinstance(gpus, list) and gpus else "unknown"
    ...
    return {"gpu": gpu, "torch": torch_version, "cuda_available": cuda_available}
```
Read `_benchmark_env` body (3 keys). `grep -n "format_report(" voyage/cli.py` → print-only sites, zero writes to `logs/` or `reports/`. `qualify.sh` final `docker run … summarize_run …` has no `> file` / `tee`.
- **Repro:** `./scripts/run.sh benchmark end-to-end --segments 1` → note absent fields (no driver/CUDA/model rev/geometry/quantization); `ls logs/` after — no benchmark artifact.
- **Fix candidates:** Build setup from run config + `model_registry` pins + `doctor.probe()` (driver/CUDA) + worker `health`; `tee` reports to `logs/benchmark-<target>-<ts>.txt` + JSON sidecar; `qualify.sh` writes `reports/qual-<backend>-<date>.json` (gitignored or committed by policy); document artifact paths in `BENCHMARKING.md`.
- **Refs:** `docs/BENCHMARKING.md:1-18`; `reports/video-backends.md:19-29` (hand-recorded env table that should be machine-generated).
