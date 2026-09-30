# 059 — `summarize_gauges()` silently drops the worker VRAM fields the supervisor collects

**Severity:** MEDIUM

**File:line:** `voyage/bench.py:40-56` (drops); `voyage/supervisor.py:564-604` (collects `video/audio/director vram_free_gib + vram_total_gib`); `voyage/cli.py` soak/benchmark report paths

**Overlaps with:** 051 (bench drops VRAM + no percentiles — near-duplicate on the gauge half; recommend merging the gauge half here, keeping percentiles in 051)
- **Description:** `_sample_gauges` records `disk_free_gib`, `rss_peak_mb`, plus per-worker `vram_free_gib`/`vram_total_gib` (when `health` answers). `summarize_gauges` aggregates only `rss_*` and `disk_*` — VRAM never reaches the soak/end-to-end trend report, even though VRAM exhaustion is the #1 GPU failure mode in this repo (16 GB budget, 13–15 GiB peaks, rebuild OOMs).
- **Rationale:** `docs/BENCHMARKING.md:28-34,48-50` promises `peak-VRAM`, `average-VRAM`, and "the soak trend shows growth". The soak report cannot show growth for the quantity that actually OOMs. GPU worker `benchmark` ops do return `vram_peak_gib`/`vram_avg_gib` (e.g. `video_ltxv.py`), but those are per-probe, not trended — the per-segment gauge stream is the only trend source and it is discarded at summary time.
- **Evidence (re-verified live 2026-09-30):** `bench.py` return dict has exactly `segments, rss_first/last/delta, disk_first/last` — zero `vram_*` keys (`voyage/bench.py:40-56`, 57-line file, verified); `grep vram voyage/` shows producers in all video workers + sfx/audio workers and the supervisor collector (`supervisor.py:557-600`, `for key in ("vram_free_gib", "vram_total_gib")`), with no consumer after collection.
- **Repro:** Run any soak with `resource_gauges` events containing `video_vram_free_gib`, then `./scripts/run.sh soak …` → printed `soak` report has no `vram_*` lines. Unit-level: `python3 -c "from voyage.bench import summarize_gauges; print(summarize_gauges([{'rss_peak_mb':1,'disk_free_gib':2,'video_vram_free_gib':3}]))"`.
- **Fix candidates:** Extend `summarize_gauges` with `vram_first/free-min` per worker (min-free = closest-to-OOM) + `vram_workers_reporting`; surface in `cmd_soak`/`cmd_benchmark end-to-end` reports; add an alert threshold line (see 061/066).
- **Refs:** `docs/BENCHMARKING.md:24-34`; `reports/video-backends.md` (`peak_vram_bytes` measured but not trended by soak).

## Progress log

- 2026-09-30 re-verified live (container `voyage:latest`): `summarize_gauges([{'rss_peak_mb':100,'disk_free_gib':10,'video_vram_free_gib':12.0,'video_vram_total_gib':15.0}])` returns exactly `segments/rss_*/disk_*` — zero `vram_*` keys. Premise CONFIRMED, no drift from the filed `bench.py:40-56` lines.
- TDD: wrote `tests/test_observability_rank2.py` first (15 tests) — 14 failed as designed (all gauge/VRAM assertions + `timing_stats_ex`/`partial_segment_ids` missing), 1 passed (legacy `timing_stats` shape pin).
- Fix in `voyage/bench.py` only (no `cli.py`/`supervisor.py`/`workers` touches per concurrency rule): new `_finite_float` guard (skips `"unknown"`/None/bool/non-finite, issue-071 precedent) hardening `rss`/`disk` plus the new VRAM series; per worker in `("video","audio","director")` the summary now carries `{worker}_vram_free_first/last/min_gib` (min-free = closest-to-OOM) + `{worker}_vram_total_gib` (last known) + `vram_workers_reporting` (workers with ≥1 numeric free sample); new `timing_stats_ex` successor (`count/mean/min/max/p50/median + p95 nearest-rank + population std`, `timing_stats` frozen for backward compat).
- Gates (container): `ruff check` + `ruff format --check` + `mypy` clean on `voyage/bench.py` + new test file; `pytest tests/test_observability_rank2.py tests/test_benchmark.py tests/test_scoreboard.py tests/test_console.py` → 41 passed.

## Resolution

- Fixed in `voyage/bench.py` (`_VRAM_WORKERS`, `_finite_float`, `timing_stats_ex`, extended `summarize_gauges`) with coverage in `tests/test_observability_rank2.py` (`test_summarize_gauges_trends_vram_per_worker`, `test_summarize_gauges_skips_unknown_strings`, `test_timing_stats_ex_reports_percentiles`, `test_timing_stats_stays_backward_compatible`).
- Residuals (other tracks): `"unknown"` producers in `voyage/workers/video.py:149-150` untouched (workers track owns the `None` swap; bench now skips them so aggregation never breaks); surfacing the new `vram_*` keys in `cmd_soak` / `cmd_benchmark end-to-end` reports untouched (`voyage/cli.py` owned by another group).
- DESIGN patch proposals (text only, no doc edits): §104 benchmark-report contract gains `vram_free_min_gib` per worker + `vram_workers_reporting` in the soak trend, and `timing_stats_ex` (`p50/p95/std`) as the percentile source for stage timings; §68 gauge section notes `"unknown"` strings are skipped at summary time (producer fix migrates them to `None`).
