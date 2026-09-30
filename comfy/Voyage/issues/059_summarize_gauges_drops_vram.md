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
