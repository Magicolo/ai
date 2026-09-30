# 051 — `bench.py` measures wall-time means only; VRAM peaks never trended

**Severity:** MEDIUM

**File:line:** `voyage/bench.py:12-21` (`timing_stats`), `40-57` (`summarize_gauges`); producers `voyage/workers/video.py:149-150`, `voyage/supervisor.py:557-599` (`_sample_gauges`)

**Description:**
`timing_stats` returns `count/mean/min/max` — no p50/p95/std, no throughput-vs-duration normalization, no quantization/mode label. `summarize_gauges` trends only `rss_peak_mb` + `disk_free_gib`; worker-reported `vram_peak_gib/vram_avg_gib` (produced by every GPU `handle_benchmark`) are dropped on the floor — soak cannot detect VRAM creep across segments. The fake video worker still reports `"vram_peak_gib": "unknown"` strings (line 149), the exact class `summarize_gauges` docstring says broke aggregation — the string-typed hole persists at the producer.

**Rationale:**
Every VRAM lesson ("record VRAM peaks + per-stage timings for GPU changes — benchmark artifacts, not claims") is unenforceable if the trending helper discards VRAM. Silent fallback to `"unknown"` strings reintroduces the typed-`None` bug one layer up.

**Live evidence (current tree):**
```python
bench.py:12-21: {"count","mean","min","max"}  # no p50/p95/std/vram
bench.py:48-57: rss/disk only; no vram_* keys read or written
bench.py:43-46: docstring admits "unknown"-string breakage history, now None-typed — producer still emits strings
workers/video.py:149: "vram_peak_gib": "unknown",
workers/video.py:150: "vram_avg_gib": "unknown",
supervisor.py:593-596: gauges[f"{name}_{key}"] only for ("vram_free_gib","vram_total_gib") ints/floats — benchmark vram_peak_* never read
```

**Repro:**
```python
from voyage.bench import summarize_gauges
print(summarize_gauges([{"rss_peak_mb":100,"disk_free_gib":10,
  "video_vram_free_gib":12.0,"video_vram_total_gib":15.0}]))
# → {'segments':1,'rss_first_mb':100.0,...}  # vram_* silently dropped (only rss/disk trended)
```

**Fix candidates:**
- Extend gauges to `vram_*` series (first/last/delta/max) with `None`-on-absent typing; replace `"unknown"` producers with `None`.
- Add `p50/p95/std` + `audio_seconds_per_wall_second`-style normalized throughput to `timing_stats` or a `timing_stats_ex` successor.
- Wire worker `stage_ms` (longlive) + finalize stage timings (050) into the soak report.

**Refs:** DESIGN §104 benchmark-report contract; `supervisor.py:557` gauge-sampling intent.

**Overlaps with:** 059 (summarize_gauges drops VRAM — near-duplicate on the gauge half; this file additionally covers timing percentiles. Recommend merging the gauge half into 059, keeping the percentile half here).
