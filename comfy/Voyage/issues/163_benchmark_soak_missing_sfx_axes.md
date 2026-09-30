# 163 — `benchmark`/`soak` have no SFX axis (workers implement `handle_benchmark`, the harnesses cannot reach it)

- **Severity:** MEDIUM (perf-blindspot — the MMAudio pass is the heaviest finalize stage on CUDA, with a two-worker sharding knob, and no harness measures it)
- **File:line:** `Voyage/voyage/cli.py:2011-2032` (benchmark parser: `video|audio|end-to-end` only) + `Voyage/voyage/cli.py:1459-1527` (`cmd_benchmark`: video/audio branches at `:1465-1488`, e2e at `:1489-1527`) + `Voyage/voyage/cli.py:2035-2043` (soak parser) + `Voyage/voyage/cli.py:1547-1576` (`cmd_soak`: stages + gauges + prefetch, no SFX) vs `Voyage/voyage/bench.py:40-56` (`summarize_gauges`: RSS/disk only); contrast `Voyage/voyage/workers/sfx.py:85` + `Voyage/voyage/workers/sfx_mmaudio.py:254` (`handle_benchmark` exists on both SFX workers)
- **Area:** benchmark/soak harness gap (SFX perf unmeasurable despite a ready worker op)
- Overlaps with 154 (sfx-benchmark half — 154 owns the CLI-target + augment-worker-op side; this file owns the soak + harness-aggregation side)

## Description

Both SFX workers implement the benchmark op (`workers/sfx.py:85`
fake, `workers/sfx_mmaudio.py:254` GPU, each with a `"benchmark"`
serve-map entry), following the same `validate_benchmark_counts` /
walls-and-peak shape the other workers use. But no CLI harness can
invoke them:

- `benchmark` (`cli.py:2011-2032`) accepts `benchmark_target` in
  `("video", "audio", "end-to-end")` only — `sfx` is rejected at
  parse time by `choices=`.
- `cmd_benchmark` (`cli.py:1459-1527`) branches `video|audio`
  (`:1465-1488`, via `supervisor._video`/`_audio`) and `end-to-end`
  (`:1489-1527`, throwaway fake run + `_stage_means` + gauges at
  `:1519-1524`); there is no SFX branch, no SFX worker lookup, no
  window/ledger timing.
- `soak` (`cli.py:2035-2043` parser, `cmd_soak` at `:1547-1576`)
  reports `stages` + `summarize_gauges` + `prefetch` + validate
  verdict — `summarize_gauges` (`bench.py:40-56`) aggregates
  `rss_peak_mb`/`disk_free_gib` only, and no SFX stage (window
  render seconds, join/mix seconds, peak VRAM per model size) is
  collected.
- The generate-time forwarding path (`cli.py:1382-1407`) already
  threads `no_sfx`/`sfx_backend`/`sfx_caption`/`sfx_device`/
  `sfx_model_size`/`sfx_workers` into finalize, and the env helper
  (`cli.py:1431-1440`) plus count guard (`cli.py:1443-1456`) are
  backend-agnostic — so the missing piece is purely harness surface,
  not plumbing.

Consequence: the `--sfx-workers 1|2` sharding decision (small_44k
across cuda:0+cuda:1), the small/medium/large model-size ladder
(2060-fit vs 4060-required), and SFX finalize latency can only be
measured ad hoc — exactly the numbers a benchmark harness exists to
produce, for the pass most likely to dominate finalize wall-time on
CUDA.

## Rationale

Unharnessed perf knobs get set by folklore. The SFX worker already
paid the `handle_benchmark` implementation cost (both variants); the
CLI just never exposes it, so the investment returns nothing and the
sharding/model-size choice has no reproducible numbers behind it.
This is the standard harness-coverage gap: worker op without a
caller.

## Live evidence

- `sed -n '2011,2032p' voyage/cli.py` — `choices=["video", "audio",
  "end-to-end"]`; no `sfx`.
- `sed -n '1459,1527p' voyage/cli.py` — video/audio branch
  `:1465-1488` (`supervisor._video`/`_audio` at `:1483`), e2e branch
  `:1489-1527` (`_stage_means` + `summarize_gauges` at `:1519-1524`);
  `sfx` appears nowhere in the function.
- `sed -n '1547,1576p' voyage/cli.py` — soak metrics are
  `segments_committed`/`stages`/gauges/`prefetch`/`validate_errors`;
  no SFX keys.
- `sed -n '40,56p' voyage/bench.py` — `summarize_gauges` returns
  segments/rss_first/rss_last/rss_delta/disk_first/disk_last; no VRAM
  (see also 059), no SFX.
- `grep -n "handle_benchmark" voyage/workers/sfx.py
  voyage/workers/sfx_mmaudio.py` — `:85` and `:254`, each with a
  `"benchmark"` serve-map entry (`sfx.py:137`,
  `sfx_mmaudio.py:372`).
- Overlap check: 051 is the bench-drops-VRAM shape (report fields,
  not target coverage); 059 is the gauges-drops-VRAM aggregation;
  060 is the benchmark-env thinness. None names the missing `sfx`
  target.

## Repro

Static: `voyage benchmark sfx --run <dir>` → argparse rejects
(`invalid choice`). `grep -n "sfx" voyage/cli.py | sed -n ...` shows
SFX flags only on finalizing verbs (`:1398-1403`, `:1773-1817`),
never in `cmd_benchmark`/`cmd_soak`. Dynamic: run any SFX dub and
time it by hand — no harness reproduces the number.

## Fix candidates

1. Add `sfx` to `benchmark_target` choices plus a `cmd_benchmark`
   branch mirroring the audio branch (load run, resolve
   `config.sfx.backend`, start workers, `call("benchmark",
   {"warmup","measured"})`, `format_report("sfx", setup, metrics)`);
   setup should record `model_size` + `sfx_workers` (the two knobs
   the numbers must discriminate).
2. Soak: add an SFX section (window render mean, join/mix seconds,
   `validate_sfx_ledger` verdict) — cheapest as a post-run pass over
   `audio/sfx/` stems + ledger, no extra renders.
3. E2E: optionally include the SFX pass in the end-to-end throwaway
   (fake SFX is CPU-cheap) so the e2e report covers the full shipped
   path; keep it behind the existing `no_sfx` gate if the throwaway
   must stay music-only.
4. Tests: `benchmark sfx` on a fake run (exit 0 + report contains the
   sfx axis); soak on an SFX-finalized run asserts the SFX section is
   present; `choices` pin so the next backend cannot silently drop
   out.

## Refs

 - `Voyage/voyage/cli.py:1382-1407,1431-1456,1459-1527,1547-1576,1773-1817,1992-2032,2035-2043`;
   `Voyage/voyage/bench.py:40-56`; `Voyage/voyage/workers/sfx.py:25,85,137`;
   `Voyage/voyage/workers/sfx_mmaudio.py:254,372`.
 - Adjacent, not overlapping: 051 (bench report shape); 059 (gauges
   VRAM); 060 (benchmark env).

## Progress log

- 2026-09-30 (Group E2): evaluated live first jointly with 154 (same filing, keep/fold: 154 owns the CLI-target half, this file the soak-only remainder). Premise CONFIRMED as-read post-080 split: `voyage/cli.py:643-646` (choices), `voyage/cli_observe.py:161-239` (`cmd_benchmark` video/audio + e2e, no SFX), `:259-298` (`cmd_soak` stages + gauges + prefetch + validate, no SFX), `bench.summarize_gauges` RSS/disk + per-worker VRAM only. The `bench.py` soak-side shapes (mine) are implemented; all CLI/soak-verb legs are residuals.

## Resolution

- Verdict: SOAK-SIDE SHAPES IMPLEMENTED in `voyage/bench.py` (shared with 154 — see that file); soak-verb legs RESIDUAL (below).
- Changes: `summarize_sfx_windows` (soak SFX section aggregation over plain window records — the post-run pass fix candidate 2 needs, with no extra renders) + `sfx_benchmark_setup` (carries `model_size` + `sfx_workers` for the branch) live in `voyage/bench.py`, tested by `tests/test_e2_bench_sfx_augment_154_163.py` (4 passed, CPU-only, `voyage:latest`).
- Test evidence: shared with 154 (same test file); `test_benchmark.py` green; ruff + format + mypy strict clean.
- DESIGN proposal (quoted text only, for the DESIGN owner — §104/§68): "Soak reports carry an SFX section (window render mean, join/mix rollup via `summarize_sfx_windows`, ledger verdict) collected as a post-run pass over stems + ledger — no extra renders; the end-to-end throwaway stays music-only behind the existing `no_sfx` gate unless the SFX pass is explicitly included."
- Residuals (out of scope, precise — for the CLI owner): (1) `voyage/cli_observe.py:259-298` (`cmd_soak`): add the SFX section (window render mean via `summarize_sfx_windows` over `audio/sfx/` stems + `sfx.jsonl`, plus a `validate_sfx_ledger` verdict); (2) e2e SFX inclusion decision behind `no_sfx`; (3) `choices`-pin test covering the full target list (154 residual 4). No supervisor SFX handle exists yet (see 154) — both residuals block on that seam.
