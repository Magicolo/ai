# 154 — `benchmark` has no SFX or augment targets though both SFX workers ship `handle_benchmark`

- **Severity:** MEDIUM (observability gap — the two newest GPU stages have no §104 probe path, so ladder/bench claims about them are unmeasurable)
- **File:line:** `Voyage/voyage/cli.py:1956-1977` (verb parsers around `benchmark`: stop/validate/finalize own SFX flags but benchmark is absent) + `:1416-1441` (`_benchmark_env`/`_check_benchmark_counts` shared helpers) → `Voyage/voyage/cli.py:2011-2032` (`_add_benchmark_parser`: `choices=["video", "audio", "end-to-end"]`) → `Voyage/voyage/workers/sfx_mmaudio.py:254-332` (`handle_benchmark`, implemented) + `Voyage/voyage/workers/sfx.py:85-115` (`handle_benchmark`, implemented) vs `Voyage/voyage/workers/augment_worker.py` + `Voyage/voyage/augment.py` (zero `benchmark` hits)
- **Area:** benchmark CLI surface (below both previous windows; 051 covers bench dropping VRAM/no-percentiles, 060 covers thin env/stdout-only artifacts — neither covers missing targets)
- Overlaps with 163 (sfx-benchmark half — 163 owns the soak + harness-aggregation side; this file owns the CLI-target + augment-worker-op side)

## Description

`voyage benchmark` offers exactly three targets (`cli.py:2015-2016`): `video`, `audio`, `end-to-end`. Both SFX workers already implement the full `handle_benchmark` contract — MMAudio times warmup+measured 8 s windows with VRAM peaks (`sfx_mmaudio.py:254-332`, `validate_benchmark_counts` at `:264`, probe render at `:276-299`, per-window `reset_peak_memory_stats`/`max_memory_allocated` at `:301/:317`), fake times noise windows (`sfx.py:85-115`) — but no CLI path can invoke them: `cmd_benchmark` (`cli.py:1459-1488`) branches on `target in ("video", "audio")` then falls through to end-to-end, so `voyage benchmark sfx` is an argparse rejection (`invalid choice`), not a missing-worker error. The augment stage is worse: neither `augment.py` (orchestration: chunk math, ffmpeg decode/encode, device selection) nor `augment_worker.py` (upscale/interpolate library) mentions `benchmark` at all (`rg benchmark` → zero hits in both files), so the Real-ESRGAN + FILM chunk path — the stage the 046/047/050 files already flag for O(N²) rescan, reload-per-call, and double-encode cost — has no measurement entry point anywhere (no worker op, no CLI target, no `bench.py` report shape).

## Rationale

§104 benchmark reports are the project's mechanism for ladder decisions (2060 vs 4060 residency, small vs large MMAudio, chunk sizes). The SFX ladder (`SFX_DUAL_MODEL_SIZE`, `small_44k` on cuda:1 fitness, 2060 6 GB ceiling at `sfx_finalize.py:283-287`) and the augment chunk tuning (`DEFAULT_CHUNK_FRAMES`, CRF, upscale factor) currently cannot produce a `benchmark <stage>` report at all — operators fall back to wall-clocking finalize runs, which conflates the stages. Shipping `handle_benchmark` in a worker without a CLI target is dead measurement code; shipping a GPU stage (augment) with neither is unmeasurable new surface.

## Live evidence

- `sed -n '2011,2032p' voyage/cli.py` — `choices=["video", "audio", "end-to-end"]`; `--run` required "for video/audio targets" (`:2019-2021`), no sfx/augment branch in `cmd_benchmark` (`:1459-1527`).
- `sed -n '254,332p' voyage/workers/sfx_mmaudio.py` — full benchmark (warmup/measured, probe video, walls+peaks, `vram_peak_gib`/`vram_avg_gib` at `:330-331`); `sed -n '85,115p' voyage/workers/sfx.py` — fake counterpart (`validate_benchmark_counts` at `:89`, `windows_per_second` at `:113`).
- `rg -n "benchmark" voyage/augment.py voyage/workers/augment_worker.py` → no hits (both files); `rg -n "benchmark" voyage/bench.py` → report formatter exists but no augment/sfx report shape feeds it.
- `sed -n '1416,1441p' voyage/cli.py` — `_benchmark_env` + `_check_benchmark_counts` are already target-agnostic shared helpers, so new targets reuse them with no new plumbing.
- Overlap check: 051 is report *content* (VRAM fields dropped, no percentiles — applies once a target runs); 060 is env thinness/stdout-only artifacts (applies to existing targets); 059 is gauge summarization. None names the missing sfx/augment target surface.

## Repro

Deterministic CLI: `voyage benchmark sfx --run <dir>` → `error: argument benchmark_target: invalid choice: 'sfx' (choose from 'video', 'audio', 'end-to-end')`. `voyage benchmark augment --run <dir>` → same rejection. Static: `python -c "from voyage.workers import sfx_mmaudio; assert hasattr(sfx_mmaudio, 'handle_benchmark')"` passes while no CLI path references it (`rg -n "sfx.*benchmark|benchmark.*sfx" voyage/cli.py` → no hits).

## Fix candidates

1. Add `sfx` target: resolve the run's `[sfx]` backend like `cmd_benchmark` resolves video/audio backends (`:1471-1479`), start a single SFX worker via the existing `SFX_WORKER_MODULES` map, call `benchmark`, print via `format_report("sfx", …)` — mirrors the video/audio branch exactly.
2. Add `augment` target in two parts: a `handle_benchmark`-shaped entry in `augment_worker.py` (time upscale+interpolate on synthetic frames with VRAM peaks, reusing `validate_benchmark_counts`) plus orchestration-level chunk reporting (per-chunk walls through `run_augment_chunks`); CLI threads it the same way.
3. Keep choices sorted and update `_add_benchmark_parser` help + `docs/BENCHMARKING.md` (091 notes the docs gap) so the new targets are discoverable; add a CLI test pinning the full choice list so future workers cannot ship headless benchmarks again.
4. Tests: `benchmark sfx` on a fake-config run returns 0 with a `windows_per_second` report; `benchmark augment` CPU path returns 0 torch-free when weights are absent (NotImplementedError → clean skip, not traceback — mirrors `_require_weights`).

## Refs

- `Voyage/voyage/cli.py:1425-1527,1956-2032`; `Voyage/voyage/workers/sfx_mmaudio.py:254-332`; `Voyage/voyage/workers/sfx.py:85-115`; `Voyage/voyage/bench.py`; DESIGN §104 (benchmarks).
- Adjacent, not overlapping: 051 (report content once a probe runs); 060 (env/artifacts for existing targets); 059 (gauge aggregation); 091 (benchmark docs).
