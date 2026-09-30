# 194 — `_benchmark_env` omits the finalize presentation floors (post-060 axis)

- Severity: LOW
- Area: bench harness — setup reproducibility
- Files (as-read 2026-09-30; concurrent uncommitted edits noted in `voyage/cli.py` — citations are as-read values):
  - `voyage/cli.py:1425-1441` (`_benchmark_env`: returns exactly `gpu/torch/cuda_available`)
  - `voyage/media.py:666-678,808-810` (floors: `PRESENTATION_MIN_FPS=24`, `AUGMENT_DEFAULT_MIN_FPS/WIDTH/HEIGHT = 32/1280/720`)
  - `voyage/cli.py:1008-1009` (floors threaded into finalize from `config.augment`)
  - `Voyage/DESIGN.md:7514-7525` (floors entry: knobs `--min-fps/--min-resolution`, `[augment]` TOML, TUI fields)

## Description

`_benchmark_env` records three setup facts — GPU name, torch version, CUDA
availability — and nothing else:

```python
# cli.py:1425-1441 (as-read)
return {"gpu": gpu, "torch": torch_version, "cuda_available": cuda_available}
```

Since 2026-09-30 the single largest determinant of finalize wall-clock and
output bytes is the presentation-floor triple (`min_fps/min_width/min_height`,
defaults 32/1280/720): it gates the stream-copy fast path off
(`plan.needs_reencode`), selects minterpolate-vs-plain-fps, and sets the final
encode geometry — a CausVid 832×480@16 native vs a floored 1280×720@32 differs
by ~2.7× pixels + 2× fps plus the minterpolate stage (091's cost note). None of
it appears in the §104 setup block: not the three floor values, not the
resulting `out_w/out_h/out_fps` plan, not `needs_minterpolate/needs_reencode`,
not the SFX/augment on-off that decides whether FILM/Real-ESRGAN weights even
load. Two benchmark reports taken on either side of a `--min-fps 0` change (or
either side of the floors' introduction) are numerically incomparable with no
field recording why — the exact "unmeasurable ladder claim" class 154/163 file
for SFX/augment targets, here for the floors that already shipped.

## Rationale

Setup blocks exist so reports reproduce. The floors are first-class run config
(TOML section + CLI flags + TUI fields + `AugmentConfig`) with first-order perf
effects; omitting them from the setup block while 091 already notes
"BENCHMARKING e2e numbers pre-floors" leaves the harness reporting numbers
whose dominant variable is unrecorded.

## Live evidence (verified live 2026-09-30, host reads)

- `sed -n '/def _benchmark_env/,/^def /p' voyage/cli.py` — 3-key dict; no
  `min_fps`, no geometry, no joint/sfx/augment flags.
- `sed -n '1000,1010p' voyage/cli.py` — finalize receives
  `min_fps=config.augment.min_fps, min_width=…, min_height=…`: the values exist
  at the call site that also spreads `**_benchmark_env()` (`:1478`, `:1517`)
  but are never added to it.
- `DESIGN.md:7514-7525` confirms the floors' scope (finalize/generate/run +
  TOML + TUI) — all report-relevant surfaces, none captured in env.

## Repro

1. `benchmark end-to-end` on the same run twice, once with default floors and
   once with `--min-fps 0 --min-resolution 0` (or pre/post-floors code).
2. Diff the two `§104` setup blocks → identical (`gpu/torch/cuda_available`);
   diff the measured stage means → widely different (re-encode vs stream-copy).
   No recorded field explains the delta.

## Fix candidates

1. (Preferred) Extend `_benchmark_env` (or a sibling `_finalize_env`) with the
   floor triple + resolved `out_w/out_h/out_fps` + `needs_reencode/
   needs_minterpolate` + sfx/augment enablement — pure data already available
   at both spread sites.
2. Pair with 091's BENCHMARKING refresh (re-baseline e2e numbers *with* floors
   recorded) so old vs new reports are visibly incomparable, not silently so.
3. Test: setup block contains the floor keys; a floors-off report round-trips
   them.

## Refs

- In-tree: `voyage/cli.py:1425-1441,1470-1527` (env + benchmark/soak spreads);
  `voyage/media.py:699-751` (`plan_augmentation` — the resolved plan worth
  recording); `voyage/bench.py:24-37` (`format_report` renders whatever setup
  it is given).
- Not-a-duplicate: 060 is the §104 GPU/setup-field list as of its sweep (thin
  env, stdout-only artifacts) — the floors did not exist then and are a new
  axis, not another §104 row; 051 is timing-math shape (means, no percentiles);
  154~163 is the SFX/augment *target* gap (which workers get benchmarked),
  deduped per brief — this file is the *setup-record* gap for the already-
  benchmarked finalize path. Fix jointly with 060's env work if convenient,
  but the floors axis is independently missing.
