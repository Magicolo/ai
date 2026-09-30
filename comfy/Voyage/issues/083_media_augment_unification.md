# 083 — Unify `media.py` finalize vs `augment.py` vs `augment_worker.py` spike

- Severity: HIGH (structure / correctness-adjacent)
- Files: `voyage/media.py:1` (1116L; `finalize_run` 259L, `build_final_audio` 164L, `plan_augmentation` 52L), `voyage/augment.py:1` (281L), `voyage/workers/augment_worker.py:1` (454L spike)
- Area: finalize/augment seam

## Description

Plan math lives in two places with a self-admitted TODO: `augment.py:10-12` "TODO (unify): if media.py ever gains chunk/frame-count plan math, move interpolated_frame_count there" vs `media.py:699` `plan_augmentation` frame math (`(n-1)*m+1` at `augment.py:71-75`). `finalize_run` carries a 12-scalar overload (`width=768,height=432,fps=24` pre-augmentation defaults — a trap under ≥1280×720@32 floors) alongside `options=` (`FinalizeOptions:807`, `blend` default; `hard-splice` = legacy). `augment_worker.py` is a vendored-minimal `RRDBNet` + `FilmNetMini` spike whose docstring admits official `film_net` weights will NOT load (`ModelCompatibilityError`) and anime_6B needs its own loader — dead arch shipped as the augment path.

## Rationale

Two plan-math homes drift (floors ×5 already — see 084). Scalar overload with stale geometry defaults invites a 768×432 call under 32fps floors. Spike arches mistaken for shipped path waste every future augment debug.

## Live evidence

- `voyage/augment.py:10-12` TODO; `:71-75` `(n-1)*m+1`; `DEFAULT_CHUNK_FRAMES=32`
- `voyage/media.py:676-678,802-810,911-913` floors; `:670` `PRESENTATION_MIN_FPS=24` vs 32fps floor (`plan_augmentation:736` max() keeps both); `:857-914` 12-scalar overload + `options=`
- `voyage/workers/augment_worker.py:68` `_require_torch`, vendored arches, `ModelCompatibilityError` branches
- `tests/test_augment_plan.py` (11), `test_augment_runner.py` (39, only 2 torch smokes), `tests/test_augment_runner.py:90-103` pins the TODO'd function

## Repro

```bash
grep -n "TODO.*unify\|interpolated_frame_count\|plan_augmentation\|PRESENTATION_MIN_FPS\|ModelCompatibilityError" voyage/augment.py voyage/media.py voyage/workers/augment_worker.py
grep -n "def finalize_run" -A 15 voyage/media.py | head -n 25
```

## Fix candidates

1. Single plan-math home (move `interpolated_frame_count` into `media.py` or extract `augment_plan.py` both import; delete TODO + duplicate).
2. Make `options=` the only `finalize_run` knob; keep scalars as a tested shim until ~10 `finalize_run(run_dir, out)` call sites migrate; fix `768/432/24` defaults or drop them.
3. Promote-or-quarantine `augment_worker.py`: either land full upstream FILM port + anime_6B loader (deletes stand-in branches) or move to `experimental/` with spike contract.
4. Gate: `gates.sh` green + augment quad (`test_augment_config/plan/models/runner`) green.

## Refs

- Issues 042 (floors vs README), 031-analog plan tests; `voyage/config.py:375-390` `AugmentConfig`

## Progress log

- 2026-09-30 (this resolution): re-verified every premise live
  (`voyage:latest`, CPU-only): `augment.py:10-12` TODO present,
  `interpolated_frame_count` (`augment.py:72-76`, `(n-1)*m+1`) with
  `media.py` carrying no copy (two plan-math homes, conceptual not
  literal duplication); `finalize_run` 12-scalar overload + `options=`
  (`media.py:1049-1066`, defaults `768/432/24`); `augment_worker.py:15-19`
  spike admission (`ModelCompatibilityError` on official weights);
  `FINALIZE_CRF_*` mirroring `CRF_*` with a stale "so media stays
  stdlib-only without importing" comment (`augment` IS stdlib-only —
  verified imports: math/os/subprocess/collections/concurrent/dataclasses
  + `voyage.errors` only, no cycle either direction).
- Defaults-trap evaluation (do NOT change): `test_finalize_fastpath.py`
  pins `finalize_run(run_dir, out, min_fps=0, min_width=0, min_height=0)`
  → stream-copy at 768x432@24 (zero-floor callers depend on the legacy
  defaults for the fast path). Raising defaults to 1280x720@32 would flip
  those callers to re-encode and break the test — verified by reading the
  test (`"Legacy native path: floors disabled so 768x432@24 fake segments
  stream-copy"`). Decision: keep `768/432/24`, document as legacy shim.
- TDD failing-first (`tests/test_media_augment_unified_083.py`, 3 tests):
  watched `AttributeError: module 'voyage.media' has no attribute
  'interpolated_frame_count'` in-container, then green.
- Implemented: (1) single plan-math home — `media.py` re-exports
  `interpolated_frame_count` (`is`-identical to `augment`'s) and aliases
  `FINALIZE_CRF_MINIMUM/MAXIMUM` to `CRF_MINIMUM/MAXIMUM` (one codec
  ladder); `augment.py` TODO closed with the single-home rule. (2) one
  knob contract — new pure `ResolvedFinalizeSettings` +
  `resolve_finalize_settings()` owns the scalar/`options=` split
  ("explicit scalar wins, `None` means use `options`"); `finalize_run`
  delegates (no behavior change) and its docstring declares `options=`
  canonical + scalars legacy shim + why defaults stay. (3) quarantine —
  `augment_worker.py` docstring gains a QUARANTINE header (spike
  stand-in, orchestration is the shipped path, full port or
  `experimental/` move still open); no code moved.
- Lesson (do not regress): `ruff check --fix` deletes bare re-export
  imports (F401) — the `from m import x as x` self-alias idiom marks
  intentional re-exports (this pass's `media.py` line was pruned once,
  restored with the idiom, gates clean after).
- Gates (touched files only): `ruff check` + `format --check` + `mypy
  strict` clean on `media.py`/`augment.py`/`workers/augment_worker.py`/
  new test; suites green: new 3 + `test_augment_plan/runner/config` +
  `test_finalize_fastpath` + `test_integration` (111 passed, 3 skipped).

## Resolution

- Verdict: FIXED (safe subset; spike promotion lands as quarantine).
- Files changed: `voyage/media.py` (+re-exports, CRF aliases,
  `ResolvedFinalizeSettings`/`resolve_finalize_settings`, docstring),
  `voyage/augment.py` (TODO→single-home rule),
  `voyage/workers/augment_worker.py` (QUARANTINE header),
  `tests/test_media_augment_unified_083.py` (new, 3 tests).
- Residual (precise): full upstream FILM port + anime_6B loader (deletes
  stand-in branches) or `experimental/` move with spike contract;
  `options=`-only finalize (drop the 12 scalars after the ~10
  `finalize_run` call sites migrate — `cli.py:1140` already passes
  explicit geometry); defaults raise to presentation floors (blocked by
  the zero-floor fast-path contract — needs a `test_finalize_fastpath`
  rewrite first).
