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
