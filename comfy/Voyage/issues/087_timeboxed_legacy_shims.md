# 087 — Time-boxed / deprecated shims: schedule removal (legacy migration, hard-splice, fps=0, tomli)

- Severity: LOW (compat debt with expiry conditions)
- Area: compatibility shims
- Decision: do not extend; delete on condition, not on sight

## Description

1. `voyage/concepts.py:34-41,125,139-140,156-171` — `LEGACY_MIGRATION_REMOVE_AFTER = "2026-12-31"`, `_migrate_legacy` warns `DeprecationWarning` per use. Threaded only at `supervisor.py:1452`, `cli.py:542,1557`. After date: delete constant + param + method + 3 call args.
2. `voyage/tui_state.py:510-515` — `run_id → name` file migration ("Legacy files (pre-Name-merge) carried run_id/output/final_video"). Companions: `config.py:542 run_id`, `persistence.py:35,104`, `supervisor.py:462`, `models.py:207`; `tui.py:972,997`, `cli.py:1336-1337` triple.
3. `voyage/supervisor.py:852-856` — flat deterministic-backend payload fields ("backward compat"). Goes with `deterministic` director if ever retired; today live (deterministic is explicit opt-out, CPU/offline path).
4. `voyage/media.py:774-782,828-830,896-897,910` — `JointStyle = Literal["blend","hard-splice"]`; `hard-splice` = "legacy behavior, clicks included"; `effective_overlap_fraction()` forces 0; `finalize_run` maps `overlap_fraction<=0 → "hard-splice"`. `blend` is default. Deleting removes literal branch + map + fn + `tests/test_integration.py:103-116`, `test_generation_stack.py:168` pins. Needs product sign-off.
5. `voyage/models.py:211-214` — `RunState.fps` unguarded ("CLI tolerates legacy fps=0 states"). `VideoConfig` now validates positive (`config.py:250`), so 0 is unproducible via validated configs. Tighten to `Field(24, ge=1)` after confirming no stored `fps=0` runs remain.
6. `voyage/config.py:20`, `voyage/tui_state.py:31` + `pyproject.toml:168-180` — stale `tomli` `type: ignore[import-not-found, no-redef]`; override silences `unused-ignore` for whole modules. Delete 2 comments + override block. `try/except ImportError` fallback stays until py310 floor raised.

## Rationale

Each shim is load-bearing until its expiry condition; extending them spreads legacy surface. Tracking expiry in one issue prevents silent permanence.

## Live evidence

- `grep -n "LEGACY_MIGRATION_REMOVE_AFTER\|_migrate_legacy\|legacy_path" voyage/concepts.py voyage/supervisor.py voyage/cli.py`
- `grep -n "hard-splice\|effective_overlap_fraction\|JointStyle" voyage/media.py tests/test_integration.py tests/test_generation_stack.py`
- `grep -n "fps.*legacy\|ge=1\|tomli" voyage/models.py voyage/config.py voyage/tui_state.py pyproject.toml`

## Repro

```bash
grep -rn "LEGACY_MIGRATION\|hard-splice\|fps=0\|tomli" voyage/ tests/ pyproject.toml | head -n 30
```

## Fix candidates

1. Calendar-delete (1) after 2026-12-31 + (2) once pre-merge installs age out; product decision for (4); stored-run scan then tighten (5); delete stale ignores (6) now.
2. Until then: ban new call sites threading `legacy_path`; ban new `overlap_fraction<=0` producers.
3. Gate: `gates.sh` green + `test_concept_integrity`, `test_state_integrity`, `test_integration` green.

## Refs

- `voyage/concepts.py`, `voyage/media.py:774-914`, `voyage/models.py:211-214`, `pyproject.toml:168-180`

## Progress log (2026-09-30, tests-only pass — EVALUATE)

- All 6 premises re-verified live (read-only): (1)
  `LEGACY_MIGRATION_REMOVE_AFTER = "2026-12-31"` (`concepts.py:34`) +
  `_migrate_legacy` (`:162`) still threaded (expiry 3 months out — not
  actionable today); (2) `tui_state.py` run_id→name migration prose still
  live; (3) flat deterministic-backend payload compat
  (`supervisor.py:852-856` region) still live (deterministic is the
  explicit CPU/offline opt-out); (4) `JointStyle hard-splice`
  (`media.py:940`, `effective_overlap_fraction`, `finalize_run` map) still
  live with `blend` default — needs product sign-off; (5) `RunState.fps`
  legacy-0 tolerance (`models.py:211-214`) still live (needs stored-run
  scan first); (6) stale `tomli` ignores (`config.py:20`,
  `tui_state.py:31`, `pyproject.toml:132-138` override) still present but
  the `try/except ImportError` fallback must stay until the py310 floor
  rises (worker image is py3.10).
- Tests-scope triage: NOTHING closable from `tests/` alone. Items 1–5 are
  calendar/product/scan-gated by definition (deleting early breaks
  compat); item 6 needs `pyproject.toml` + `voyage/` ownership (frozen)
  and the floor decision. A tests-only "ban new call sites" lint without
  the owner-enforced gate is advisory text, not a fix. No files changed —
  timeboxed shims need the owner, not a shim.
- Gate evidence: n/a.

## Resolution (2026-09-30, tests-only pass)

- Verdict: accepted (formal accept-residual with rationale — expiry
  conditions, not neglect). Files changed: none. DESIGN proposals: none.
- Rationale: each shim is load-bearing until its stated condition
  (calendar date / install age-out / product decision / stored-run scan /
  py310 floor raise); extending nothing and deleting nothing today IS the
  correct action. Pretending a tests-pass "closed" a calendar gate would
  be dishonest.
- Residuals (exact handoff, per-item owners): (1) concepts owner —
  calendar-delete after 2026-12-31 (constant + param + method + 3 call
  args); (2) tui owner — delete once pre-merge installs age out; (3)
  supervisor/director owner — goes with `deterministic` if ever retired;
  (4) product + media owner — sign-off then remove literal branch + map +
  fn + `tests/test_integration.py:103-116` + `test_generation_stack.py`
  pins in the same commit; (5) models owner — stored-run `fps=0` scan
  then tighten to `Field(24, ge=1)`; (6) toolchain owner — delete 2
  comments + override block when the py310 floor rises (fallback stays).
  Until then: ban new `legacy_path` threads + new `overlap_fraction<=0`
  producers (owner-enforced).
