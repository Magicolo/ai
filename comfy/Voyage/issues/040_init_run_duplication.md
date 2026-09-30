# 040 — `_init_run` duplication: 121 references, 20+ files, helper adopted by 2

- Severity: MEDIUM (dead-code / convergence debt)
- File: `tests/conftest.py:80-129` (helper) vs private `_init_run` copies in 20 test files
- Area: tests / structure
- Overlaps with: 088 (tests fold 86→65 — convergence execution may absorb this migration; not a duplicate)

## Description

`conftest.py:1-12` documents the convergence plan ("over a dozen test
modules each carry a private `_init_run` copy … New tests should use
`initialize_run_directory` … two converted as proof"). Measured today: 121
`_init_run` references across 20+ files. Only 2 converted per the docstring
— the migration stalled with no ratchet.

## Rationale

The docstring's own argument applies: "a layout change would need thirteen
[now twenty] matching edits" with cosmetic drift already observed
(top-level vs function-level imports, seed 7 vs 11). This is §12 "wrong
abstraction costs less than duplication past ~3x" territory — 20x demands
convergence.

## Live evidence (re-verified 2026-09-30)

Host `rg` today:

```
count: 121  (rg -n "_init_run" tests/*.py | wc -l)
files: tests/test_phase2.py, test_concept_integrity.py, test_commit_split.py,
  test_observability.py, test_av_alignment_consumer.py, test_integration.py,
  test_scoreboard.py, test_commit_hardening.py, test_failure_policy.py,
  conftest.py, test_inspector_wiring.py, test_run_relative_consumer.py,
  test_recovery.py, test_generation_stack.py, test_qualification.py,
  test_state_integrity.py, test_benchmark.py, test_stage_timings.py,
  test_crash_matrix.py, test_finalize_fastpath.py, test_console.py
```

`tests/conftest.py:80-129` read live: `initialize_run_directory` +
`run_directory_factory` (fresh `tmp_path` per call — correctly isolated,
not shared across Hypothesis examples).

## Repro

```bash
rg -n "_init_run" tests/*.py | wc -l
rg -l "_init_run" tests/*.py
```

## Fix candidates

Convert 2-3 files per pass (mechanical: replace body with helper call, keep
seed/style args explicit); add a CI `rg "_init_run" --count` ratchet that
can only fall; delete `run_directory_factory`-bypassing ad-hoc scaffolds.

## Refs

- AGENTS.md §12 pragmatic DRY ("tolerate ~3x duplication before
  abstracting"); pytest `tmp_path` per-test isolation docs.
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §10.

## Progress log (2026-09-30, resolution pass)

- As-read counts drifted: `rg -n "_init_run" tests/*.py | wc -l` → **153**
  across **25 files** (issue said 121 / 20+); test files 116, test fns
  1283. Growth is concurrent-track scaffolds, same pattern.
- Premise update: every remaining `def _init_run` is already a 1–2 line
  delegation to `initialize_run_directory` (verified via `rg -A12`) — the
  full-body duplication the issue describes already converged
  semantically; what remains is wrapper-per-file (distinct default
  `run_id`s) plus one cross-module import (`test_phase2` importing
  `test_recovery._init_run`).
- Fix applied, TDD: wrote `tests/test_init_run_ratchet.py` first with the
  cap at the 147 target → failed as designed (`153 exceeds the ratchet
  147`), structural delegation test passed. Then converted, keeping
  run_ids explicit: (1) folded `tests/test_phase2.py` (3 tests) verbatim
  into `tests/test_recovery.py` under an 088-fold header (its `_tape_run`
  already used recovery's helper — zero semantic change), deleted the
  file; (2) inlined the `timings` wrapper in `test_stage_timings.py`;
  (3) inlined the `concepts` wrapper in `test_concept_integrity.py`;
  (4) inlined the `alignment` wrapper (def + 3 calls) in
  `test_av_alignment_consumer.py`. Deletion safety: `test_phase2` last
  touched 2026-09-23 (mtime + `git log -3` checked), zero importers.
- As-left: **144 references / 21 files**, 116 test files, 1285 fns; cap
  set to 144. New module avoids the literal needle by concatenation so it
  never counts itself.
- Evidence in-container (`voyage:latest`): ratchet + recovery +
  stage-timings + concept-integrity + av-alignment → 23 passed (8.4 s);
  the pre-existing supervisor `ConceptStore legacy_path`
  DeprecationWarning is unrelated (voyage/ scope). `ruff check` +
  `format --check` + `mypy` green on all touched files.

## Resolution

- Resolved (convergence step + ratchet landed). Residual: 144
  references remain — continue 2–3 files per pass per the issue's own
  cadence and lower the cap each time (never raise it); coordinate with
  concurrent tracks adding new scaffolds (their files then fail the
  ratchet by design — converge, don't bump the cap).
