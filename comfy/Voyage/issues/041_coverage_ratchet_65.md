# 041 — Coverage ratchet at 65% with GPU paths structurally uncovered + slow full-suite signal

- Severity: LOW
- Files: `Voyage/pyproject.toml:190-203`, `Voyage/scripts/gates.sh:28`
- Area: tests / toolchain
- Overlaps with: 094 (toolchain ruff/mypy ratchet — sibling ratchet, not a duplicate)

## Description

`fail_under = 65` with the comment "measured 78% total with the fake
backends; GPU worker bodies are uncovered by design". The floor sits ~13
points under measured, so a 12-point regression still passes. Meanwhile
gates deselect `gpu` and the full coverage run exceeds the 120 s tool timeout
in this environment — the slowest quality signal doubles as the least
sensitive.

## Rationale

Coverage docs position `fail_under` as a regression tripwire; a floor far
below measured plus permanently uncovered worker bodies means the number can
only catch catastrophic loss. The AGENTS log shows the floor was already
raised once — it should keep tracking measured.

## Live evidence (re-verified 2026-09-30)

`Voyage/pyproject.toml:197-203` read live:

```toml
[tool.coverage.report]
show_missing = true
# Issue 041: modest ratchet — 65% (measured 78% total with the fake
# backends; GPU worker bodies are uncovered by design). Raise toward the
# measured total as worker coverage grows; the floor fails
# `coverage report` on any regression.
fail_under = 65
```

Track-C sweep capture (preserved): collect line `1076/1077 tests collected
(1 deselected)`; full `coverage run -m pytest -q -m 'not gpu'` timed out at
120 s in-container here (signal cost, not failure).

## Repro

```bash
coverage run -m pytest -q -m 'not gpu' && coverage report   # allow >120 s
coverage report | tail
```

## Fix candidates

(a) Raise `fail_under` to measured-minus-small-slack (e.g. 75) now, toward
78+ as worker coverage grows (the file's own plan).
(b) Add `--fail-under` per-directory floors for `voyage/` cores vs workers
so worker growth is visible.
(c) Record per-gate timings to catch suite-bloat.

## Refs

- Coverage.py `fail_under` semantics.
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §11.

## Progress log (2026-09-30, toolchain track)

- Re-measured live in-container (`voyage:latest`, CPU-only,
  `coverage run -m pytest -q -m 'not gpu' -p no:cacheprovider`):
  **TOTAL 10342 stmts / 2519 miss / 76%** (was 78% at issue time —
  tree grew, incl. the half-landed cli/registry decomposition).
  Two attempts: the first died at 86% when its container vanished
  under box contention (concurrent agents' containers active); the
  second ran with `--continue-on-collection-errors` +
  `--ignore` for the two collection-broken in-flight files
  (test_cli_split, test_registry_split — concurrent migration) and
  completed. PROBE_EXIT=1 with **9 failed** (all AttributeError-class
  in foreign in-flight files, e.g. test_augment_models,
  test_cli_validate_handoff, test_generate_ensure) — failures are
  concurrent-track breakage, verified foreign, untouched here; they
  marginally depress the total (failing tests abort mid-file).
- Per-group table from the same run (grounds the per-directory
  proposal): workers 63.8% (2899 stmts), audio 67.9% (408),
  top-level cores mostly 80-100% (supervisor 89.6, media 88.7,
  config 89.8, tui 81.1, doctor 79.6, console 78.3, sfx_finalize
  79.3; 100%: errors/hashing/seeds/__init__; lows: cli_models 28.8%
  (new migration file), model_registry 54.9%, tui_state 63.0%).
- Raised `fail_under` 65 → **73** (pyproject.toml): the issue's own
  formula, measured-minus-small-slack (76 − 3). Slack cut 13→~3
  points — a >3-point regression now fails `coverage report`, while
  3 points of headroom absorb concurrent-tree wobble (the issue's
  literal example 75 assumed measured-78; with measured-76 that would
  leave 1 point under active churn — too tight, so 73). Comment
  updated with the measured total, stmt count, group split, and the
  per-directory pointer. Verified arithmetically against the
  same-day report (75.6 ≥ 73 → passes); the next full gates.sh run
  enforces it for real.
- Per-directory floors (candidate (b)): NOT implemented — coverage.py
  has no native per-dir fail_under, and a bash-parsed
  `coverage report --include=` gate would be fragile under the same
  churn (plus a new script file is discouraged and possibly another
  track's). Concrete proposal recorded: workers ≥ 60, audio ≥ 65,
  voyage top-level (excl. workers/audio/vision) ≥ 85, enforced via
  `coverage json` + a small checker when the tree settles.
- Timings (candidate (c)): full pytest leg is ~7-8 min wall under box
  contention (first attempt ~5 min to 86% before eviction) — confirms
  the issue's ">120 s tool timeout" note; slowest signal stands.
- TDD/regression: the raised floor IS the tripwire — `coverage
  report` in gates.sh fails any regression past 73%.

## Resolution

- Floor raised 65 → 73 with measured-76 evidence and a per-directory
  proposal (workers 60 / audio 65 / cores 85 via coverage-json
  checker, deferred to a settled tree). Timings recorded.
- Files changed: `pyproject.toml` (fail_under + comment). Gate
  evidence: same-day full-run TOTAL 76% in-container (log excerpts
  above); 75.6 ≥ 73 arithmetic pass; foreign 9-failure/2-ignore
  context documented so the next runner isn't surprised. DESIGN
  proposals: none. Residuals: per-dir floors unimplemented (proposal
  above); floor should track upward as worker coverage grows (the
  file's own plan).
