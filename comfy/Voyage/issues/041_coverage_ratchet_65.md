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
