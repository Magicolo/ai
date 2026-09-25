# 088 — 13 copy-pasted `_init_run` scaffolds across test modules (drift already visible)

- Status: open
- Severity: low (persistence-format change must be edited 13 times; no
  `conftest.py`)
- Area: tests — 13 sites (see list)
- Rank rationale: pass-2 finding; test-scaffold duplication is distinct from
  019's worker triplication.

## Technical description

13 sites (`rg -n "def _init_run" tests/*.py` → 13 hits, verified live by
orchestrator 2026-09-25): `test_qualification.py:172`,
`test_generation_stack.py:86`, `test_recovery.py:25`,
`test_inspector_wiring.py:43`, `test_integration.py:24`,
`test_benchmark.py:30`, `test_scoreboard.py:14`, `test_stage_timings.py:25`,
`test_state_integrity.py:24`, `test_observability.py:31`,
`test_console.py:30`, `test_crash_matrix.py:28`, `test_failure_policy.py:38`.
All do the same mkdir/write-toml/load-config/write-manifest/write-state/
concepts sequence with already-diverged signatures (`run_id` defaults
`benchmark`/`console`/`crash-matrix`/…, `inspector` kwarg variants, style
overrides). No `conftest.py` (confirmed absent) and no shared helper.

## Why this is an issue

A persistence-format change must be edited thirteen times across divergent
signatures — and drift is already visible (`run_id` defaults, `inspector`
kwargs, style overrides differ per copy), so the copies are no longer even
identical. With no `conftest.py` or shared helper to converge on, each new
test module adds a fourteenth copy and the next format migration gets more
expensive instead of cheaper.

## Evidence

`rg` count above.

## Reproduction

`rg -n "def _init_run" Voyage/tests/*.py`.

## Source references

- Files/lines above.

## Resolution candidates

Single `_init_run` in a new `tests/helpers.py` (or `conftest.py` fixture) with
keyword options; keep per-module wrappers only where behavior differs.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep; count re-verified live.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  all 13 sites live (every `file:line` matches `rg` output; no `conftest.py`).
- Open: extract helper.
