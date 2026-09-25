# 035 — `mypy voyage` only: tests/scripts/workers untyped, escapes accumulate

- Status: open
- Severity: medium-high (typed core, untyped perimeter; 11 invisible ignores)
- Area: standards — `Voyage/scripts/gates.sh:7` (+ `build.sh`, `build-director.sh`)
- Rank rationale: `warn_unused_ignores=true` would flag the 11 test ignores as
  unused if tests were checked — instead they are invisible.

## Technical description

All three gate scripts run `mypy voyage`; Zoomy runs `mypy zoomy tests` with
`mypy_path=["scripts"]` (`Zoomy/pyproject.toml:52-56`). Evidence tests are outside:

- `Voyage/tests/test_draft.py:17,32,37,49,58,64,79` —
  `def _base_config(tmp_path): # type: ignore[no-untyped-def]` ×7
- `Voyage/tests/test_precision.py:20,51,58` — same pattern ×3
- `Voyage/tests/test_ltxv.py:101` —
  `def _ltxv_config(tmp_path: Path): # type: ignore[no-untyped-def]`

`rg -n "no-untyped-def" Voyage/tests` → 11 hits (sweep output).

## Why this is an issue

A typed core with an untyped perimeter rots from the edges: test helpers drift out of sync with the signatures they exercise, and the 11 invisible `type: ignore` comments accumulate precisely because nothing flags them — `warn_unused_ignores` cannot fire on files mypy never sees. Each untyped helper is a small lie about the contract it tests, and the next refactor pays for all of them at once when tests fail for reasons the type checker would have caught.

## Evidence

`rg -n "mypy voyage" Voyage/scripts/*.sh`; `rg -n "no-untyped-def" Voyage/tests`.

Verified live 2026-09-25: all three gate scripts (`gates.sh:7`, `build.sh:6`,
`build-director.sh:8`) run bare `mypy voyage`; `rg -n "no-untyped-def" tests/`
→ 11 hits (`test_draft.py` ×7 at :17,32,37,49,58,64,79; `test_precision.py` ×3
at :20,51,58; `test_ltxv.py:101` ×1). Refs current.

## Reproduction

Add `tests` to the mypy invocation and watch the 11 ignores light up (then delete
them by annotating).

## Source references

- `Voyage/scripts/gates.sh:7`, `build.sh:6`, `build-director.sh:8`; test files
  above; `Voyage/pyproject.toml:43` (`warn_unused_ignores`).

## Resolution candidates

`mypy voyage tests` (+ `mypy_path` if scripts get typed helpers); annotate
`tmp_path: Path` + return types and delete the 11 ignores. Zoomy holds this bar
(no `no-untyped-def` ignores).

## Investigation / progress / resolution log

- 2026-09-25: found by standards sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; gate-script
  invocations + all 11 `no-untyped-def` sites re-verified live, current; pasted
  output into Evidence.
- Open: extend mypy scope + annotate.
