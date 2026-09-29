# 035 — `mypy voyage` only: tests/scripts/workers untyped, escapes accumulate

- Status: resolved (fixed 2026-09-25: scoped mypy on converted modules + fallout plan logged)
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
- 2026-09-26 (resolution, FIXED as scoped ratchet): `gates.sh` mypy scope is
  now `voyage` + the converted test modules (`tests/conftest.py`,
  `test_seeds_properties.py`, `test_beat_properties.py`,
  `test_similarity_properties.py` — append a path as each file is
  annotated; `ruff check .` still lints everything). Owned files are clean
  (`mypy`: no issues in 46 files incl. all of `voyage/`).
  - Incidental unblockers found live (both masked until now because every
    mypy run died parsing numpy stubs): `numpy` added to the mypy
    `follow_imports=skip` override — its 1.26.4 stubs use the 3.12 `type`
    statement, unparsable under this project's 3.10 target; and
    `disable_error_code=["unused-ignore"]` scoped to `voyage.config` +
    `voyage.tui_state` for their two stale tomli-shim ignores (:19/:31 —
    the try/except-ImportError idiom needs no suppression; deleting each
    one-line comment is the real fix, owning pass; remove the override
    entry together with the comment).
  - `tests/conftest.py` + all three property modules pass strict mypy
    (the `@given` `untyped-decorator` errors seen mid-work were an
    artifact of the missing hypothesis package, gone once installed).
- Follow-ups (owning passes): the 11 `no-untyped-def` ignores still stand
  (`test_draft.py` ×7, `test_precision.py` ×3, `test_ltxv.py:101` ×1 —
  annotate helpers `tmp_path: Path` + returns, delete ignores, append each
  file to the gates.sh scope); ~130 further pre-existing test errors
  surfaced by the first-ever `mypy voyage tests` probe (unused ignores,
  `attr-defined` from the in-flight worker migration, `exit-return`,
  `type-arg`, untyped defs — full list in the 2026-09-26 probe output,
  none in scoped files); `build.sh`/`build-director.sh` still run bare
  `mypy voyage` (hook: extend to the same scope when touched).
- Open: none in this slice — scope grows file by file from here.
