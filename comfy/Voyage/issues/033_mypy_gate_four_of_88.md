# 033 — Mypy gate covers 4/88 test files; full `mypy tests` is 108 errors red

- Severity: LOW (typing gate scope — regressions outside 4 files invisible)
- Group: standards/typing — Rank: 4/5
- File:line: `Voyage/scripts/gates.sh:28`
- Overlaps: 031/032/034 cluster — same ratchet; 034 lists the wrong-code files blocking this gate.

## Description

The gate runs mypy over `voyage` plus only 4 test modules (53 files, green).
Full `mypy tests` reports ~108 errors in ~25 files, including wrong-code
ignores, invalid type aliases, non-overlapping comparisons, and untyped defs.
The gate comment admits the gap ("append a module path here as each file is
annotated") but there is no ratchet list, owner, or count — regressions
outside the 4 files are invisible.

## Rationale

Mypy docs explicitly support per-module incremental adoption
(`[mypy-package_to_fix_later.*] ignore_errors=True`, then invert). The
current binary split (4 in / 84 out) with no tracking is the worst of both:
strict where easy, silent where hard.

## Live evidence (re-verified 2026-09-30)

`Voyage/scripts/gates.sh:18-28` read live:

```bash
# mypy scope (issue 035): `voyage` plus the converted test modules. ...
mypy voyage tests/conftest.py tests/test_seeds_properties.py tests/test_beat_properties.py tests/test_similarity_properties.py
```

Track-C sweep capture (in-container `mypy tests`, quoted from preserved Task
output `ses_f10013fc5ffeLLDtqZEwFbf3JR`):

```
tests/test_commit_hardening.py:168: Module "voyage.supervisor" does not explicitly export attribute "validate_video" [attr-defined]
tests/test_tui_app.py:532: "object" has no attribute "query" [attr-defined]; note: not covered by "type: ignore[union-attr]"
tests/test_crash_matrix.py:89: … note: not covered by "type: ignore[method-assign]"
tests/test_qualification.py:94,139,140: Variable "…Frame" is not valid as a type [valid-type]
tests/test_tui_state.py:146,151,199: Non-overlapping equality check … [comparison-overlap]
Found 108 errors in 25 files (checked 88 source files)
```

Spot re-verification today: `tests/test_tui_app.py:532` still carries
`# type: ignore[union-attr]` on the `app.query` line; `tests/test_crash_matrix.py:89`
still carries `# type: ignore[method-assign]`.

## Repro

```bash
docker run --rm -v "$PWD:/app" voyage:latest bash -c "mypy tests 2>&1 | tail -n 50"
```

## Fix candidates

(a) Add a `tests/` ratchet file listing converted modules.
(b) Fix the wrong-code ignores first (zero-behavior).
(c) Enable `mypy --warn-unused-ignores` over the full `tests/` tree in CI as
a non-blocking report so the count can only fall.

## Refs

- `Voyage/scripts/gates.sh:18-28`
- Mypy `existing_code.rst` ("Start small … per-module ignore_errors … aim
  for mypy --strict"); `error_codes.rst`.
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §3.
