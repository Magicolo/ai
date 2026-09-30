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

## Progress log (2026-09-30, toolchain track)

- Re-verified premises live in-container (`voyage:latest`, CPU-only):
  full `mypy tests` now reports **140 errors in 38 files (checked 124
  source files)** — drifted from the preserved 108/25/88 (tree grew to
  ~125 test files under concurrent churn). Error-file histogram
  re-captured (top: test_media_memory.py 20, test_failure_policy.py 14,
  test_causvid_worker.py 14, test_commit_split.py 12,
  test_issue_014_embed_restore.py 10, …). Five newly-erroring files are
  concurrent-track in-flight work (test_cli_split, test_registry_split,
  test_single_source, test_longlive2_deprecation_079,
  test_media_augment_unified_083).
- Candidate selection: all on-disk tests/*.py minus the 38 error files,
  minus empty tests/__init__.py, minus the 4 already-gated files, minus
  untracked in-flight files (test_ltxv_stage_ms.py,
  test_init_run_ratchet.py, test_worker_validators_unified_084.py —
  clean but owned by the concurrent track; gating untracked files would
  couple this change to their uncommitted renames), minus 150's four
  files (all carry errors anyway: 014 ×10, 029 ×4, 030 ×7,
  director_models_dir ×1 — no 150 conflict). Net: **78 candidates**.
  One mid-verification casualty: test_repaint_similarity_gate.py passed
  the full-tree run but failed the gate-context run
  (`AudioConfig has no attribute repaint_similarity_threshold`) — a
  concurrent agent edited it between the two runs; dropped from the
  list and recorded here.
- Strong verification (TDD: the gate list IS the regression test — any
  future annotation regression in these files fails gates.sh): ran the
  exact proposed command in-container —
  `mypy voyage <4 existing> <78 candidates>` → **Success: no issues
  found in 133 source files**. The gates.sh mypy file set was then
  diffed against the verified set: **SETS IDENTICAL** (order differs,
  mypy is order-insensitive).
- Expanded `scripts/gates.sh`: 4 → 82 test modules in the mypy
  invocation (5-per-line wrapped, sorted), comment rewritten to state
  the new scope (package + 82), the exclusion rule (untracked/in-flight
  files stay out until committed AND clean), and the ratchet rule
  (append as each file is annotated). `bash -n scripts/gates.sh` clean;
  the pre-existing gate scope (`mypy voyage` + 4 files) re-verified
  green first (55 source files, includes this track's 034/038 edits).
- Scope note: no test file *contents* were edited (gate-membership via
  gates.sh only, per track contract; test restructuring belongs to
  039/040/086/088/089).

## Resolution

- Gate expanded 4 → 82 test modules (78 newly gated, all verified
  clean in the exact gate configuration). Regressions in any of the 82
  now fail `gates.sh` — this is the TDD ratchet artifact for 033.
- Files changed: `scripts/gates.sh` only. Gate evidence: `mypy voyage
  + 82 files` → Success, 133 source files, in-container; `bash -n`
  clean; set-equality proof vs the verified file list. DESIGN
  proposals: none. Residuals: 38 error files stay out (list above —
  includes all four 150-owned files, owned by their passes); 3 clean
  untracked files await their owners' commits; full `mypy tests`
  still 140 errors (owning passes' scope).
