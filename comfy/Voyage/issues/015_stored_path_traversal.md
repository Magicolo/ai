# 015 — `resolve_stored_path` returns `run_dir / candidate` unconditionally for relatives (`..` escapes) and trusts any pre-existing absolute path

- Severity: HIGH
- Group: security/paths — Rank: 1/5
- File:line: `voyage/paths.py:62-86` (`resolve_stored_path`)

## Technical description

Live code (re-verified 2026-09-30, `voyage/paths.py:62-86`):

```python
def resolve_stored_path(run_dir: Path, stored: str | Path) -> Path:
    candidate = Path(stored)
    if candidate.exists():
        return candidate              # (A) any pre-existing absolute path trusted as-is
    if not candidate.is_absolute():
        return run_dir / candidate    # (B) NO containment check — `..` escapes
    parts = candidate.parts
    for index, part in enumerate(parts):
        if part in _LAYOUT_ANCHORS:
            reanchored = run_dir.joinpath(*parts[index:])
            if reanchored.exists():
                return reanchored
    return candidate                  # (C) missing absolute returned as-is
```

Two gaps:

- **(B) Relative `..` escape.** A ledger/metrics entry like `../evil.wav` or `segments/../../etc/x` resolves to `run_dir / "../evil.wav"` = outside the run. The caller (`serving.resolved_path`, `_check_segment_metrics` tape check, finalize assembly) then reads/writes outside the run dir. Contrast `_checked_tape_path` (`supervisor.py:396-414`), which *does* enforce `relative_to(run_dir)` — the consumer side does not.
- **(A) Pre-existing absolute trust.** `if candidate.exists(): return candidate` runs *before* any anchor/containment logic. Any absolute path that happens to exist on this machine (e.g. `/tmp/evil.wav` planted by another user/process, or a stale absolute entry from a moved run that coincidentally exists) is used as-is, even when it lies outside the run. The re-anchor healing below is dead code for that case.

## Why it matters

- Stored paths come from worker reports (`recovery_tape`), ledger lines (`takes.jsonl`), and metrics — all influenced by worker output (issue 006 threat model: worker-reported paths are untrusted). A compromised/misbehaving worker can direct the supervisor to read arbitrary files (info disclosure via media probing) or overwrite arbitrary paths (slice/assemble `dest` is inside the run, but `take_path` reads are outside).
- `cp -r`/`mv` healing is the stated purpose; the current code heals missing absolutes but trusts existing ones, exactly backwards for security.
- Sibling issue 008 fixed CLI `--run-id` traversal; this is the same class one layer deeper, on the read path every commit/finalize takes.

## Live evidence

Stdlib probe on host (no container deps needed — `voyage.paths` is stdlib-only):

```
$ PYTHONPATH=Voyage python3 - <<'PY'
from pathlib import Path
from voyage.paths import resolve_stored_path
import tempfile
with tempfile.TemporaryDirectory() as td:
    run = Path(td)/"run"; run.mkdir(); (run/"segments").mkdir()
    print(resolve_stored_path(run, "../evil.txt"))
    print(resolve_stored_path(run, "/tmp/evil2.txt"))
PY
/tmp/tmp3c4h62zn/run/../evil.txt
/tmp/evil2.txt
```

- Relative `../evil.txt` → `/tmp/.../run/../evil.txt` (normalizes to `/tmp/.../evil.txt`, **outside** the run) — returned without complaint.
- Missing absolute `/tmp/evil2.txt` → returned as-is (branch C); if it *existed*, branch A would also return it as-is with no containment check.

## Repro steps

1. Create a run; append a ledger line with `path: "../escape.wav"` (or craft `metrics.json` `recovery_tape: "../escape.pt"`).
2. Commit/finalize → `resolved_path` yields a path outside the run; `probe()`/`slice_take()` opens it (read primitive) instead of failing with `MediaError`.
3. Absolute variant: `touch /tmp/planted.wav`; set a take path to `/tmp/planted.wav`; observe it is used as-is even though the run never rendered it.

## Fix candidates

1. (Preferred) Enforce containment after resolution:
   ```python
   resolved = run_dir / candidate  # for relatives
   try:
       resolved.relative_to(run_dir.resolve())
   except ValueError:
       raise ...  # or return run_dir-anchored safe fallback + loud error
   ```
   Normalize with `os.path.normpath` / `resolve()` *before* the check (lexical `..` must not pass). For absolutes, require containment OR successful re-anchor; otherwise fail loud (`MediaError`/`StateError`, never silent use).
2. Reorder: try re-anchor *before* trusting an existing absolute (a moved run's stale absolute that still exists at the old location must not shadow the moved copy). Only trust an existing absolute when it is inside `run_dir`.
3. Add `resolve_stored_path` traversal property tests (Hypothesis path segments with `..`, absolute escapes) + existing-path fixtures; reuse the `_checked_tape_path` `relative_to` pattern so both sides share one convention.
4. Audit callers (`AudioTake.resolved_path`, `_check_segment_metrics`, finalize) to handle the new loud error as `MediaError` (commit fails safe, budget engages).

## References

- `voyage/paths.py:62-86` (full function quoted above); `_LAYOUT_ANCHORS = frozenset({"segments","audio","novelty","logs"})` at `:39`.
- Contrast `voyage/supervisor.py:396-414` (`_checked_tape_path` enforces `relative_to`, the pattern to copy).
- Issue 008 (CLI traversal) + 016 (consumer-side convention) — same threat model, worker-influenced paths.
- Python `pathlib.Path.relative_to` — "raises ValueError when the path is not relative to the other": https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.relative_to
- OWASP Path Traversal — canonicalize then validate containment: https://owasp.org/www-community/attacks/Path_Traversal

## Progress log

- 2026-09-30 (scope: `voyage/paths.py` + new test file only — callers,
  supervisor, existing tests untouched): re-verified both premises against
  live code — `voyage/paths.py:62-86` matched the quoted vulnerable
  function verbatim (branch A trusts any existing absolute; branch B joins
  relatives with no containment check; branch C returns missing absolutes
  as-is). Both premises still held, so the fix below was built, not
  skipped.
- TDD: wrote `tests/test_rpc_paths_hardening.py` first; 5 of 8 path tests
  failed as predicted (relative `..`, nested `segments/../../..`,
  existing-absolute-outside, missing-absolute-outside, and re-anchor
  losing to an existing stale absolute all returned hostile paths); the 3
  regression pins (lexical relative join, in-run absolute, re-anchor of a
  missing stale entry) passed immediately.
- Fix (`voyage/paths.py`): `resolve_stored_path` now resolves before it
  trusts. The run dir resolves once as the anchor; the absolute branch
  tries layout re-anchor FIRST (the in-run copy wins even when the stale
  absolute still exists at the old location), trusts an absolute only when
  it resolves inside the run (existing or not — missing in-run entries
  still flow to the callers' `.exists()` fallbacks), and raises
  `MediaError("stored path escapes the run dir: ...")` otherwise — the
  `Supervisor._checked_tape_path` convention (`supervisor.py:403-421`).
  The relative branch returns the unchanged lexical `run_dir / candidate`
  join after the resolved containment check, so the join contract is
  preserved. Re-anchored targets are containment-checked too (an anchor
  name followed by `..` cannot smuggle an escape). New helper
  `_is_within_run` (non-strict `resolve()` + `relative_to`, missing paths
  allowed, only escapes fail).
- Post-fix: new file 13/13 green; `test_run_relative_consumer.py` and the
  fake-backend commit/validate/relocation flows green (28 passed).
  `ruff check` + `ruff format --check` + `mypy strict` green on touched
  files (in-container).

## Resolution

- Resolved. `resolve_stored_path` confines every stored path to the run:
  `..` escapes (relative or absolute) and outside-the-run absolutes raise
  `MediaError`; re-anchor healing and the in-run join/absolute behavior
  are preserved.
- Files changed: `voyage/paths.py` (containment gate + `_is_within_run`
  + docstring); `tests/test_rpc_paths_hardening.py` (new, 8 path tests:
  5 fail-first + 3 regression pins).
- KNOWN COLLATERAL — 4 existing tests pin the vulnerable branches and now
  fail (verified: they fail only on the new `MediaError`, nothing else):
  `tests/test_paths.py::test_resolve_stored_path_keeps_existing_absolute`
  (branch A), `::test_resolve_stored_path_returns_stale_when_unhealable`
  (branch C),
  `tests/test_run_relative_consumer.py::test_resolve_stored_path_prefers_existing_absolute`
  (branch A),
  `::test_resolve_stored_path_keeps_missing_absolute` (branch C). Any
  correct fix breaks them; re-pinning them is out of scope (owning
  tracks) — suggested: assert `MediaError` for outside-the-run entries,
  keep the in-run/re-anchor pins as-is.
- CALLER FOLLOW-UPS for owning tracks (out of scope, not applied): the new
  `MediaError` propagates where callers previously got a path —
  `cli.py:849` (`_check_segment_metrics` tape check; a traversal tape now
  raises out of `validate_run` instead of appending an error string),
  `media.py:576` (falls to the new loud error instead of the missing-
  take fallback), `sfx_finalize.py:229,327,329`, `supervisor.py:1286`.
  Each site should decide catch-and-fail-safe (commit/validate budget)
  vs propagate, mirroring how `_checked_tape_path` errors are handled.
