# 024 — Unbounded `--output/--run/--final-video/--video` paths: `--run-id` is traversal-guarded, `--output` is not

- Severity: MEDIUM (integrity/robustness: writes escape the project tree by design)
- Group: security/CLI — Rank: 3/5
- File:line: `voyage/cli.py:83-95` (`resolve_run_dir`), `voyage/cli.py:101-125` (`is_flat_folder_name`), `voyage/cli.py:172-180` (`cmd_init`)

## Description

`init`/`generate` reject `../` in `--run-id` but accept any `--output` (`/tmp/evil-run`, `../../tmp/evil`, `/etc/cron.d/x`). `cmd_init` then `mkdir -p`s and writes `voyage.toml` + `state.json` + manifest there. Same for `--final-video`, `--video`, `--output` in `finalize`/`sfx` (absolute `resolve()`, `mkdir -p`, `shutil.copy2`). No containment check, no "outside `output/`" warning. The TUI is safe only because it hardcodes `output/<name>`.

Path traversal guards must cover the path actually used for writes. Guarding the display name while the write path is free is a classic incomplete-fix shape (same class as the `is_flat_folder_name` work that guarded only the id).

## Rationale

- The guarded field (`--run-id`) is cosmetic once `--output` is free: the attacker/accident vector just moves one flag over.
- `mkdir -p` + write + `copy2` on unbounded paths turns a typo (`--output /`) into tree-scale damage with exit 0.
- One funnel already exists (`resolve_run_dir`, used by every verb) — the fix is a single containment predicate, not per-verb plumbing.

## Live evidence (read + grep, 2026-09-30)

`voyage/cli.py:73-88`:

```python
def _run_dir_arg(value: str) -> Path:
    # Absolute: workers spawn with CWD=run_dir, ...
    return Path(value).resolve()


def resolve_run_dir(value: str) -> Path:
    """Canonical run-dir resolution (issue 008/057: one helper, every verb). ..."""
    return _run_dir_arg(value)
```

`voyage/cli.py:168-175`:

```python
def cmd_init(args: argparse.Namespace) -> int:
    if _check_run_id(args.run_id) != 0:
        return 2
    run_dir = resolve_run_dir(args.output)
    ...
    run_dir.mkdir(parents=True, exist_ok=True)
```

No output guard between the run-id check and the `mkdir`. `cmd_init` guards output? `False`; checks run_id only? `True` (sweep probe, still accurate — no output check was added since).

Sweep probe (preserved Track B result):

```
run-id ../../evil flat? False     # guarded
resolved --output /tmp/evil-run: /tmp/evil-run     # unguarded, used for mkdir+write
resolved --output ../../tmp/evil: /tmp/evil        # escapes cwd
cmd_init guards output? False | checks run_id only: True
```

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from pathlib import Path
from voyage.cli import is_flat_folder_name, resolve_run_dir
print('run-id ../../evil flat?', is_flat_folder_name('../../evil'))
print('output ../../tmp/evil ->', resolve_run_dir('../../tmp/evil'))"
grep -n "resolve()" voyage/cli.py   # every unbounded sink
```

## Fix candidates

1. Single `resolve_run_dir` containment rule: warn/error when the resolved path escapes `output/` (or require `--force` for absolute escapes); implement via `Path.relative_to`.
2. Apply the same predicate to `--final-video`/`--video`/`--output` in `finalize`/`sfx`/`generate` (they all funnel through one or two helpers).
3. Add traversal tests for `--output`/`--final-video`/`--video` mirroring the existing `--run-id` tests.

## Refs

- Issue 008's class (traversal guards must cover the write path, not the display name) — `is_flat_folder_name` docstring at `voyage/cli.py:99-113` states the intent; this issue is the uncovered remainder.

## Progress log (2026-09-30, tests-only pass — VERIFY)

- Concurrent-owner work checked live (read-only; post-split anchors):
  `voyage/cli_paths.py:24` `resolve_run_dir` still bare
  `Path(value).resolve()` (no containment predicate, no `relative_to`,
  no `--force` gate); `voyage/cli_run_ops.py:36-65` `cmd_init` still
  guards `--run-id`/`--name` via `_check_run_id` then `mkdir(parents=True)`
  on the unbounded `resolve_run_dir(output)` with no output check between;
  `voyage/cli_generate.py:104,257-259` (`resolve_run_dir` + `final_video`
  `Path.resolve()` + `mkdir`) and `voyage/cli_finalize.py:125`
  (`output.parent.mkdir`) likewise unbounded. `rg
  relative_to|outside.*output|warn.*output voyage/cli_paths.py
  voyage/cli_run_ops.py voyage/cli_generate.py voyage/cli_finalize.py` →
  zero hits (no predicate landed anywhere on the write path).
- Behavioral cross-check (in-container, `voyage:latest`, CPU-only):
  `tests/test_cli_hardening.py` 45/45 green — the existing `--run-id`
  traversal guards hold, but no `--output`/`--final-video`/`--video`
  containment test exists (none found by name/grep), consistent with the
  fix not landing. The issue's sweep probe semantics still hold:
  `--run-id ../../evil` rejected, `--output /tmp/evil-run` /
  `../../tmp/evil` resolved + used for `mkdir+write` unguarded.
- Verdict: NOT fixed in-tree. No code change here (all fix candidates
  write `voyage/cli_paths.py` + verb modules — frozen `voyage/` scope and
  the concurrent owner's regions; touching them would collide).
  Leave OPEN with this status note.

## Resolution (2026-09-30, tests-only pass)

- Verdict: verified-open (still reproduces, owner regions untouched).
  Files changed: none. DESIGN proposals: none.
- Residuals (exact handoff, CLI-paths owner): single `resolve_run_dir`
  containment rule (`voyage/cli_paths.py:24` — warn/error on escape from
  `output/`, `Path.relative_to`, `--force` for absolute escapes) applied
  to `--output` (`cli_run_ops.py:48`, `cli_generate.py:104`) +
  `--final-video`/`--video`/`--output` in `finalize`/`sfx`/`generate`
  (`cli_generate.py:257-259`, `cli_finalize.py:125`, sfx `Path.resolve()`
  sites) + traversal tests mirroring the `--run-id` tests
  (`tests/test_cli_hardening.py` pattern, 45 tests green as base).

## Progress log (2026-09-30, containment pass)

- Premise re-verified live before touching code (in-container
  `voyage:latest`, CPU-only): `resolve_run_dir('/tmp/evil-run')` →
  `/tmp/evil-run`, `resolve_run_dir('../../tmp/evil')` → `/tmp/evil`;
  source is bare `Path(value).resolve()` (`cli_paths.py:21`); grep for
  `relative_to|outside.*output` across all four verb modules → zero
  hits. Verdict at read time: OPEN.
- Concurrency check: `git status` clean across `voyage/`/`tests/`
  (only untracked `../../tango/Tango` outside the tree) — no concurrent
  hunks in the write path, so the fix is disjoint and safe to land in
  own files (`cli_paths`/`cli_run_ops`/`cli_generate`/`cli_finalize` +
  new tests).
- TDD red: new `tests/test_output_containment.py` (6 tests) failed at
  collection (`ImportError: cannot import name
  'is_outside_output_dir'`) before the fix; green after.
- Policy decision (warn-only — recorded so it is not "strengthened"
  later without context): rejecting outside-`output/` would break the
  documented `/tmp/vdemo` flows (cheat sheet) and 40+ tmp_path-based
  tests that init outside the cwd tree without `--force`. An
  error-with-`--force`-escape was rejected: `--force` currently means
  "allow non-empty dir", and redefining it is a breaking contract
  change for the CLI owner with a migration, not a compatible
  hardening. The warning is the typo guard (a bare `--output /` can no
  longer scatter writes silently with exit 0).
- `--video` (sfx input) deliberately not warned: it is read-only; the
  derived/default `--output` beside it is what gets warned.

## Resolution (2026-09-30, containment pass)

- Verdict: fixed (warn-only containment). Files changed:
  `voyage/cli_paths.py` (+`output_root`/`is_outside_output_dir`/`warn_if_outside_output_dir`),
  `voyage/cli_run_ops.py` (`cmd_init` warns on `--output`),
  `voyage/cli_generate.py` (`--final-video` warns; run dir warns via
  the `cmd_init` funnel), `voyage/cli_finalize.py` (finalize
  `--output` + sfx explicit `--output` warn), plus new
  `tests/test_output_containment.py` (6 tests).
- Test evidence: new 6 + `test_cli_hardening` 45 = 51 passed, no
  regressions; per-file `ruff check` + `format --check` + `mypy`
  green on all 5 files (the new test module is also mypy-strict
  clean, ready for the 033 scope list); `mypy voyage` 63 files clean;
  full suite 1702 passed + 8 skipped with only foreign
  exclusions/failures (untracked `test_augment_contract_166.py`
  format violation, untracked `test_supervisor_proposal_helpers.py`
  collection error against missing `voyage.supervisor_proposal`, 3
  `test_worker_perf_rank2` load-flakes green in isolation with zero
  CLI refs — none touched, per §9).
- DESIGN proposals (quoted, for the owner — DESIGN.md untouched):
  "Unbounded `--output`/`--final-video` paths warn on stderr when the
  resolved target escapes `./output/`; absolute outside-tree paths
  remain legal. `--run-id` traversal stays a hard error (exit 2)."
- Residuals: (1) error-semantics upgrade (reject outside `output/`
  unless `--force`) belongs to the CLI owner with a test migration
  (every tmp_path init would need `--force`); (2) new test module for
  the 033 mypy-scope list once committed.
