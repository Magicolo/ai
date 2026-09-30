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

## Progress log (2026-09-30, follow-up evaluation — warn-only vs hard error)

- Follow-up evaluated live against the batch-10 evidence (read-only;
  no behavior change): `voyage/cli_paths.py:97-127`
  (`is_outside_output_dir` pure `relative_to` predicate +
  `warn_if_outside_output_dir` warn-only returning Bool);
  `voyage/cli_run_ops.py:54-59` (`cmd_init` warns on `--output`,
  then `mkdir(parents=True)`); `voyage/cli_generate.py:108-109,262-267`
  (run dir via the `cmd_init` funnel, `--final-video` warns only when
  explicit); `voyage/cli_finalize.py:36-39,124-129` (finalize
  `--output` + sfx explicit `--output` warn; `--video` read-only never
  warned, derived defaults never warned).
- Batch-10 premise re-confirmed in-container (`voyage:latest`,
  CPU-only): `output_root()` tracks the cwd (`/app/output`);
  `is_outside_output_dir('output/run')` False,
  `is_outside_output_dir('/tmp/vdemo')` True with a stderr warning and
  a True return; `tests/test_output_containment.py` + 
  `tests/test_cli_hardening.py` 51/51 green; per-file `ruff check` +
  `format --check` + `mypy` green on all 5 scope files (unchanged).
- Hard-error migration cost (why path (a) is out of scope and over
  the >3-break rule): `--force` today means exactly "allow init into
  a non-empty directory" (`voyage/cli.py:289` init, `:530` generate —
  outside this task's file scope, as are `docs/OPERATIONS.md` and
  every test module but `test_output_containment.py`). Broadening it
  to also mean "allow outside `./output/`" is a breaking contract
  change needing parser help-text + OPERATIONS + test migration.
  `finalize` (`cli.py:612-625`) and `sfx` (`cli.py:628-644`) have NO
  `--force` flag at all, so a `--force`-gated error is unimplementable
  there without new parser surface — and an ungated error breaks the
  documented `finalize --run /tmp/vdemo --output /tmp/vdemo/final.mp4`
  publish flow (`README.md:81`) with no escape. Call sites that assert
  exit 0 on outside-tree targets without `--force` and would flip to
  exit 2 under a hard error include `test_cli_hardening.py`
  (`_init_fake_run` x9 on `tmp_path/"run"`, `test_init_accepts_absolute_output`
  on `tmp_path/"sub"/"run"`, `test_init_then_run_with_relative_paths`
  on `rel-run` outside `tmp_path/output`), `test_generate.py`
  (`_generate_args(tmp_path/"run")` end-to-end + nonempty-guard),
  `test_generation_stack.py:233-251`
  (`generate --output tmp_path/"run"` asserts exit 0 + `final.mp4`),
  `test_generate_ensure.py:297-301` (same shape),
  `test_cli_group_a.py` (3 `main init` + 2 direct `cmd_init` on
  `tmp_path/"run"`), `test_cli_run_ops_pruning.py:28-36` (2 direct
  `cmd_init`), `test_cli_validate_handoff.py:116`,
  `test_sfx_finalize.py:145-159` (parser accepts `/tmp/x.mp4`) —
  far above the >3-unrelated-break threshold. No TDD red was written:
  no behavior changes, so no failing test was needed.

## Resolution (2026-09-30, follow-up evaluation)

- Verdict: decided-not-changed (warn-only is the binding accepted
  behavior; the error-upgrade follow-up is closed). Files changed:
  this issue file only (append-only; no `voyage/` or `tests/` edits).
- Accepted behavior (exact enumeration — all legal, all exit 0):
  (1) `init --output` outside `./output/` (absolute `/tmp/vdemo`,
  `tmp_path/run`, relative `rel-run` resolving outside) warns on
  stderr and proceeds — documented `/tmp/vdemo` quickstart +
  tmp_path test isolation, and the path is user-explicit so the typo
  guard is the warning; (2) `generate --output` outside warns via the
  `cmd_init` funnel and proceeds — same reason, pinned by
  exit-0 end-to-end tests; (3) `generate --final-video` outside warns
  (explicit only) and proceeds — user-explicit publish path;
  (4) `finalize --output` outside warns and proceeds — the README
  publish flow has no `--force` escape to gate on; (5) `sfx --output`
  explicit-outside warns and proceeds, `--video` (read-only) never
  warns, derived defaults beside the input never warn; (6) targets
  under `./output/` (TUI `output/<name>`, generate defaults,
  contained inits) stay quiet; (7) `--run-id`/`--name` traversal
  (`../`, slashes, reserved names) stays a hard error (exit 2, no
  writes) — the display-name guard has no legitimate absolute-path
  use, unlike write targets. `--force` keeps its single narrow meaning
  ("allow init into a non-empty directory") and is NOT an
  outside-tree escape.
- Test evidence: `test_output_containment.py` 6 + `test_cli_hardening.py`
  45 = 51 passed (in-container `voyage:latest`, CPU-only); per-file
  `ruff check` + `ruff format --check` + `mypy` green on
  `cli_paths.py`/`cli_run_ops.py`/`cli_generate.py`/`cli_finalize.py`/
  `test_output_containment.py` (all unchanged).
- DESIGN proposal (quoted, one line, for §58):
  "Outside-tree `--output`/`--final-video`/`--output` (finalize/sfx) targets are legal and warn on stderr; only `--run-id`/`--name` traversal is a hard error (exit 2)."
- Residuals: none open on this issue — the upgrade is rejected, not
  deferred. A future revisit requires a CLI-owner migration proposal
  (new or redefined flag + parser help + OPERATIONS contract + every
  tmp_path/finalize publish-path migration) and is out of scope here.
