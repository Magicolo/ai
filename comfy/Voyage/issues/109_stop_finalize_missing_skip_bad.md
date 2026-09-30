# 109 — `voyage stop --run X --finalize` always crashes: stop parser lacks `--skip-bad`, `cmd_finalize` reads it directly

- **Severity:** HIGH (advertised flag path is 100% broken: every invocation ends in an unhandled traceback, after mutating run state)
- **Track:** second-pass TUI/CLI edges (cross-verb namespace contract)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`, tree as-read; `git status` shows uncommitted concurrent edits in `voyage/cli.py` — lines below are as-read, not HEAD)

## File:line (live-verified)

- `voyage/cli.py:1955-1965` (`_add_stop_parser`: `--run`, `--finalize`, `_add_sfx_args` only — no `--skip-bad`, no augment args, no console args)
- `voyage/cli.py:770-774` (`cmd_stop`: on `--finalize` sets `args.output` and calls `cmd_finalize(args)` with the stop-parser namespace)
- `voyage/cli.py:1001` (`cmd_finalize`: `skip_bad=args.skip_bad` — direct attribute access, no `getattr` fallback)
- Safe siblings: `_augment_overrides` (`:920-936`, all `getattr`), `get_console` (`:135-140`, `getattr` defaults), every `sfx_*` read in `cmd_finalize` (`:975-992`, `getattr`) — `skip_bad` is the only direct access on the stop path

## Description

`stop --finalize` is documented ("run the finalizer inline after requesting stop") and wires SFX flags forward (`_add_sfx_args(stop)`), so the handoff looks complete. But the stop parser never defines `--skip-bad` while `cmd_finalize` reads `args.skip_bad` directly. Every `stop --finalize` invocation therefore raises `AttributeError` inside `finalize_run` kwarg evaluation — after `_set_status` already wrote `STOP_REQUESTED` and after `resolve_config` succeeded. The operator gets a traceback (uncaught: `main` catches only `VoyageError`, `:2043-2047`) instead of a finalized video, on a run whose status was just flipped.

## Rationale

- A flag that crashes on 100% of invocations is worse than a missing flag: `--help` advertises it, the SFX wiring suggests it was tested, and the failure mode (traceback, exit via unhandled exception) violates the exit-code contract every neighboring verb honors (2 = usage, 1 = runtime failure with a message).
- The partial side effect (status flipped, no finalize) is the confusing half: a retry of plain `stop` reports success while nothing was finalized, and the operator must discover `finalize` separately.
- The codebase already solved this exact problem one line down: every other cross-verb read in `cmd_finalize` uses `getattr(args, ..., default)`. `skip_bad` is the single holdout.

## Live evidence (container, 2026-09-30)

```
  $ docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "..."
  init: 0
  status -> STOP_REQUESTED
  Traceback (most recent call last):
    File "/app/voyage/cli.py", line 774, in cmd_stop
      return cmd_finalize(args)
    File "/app/voyage/cli.py", line 1001, in cmd_finalize
      skip_bad=args.skip_bad,
  AttributeError: 'Namespace' object has no attribute 'skip_bad'
  ```
- **Overlaps with:** 138 (`finalize --skip-bad` covers only one of three failure legs — the finalize-side half; this file is the stop-side crash); 022 (same handoff class: inner-namespace slice silently incomplete).

Namespace probe: `build_parser().parse_args(['stop','--run','/tmp/x','--finalize'])` has `skip_bad=False` (hasattr), `min_fps=False`, `no_augment=False`, `verbose=False` — all absent; only `skip_bad` is read directly, so it is the sole crash point (augment/console/SFX reads are all `getattr`-guarded).

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
import argparse
from voyage import cli
ns = cli.build_parser().parse_args(['stop','--run','/tmp/any','--finalize'])
print('has skip_bad:', hasattr(ns, 'skip_bad'))  # False -> cmd_finalize crashes at cli.py:962
"
# End-to-end: init any run, then `voyage stop --run <dir> --finalize` -> status flips, then AttributeError traceback.
```

## Fix candidates

- Minimal: `skip_bad=getattr(args, "skip_bad", False)` at `cli.py:962-963` (matches every neighboring read).
- Structural (preferred, same class as 022): add `--skip-bad` (+ `_add_augment_args`, which `finalize` owns and `stop --finalize` silently drops — a stop-finalize can never set floors today) to the stop parser via the shared helpers, so the finalizing verbs share one flag surface and no `getattr` default can drift from the parser default.
- Test: `stop --finalize` forwarding-contract test asserting the stop namespace satisfies every attribute `cmd_finalize` reads directly (the 022 pattern: enumerate direct `args.X` accesses in `cmd_finalize`, assert presence on all three finalizing parsers).

## Refs

- `voyage/cli.py:1718-1763` (`_add_sfx_args` docstring already states the one-helper rule "so the flags (and their defaults) cannot drift apart across verbs — a missing flag on any finalizing verb is an AttributeError at finalize time" — this issue is that sentence coming true for `--skip-bad`).
- Issue 022 (same handoff class: inner-namespace slice silently incomplete); `clig.dev` ("validate user input… check early and bail out before anything bad happens").

## Progress log

- 2026-09-30 (Group A): premise re-verified live in-container — stop
  namespace still lacked `skip_bad` and `cmd_finalize` still read
  `args.skip_bad` directly (code has since split: `cmd_finalize` lives
  in `voyage/cli_finalize.py:50`, parsers in `voyage/cli.py`).
- Wrote failing test first
  (`tests/test_cli_group_a.py::test_stop_finalize_end_to_end_without_crash`):
  red with the exact `AttributeError: 'Namespace' object has no attribute
  'skip_bad'` after `status -> STOP_REQUESTED`.
- Fixed structurally (the issue's preferred candidate) + defensively.

## Resolution: FIXED

- `voyage/cli.py:594-601` (`_add_stop_parser`): added `--skip-bad` plus
  `_add_augment_args(stop)` and `_add_console_args(stop)` (shared
  helpers — the stop-finalize surface now matches `finalize`; this also
  resolves 179 in the same hunk).
- `voyage/cli_finalize.py:50`: `skip_bad=getattr(args, "skip_bad",
  False)` — belt-and-braces for hand-built namespaces, matching every
  neighboring read.
- Test evidence: `tests/test_cli_group_a.py` 17 passed in-container;
  neighbors (`test_cli_hardening`, `test_cli_split`,
  `test_cli_tui_split`, `test_generate`, ...) green.
- Gates: `ruff check` + `ruff format --check` + `mypy strict` green on
  all touched files.
- Residuals: none.
