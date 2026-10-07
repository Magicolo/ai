# 298 — `cmd_validate` orphan in `voyage/cli_validate.py:375`

Severity: MEDIUM (pass-2 TUI-remnant sweep).

## Technical description

`validate` verb deleted, but `cmd_validate(args)` remains, resolving
`resolve_run_ref(run=..., name=...)` with a `--run` flag no live parser provides. Library
`validate_run` is live (imported by `cli_generate.py:24` + 12 test modules);
`cmd_validate` has zero callers.

## Rationale

Dead CLI surface + traversal-flag confusion (accepts a `--run` no parser supplies).

## Live evidence

```
$ rg -n "cmd_validate" voyage tests scripts
voyage/cli_validate.py:375:def cmd_validate(args: argparse.Namespace) -> int:
(no other hits; cf. validate_run → 15+ importers)
$ rg -n "set_defaults\(func" voyage/*.py
voyage/cli.py:324: conf.set_defaults(func=cmd_configure)
voyage/cli.py:388: gen.set_defaults(func=cmd_generate)
```

`voyage/cli_validate.py:375-380` body: `resolve_run_ref(run=getattr(args,"run",None),
name=getattr(args,"name",None))`.

Repro: `rg cmd_validate` above; parser list → no `validate`; no caller wires it.

## Source refs

`voyage/cli_validate.py:375-380`.

## Online sources

- None (in-tree parser list is the anchor).

## Fix candidates

- Delete `cmd_validate` (keep `validate_run` + helpers), or re-expose as `python -m
  voyage.cli_validate --run` mirroring `boundary_metrics.main` if a read-only check is
  wanted.

## Log

- 2026-10-07: filed from read-only pass-2 TUI-remnant sweep; no code touched.

## Evaluation

- 2026-10-07 (Group L): re-read live — `voyage/cli_validate.py:375`
  `cmd_validate` still present; `rg cmd_validate voyage tests scripts` →
  definition only, zero callers; `voyage/cli.py` wires only
  `cmd_configure`/`cmd_generate` (no validate verb). `validate_run` stays
  live (imported by `cli_generate.py` + 12 test modules). Issue is LIVE.
  `python -m` wiring is NOT trivial: no parser builder exists for it and
  parser construction lives in `voyage/cli.py` (concurrent scope), so
  delete-only per the issue's first candidate.

## Progress log

- 2026-10-07 (Group L): deleted `cmd_validate`, kept `validate_run` +
  all helpers; removed the two imports it alone used (`argparse`,
  `resolve_run_ref`); all remaining imports still referenced. No test
  imports `cmd_validate` (all import `validate_run`/helpers), so no test
  updates needed. Note: `voyage/cli_validate.py` matches the `cli*.py`
  concurrent-ownership glob, but the issue explicitly scopes it, it
  carries zero concurrent diff, and no forbidden file was touched
  (verified via `git status` before/after).

## Resolution (2026-10-07)

- RESOLVED. Files changed: `voyage/cli_validate.py` only. Verification:
  `rg cmd_validate` over voyage/tests/scripts/docs → zero source hits;
  `py_compile` clean. Left open: nothing (read-only checks remain
  available as `from voyage.cli_validate import validate_run`).
