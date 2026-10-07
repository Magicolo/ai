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
