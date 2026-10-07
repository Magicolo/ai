# 243 — Mutual exclusion done by hand in 5 places instead of `argparse.add_mutually_exclusive_group`; `store_true(default=None)` tri-state + double error prints

Severity: MEDIUM (track B-07).

## Technical description

No `add_mutually_exclusive_group` anywhere in `cli.py`. Conflicts
(`--segments/--duration`, `--low/--medium/--high-definition`,
`--prompt-enhance/--no-prompt-enhance`, `--sfx-dual-pan/--no-sfx-dual-pan`,
`--run/--name`) are hand-checked in `cmd_configure`/`_resolve_segments`/`_extend_plan`/
`resolve_run_ref`. `--prompt-enhance` uses `action="store_true", default=None`
(absent=None, present=True) so `is_provided` works — clever but invisible in `--help`
usage line. Invalid `--segments 0` prints twice on create (`must be positive` from
`_resolve_segments` + `pass one of --segments or --duration` from the caller, since both
map to `None`).

## Rationale

Hand-rolled exclusion duplicates the parser's job, drifts (help shows no `|` group), and
conflates "absent" with "invalid" through a shared `None` return.

## Live evidence

```
NO mutually_exclusive groups in cli.py
"pass only one of" sites: cli_configure.py:72,173,181,187; cli_generate.py:360; cli_paths.py:56
sub = parser.add_subparsers(dest="command", required=False)  # voyage/cli.py:395
```

Repro: `configure --segments 0` → two stderr lines; `configure --prompt-enhance
--no-prompt-enhance` → custom message instead of `error: argument --no-prompt-enhance:
not allowed with argument --prompt-enhance` + usage.

## Source refs

`voyage/cli.py:65-196,237-388,391-400`; `voyage/cli_configure.py:67-85,167-197`;
`voyage/cli_generate.py:339-392`; `voyage/cli_core.py:34-75`.

## Online sources

- `https://docs.python.org/3/library/argparse.html#mutual-exclusion`.
- `https://stackoverflow.com/questions/7869345/how-to-make-python-argparse-mutually-exclusive-group-arguments-without-prefix`.

## Fix candidates

- `add_mutually_exclusive_group()` for each pair/triple (keep `is_provided` for
  hand-built namespaces); make `_resolve_segments` return a `(ok, value, error)` triple
  instead of `None`-for-both-absent-and-invalid; keep `cmd_configure` pre-checks only as
  belt-and-braces for direct callers.

## Log

- 2026-10-07: filed from read-only Track B sweep; no code touched.

## Consolidated from 227_handrolled_mutex_groups (2026-10-07)

227 was a concurrent duplicate of this finding on the same topic (hand-rolled argparse mutual exclusion + duplicate contradictory error prints); 243 is the richer copy. Content unique to 227, preserved:

- Provenance: the absence of any `add_mutually_exclusive_group` was "verified live 2026-10-07: only two `add_parser` hits at `:239,329`" in `voyage/cli.py` — verified live by the orchestrator.
- 227's Live evidence section read "See grep/site list above." (the same grep/site list already reproduced in this file) and its log read: "Track B sweep, 2026-10-07. No-group verified live by orchestrator. Read-only; nothing fixed."

Everything else in 227 (technical description, rationale, repro, source refs, online sources, fix candidates) is already present verbatim-equivalent above.
