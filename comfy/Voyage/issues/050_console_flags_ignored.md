# 050 — `--verbose`/`--no-color` parsed but ignored on 3 verbs

- Status: resolved (2026-09-25, CLI track — remove + docs, per cheapest candidate)
- Severity: medium (silent flag-dropping; docs claim "two verbosity levels
  everywhere")
- Area: CLI polish — `voyage/cli.py:76-88,1254-1311`, `voyage/console.py:86-108`
- Rank rationale: accepting then ignoring flags is worse than not accepting them.

## Technical description

`_add_console_args()` is attached to
`run/generate/status/validate/finalize/benchmark/soak/inspect`, and OPERATIONS
documents all of them. But `get_console()` is called only in
`cmd_run/cmd_finalize/cmd_generate/cmd_soak` (`cli.py:318,659,827,1006`).
`cmd_status/cmd_validate/cmd_inspect/cmd_benchmark` never call it — flags
accepted then silently dropped. Conversely `pause/resume/stop/doctor/models`
don't accept them at all (fine, but inconsistent with the framing).

## Why this is an issue

Accepting a flag and then ignoring it is worse than not accepting it: users
pass `--verbose` to `status` during an incident, see no additional detail,
and conclude there is nothing more to see — when the information they need
was simply never wired. Scripts that add `--no-color` for log capture get
ANSI escapes anyway on three verbs, breaking parsers. Blast radius is four
CLI verbs; the honest fix is a few lines (wire or remove + docs correction).

## Evidence

Re-verified live 2026-09-25 (correction to the original report — `benchmark`
is a FOURTH silent verb, not a wired one):

```
$ rg -n "get_console" Voyage/voyage/cli.py
68:def get_console(args: argparse.Namespace) -> VoyageConsole:
318:     console = get_console(args)    # cmd_run (:283)
659:     console = get_console(args)    # cmd_finalize (:635)
827:     console = get_console(args)    # cmd_generate (:783)
1006:     console = get_console(args)   # cmd_soak (:997)
$ rg -n "_add_console_args" Voyage/voyage/cli.py | tail -8
1177:    _add_console_args(run)
1251:    _add_console_args(gen)
1256:    _add_console_args(status)
1274:    _add_console_args(validate)
1285:    _add_console_args(finalize)
1296:    _add_console_args(benchmark)
1302:    _add_console_args(soak)
1310:    _add_console_args(inspect)
```

`cmd_status` (:396), `cmd_validate` (:607), `cmd_inspect` (:1028) and
`cmd_benchmark` (:913) accept the flags but contain no `console` reference.

## Reproduction

`voyage status --verbose --run X` vs `voyage run --verbose ...` — only the latter
changes output.

## Source references

- Files/lines above.

## Resolution candidates

Either remove `_add_console_args` from the four silent verbs or wire it (e.g.
`status --verbose` shows config digest + manifest). Cheapest honest fix is the
former + docs correction.

## Investigation / progress / resolution log

- 2026-09-25: found by docs sweep.
- 2026-09-25 (repair): re-verified live — `cmd_benchmark` (:913) contains NO
  `get_console` call, so the silent set is status/validate/inspect/benchmark
  (four verbs, not three); description and candidates corrected. Added
  `## Why this is an issue` + real `rg` output.
- 2026-09-25 (CLI track): FIXED via remove + docs correction. `_add_console_args`
  removed from the `status`/`validate`/`benchmark`/`inspect` subparsers
  (`voyage/cli.py` `build_parser` — wired set is now exactly
  run/finalize/generate/soak, matching the four `get_console` call sites);
  `docs/OPERATIONS.md:39-41` corrected to "honored on run, generate,
  finalize, and soak only". Tests: `tests/test_console.py`
  `test_run_cli_accepts_verbose_and_no_color` narrowed to run-only plus new
  `test_status_cli_rejects_console_flags` (status/validate/inspect reject
  the flags with exit 2). Gates green (ruff + format + mypy + pytest).
