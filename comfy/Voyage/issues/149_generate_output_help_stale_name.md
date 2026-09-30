# 149 — `generate --output` help still says `output/<run-id>` (stale since the `--name` migration)

- Severity: LOW (help text contradicts actual routing; no behavioral bug)
- Area: config/CLI — help parity after the `--name` migration
- Files (as-read 2026-09-30; concurrent uncommitted `--name` edits in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py` — lines as-read):

## File:line

- `voyage/cli.py:1899` (`generate --output`: `help="run directory (default output/<run-id>)"` — the stale string)
- `voyage/cli.py:1893-1898` (`generate --run-id` legacy + `--name` primary-wins pair — the migration context)
- `voyage/cli.py:1249-1255` (`cmd_generate`: `run_id = _effective_run_id(args)`; default dir is `Path("output") / run_id` where run_id is the EFFECTIVE (name-wins) id)
- `voyage/tui_state.py:72` (`FIELD_HELP["name"]`: "the run lands in output/<name>/ with final.mp4 inside" — the new spelling, contradicting the CLI help)
- `tests/test_generate.py:296-395` (new `--name` tests: routes-to-`output/boba`, wins-over-`run-id`, rejects traversal, defaults-to-`output/<run-id>` when only legacy given)

## Description

After the `--name` migration the effective default is `output/<effective-id>` (name when given, else run-id). The TUI help (:72) and the routing tests (`test_generate_name_routes_to_output_name`, `test_generate_name_wins_over_run_id`) all speak `<name>`, but the CLI `--output` help at :1899 still reads `default output/<run-id>`. A user passing `--name boba` without `--output` reads the help, expects `output/<run-id>` (or wonders which id), and lands in `output/boba`. Docs-vs-behavior drift in the exact line users read when deciding where their video goes.

## Rationale

- Help text is a contract for path-affecting flags: a stale default path misdirects cleanup/archival scripts built from `--help`.
- The migration updated the parser (`--name` flag), the resolver (`_effective_run_id`), the TUI help, and the tests — the one help string that names the default dir was missed.
- Fix is a one-word edit; the test already pins the behavior, so only the prose needs to catch up.

## Live evidence

```
cli.py:1899  gen.add_argument("--output", default=None,
              help="run directory (default output/<run-id>)")
cli.py:1254  output = Path(args.output) if args.output else Path("output") / run_id  # effective
tui_state.py:72  "name": "Run + folder name. ... lands in output/<name>/ ..."
tests/test_generate.py:296-319  --name boba → output/boba/final.mp4
tests/test_generate.py:322-346  --run-id legacy + --name boba → output/boba only
```

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest \
  python3 -m voyage.cli generate --help | grep -A1 -- '--output'
# Shows 'default output/<run-id>' while '--name boba' routes to output/boba.
```

## Fix candidates

- `help="run directory (default output/<name>)"` (or `output/<name|--run-id>` if the legacy alias must stay visible). Keep the `init --output` (required, no default) and `run --run` wording untouched.
- Test: help-text assertion pins the new default spelling alongside the existing routing tests.

## Refs

- `voyage/cli.py:1893-1899`; `voyage/cli.py:129-134` (`_effective_run_id`); `tests/test_generate.py:296-395`.
