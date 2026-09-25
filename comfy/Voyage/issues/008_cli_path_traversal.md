# 008 — CLI path traversal: `--run-id`/`--output`/`--run`/`--final-video` unvalidated

- Status: resolved (2026-09-25, CLI track)
- Severity: major (security / robustness)
- Area: supply chain + CLI input validation
- Rank rationale: writes outside the run tree from a crafted flag; the TUI
  already has the right check, the CLI doesn't use it.

## Technical description

- TUI validates `name` with `_flat_folder_name` (rejects `/`, `\`, `..` —
  `voyage/tui_state.py:118-121`), but CLI `cmd_init` (`Path(args.output)` raw,
  `voyage/cli.py:101-102`) and `cmd_generate` (`Path("output")/args.run_id`, no
  check) don't. `--run-id ../../tmp/evil-run` escapes; `--output /etc/cron.d/x`
  + `--force` writes `voyage.toml`/`state.json` anywhere the container user can
   write; `--final-video` in `cmd_finalize` (`Path(args.output)`, `cli.py:638`)
   likewise unchecked.

## Why this is an issue

A crafted `--run-id` or `--output` writes `voyage.toml` and `state.json`
outside the run tree — anywhere the container user can write — so one
unsanitized variable in a wrapper script escapes the output directory and can
clobber unrelated files. The TUI already rejects `/`, `\` and `..`, which
makes the CLI strictly weaker than its sibling interface for the same
operation. Since runs are routinely created programmatically
(`generate --run-id` from scripts and launchers), this is a live
confused-deputy gap rather than a theoretical one. The cost lands on whoever
shares the host or reuses run directories when a stray traversal silently
scatters state.

## Evidence

```
$ python3 -c "from pathlib import Path; print((Path('output')/'../../tmp/evil-run').resolve())"
/home/goulade/Projects/ai/tmp/evil-run
```

Escapes `Voyage/output/` to a sibling of the repo checkout.

## Reproduction

1. `voyage init --output output/../../tmp/evil-run --run-id x --style y --force`
   → writes outside the project tree.
2. `voyage generate --run-id ../../tmp/evil-run ...` → run dir outside `output/`.

## Source references

- `voyage/cli.py:101-102,638,793`; `voyage/tui_state.py:118-121` (good pattern);
  `voyage/supervisor.py:154-179` (run-dir consumers).

## Resolution candidates

1. Apply `_flat_folder_name` (or equivalent) to `--run-id`/TUI `name` at the CLI
   layer; resolve + constrain `--output/--run/--final-video` (warn/refuse escapes).
2. Centralize in one `resolve_run_dir()` helper used by every verb (also fixes 057).
3. Tests: traversal `--run-id` rejected; absolute `--output` inside tree accepted.

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep; traversal probe re-verified live.
- Open: implement + tests.
- 2026-09-25 (repair pass): added `## Why this is an issue`; traversal probe
  re-run live (escapes `Voyage/output/`); refs verified current
  (`cli.py:101-102,638,793`; `tui_state.py:118-121`).
- 2026-09-25 (CLI track): FIXED. `voyage/cli.py` gained `resolve_run_dir()`
  (:70, canonical alias of `_run_dir_arg` — new code calls this),
  `is_flat_folder_name()` (:80, mirrors TUI `_flat_folder_name` locally;
  importing the TUI would be circular since it imports `parse_duration`
  from the CLI), and `_check_run_id()` (:93, stderr + exit 2).
  `cmd_init` (:138-140) validates `--run-id` and resolves `--output`
  through `resolve_run_dir` (also closes 057's one-liner); `cmd_generate`
  (:913-918) validates `--run-id` and resolves via the helper;
  `cmd_finalize` (:734) and generate's `--final-video` (:1002) resolve to
  absolute. Explicit absolute `--output` (e.g. `/tmp/...`) stays allowed —
  the constraint targets `--run-id` traversal, not user-chosen absolute
  dirs. Tests: `tests/test_cli_hardening.py` (traversal init/generate
  rejected with "flat folder" on stderr; absolute `--output` accepted;
  helper returns absolute). Gates: ruff + format + mypy strict green;
  pytest 562 passed + new tests (two full-suite runs each flaked a
  different TUI Pilot test owned by another track — both pass alone).
