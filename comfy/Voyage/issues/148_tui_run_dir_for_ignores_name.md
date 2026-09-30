# 148 — TUI `_run_dir_for` reads `run_id`, ignoring `--name` (TUI resolves the wrong dir when both set)

- Severity: LOW (TUI Generate lands in `output/<run-id>` while CLI/`tui_state` land in `output/<name>`)
- Area: config/CLI/TUI — name parity
- Files (as-read 2026-09-30; concurrent uncommitted `--name` edits in `voyage/cli.py`, `voyage/tui_state.py` — lines as-read):

## File:line

- `voyage/tui.py:971-973` (`_run_dir_for`: `output = args.output or str(Path("output") / args.run_id)`; `return Path(output).resolve()`)
- `voyage/cli.py:129-134` (`_effective_run_id`: `--name` wins, `--run-id` is the legacy alias)
- `voyage/cli.py:1249-1255` (`cmd_generate`: `run_id = _effective_run_id(args)`; `output = Path(args.output) if args.output else Path("output") / run_id`)
- `voyage/tui_state.py:271-279` (`to_generate_namespace`: `name = state.name.strip()`; `output = str(Path("output") / name)`; sets BOTH `run_id=name` and `name=name`)

## Description

Since the `--name` migration, the run identity is `_effective_run_id` (name-wins). `cmd_generate` resolves the default dir from the EFFECTIVE id (:1254), and `to_generate_namespace` sets `run_id == name` always (:277-279), so the CLI path is consistent. But the TUI's `_run_dir_for` (:971-973) reads `args.run_id` directly and never consults `args.name`. Today the two fields coincide (the TUI always sets both to `name`), so the bug is latent — it fires the moment any caller passes a namespace where they differ (legacy `--run-id` + `--name`, a test double, a future TUI field split): the TUI thread then watches/streams the wrong directory while `cmd_generate` writes the right one (progress bar + final video reported from an empty dir).

## Rationale

- Single-source rule (024 precedent): exactly one function decides "which name wins". Two call sites now encode it (`_effective_run_id` vs inline `args.run_id`) and they disagree by construction.
- Latent-but-load-bearing: today's equality (`run_id == name` in every TUI namespace) is a coincidence of the current form, not an invariant — the next edit that lets them differ activates the bug silently.
- Cheap fix, real confusion avoided (wrong-dir watch looks like a hung generation).

## Live evidence

```
tui.py:971-973       output = args.output or str(Path("output") / args.run_id)
cli.py:129-134       named = getattr(args, "name", None); if ... != "": return named; return str(args.run_id)
cli.py:1254          output = Path(args.output) if args.output else Path("output") / run_id  # effective
tui_state.py:277-279 run_id=name, name=name, output=output  # today equal, masking the gap
```

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "
import argparse
from pathlib import Path
from voyage.tui import VoyageApp
ns = argparse.Namespace(output=None, run_id='legacy', name='boba')
print(VoyageApp._run_dir_for(VoyageApp(), ns))  # .../output/legacy (wrong)
from voyage.cli import _effective_run_id
print(Path('output') / _effective_run_id(ns))   # output/boba (right)
"
```

## Fix candidates

- One-liner: `_run_dir_for` uses `_effective_run_id(args)` (import from `voyage.cli`, the CLI single source) instead of `args.run_id`.
- Test: namespace with divergent `run_id`/`name` resolves to `output/<name>`; explicit `--output` still wins.

## Refs

- `voyage/tui.py:971-973` vs `voyage/cli.py:129-134` + `:1249-1255`; `tests/test_generate.py:322-346` (name-wins-over-run-id CLI precedent).
