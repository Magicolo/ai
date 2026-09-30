# 143 — `cmd_init` mkdirs before validating: bad config leaves a partial run dir

- Severity: MEDIUM (failed init litters `segments/`+`logs/`+`voyage.toml`; retry trips over own litter)
- Area: config/CLI — init ordering
- Overlaps with 110 (same validate-after-mutate class, one layer up)
- Files (as-read 2026-09-30; concurrent uncommitted `--name` edits in `voyage/cli.py` — lines as-read):

## File:line

- `voyage/cli.py:170-192` (`cmd_init`: `_check_run_id` → non-empty guard → `mkdir` run/`segments`/`logs` → `default_config_toml` → `write_text voyage.toml` → `load_config`)
- `voyage/config.py:560-565` (`ProjectConfig.style_required`: empty style raises — fires at `load_config`, AFTER the mkdirs)

## Description

`cmd_init` creates directories first (`run_dir.mkdir(parents=True, exist_ok=True)` at :178, `segments`/:179, `logs`/:180) and only then materializes + validates the config (`write_text` :192, `load_config` :193 → pydantic validators such as `style_required`). Any validation failure after :178 (empty/whitespace style via a hand-built namespace, bad backend string, future validator tightening) raises out of `cmd_init` leaving a partial run dir behind (`voyage.toml` possibly written, `segments/`+`logs/` created, no `state.json`/manifest). The natural retry then hits the non-empty guard at :175 (`refusing to init non-empty directory ... (use --force)`) — a second error blaming directory state instead of the original bad value. Same class as 110 (generate validates overrides post-init), one layer down.

## Rationale

- Validate-before-mutate: pure config math (style non-empty, backend literal, TOML render) is knowable before the first `mkdir`. Mutating first inverts the contract.
- The litter is silent: the traceback says what was invalid, never that it left `segments/`+`logs/` behind for the retry to trip over.
- CLI-argparse callers rarely hit it (`--style` is required, `--backend` has `choices=`), so the blast radius is programmatic/TUI-adjacent callers and future validators — exactly where a confusing retry costs the most.

## Live evidence

```
cli.py:178  run_dir.mkdir(parents=True, exist_ok=True)
cli.py:179  (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
cli.py:180  (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
cli.py:192  (run_dir / paths.CONFIG_FILENAME).write_text(config_text, ...)
cli.py:193  config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
config.py:560-565  style_required: not value.strip() → ValueError
```

No `try/except` between :178 and :193 compensates (no rollback); `write_manifest`/:196 + `write_state`/:197 + `concepts.jsonl`/:198 never run on the failure path.

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "
import argparse, tempfile
from pathlib import Path
from voyage import cli
tmp = Path(tempfile.mkdtemp())
ns = argparse.Namespace(output=str(tmp/'r'), run_id='r', name='r',
    style='   ', seed=0, force=False, backend='ltxv',
    director='qwen', director_device='cuda:1')
try: cli.cmd_init(ns)
except Exception as e: print(type(e).__name__, e)
print(sorted(p.name for p in (tmp/'r').iterdir()))
print('retry:', cli.cmd_init(argparse.Namespace(**{**vars(ns), 'style':'ok'})))
"
# Expect: validation error, dir contains [segments, logs, voyage.toml],
# retry → 2 'refusing to init non-empty directory'.
```

## Fix candidates

- Build + validate the config fully in memory BEFORE any `mkdir` (render TOML → parse/validate → only then create dirs + write files).
- Alternatively, remove the created dirs when post-mkdir validation fails (compensating rollback) and say so in the message. Weaker: crash between mkdir and cleanup still litters.
- Test: `cmd_init` with each invalid field asserts exit/exception AND `not run_dir.exists()`.

## Refs

- `voyage/cli.py:170-200`; `voyage/config.py:560-565`; issue 110 (same validate-after-mutate class one layer up).

## Progress log

- 2026-09-30 (Group A): premise re-verified live on the split tree
  (`cmd_init` now `voyage/cli_run_ops.py:35-64`): blank style raised
  `ConfigurationError` post-write, littering `segments/`+`logs/`+
  `voyage.toml`, and the retry hit the non-empty guard. Note: the
  concurrent uncommitted hunk in this file (root-concepts removal) sits
  below the init guards — kept disjoint, untouched.
- Wrote failing test first
  (`tests/test_cli_group_a.py::test_cmd_init_rejects_blank_style_without_litter`):
  red (raised instead of returning 2, dir littered).
- Fixed with the first candidate (full in-memory validation before any
  mkdir), folded with 185's missing-attribute trigger in the same hunk.

## Resolution: FIXED

- `voyage/cli_run_ops.py:35-63`: `cmd_init` now validates output
  presence, run-id (existing), style non-blank, seed int (bool
  rejected), and backend membership in `BACKEND_REGISTRY` (single
  source) — all before the first `mkdir`, each a stderr message + exit
  2. Post-write `load_config` stays as the full validator.
- `BACKEND_REGISTRY` added to the `voyage.config` imports; `cast` added
  for the `VideoBackendName` narrowing (`mypy strict` clean).
- Test evidence: blank-style and missing-attribute cases assert exit 2
  AND `not run_dir.exists()`; pre-existing `test_cli_run_ops_pruning`
  + `test_concepts_pruning` still green.
- Gates: `ruff check` + `ruff format --check` + `mypy strict` green.
- Residuals: none. (Director backend/device strings are free-form in
  the config model — inventing a pre-mkdir rejection for them would be
  new behavior, deliberately not added.)
