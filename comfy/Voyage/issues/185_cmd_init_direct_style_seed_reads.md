# 185 — `cmd_init` reads `args.style`/`args.seed` directly while sibling overrides use `getattr`: hand-built namespaces crash with `AttributeError` after the mkdirs

- **Severity:** LOW (robustness — programmatic/TUI-adjacent callers get a traceback + a littered run dir instead of a clean error; CLI parsers always supply both, so the CLI path never fires)
- **Track:** B (CLI init boundary — below 143/110/109)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`, tree as-read; concurrent uncommitted edits in `voyage/cli.py` — lines as-read)

## File:line (live-verified)

- `voyage/cli.py:181-183` (`cmd_init`: `backend = getattr(args, "backend", None) or "ltxv"`, `director_backend = getattr(args, "director", None) or "qwen"`, `director_device = getattr(args, "director_device", None) or "cuda:1"` — all defensive)
- `voyage/cli.py:184-191` (`config_text = default_config_toml(run_id, args.style, args.seed, ...)` — direct reads, no fallback)
- `voyage/cli.py:174-180` (guards + mkdirs run BEFORE the direct reads: `_check_run_id` at `:172`, non-empty guard at `:175`, `mkdir` run/`segments`/`logs` at `:178-180`)
- `voyage/cli.py:175` (`... and not args.force` — `args.output` at `:174` via `resolve_run_dir(args.output)` is also direct)

## Description

Three of `cmd_init`'s inputs are read defensively (`getattr` + default), four are read directly (`args.output`, `args.force`, `args.style`, `args.seed`). A hand-built namespace missing `style` (e.g. a test double or a future TUI field split that forgets the key) raises `AttributeError: 'Namespace' object has no attribute 'style'` — AFTER lines 178-180 created `run/`, `segments/`, `logs/`. The failure therefore litters exactly like 143's validate-after-mkdir (partial dir, retry trips over the non-empty guard), but the trigger is a missing attribute rather than an invalid value, and the traceback names no user-facing cause. The neighboring `getattr` lines prove the defensive idiom was known at this call site — `style`/`seed` are the holdouts (same shape as 109's `skip_bad` holdout one verb over).

## Rationale

- Boundary rule for cross-verb/programmatic entry points: validate-then-mutate, and never let an `AttributeError` be the validator. `cmd_finalize` already follows it for every sfx/augment read (all `getattr`); `cmd_init` follows it for 3 of 7.
- The ordering makes it worse than a plain crash: the mkdirs precede the reads, so the crash always leaves state behind (143's litter without 143's explanatory `ValueError`).
- Fix is two `getattr`s + an explicit missing-value error before the first `mkdir` (pairs with 143's validate-before-mutate fix).

## Live evidence (container, 2026-09-30)

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
import argparse
from voyage import cli
ns = argparse.Namespace(output='/tmp/x_init_probe', run_id='p', name='p', force=True, backend='ltxv')
try:
    cli.cmd_init(ns)
except AttributeError as e:
    print('AttributeError:', e)"

AttributeError: 'Namespace' object has no attribute 'style'
```

Source order (live `sed -n '170,200p'`): guards `:172-177` → mkdirs `:178-180` → getattr trio `:181-183` → direct `args.style, args.seed` at `:186-187` → write/validate `:192-193`.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
import argparse
from voyage import cli
ns = argparse.Namespace(output='/tmp/x_init_probe', run_id='p', name='p', force=True, backend='ltxv')
cli.cmd_init(ns)"  # AttributeError: style (plus littered /tmp/x_init_probe/{segments,logs} if /tmp is shared)
sed -n '170,200p' Voyage/voyage/cli.py
```

## Fix candidates

1. Read `style`/`seed` via `getattr(args, "style", None)` / `getattr(args, "seed", None)` and emit the same `error: ...` + exit-2 shape as `_check_run_id` BEFORE any `mkdir` (closes this hole and half of 143 in one move).
2. Same pass: `args.output`/`args.force` direct reads (`:174-175`) deserve the same treatment for hand-built namespaces (missing `output` crashes inside `resolve_run_dir` before any message).
3. Test: `cmd_init` with each required attr missing asserts exit 2 AND `not run_dir.exists()` (pairs with 143's proposed litter test).

## Refs

- `voyage/cli.py:170-200`; issue 143 (same mkdirs, invalid-value trigger — this file is the missing-attribute trigger); issue 109 (same direct-vs-getattr holdout shape on the stop path); issue 022 (hand-built-namespace contract tests as the fix pattern).
