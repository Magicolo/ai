# 179 — `stop --finalize` runs the finalizer without console flags: stop parser never defines `--verbose`/`--no-color`

- **Severity:** LOW (finalize third of a `stop --finalize` run is always non-verbose; the flag cannot even be requested)
- **Track:** B (CLI stop→finalize handoff — below 109/147/115)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`, tree as-read; concurrent uncommitted edits in `voyage/cli.py` — lines as-read)

## File:line (live-verified)

- `voyage/cli.py:1956-1967` (`_add_stop_parser`: `--run`, `--finalize`, `_add_sfx_args(stop)` only — no `--skip-bad`, no `_add_augment_args`, no `_add_console_args`)
- `voyage/cli.py:771-776` (`cmd_stop`: on `--finalize` sets `args.output` and calls `cmd_finalize(args)` with the SAME stop-parser namespace)
- `voyage/cli.py:137-142` (`get_console`: `verbose=getattr(args,"verbose",False)`, `no_color=getattr(args,"no_color",False)` — missing attrs silently default off)
- `voyage/cli.py:1037` (`cmd_finalize`: `console = get_console(args)` — resolves to off/off on the stop path, always)
- Contrast: `_add_finalize_parser` (`:1976-1990`) wires `_add_console_args`; `_add_run_parser` (`:1847-1865`) and `_add_generate_parser` (`:1874-1933`) wire it too — `stop` is the only finalizing verb without it.

## Description

`cmd_stop --finalize` reuses the stop namespace directly (`return cmd_finalize(args)`), so whatever the stop parser defines is what finalize sees. The stop parser defines no console flags, and `get_console`'s `getattr(..., False)` defaults mask the absence — finalize runs, but its `console.ok("finalized -> ...")` line and any future verbose output always render in non-verbose mode. Worse than the `generate`→`finalize` drop (147): there the user CAN pass `--verbose` (it just doesn't reach finalize); here `--verbose` is not even a recognized flag (`stop --finalize --verbose` → exit 2, unrecognized arguments), so there is no way to get a verbose stop-finalize at all.

## Rationale

- Finalizing-verb parity: every verb that runs the finalizer (`finalize`, `generate`, `stop --finalize`) should offer the same console surface. Two do, one doesn't.
- `getattr` defaults mask the gap (no `AttributeError`, just quiet behavior) — the same masking pattern as 145/147.
- The `_add_sfx_args` docstring (`:1773-1782`) already states the one-helper rule for finalizing verbs ("a missing flag on any finalizing verb is an AttributeError at finalize time"); console flags deserve the same shared-helper treatment (`_add_console_args` exists for exactly this).

## Live evidence (container, 2026-09-30)

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage import cli
ns = cli.build_parser().parse_args(['stop','--run','/tmp/any','--finalize'])
print('stop ns keys:', sorted(vars(ns).keys()))
print('has verbose:', hasattr(ns,'verbose'), 'has no_color:', hasattr(ns,'no_color'))
from voyage.cli import get_console
c = get_console(ns)
print('console from stop ns: verbose=%s no_color=%s' % (c.verbose, c._no_color))"

stop ns keys: ['command', 'finalize', 'func', 'no_sfx', 'run', 'sfx_backend', 'sfx_caption', 'sfx_device', 'sfx_model_size', 'sfx_workers']
has verbose: False has no_color: False
console from stop ns: verbose=False no_color=False
```

`generate` parser for contrast: `has verbose: True` (value `False` until passed).

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage import cli
ns = cli.build_parser().parse_args(['stop','--run','/tmp/any','--finalize'])
print(hasattr(ns, 'verbose'), hasattr(ns, 'no_color'))  # False False
"
docker run --rm -v "$PWD:/app" -w /app voyage:latest --entrypoint voyage voyage:latest stop --help
# Shows --run/--finalize/SFX only — no --verbose/--no-color, unlike finalize --help.
```

## Fix candidates

1. Add `_add_console_args(stop)` in `_add_stop_parser` (one line, mirrors every other finalizing verb) — `stop --finalize --verbose` then works and `cmd_finalize`'s `get_console` picks it up with no further change.
2. Same pass: consider `_add_augment_args(stop)` + `--skip-bad` (109's structural fix) so the stop-finalize surface matches `finalize` fully; keep this file's scope to console flags and cross-ref 109.
3. Test: stop namespace carries `verbose`/`no_color` defaults; `stop --finalize --verbose` parses.

## Refs

- `voyage/cli.py:1956-1967` vs `:1976-1990` (finalize) / `:1847-1865` (run) / `:1874-1933` (generate); `:771-776` (same-namespace handoff); `:137-142` (getattr masking).
- Not-a-duplicate: 109 (stop→finalize `skip_bad` crash + augment absence — console flags never named); 147 (generate→finalize namespace drop — different call site: new-namespace build vs same-namespace passthrough); 115 (soak/benchmark CUDA preflight — different flags).

## Progress log

- 2026-09-30 (Group A): premise re-verified live — stop namespace
  keys had no `verbose`/`no_color`, and `stop --finalize --verbose`
  exited 2 (unrecognized arguments).
- Wrote failing test first (joint with 109:
  `tests/test_cli_group_a.py::test_stop_parser_offers_full_finalize_surface`
  parses `--verbose/--no-color/--skip-bad/--min-fps` on `stop`): red.
- Fixed with candidate 1, folded into 109's structural hunk (same
  lines, one review).

## Resolution: FIXED (folded into 109's hunk)

- `voyage/cli.py:594-601` (`_add_stop_parser`): `_add_console_args(stop)`
  added alongside `--skip-bad` + `_add_augment_args(stop)` — `stop` is
  no longer the only finalizing verb without console flags, and
  `cmd_finalize`'s `get_console(args)` picks them up with no further
  change (same-namespace handoff).
- The 109 augment consideration (candidate 2 there) landed too, so the
  stop-finalize surface now matches `finalize` fully: skip + sfx +
  augment + console.
- Test evidence: `stop --finalize --verbose` parses; flags default off.
- Gates: `ruff check` + `ruff format --check` + `mypy strict` green.
- Residuals: none.
