# 147 — `generate`→`finalize` namespace drops `verbose`/`no_color` (finalize stage ignores console flags)

- Severity: LOW (finalize third of a `generate` run ignores `--verbose`/`--no-color`)
- Area: config/CLI — console-flag plumbing
- Files (as-read 2026-09-30; concurrent uncommitted edits in `voyage/cli.py` — lines as-read):

## File:line

- `voyage/cli.py:1358-1377` (`cmd_generate`→`cmd_run` namespace: carries `verbose=console.verbose`, `no_color=...`, `progress_sink=sink` — the forwarding precedent)
- `voyage/cli.py:1390-1408` (`cmd_generate`→`cmd_finalize` namespace: `run/output/skip_bad/no_sfx/sfx_*/min_fps/min_resolution/no_augment` — NO `verbose`/`no_color`/`progress_sink`)
- `voyage/cli.py:137-142` (`get_console`: `verbose=getattr(args,"verbose",False)`, `no_color=getattr(args,"no_color",False)` — missing attrs silently default off)
- `voyage/cli.py:979-997` (`cmd_finalize`: resolves augment overrides, `Path(args.output).resolve()`, `finalize_run(... width=config.video.width ...` — never reads a console/verbosity flag)

## Description

`cmd_generate` builds a console from the user's flags (`console = get_console(args)` at :1300) and forwards verbosity into the run stage (`verbose=console.verbose`, `no_color`, `progress_sink` at :1373-1375). The finalize stage built ten lines later (:1393-1407) forwards none of them. Combined with `get_console`'s `getattr(..., False)` defaults, whatever `cmd_finalize` (or its callees) reads for verbosity resolves to off — so `voyage generate --verbose` is verbose for init/run/ensure and silent for finalize, and `--no-color` is honored everywhere except finalize. The user-visible symptom is mild (less detail + possible color/animations during the final stage), but the plumbing asymmetry guarantees any future console-aware finalize work also arrives muted.

## Rationale

- Sibling-call parity: two child stages built in the same function from the same `console` object should receive the same console context. One does, one doesn't.
- `getattr` defaults mask the gap (no `AttributeError`, just quiet behavior) — the same masking pattern as 145.
- TUI inherits it: `to_generate_namespace` sets `verbose`/`no_color` (:300-301), the run stage respects them, the finalize stage drops them.

## Live evidence

```
cli.py:1358-1376  cmd_run(Namespace(..., verbose=console.verbose,
                 no_color=getattr(args, "no_color", False), progress_sink=sink))
cli.py:1393-1407  cmd_finalize(Namespace(run=..., output=..., skip_bad=...,
                 no_sfx=..., sfx_*=..., min_fps=..., min_resolution=...,
                 no_augment=...))  # no verbose / no_color / progress_sink
cli.py:137-142  get_console: getattr(args, "verbose", False) / getattr(args, "no_color", False)
```

`grep -n 'verbose\|no_color\|progress_sink' cli.py` shows the finalize-construction site as the only child-stage call without all three.

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "
import argparse
ns = argparse.Namespace(run='r', output='o.mp4', skip_bad=False, no_sfx=True,
    sfx_backend=None, sfx_caption=None, sfx_device=None, sfx_model_size=None,
    sfx_workers=1, min_fps=None, min_resolution=None, no_augment=False)
from voyage.cli import get_console
print(vars(get_console(ns)))  # {'verbose': False, 'no_color': False} — always off
"
```

## Fix candidates

- Forward the same three keys into the finalize namespace (`verbose=console.verbose`, `no_color=...`, `progress_sink=sink`), mirroring the `cmd_run` call.
- Alternatively, pass the `console` object itself if `cmd_finalize` grows console-aware. Either way, keep the two sibling calls symmetric.
- Test: `generate --verbose/--no-color` namespace capture asserts the finalize namespace carries both.

## Refs

- `voyage/cli.py:1300-1301` (console build) vs `:1358-1377` (run forward) vs `:1390-1408` (finalize drop); issue 145 (same `getattr`-masks-omission pattern).
