# 024 — Unbounded `--output/--run/--final-video/--video` paths: `--run-id` is traversal-guarded, `--output` is not

- Severity: MEDIUM (integrity/robustness: writes escape the project tree by design)
- Group: security/CLI — Rank: 3/5
- File:line: `voyage/cli.py:83-95` (`resolve_run_dir`), `voyage/cli.py:101-125` (`is_flat_folder_name`), `voyage/cli.py:172-180` (`cmd_init`)

## Description

`init`/`generate` reject `../` in `--run-id` but accept any `--output` (`/tmp/evil-run`, `../../tmp/evil`, `/etc/cron.d/x`). `cmd_init` then `mkdir -p`s and writes `voyage.toml` + `state.json` + manifest there. Same for `--final-video`, `--video`, `--output` in `finalize`/`sfx` (absolute `resolve()`, `mkdir -p`, `shutil.copy2`). No containment check, no "outside `output/`" warning. The TUI is safe only because it hardcodes `output/<name>`.

Path traversal guards must cover the path actually used for writes. Guarding the display name while the write path is free is a classic incomplete-fix shape (same class as the `is_flat_folder_name` work that guarded only the id).

## Rationale

- The guarded field (`--run-id`) is cosmetic once `--output` is free: the attacker/accident vector just moves one flag over.
- `mkdir -p` + write + `copy2` on unbounded paths turns a typo (`--output /`) into tree-scale damage with exit 0.
- One funnel already exists (`resolve_run_dir`, used by every verb) — the fix is a single containment predicate, not per-verb plumbing.

## Live evidence (read + grep, 2026-09-30)

`voyage/cli.py:73-88`:

```python
def _run_dir_arg(value: str) -> Path:
    # Absolute: workers spawn with CWD=run_dir, ...
    return Path(value).resolve()


def resolve_run_dir(value: str) -> Path:
    """Canonical run-dir resolution (issue 008/057: one helper, every verb). ..."""
    return _run_dir_arg(value)
```

`voyage/cli.py:168-175`:

```python
def cmd_init(args: argparse.Namespace) -> int:
    if _check_run_id(args.run_id) != 0:
        return 2
    run_dir = resolve_run_dir(args.output)
    ...
    run_dir.mkdir(parents=True, exist_ok=True)
```

No output guard between the run-id check and the `mkdir`. `cmd_init` guards output? `False`; checks run_id only? `True` (sweep probe, still accurate — no output check was added since).

Sweep probe (preserved Track B result):

```
run-id ../../evil flat? False     # guarded
resolved --output /tmp/evil-run: /tmp/evil-run     # unguarded, used for mkdir+write
resolved --output ../../tmp/evil: /tmp/evil        # escapes cwd
cmd_init guards output? False | checks run_id only: True
```

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from pathlib import Path
from voyage.cli import is_flat_folder_name, resolve_run_dir
print('run-id ../../evil flat?', is_flat_folder_name('../../evil'))
print('output ../../tmp/evil ->', resolve_run_dir('../../tmp/evil'))"
grep -n "resolve()" voyage/cli.py   # every unbounded sink
```

## Fix candidates

1. Single `resolve_run_dir` containment rule: warn/error when the resolved path escapes `output/` (or require `--force` for absolute escapes); implement via `Path.relative_to`.
2. Apply the same predicate to `--final-video`/`--video`/`--output` in `finalize`/`sfx`/`generate` (they all funnel through one or two helpers).
3. Add traversal tests for `--output`/`--final-video`/`--video` mirroring the existing `--run-id` tests.

## Refs

- Issue 008's class (traversal guards must cover the write path, not the display name) — `is_flat_folder_name` docstring at `voyage/cli.py:99-113` states the intent; this issue is the uncovered remainder.
