# 110 — `generate` validates numeric overrides AFTER `init`: bad flags leave an initialized run dir that breaks the retry

- **Severity:** MEDIUM (failed command leaves state behind; the retry fails with a misleading second error)
- **Track:** second-pass TUI/CLI edges (generate config-resolution ordering)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`, tree as-read; concurrent uncommitted edits in `voyage/cli.py` — lines as-read)

## File:line (live-verified)

- `voyage/cli.py:1240-1266` (`cmd_generate`: builds `init_args`, calls `cmd_init` — writes `voyage.toml`/`state.json`/manifest/`concepts.jsonl`/`segments/`/`logs/`)
- `voyage/cli.py:1279-1295` (`cmd_generate`: `apply_draft_overrides(...)` in `try/except (ValidationError, ValueError)` → `return 2` — only AFTER init)
- `voyage/cli.py:168-176` (`cmd_init`: `refusing to init non-empty directory {run_dir} (use --force)` → `return 2`)
- Same path serves the TUI: `tui.py:975-984` (`_generate_in_thread` → `cmd_generate`), so a TUI Generate with a bad numeric field inits then fails identically

## Description

`cmd_generate` writes a fresh run config at `init` and only afterwards resolves the numeric overrides (`--blocks/--take-seconds/--quantization/--beats-per-segment/--drift-every-n/--min-fps/--min-resolution`). Any invalid override (e.g. `--take-seconds 5`, which violates the `take_seconds > ahead_seconds (20.0)` model validator) exits 2 — leaving a fully initialized but empty run directory behind. The user's natural retry (same command, fixed value) then fails with `refusing to init non-empty directory ... (use --force)`, an error that blames directory state instead of restating the original problem. The operator must infer that the first failure littered, then choose between `--force` (which re-inits over the litter — safe here, but the flag reads as destructive) or manual `rm -rf`.

## Rationale

- Validate-before-mutate is the standard CLI contract (`clig.dev`: "check early and bail out before anything bad happens"; argparse validator-chain pattern: "validators gate, handlers act" — pure, no side effects). Mutating (init) before validating (overrides) inverts it.
- The failure is silent about its own litter: neither the `return 2` message nor the retry error mentions that the first attempt created the directory the second attempt trips over. Two individually reasonable messages compose into a confusing sequence.
- The TUI inherits it: TUI field validation (see 111) passes `take_seconds=5`, `_start_generation` saves settings + opens the run view, then the worker exits 2 and the form restores with `invalid numeric override` — after an `output/<name>/` dir was created the user never asked to keep.

## Live evidence (container, 2026-09-30)

```
$ ... python3 -c "cmd_generate(... take_seconds=5.0, backend='fake', no_download=True, ...)"
initialized voyage run at /tmp/tmphd1wdzw7/orphan1
exit code: 2
orphan dir left behind: [concepts.jsonl, logs, run_manifest.json, segments, state.json, voyage.toml]
error: invalid numeric override: 1 validation error for AudioConfig
  Value error, take_seconds (5.0) must exceed ahead_seconds (20.0): ...
```

No workers, no ffmpeg, no downloads were touched — the failure is pure config math that was knowable before the first byte was written.

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "
import argparse, tempfile
from pathlib import Path
from voyage import cli
tmp = Path(tempfile.mkdtemp())
base = dict(backend='fake', duration=5.0, style='x', run_id='o', name='o',
            output=str(tmp/'o'), seed=0, force=False, final_video=None, skip_bad=False,
            no_download=True, no_sfx=True, sfx_backend=None, sfx_caption=None,
            sfx_device=None, sfx_model_size=None, sfx_workers=1, min_fps=None,
            min_resolution=None, no_augment=False, draft=False, director='deterministic',
            blocks=None, take_seconds=5.0, quantization=None, beats_per_segment=None,
            drift_every_n=None, music_caption=None, video_caption=None,
            verbose=False, no_color=True)
print('first:', cli.cmd_generate(argparse.Namespace(**base)))   # 2, but inits
print('retry (fixed):', end=' ')
base.update(take_seconds=45.0)
print(cli.cmd_generate(argparse.Namespace(**base)))             # 2: refusing non-empty
"
```

## Fix candidates

- Validate overrides against the preset-resolved config BEFORE `cmd_init` (resolve twice — once pre-init for gating, once post-init as today — or split `resolve_config` validation from application). Pre-init gating needs no run dir: backend preset + overrides are all in-memory.
- Alternatively, remove the created run dir when the post-init override validation fails (compensating rollback), and say so in the message. Weaker: a crash between init and cleanup still litters.
- Test: `generate` with each invalid numeric override asserts exit 2 AND `not run_dir.exists()`.

## Refs

- `voyage/config.py:348-355` (the `take_seconds > ahead_seconds` validator); `voyage/cli.py:1236-1254` (post-init gate); https://clig.dev/ (validate early, bail before state changes).

## Progress log

- 2026-09-30 (Group A): premise re-verified live in-container —
  `cmd_generate` (now `voyage/cli_generate.py`) still called `cmd_init`
  at line 71-equivalent before `apply_draft_overrides` at
  line 84-equivalent; repro left the exact orphan dir
  (`logs`, `run_manifest.json`, `segments`, `state.json`, `voyage.toml`)
  with exit 2.
- Wrote failing test first
  (`tests/test_cli_group_a.py::test_generate_rejects_bad_take_seconds_before_init`):
  red — orphan dir existed after exit 2.
- Fixed with the first candidate (validate pre-init, resolve twice).

## Resolution: FIXED

- `voyage/cli_generate.py:30-84` (new `_pre_init_override_gate`): renders
  the exact TOML `cmd_init` would write, validates it into a
  `ProjectConfig`, and dry-runs the same `apply_draft_overrides` call
  the post-init path makes — pure, no directory touched. Failure exits
  2 with the same `invalid numeric override` message before init.
- Blank styles skip the gate (`cmd_init` reports those itself,
  litter-free since the 143/185 fix); the post-init gate stays as the
  second resolution (configs could theoretically differ — they cannot
  today, same inputs, same function).
- Test evidence: new test green; full `test_cli_group_a.py` 17 passed.
- Gates: `ruff check` + `ruff format --check` + `mypy strict` green.
- Residuals: none.
