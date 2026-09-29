# 097 — TUI/`generate`/`stop --finalize` namespaces lacked the SFX-pass flags `cmd_finalize` reads

- Status: resolved-superseded (bug real, fixed differently upstream: `getattr`-default forwarding; own helper dropped, see log)
- Severity: medium (3 E2E failures: `generate` and `stop --finalize` crashed
  with `AttributeError: 'Namespace' object has no attribute 'no_sfx'`)
- Area: CLI — parser/forwarding contract between finalizing verbs
- Rank rationale: found during final-issues review (045) while triaging gate
  failures first attributed to the Unset change; root cause is a concurrent
  SFX-pass feature collision, not the refactor.

## Technical description

The concurrent SFX finalize pass added `--no-sfx` / `--sfx-device` /
`--sfx-model-size` / `--sfx-workers` to the `finalize` subparser and reads
them in `cmd_finalize` (`cli.py:880,889,890`), but neither the `generate`
subparser, the `stop` subparser, `cmd_generate`'s internal finalize
Namespace, nor `tui_state.to_generate_namespace` carried them — so every
`generate`, `stop --finalize`, and TUI run crashed at finalize time with
`AttributeError` instead of rendering.

## Why this is an issue

`generate` is the primary entry point (one-shot E2E + TUI backend); a crash
at the finalize step wastes the entire GPU render that preceded it. The
failure mode (AttributeError, not a clean exit-2 usage error) also hides
which flag contract broke.

## Evidence

```
AttributeError: 'Namespace' object has no attribute 'no_sfx'
voyage/cli.py:880: AttributeError
```
plus the `cmd_generate` internal `argparse.Namespace(run, output, skip_bad)`
carrying only 3 of the 7 fields `cmd_finalize` reads. Reproduced live in
gates (`test_generate_fake_end_to_end_validated_finalized`,
`test_generate_defaults_to_output_run_id`).

## Reproduction

`voyage generate --backend fake ...` (or any TUI run) on a tree where
`cmd_finalize` reads SFX flags: finalize step raises AttributeError.

## Source references

- `voyage/cli.py` (`cmd_finalize`, `cmd_generate` internal Namespace,
  `_add_finalize_parser` vs `_add_generate_parser`/`_add_stop_parser`,
  `tui_state.to_generate_namespace`).

## Resolution candidates

Done: shared `_add_sfx_args(parser)` helper on all three parsers (single
source for the four flags + defaults) + pass-through in `cmd_generate`'s
internal Namespace + TUI namespace defaults + parser-parity tests.

## Investigation / progress / resolution log

- 2026-09-29 (orchestrator): found while triaging 3 gate failures first
  blamed on the 045 Unset change; isolated to the SFX-pass collision by
  reading the traceback (`no_sfx`, not Unset). Fixed same session:
  `_add_sfx_args` + pass-through + TUI defaults +
  `tests/test_sfx_parser_parity.py` (parity + overrides tests). Gates
  green. Filed to record the collision pattern (see AGENTS.md lesson on
  concurrent feature collisions).
- 2026-09-29 (orchestrator, supersession note): the `_add_sfx_args`
  fix recorded below was superseded before commit by the concurrent
  track's `getattr(args, ..., default)` forwarding (every SFX read now
  tolerates missing attrs, plus new sfx/augment flags). No revert risk
  was taken: none of this issue's code hunks ship in this commit.
  Keeping the file as the collision record — the pattern (feature adds
  reads without updating all forwarding sites) is the lesson.
