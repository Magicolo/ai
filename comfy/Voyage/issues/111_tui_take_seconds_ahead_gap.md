# 111 — TUI live validation misses the `take_seconds > ahead_seconds` invariant: field-valid forms fail at Generate with exit 2

- **Severity:** LOW-MEDIUM (the TUI's core promise is "invalid never reaches Generate"; this invariant walks straight through it)
- **Track:** second-pass TUI/CLI edges (tui_state validation math)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`; concurrent uncommitted edits in `voyage/tui_state.py` — lines as-read)

## File:line (live-verified)

- `voyage/tui_state.py:198-211` (`field_errors`: `take_seconds` checked only for `float()` parse + `math.isfinite(take) and take > 0`)
- `voyage/config.py:348-355` (`AudioConfig` model validator: `take_seconds` must EXCEED `ahead_seconds` (default 20.0), with a rationale about per-segment GPU swaps)
- `voyage/cli.py:1238-1254` (where the invariant actually fires at runtime: `apply_draft_overrides` → `ValidationError` → `error: invalid numeric override` → exit 2, post-init — see 110)

## Description

The TUI validates every keystroke (`_refresh_plan_and_errors`), and `to_generate_namespace` refuses to build on any `field_errors` entry — so a user reasonably believes a clean form (empty errors line, live plan) will generate. But `field_errors` accepts any positive finite `take_seconds`, including 5.0, while the runtime requires `take_seconds > ahead_seconds` (20.0 default). The failure surfaces only after Generate: settings saved, run view opened, worker exits 2, form restored with `invalid numeric override: ... take_seconds (5.0) must exceed ahead_seconds (20.0)`. The ahead window (20.0) is not user-visible anywhere in the form — the field's own help text (`FIELD_HELP["take_seconds"]`, `tui_state.py:77-78`) says only "Must stay above the 20s audio-ahead window" as prose, which live validation then fails to enforce.

Note the asymmetry the fix must respect: `nan` IS caught at field level (`take_seconds='nan'` → `take-seconds must be a positive finite number`, verified live — the 062/issue-100 finite-check landed here), but `5.0` — a far more likely user value — is not. Finiteness without the domain floor gives false confidence.

## Rationale

- Live per-field validation that misses the one runtime invariant for that field is worse than no validation: the empty errors line is an explicit "ready" signal.
- The invariant threshold lives one layer away (`AudioConfig.ahead_seconds`, tunable via stored TOML `[audio]`, not visible in the form). Hardcoding 20.0 into `field_errors` would reintroduce the duplication 024 just killed for frame math — the fix needs the same single-source treatment (validate against the resolved config, or surface `ahead_seconds`).
- The TUI already imports the pattern it needs: `_planning_frames_and_fps` resolves CLI truth at plan time instead of duplicating constants.

## Live evidence (container, 2026-09-30)

```
from voyage.tui_state import GenerateFormState, field_errors
field_errors(GenerateFormState(style='x', name='t', take_seconds='5'))    # -> {} (no error)
field_errors(GenerateFormState(style='x', name='t', take_seconds='nan'))  # -> {'take_seconds': '...positive finite number...'}
```

So `5` sails through the form and dies in `cmd_generate` (see 110's transcript for the exact exit-2 message).

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "
from voyage.tui_state import GenerateFormState, field_errors, to_generate_namespace
s = GenerateFormState(style='x', name='t', take_seconds='5')
print('field errors:', field_errors(s))  # {} — namespace builds fine...
ns = to_generate_namespace(s)            # ...and only cmd_generate rejects it (exit 2, post-init)
"
```

## Fix candidates

- Field-level: resolve the effective `ahead_seconds` (stored `[audio]` default 20.0 via the same config path `cmd_generate` uses) and reject `take_seconds <= ahead` with the concrete threshold in the message (`take-seconds must exceed the audio-ahead window (20.0s), got '5'`).
- Keep it single-sourced (024 precedent): no literal `20.0` in `tui_state.py` — read it from the preset/config, with the current `Unset`-blank → preset-default path untouched (blank stays valid).
- Test: `field_errors` pins `take_seconds` at/below/above the resolved floor; TUI Generate with `take_seconds=5` never opens the run view.

## Refs

- Issue 110 (the runtime half: post-init exit 2 + orphan dir); issue 100 (the `nan/inf` precedent — finiteness landed, the domain floor did not); `voyage/config.py:266-271` (`take_seconds`/`ahead_seconds` semantics).

## Progress log

- 2026-09-30 (Group A): premise re-verified live —
  `field_errors(GenerateFormState(style='x', name='t', take_seconds='5'))`
  returned `{}` while the runtime requires `> 20.0`.
- Wrote failing test first
  (`tests/test_cli_group_a.py::test_tui_rejects_take_seconds_at_or_below_ahead_window`):
  red on the `5` and `20` cases.
- Fixed at field level with the 024 single-source treatment.

## Resolution: FIXED

- `voyage/tui_state.py:236-258` (`field_errors`): after the existing
  finite/positive checks, `take_seconds` is rejected at or below the
  resolved floor read from `AudioConfig.model_fields["ahead_seconds"]`
  `.default` — no restated `20.0` literal; blank stays valid (Unset
  path untouched). Message names the concrete threshold:
  `take-seconds must exceed the audio-ahead window (20.0s), got '5'`.
- Test evidence: pins at/below/above the floor + blank-valid; full
  `test_cli_group_a.py` 17 passed; `test_tui_state.py`,
  `test_tui_absent_defaults_023.py` green.
- Gates: `ruff check` + `ruff format --check` + `mypy strict` green.
- Residuals: none. (Stored-TOML `[audio] ahead_seconds` tuning affects
  existing runs only — new runs always render the default, which is
  what the form validates against.)
