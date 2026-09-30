# 023 — TUI always-override vs CLI absent-semantics: untouched TUI form clobbers stored `quantization`/augment floors

- Severity: MEDIUM (silent config clobber on the happy path)
- Group: TUI/config — Rank: 2/5
- File:line: `voyage/tui_state.py:252-310` (`to_generate_namespace`), `voyage/tui_state.py:98-115` (form defaults), `voyage/config.py:74-85` (`is_provided`)

## Description

CLI optional flags default to `None` (= absent → stored TOML wins). The TUI namespace path is mixed:

- `blocks`, `take_seconds`, `beats_per_segment`, `drift_every_n` honor blank→`Unset` (absent).
- `quantization` always emits the concrete string (`quantization=state.quantization`, default `"fp8"`) — there is no blank/`Unset` path.
- `min_fps`/`min_resolution` honor blank→`Unset`, but the form defaults are prefilled with `"32"`/`"1280x720"`, so blank is unreachable without the user actively clearing the field. An untouched form emits concrete floors.
- `backend`/`director` always emit concrete values.

Opening the TUI on a run customized to `quantization=bf16, min_fps=60, 1920x1080` and pressing Generate with untouched fields silently reverts all three to the form defaults. The comment at `:290-295` ("blank means run TOML default … an untouched form resolves to the same floors") is true only when the TOML already equals the form defaults — otherwise untouched means clobber.

## Rationale

- Two encodings for "default" (CLI `None`-absent vs TUI always-set) defeat `resolve_config`'s explicit-flags-win contract, which can only honor absence it can see.
- The failure is silent and on the most common path (accept the defaults, press Generate).
- The `Unset` sentinel (issue 045) was introduced precisely to unify this; these three fields were left behind.

## Live evidence (read, 2026-09-30)

`voyage/tui_state.py:98-115` (prefilled defaults):

```python
    backend: str = "ltxv"
    ...
    quantization: str = "fp8"
    ...
    min_fps: str = "32"
    min_resolution: str = "1280x720"
```

`voyage/tui_state.py:273-298` (namespace — note `:288` is unconditional):

```python
    return argparse.Namespace(
        backend=state.backend,
        ...
        quantization=state.quantization,
        ...
        min_fps=int(min_fps_raw) if min_fps_raw else Unset,
        min_resolution=min_resolution_raw if min_resolution_raw else Unset,
```

Sweep probe (preserved Track B result, in-container):

```
quantization: 'fp8' (CLI default None = absent)
min_fps: 32 min_res: '1280x720'
after TUI resolve quant: fp8 (was bf16 -> clobbered to fp8)
after TUI floors: 32 1280 720 (were 60/1920/1080)
```

Repro sketch (same probe, live tree): `GenerateFormState(style='x', name='n')` → `to_generate_namespace` → `resolve_config` over a `bf16/60/1920x1080` base → `fp8/32/1280x720`.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.tui_state import GenerateFormState, to_generate_namespace
ns = to_generate_namespace(GenerateFormState(style='x', name='n'))
print('quantization:', repr(ns.quantization))
print('min_fps:', repr(ns.min_fps), 'min_resolution:', repr(ns.min_resolution))"
# Buggy: 'fp8', 32, '1280x720' — concrete, so resolve_config treats them as explicit
```

## Fix candidates

1. Emit `Unset` for untouched-at-default fields (track dirtiness per field) or add a tri-state (blank = inherit) for `quantization`/`backend`/`director` matching the numeric fields.
2. Prefill TUI fields from the target run's TOML instead of hardcoded defaults, so "untouched" literally means "stored".
3. Document the precedence once (Flags > TOML > Defaults) next to `resolve_config` and point both CLI and TUI at it; add a test that an untouched-form namespace resolves to the stored config unchanged.

## Refs (with links/quotes)

- "A professional CLI should allow configuration … with a clear order of precedence … 1. flag 2. env 3. config file 4. default … `viper.BindPFlags` … flags get top priority … command logic just asks Viper." — https://cobra.dev/docs/tutorials/12-factor-app/
- "Highest to lowest: CLI flags, then OS environment variables, then … config file, then hard-coded defaults." — https://python-config-secrets-hub.com/core-configuration-patterns-file-formats/configuration-precedence-rules/

## Progress log (2026-09-30, resolution track)

- TDD: wrote `tests/test_tui_absent_defaults_023.py` (4 tests) first,
  watched the 2 untouched-form tests fail (explicit-value + blank-floor
  tests already passed as correct-path anchors), then fixed.
- Fix in `voyage/tui_state.py` (`to_generate_namespace` — the only hook
  the live TUI calls, `voyage/tui.py:902`, and `tui.py` is out of scope
  so widget-level dirtiness tracking is impossible here):
  `_DEFAULT_QUANTIZATION`/`_DEFAULT_MIN_FPS`/`_DEFAULT_MIN_RESOLUTION`
  constants; default-valued `quantization`/`min_fps`/`min_resolution`
  emit `Unset` (stored TOML wins); blank floors keep the pre-existing
  `Unset` path; `0`/`"0"` still disable; non-default values still
  override. `backend`/`director` stay concrete — correct for
  `generate`, which inits a fresh config from the preset. `FIELD_HELP`
  updated for the three fields.
- Documented tradeoff: explicitly re-selecting a default value via the
  TUI (stored `bf16` back to `fp8`) is indistinguishable from untouched
  and now inherits — that downgrade needs the CLI flags
  (`--quantization`/`--min-fps`/`--min-resolution`). Full dirtiness
  tracking (per-field touched flags set by `tui.py` change events)
  would remove the tradeoff and is left as a follow-up.
- Evidence: 15/15 new tests green in-container (all three new files);
  `ruff check` + `ruff format --check` + `mypy` strict green on all
  touched files.

## Resolution: FIXED (with one known test residual)

- Verdict: fixed — an untouched TUI form resolves stored
  `quantization`/`min_fps`/`min_resolution` unchanged (proven by
  `test_untouched_form_preserves_stored_quantization_and_floors` over a
  `bf16`/`60`/`1920x1080` base).
- Residual (expected, for the orchestrator):
  `tests/test_augment_config.py::test_tui_namespace_carries_augment_attrs`
  asserts the OLD buggy contract (untouched form emits concrete
  `32`/`"1280x720"`) and now fails. It encodes the exact behavior this
  issue required changing, so it needs updating to the inherit
  contract — deliberately left failing (existing-test edits are outside
  this track's file contract). All other adjacent suites green (167
  passed in the `test_tui`/`test_tui_state`/`test_augment_config`/
  `test_unset`/`test_config_resolution`/`test_backend_registry`/
  `test_precision`/`test_cli_tui_split` batch).
