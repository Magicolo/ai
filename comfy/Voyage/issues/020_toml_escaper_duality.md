# 020 — TOML escaper duality: `config._toml_basic_string` emits raw C0 controls → `init` writes unparseable `voyage.toml`

- Severity: HIGH (integrity: a creator free-text style/run-id can brick the run config at write time)
- Group: config/robustness — Rank: 2/5
- File:line: `voyage/config.py:588-600` (weak escaper), `voyage/tui_state.py:419-438` (strong escaper), `voyage/config.py:634-635` (consumer)

## Description

There are two TOML basic-string escapers with different strength. The config one escapes only backslash, double-quote, newline, carriage-return, and tab. The TUI one additionally emits every other C0 control (`U+0000-U+001F` minus `\n\r\t`) as `\uXXXX`. A style or run-id containing e.g. BEL (`\x07`) or ESC (`\x1b`) round-trips through the TUI saver but produces a `voyage.toml` that `tomllib` rejects when `cmd_init` writes it via `default_config_toml`.

TOML forbids raw control characters in every string form; the only valid representation is an escaped basic string. Two escapers is exactly the drift the codebase otherwise avoids via single-source helpers (cf. `BACKEND_REGISTRY`, `resolve_config`).

## Rationale

- A config writer that can emit a file its own reader rejects is a durability bug: exit 0 at `init`, hard failure at the next `load_config`.
- The attack/accident surface is creator free-text (`--style`, `--run-id`), the least sanitized input in the system.
- The codebase already converged on single-source resolvers for backends, presets, and absent-encoding (`Unset`); escapers are the remaining hand-duplicated pair, and they have already diverged.

## Live evidence (read, not memory)

`voyage/config.py:571-587`:

```python
def _toml_basic_string(raw_value: str) -> str:
    """Quote free-text as a TOML basic string.
    ...
    """
    escaped_value = (
        raw_value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped_value}"'
```

`voyage/tui_state.py:419-437`:

```python
def _toml_string(raw: str) -> str:
    """Quote a string as a TOML basic string (every control escaped). ..."""
    escaped = (
        raw.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    for code in range(0x20):
        control = chr(code)
        if control in ("\n", "\r", "\t"):
            continue
        escaped = escaped.replace(control, f"\\u{code:04X}")
    return f'"{escaped}"'
```

`voyage/config.py:616-617` (consumer uses the weak one):

```python
    escaped_run_id = _toml_basic_string(run_id)
    escaped_style = _toml_basic_string(style)
```

Sweep probe (in-container, `voyage:latest`, quoted from the preserved Track B result):

```
config escaper repr: '"evil\x07bell\x1bX"'
tui escaper repr:    '"evil\\u0007bell\\u001BX"'
config TOML parse FAIL: TOMLDecodeError Illegal character '\x07' (at line 3, column 19)
```

`default_config_toml('r','style'+raw,0)` → `tomllib.loads` raises.

## Repro

```bash
# Host has no pydantic by design; run inside the image against the live tree:
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.config import default_config_toml
import tomllib
toml_text = default_config_toml('r', 'style\x07bell', 0)
print(tomllib.loads(toml_text))"
# Expected before fix: tomllib.TOMLDecodeError: Illegal character '\x07'
```

Grep to find both escapers:

```bash
grep -rn "_toml_basic_string\|def _toml_string" voyage/config.py voyage/tui_state.py
```

## Fix candidates

1. Delete `config._toml_basic_string`; import/share `tui_state._toml_string` (or move both into a new `voyage/toml_util.py` with zero other dependencies so neither direction risks a cycle).
2. Keep the docstring's rationale updated — the "later batch" has landed; the split is now pure debt.
3. Add a round-trip property test: Hypothesis text with C0 controls → `tomllib.loads(default_config_toml(...))` parses and the style survives (`tests/test_config_toml_roundtrip.py`).
4. Consider also escaping DEL (`U+007F`) explicitly if the chosen TOML spec revision forbids it raw (harmless either way inside a basic string).

## Refs (with links/quotes)

- TOML v1.1.0: "Basic strings … Any Unicode character may be used except those that must be escaped: quotation mark, backslash, and the control characters other than tab (U+0000 to U+0008, U+000A to U+001F, U+007F)." — https://toml.io/en/v1.1.0
- Same bug class upstream: "TOML forbids raw control characters … in every string form … so a description … containing e.g. a NUL or escape byte produced a TOML file that `tomllib` rejects … route such strings through a shared `toml_escape_basic` helper." — https://github.com/github/spec-kit/pull/3402
- Precedence/single-source discipline (applies to helpers as much as values): "Encode that order in one place … with no `if ENV == …` branching anywhere." — https://python-config-secrets-hub.com/core-configuration-patterns-file-formats/configuration-precedence-rules/

## Progress log (2026-09-30, cli/config/tui_state track — this change)

- Premise re-verified against CURRENT live code (as-read, in-container):
  duality still holds — `voyage/config.py:588-604` (`_toml_basic_string`,
  escapes only backslash/quote/newline/return/tab) vs
  `voyage/tui_state.py:419-438` (`_toml_string`, plus the full C0 loop),
  with `default_config_toml` (`voyage/config.py:634-635`) consuming the
  weak one. Live probe: `default_config_toml('r', 'style\x07bell', 0)` →
  `tomllib.TOMLDecodeError: Illegal character '\x07'`. Import-direction
  check: `tui_state` already imports `config` lazily in four places
  (`tui_state.py:235,255,327,365`) and `cli` lazily, while `config`
  imports only `errors` at top level — so the single home must be
  `config.py` (`config` importing `tui_state` would add a reverse edge;
  a new `toml_util.py` is disallowed by this task's file scope, which
  restricts new helpers to `cli.py`/`config.py`).
- TDD: new `Voyage/tests/test_cli_validate_handoff.py` — BEL/ESC pins,
  DEL/NUL pins, and a Hypothesis round-trip property (any sub-ASCII text
  with controls through `default_config_toml` → `tomllib.loads` parses +
  style survives) all failed first with `TOMLDecodeError`, pass after.
- Fix: `config._toml_basic_string` upgraded to the full-C0 version (short
  escapes + `\uXXXX` for every other C0 control + `\u007F` for DEL, per
  the TOML v1.1 control rule cited above); `tui_state._toml_string`
  is now a thin alias delegating via lazy import (preserves its
  stdlib-only import time). One implementation, two names (back-compat).
- Evidence: `test_cli_validate_handoff.py` 6 passed; related suites
  (`test_config_resolution`, `test_tui_state`, `test_generate`,
  `test_av_alignment_consumer`, `test_sfx_finalize`) green. `ruff check`
  + `ruff format --check` + `mypy strict` clean on `voyage/config.py`,
  `voyage/tui_state.py`, and the new test file.

## Resolution

- Status: resolved. Single shared escaper lives in `voyage/config.py`
  (`_toml_basic_string`, full-C0 + DEL); `voyage/tui_state.py`
  (`_toml_string`) delegates to it. No `DESIGN.md` edit made here —
  proposal: in the config/TUI as-built, note that all TOML basic-string
  quoting flows through the one `config._toml_basic_string` helper (C0 +
  DEL escaped), so creator free-text (`--style`, `--run-id`, TUI fields)
  can never emit a file the reader rejects.
