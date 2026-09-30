# 034 — Stale / wrong-code `type: ignore`s; `warn_unused_ignores` debt is real

- Severity: LOW (typing hygiene — wrong-code ignores suppress nothing)
- Group: standards/typing — Rank: 4/5
- File:line: `tests/test_commit_hardening.py:168,175`; `tests/test_tui_app.py:532`; `Voyage/pyproject.toml:175-180`
- Overlaps: 031/032/033 cluster — fix wrong codes first (zero-behavior), then expand 033 gate.

## Description

Three rot modes coexist: (a) unused ignores; (b) wrong-code ignores where
the code does not cover the error (`method-assign` masking
`assignment`/`attr-defined`; `union-attr` vs `attr-defined`); (c)
config-level `disable_error_code=["unused-ignore"]` for
`voyage.config`/`voyage.tui_state` (`pyproject.toml:175-180`) that silences
the detector instead of deleting two one-line comments.

## Rationale

Mypy documents `warn_unused_ignores` as the upgrade-hygiene flag. Wrong-code
ignores are worse than none: they signal triage while suppressing nothing,
and `warn_unused_ignores=true` (already set, `pyproject.toml:119`) cannot
fire where the gate never looks (issue 033) or where the code is disabled by
override.

## Live evidence (re-verified 2026-09-30)

Host `rg` today: 175 total `type: ignore` hits (`voyage/` + `tests/`).

```
tests/test_tui_app.py:532:        rows = app.query(".field-row")  # type: ignore[union-attr]
tests/test_crash_matrix.py:89:        supervisor._video.call = _crash_once  # type: ignore[method-assign]
tests/test_precision.py:20:  def _base_config(tmp_path):  # type: ignore[no-untyped-def]
tests/test_precision.py:51:  def test_draft_preserves_quantization(tmp_path) -> None:  # type: ignore[no-untyped-def]
tests/test_precision.py:58:  def test_quantization_override(tmp_path) -> None:  # type: ignore[no-untyped-def]
tests/test_commit_hardening.py:168:  supervisor_module.validate_video = _zero_divide  # type: ignore[method-assign]
tests/test_commit_hardening.py:175:  (restore line, same wrong code)
tests/test_commit_split.py:108,116,245:  …  # type: ignore[method-assign]
tests/test_commit_split.py:198,264:  config.video.backend = "causvid"  # type: ignore[assignment]
voyage/config.py:20:    import tomli as tomllib  # type: ignore[import-not-found, no-redef]
voyage/tui_state.py:31:    import tomli as tomllib  # type: ignore[import-not-found, no-redef]
```

`Voyage/pyproject.toml:175-180` override block read live:

```toml
[[tool.mypy.overrides]]
module = ["voyage.config", "voyage.tui_state"]
disable_error_code = ["unused-ignore"]
```

Sweep histogram (preserved output): `attr-defined 66, method-assign 31,
arg-type 28, import-not-found 21, no-untyped-def 11 …`.

## Repro

```bash
rg -n "type: ignore" voyage/ tests/ | wc -l
mypy tests 2>&1 | grep -B1 "not covered"
```

## Fix candidates

(a) Delete the `voyage.config`/`voyage.tui_state` override block together
with the two shim comments (the pyproject comment itself prescribes this).
(b) Bulk-fix wrong codes (`method-assign`→`assignment`,
`union-attr`→`attr-defined` or real narrowing).
(c) Run `mypy --warn-unused-ignores` over the full tree and delete what fires.

## Refs

- Mypy `command_line.rst` (`--warn-unused-ignores`).
- `Voyage/pyproject.toml:119,175-180`
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §4.
