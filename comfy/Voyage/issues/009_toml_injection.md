# 009 — TOML injection / self-DoS via unescaped `style`/`run_id` in `default_config_toml`

- Status: fixed
- Severity: major (config integrity)
- Area: robustness — config generation
- Rank rationale: a style string with a quote breaks the generated TOML or
  injects live tables; the correct escaper already exists in the TUI.

## Technical description

`voyage/config.py:271,287-292` interpolates raw values:

```python
run_id = "{run_id}" ...
style = "{style}"
```

The TUI already has a correct escaper (`_toml_string`: backslash/quote/newline/CR/
tab) used for `~/.config/voyage/tui-last.toml` (`voyage/tui_state.py:285-295`) —
`config.py` doesn't use it.

## Why this is an issue

Style is free-text creator input that flows straight into the persisted run
charter, so any quote, newline, or pasted prose with brackets either breaks the
generated TOML or injects live tables — including a `[video]` override that
self-DoSes init with a duplicate-table error. The charter carries the config
digest that pins reproducibility for the entire run, so corrupting it at
creation poisons provenance from segment zero. The correct escaper already
exists in the TUI, making this a reuse gap rather than a design problem.
Creators pay with failed inits at the exact moment they write an expressive
style string.

## Evidence

```
$ python3 -c "...default_config_toml('voyage', evil, 0)..."
injection-parses: True
```

where `evil = 'x"\n[evil]\npwned=true\n#'`. The generated TOML contains a live
`[evil]` table (parses OK — pydantic silently ignores extra keys). A second probe
from the supply-chain sweep showed `style='x"\n[video]\nbackend="ltxv'` raises
`ConfigurationError: Cannot declare ('video',) twice` — an init-time self-DoS on
a crafted `--style`. (Note: `load_config` in this tree takes a `Path`, so the
probe's `str` call failed with `AttributeError: 'str' object has no attribute
'read_bytes'` — that call-shape papercut is incidental to this issue.)

## Reproduction

1. `voyage init --style 'x"\n[video]\nbackend="ltxv' ...` → generated TOML
   either carries a foreign table or fails to parse (DoS).
2. Any style containing `"` or newline is affected; styles are user free-text and
   persisted into the run charter.

## Source references

- `voyage/config.py:271,287-292,359`; `voyage/tui_state.py:285-295` (good escaper).

## Resolution candidates

1. Reuse `_toml_string` (move to a shared helper, e.g. `voyage/toml_util.py`) or
   build TOML via `tomli-w`.
2. Add tests: quote/newline/`[video]` styles round-trip through
   `default_config_toml` → `load_config` byte-identical.

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep; injection re-verified live
  (`injection-parses: True`).
- Open: implement + tests.
- 2026-09-25 (repair pass): added `## Why this is an issue`; injection probe
  re-run live (`injection-parses: True`, foreign `[evil]` table); refs verified
  current (`config.py:271,289-290`; `tui_state.py:285`).
- 2026-09-25 (fix): FIXED — added local `_toml_basic_string` escaper in
  `voyage/config.py:272-288` (backslash/quote/newline/CR/tab, mirrors the TUI
  escaper; kept local per the batch split — `tui_state.py` untouched, a later
  batch owns the shared-helper move) and applied it to `run_id`/`style` in
  `default_config_toml` (`voyage/config.py:307-313`). Verified live: hostile
  styles (`x"[video]...`, `x"[evil]...`) parse with no foreign table and
  round-trip byte-identical through `load_config`; safe styles render
  identically to before. Tests: `test_config_toml_escapes_style_injection` +
  `test_config_toml_escapes_quotes_and_newlines` in `tests/test_unit.py`.
  Gates: ruff + format + mypy strict clean on all scope files, 424 pytest
  passed in-container (full `gates.sh` still flags 18 pre-existing
  `issues/*.md` format nits outside this batch's scope).
- Resolution: fixed as above; no foreign-table injection, round-trip pinned.
