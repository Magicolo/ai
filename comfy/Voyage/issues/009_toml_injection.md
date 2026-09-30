# 009 — TOML injection / self-DoS via unescaped `style`/`run_id` in `default_config_toml`

- Status: resolved in live tree (`_toml_basic_string` escaper landed)
- Severity: HIGH (config integrity; resolved, record only; residual C0 gap → 020)
- Group: robustness/config — Rank: 2/5 (fixed; remainder in 020)
- Area: robustness — config generation (`voyage/config.py`)
- Rank rationale: a style string with a quote broke the generated TOML or injected
  live tables; the correct escaper already existed in the TUI.

## Technical description

Pre-fix, `voyage/config.py:271,287-292` interpolated raw values:

```python
run_id = "{run_id}" ...
style = "{style}"
```

The TUI already had a correct escaper (`_toml_string`: backslash/quote/newline/CR/
tab) used for `~/.config/voyage/tui-last.toml` (`voyage/tui_state.py:285-295` at
pass 1) — `config.py` didn't use it.

Live state (re-verified 2026-09-30): `voyage/config.py:571-587`
(`_toml_basic_string`: `\`, `"`, `\n`, `\r`, `\t` escaped) applied to both
free-text fields (`:616-617`, `run_id = {escaped_run_id}` / `style = {escaped_style}`).

## Why this is an issue

Style is free-text creator input that flows straight into the persisted run
charter, so any quote, newline, or pasted prose with brackets either broke the
generated TOML or injected live tables — including a `[video]` override that
self-DoSes init with a duplicate-table error. The charter carries the config
digest pinning reproducibility for the entire run, so corrupting it at creation
poisoned provenance from segment zero. The correct escaper already existed in the
TUI, making this a reuse gap rather than a design problem.

## Evidence

Live verification 2026-09-30 (in-container, `voyage:latest`):

```
$ docker run ... python3 -c "from voyage.config import default_config_toml; ..."
evil = 'x"\n[evil]\npwned=true\n#'
has-live-evil-table: True        # substring inside the QUOTED value (expected)
top-level-keys: ['audio', 'augment', 'director', 'draft', 'experimental',
  'min_free_space_gib', 'run_id', 'schema_version', 'seed', 'sfx',
  'style', 'video', 'voyage']
```

No top-level `evil` table survives parsing — the `[evil]` substring is inert
inside the escaped basic string. Pre-fix probes showed `injection-parses: True`
with a live `[evil]` table, and `style='x"\n[video]\nbackend="ltxv'` raised
`ConfigurationError: Cannot declare ('video',) twice` (init-time self-DoS).

## Reproduction

1. Pre-fix: `voyage init --style 'x"\n[video]\nbackend="ltxv' ...` → generated TOML
   either carried a foreign table or failed to parse (DoS).
2. Any style containing `"` or newline was affected; styles are user free-text and
   persisted into the run charter. Post-fix: quote/newline/`[video]` styles
   round-trip through `default_config_toml` → `load_config` byte-identical.

## Source references

- `voyage/config.py:571-587` (escaper), `:590-621` (use);
  `voyage/tui_state.py:419-438` (`_toml_string`, TUI twin).

## Resolution candidates

1. (Landed) Local `_toml_basic_string` in `config.py` (mirrors the TUI escaper;
   kept separate per batch scope) — or build TOML via `tomli-w` as a follow-up.
2. Tests: quote/newline/`[video]` styles round-trip through
   `default_config_toml` → `load_config` byte-identical.

## Online references

- TOML v1.0.0, basic strings — "Any Unicode character may be escaped ... `\"`,
  newlines must be escaped as `\n`":
  https://toml.io/en/v1.0.0
- TOML v1.0.0, tables — a `[table]` header opens a new table (the injection
  primitive when quoting is skipped): https://toml.io/en/v1.0.0

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep; injection re-verified live.
- Resolution batch 1: `_toml_basic_string` + round-trip tests landed.
- 2026-09-30: re-verified live (escaper + parsed-keys evidence above);
  reconstructed from archived pass-1 text (commit `b5d7dda`). Status → resolved.
