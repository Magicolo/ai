# 072 — `tui-last.toml` control characters corrupt the file; next launch silently resets the whole form

- Status: resolved (fixed 2026-09-25, TUI track)
- Severity: low-medium (total form reset, not just the offending field)
- Area: TUI settings persistence — `voyage/tui_state.py:285-294`
  (`_toml_string`), `:350-363` (`load_last_settings`)
- Rank rationale: pass-2 finding; different payload (`_toml_string`, last-settings)
  and consequence (total reset) from 009's run-config injection.

## Technical description

```python
escaped = (raw.replace("\\", "\\\\").replace('"', '\\"')
    .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
return f'"{escaped}"'
```

No escaping of `\x00-\x08\x0b\x0c\x0e-\x1f`. `load_last_settings` catches
`TOMLDecodeError` by returning full defaults.

## Why this is an issue

One control character in a style or name string corrupts the persisted TOML,
and the loader's answer is a silent total reset — every field back to defaults,
not just the offending one. The user loses all settings with no message naming
the culprit, and the corruption recurs on the next save until the character is
found by hand.

## Evidence (live probes by pass-2 sweep)

```
style='a\x01b' → file bytes contain raw \x01 → tomllib: TOMLDecodeError: Illegal character '\x01'
loaded backend: 'ltxv' name: 'voyage' style: ''   (everything reset, not just style)
style='hello\x00world' → roundtrip=False, raw contains NUL
```

## Reproduction

Save `GenerateFormState(style='a\x01b', backend='fake', name='myrun')` via
`save_last_settings`, then `load_last_settings` → defaults.

## Source references

- Files/lines above.

## Resolution candidates

Escape remaining controls as `\uXXXX` in `_toml_string` (or reject control chars
in `field_errors` for style/name). Share the escaper with 009's config fix.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI sweep with live probes.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`tui_state.py:285-294` `_toml_string`, `:350-363`
  `load_last_settings` — both match; unescaped `\x00-\x1f` range confirmed).
- Open: implement + roundtrip tests.
- 2026-09-25 (fix, TUI track): relevance re-verified live (raw `\x01`
  in file bytes, load reset every field to defaults). Implemented the
  `\uXXXX` candidate in `_toml_string` (`voyage/tui_state.py`): short
  escapes kept for backslash/quote/`\n`/`\r`/`\t`, every other C0
  control (`\x00-\x1f`) emitted as `\uXXXX`. Live probe: file holds no
  raw control byte and `style='a\x01b\x00c\x0bd\x0ce\x1ff'` round-trips
  exactly with sibling fields intact. Tests: `test_tui_state.py`
  `test_last_settings_round_trip_control_characters` (5 payloads incl.
  NUL/`\x01`/VT/FF/US/tab — byte-clean file + exact round-trip).
  Gates: `scripts/gates.sh` GREEN (ruff + format + mypy strict +
  563 pytest).
