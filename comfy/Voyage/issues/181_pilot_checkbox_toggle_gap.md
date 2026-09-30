# 181 — No Pilot test ever toggles a TUI checkbox: `_read_form` flag wiring is interaction-untested

- **Severity:** LOW-MEDIUM (tests — a swapped flag id in `_read_form` passes the whole suite; the 131 class on a different widget family)
- **Track:** A (TUI Pilot failure matrices — below 130/131/114)
- **Verified live:** 2026-09-30 host read (tree as-read; no edits in these ranges)

## File:line (live-verified)

- `voyage/tui.py:667-691` (five `Checkbox` constructions: `#flag-draft`, `#flag-force`, `#flag-skip-bad`, `#flag-verbose`, `#flag-no-color`)
- `voyage/tui.py:760-780` (`_read_form`: `force=_read_flag_field(self, "#flag-force")` at `:767`, `skip_bad` at `:768`, `draft` at `:769`, `verbose` at `:778`, `no_color` at `:779`)
- `voyage/tui.py:288-290` (`_read_flag_field`: `return app.query_one(widget_id, Checkbox).value`)
- `Voyage/tests/test_tui_app.py` (33 tests): zero `click` on any `#flag-*`, zero `Checkbox` queries, zero toggles — `rg "Checkbox|flag-draft|flag-force|flag-skip|flag-verbose|flag-no-color|click\("` shows `click` only at `:56` (`#button-generate` helper); all 9 `press` calls are text or `ctrl+g`/`ctrl+x`/`b`/`tab`
- `Voyage/tests/test_tui_state.py:35-45` (flags covered at state→namespace level only — `force=True, skip_bad=True, draft=True, verbose=True, no_color=True` into `to_generate_namespace`; the widget→state half in `_read_form` has no caller)

## Description

131 proved the backend `Select` is only tested programmatically (`.value =` assignment, never the overlay). The five checkboxes are worse: they are not tested AT ALL — not programmatically, not by interaction. `test_tui_state.py` covers state→namespace (given `force=True`, the namespace carries it), but nothing covers widget→state (given the user checks "Force", `_read_form().force` becomes `True`). Concretely uncovered: a transposition in `_read_form` (e.g. `force=_read_flag_field(self, "#flag-draft")`) passes every test, because no test ever sets a checkbox to a non-default value and reads it back. The default-off coincidence (`GenerateFormState` defaults all five to `False`, matching unchecked boxes) masks the wiring the same way `run_id == name` masks 148.

## Rationale

- The 131 rationale applies verbatim with "overlay" replaced by "toggle": the test suite proves the logic downstream of the interaction while the interaction itself (the layer users actually touch) is uncovered.
- Flags are load-bearing: `force` decides init-into-non-empty, `skip_bad` decides abort-vs-salvage, `draft` switches geometry, `verbose`/`no_color` switch the console. A silent swap ships the wrong run.
- Fix is five toggle-and-read assertions following the existing `_read_form().backend == "fake"` pattern at `test_tui_app.py:164`.

## Live evidence (host read, 2026-09-30)

```
$ rg -n "Checkbox|flag-draft|flag-force|flag-skip|flag-verbose|flag-no-color" Voyage/tests/test_tui_app.py
# (no output — zero hits)

$ rg -n "click\(|press\(" Voyage/tests/test_tui_app.py | head -12
56:        if await pilot.click("#button-generate"):
104:            await pilot.press(*"quiet harbor")
...
# clicks: 1 (generate button only); presses: text/ctrl+g/ctrl+x/b/tab — no flag toggle.

$ rg -n "_read_form|_read_flag" Voyage/tests/*.py
Voyage/tests/test_tui_app.py:164:            assert app._read_form().backend == "fake"
# The ONLY _read_form assertion in the suite — a Select, never a flag.
```

## Repro

```bash
rg -n 'Checkbox|flag-' Voyage/tests/test_tui_app.py Voyage/tests/test_tui.py
# Empty: no Pilot or unit test references any checkbox widget id.
sed -n '760,780p' Voyage/voyage/tui.py  # the five untested flag reads
# Mutation proof: swap two flag ids in _read_form, run the TUI suite — still green.
```

## Fix candidates

1. Pilot toggle test per flag (or one parametrized test over the five ids): check the box via `pilot.click("#flag-force")`, assert `_read_form().force is True` + plan/errors refresh; uncheck, assert `False`. Follows the `:160-167` backend-propagation pattern.
2. If click-toggling proves version-fragile, set `.value = True` in-process (the 131 programmatic pattern) AND rename honestly (`test_flag_values_propagate_from_widgets`), keeping this file as the interaction-gap record.
3. Assert the checkbox count/ids (`5` boxes, exact id set) so a deleted/renamed flag fails loudly instead of silently unwiring.

## Refs

- `voyage/tui.py:667-691,760-780,288-290`; `tests/test_tui_app.py:160-167` (the lone `_read_form` assertion); `tests/test_tui_state.py:35-45` (state→namespace half, widget half missing).
- Not-a-duplicate: 131 (backend-Select overlay operation — different widget family, different interaction); 130 (Pilot sleep budgets — timing, not coverage); 114 (checkbox HELP — explicitly scopes out handlers/wiring: "help coverage only").
