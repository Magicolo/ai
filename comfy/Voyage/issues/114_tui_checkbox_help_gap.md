# 114 — Five TUI checkbox flags have no help: no tooltips, no `FIELD_HELP` keys, help panel falls back to overview

- **Severity:** LOW (usability: the focus-driven help panel — the form's replacement for field descriptions — goes silent on exactly the flags that need one sentence each)
- **Track:** second-pass TUI/CLI edges (TUI help coverage)
- **Verified:** 2026-09-30 by source read (no import needed; lines as-read)

## File:line (live-verified)

- `voyage/tui.py:667-691` (four `Checkbox(...)` constructions — no `tooltip=` argument, unlike every `Input`/`Select`/`TextArea` on the form, which all pass `tooltip=FIELD_HELP[...]`)
- `voyage/tui_state.py:66-86` (`FIELD_HELP`: 12 keys for the 13 text/select fields — no `draft`/`force`/`skip_bad`/`verbose`/`no_color`)
- `voyage/tui.py:84-99` (`FIELD_WIDGET_IDS` → `WIDGET_FIELD_NAMES`: only the 13 text/select widget ids; `flag-draft`/`flag-force`/`flag-skip-bad`/`flag-verbose`/`flag-no-color` map to nothing)
- `voyage/tui.py:802-820` (`_refresh_help`: unmapped widget id → `_HELP_OVERVIEW`; only `_BUTTON_HELP` (`:101-107`) gets the same button treatment)

## Description

Focusing any text/select field shows its description in the side help panel (plus any validation error). Focusing a checkbox — Draft profile, Force, Skip bad, Verbose, No color — shows the generic overview ("Focus any field to see what it does…"), because checkboxes have no tooltip, no `FIELD_HELP` entry, and no id mapping. The flags that most need one line (what does "Draft profile" change? what does Force risk? what does Skip bad salvage?) are the ones without it. `--help` documents the CLI equivalents, but the TUI user is by definition not reading `--help`.

Deliberately out of scope: there is no `on_checkbox_changed` handler (toggles don't refresh plan/errors/help). Checked live: plan (`_plan_details`) depends only on backend/blocks/duration and errors only on text/select fields, so the missing handler has zero visible effect today — not filed. This issue is help coverage only.

## Rationale

- The help panel exists because the compact form dropped card descriptions (per `test_help_panel_describes_focused_field`). A panel that answers 13 of 18 focusable inputs keeps 5 silent — including Draft, whose geometry/take-length effects are otherwise invisible in the plan line (see the draft-analysis note in evidence).
- Fix is five strings + five id mappings + (optionally) `tooltip=` args, following the exact pattern the file already uses for every other widget. Test pattern exists (`test_help_panel_describes_focused_field` — extend to one checkbox).

## Evidence (source, 2026-09-30)

```python
# tui.py:667-671 — no tooltip, unlike the field rows above it
yield Checkbox(
    "Draft profile (fast low-res iteration)",
    value=self.initial_state.draft,
    id="flag-draft",
)
# tui_state.py:66-86 FIELD_HELP keys: backend/duration/style/name/seed/
#   director/blocks/take_seconds/quantization/beats_per_segment/
#   drift_every_n/min_fps/min_resolution — no draft/force/skip_bad/verbose/no_color
# tui.py:813-816 — fallback for unmapped ids:
field = WIDGET_FIELD_NAMES.get(widget_id or "")
if field is None or field not in FIELD_HELP:
    body.update(_HELP_OVERVIEW)
```

## Repro

Pilot (headless): mount `VoyageApp`, `set_focus` on `#flag-draft`, assert `#help-body` content — currently the overview text, not a draft description. (Source read above is already conclusive; the Pilot form follows `test_help_panel_describes_focused_field`.)

## Fix candidates

- Add five `FIELD_HELP` entries (one line each; draft's should state geometry 640×352 + take-length 45s + "iteration only, never finals" per `DraftConfig` docs), map `flag-*` ids in a checkbox help table (or extend `FIELD_WIDGET_IDS` handling), pass `tooltip=` on the five `Checkbox` constructors.
- Test: focus each checkbox → help body contains its keyword; matches the existing help-panel test shape.

## Refs

- `tests/test_tui_app.py:172-194` (help-panel test — text fields only); `voyage/config.py:494-516` (`DraftConfig` semantics the draft line should summarize).

## Progress log

- 2026-09-30 (console/TUI track): re-verified live first — premise HOLDS as-read (`voyage/tui.py:669-705` pre-fix: 5 of 7 `Checkbox` constructions with no `tooltip=`; `voyage/tui_state.py:87-114` `FIELD_HELP` with 15 keys — `draft`/`force`/`skip_bad`/`verbose`/`no_color` absent; `_refresh_help` maps only `WIDGET_FIELD_NAMES`, so focused checkboxes fell back to `_HELP_OVERVIEW`). Batch-8 boxes (`no_download`/`no_sfx`) already carry tooltips + `FIELD_HELP` keys — new keys chosen disjoint (`draft`/`force`/`skip_bad`/`verbose`/`no_color`), no collision. TDD red-first: `tests/test_tui_checkbox_help_114.py` (3 tests) failed 3/3 in-container before the fix, green after. No concurrent hunks in owned files.

## Resolution

- Verdict: FIXED.
- Files changed: `voyage/tui_state.py` (5 new `FIELD_HELP` entries — draft states 640×352 + 45s takes + iteration-only per `DraftConfig`; force/skip_bad/verbose/no_color one line each); `voyage/tui.py` (`tooltip=FIELD_HELP[...]` on the 5 flag `Checkbox` constructors + new `FLAG_HELP_FIELDS` id→key table consulted by `_refresh_help` before the overview fallback; batch-8 `flag-no-download`/`flag-no-sfx` mapped too since they had tooltips but the same overview fallback on focus — same gap class, 2 extra dict entries, no behavior change elsewhere).
- Test evidence: new `tests/test_tui_checkbox_help_114.py` (3 passed — FIELD_HELP keys + tooltip args + focused-checkbox help-panel content, batch-8 keys pinned intact); `tests/test_tui_app.py` full file (33 passed); full suite 1690 passed / 8 failed, all foreign (see issue 113 log). Gates on touched files: `ruff check` + `ruff format --check` + `mypy` clean.
- DESIGN proposal (quoted text only, for the DESIGN owner — launcher-TUI help panel): "Every focusable form input — text, dropdown, and flag checkbox — resolves to a FIELD_HELP entry in the side help panel; checkbox ids live in tui.py FLAG_HELP_FIELDS (kept out of FIELD_WIDGET_IDS, which drives text/select error styling only)."
- Residuals: none — the deliberately-out-of-scope `on_checkbox_changed` handler stays out (plan/errors still depend only on text/select fields, verified live; toggling flags needs no refresh).
