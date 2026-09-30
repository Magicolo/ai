# 131 — `test_backend_select_is_operable` never operates the dropdown: programs `.value`, bypasses the overlay path

- **Severity:** LOW-MEDIUM (tests — a test named "operable" proves programmatic set, not operation; the blank-dropdown overlay class is uncovered)
- **File:line:** `Voyage/tests/test_tui_app.py:149-169` (sets `app.query_one("#field-backend", Select).value = "fake"` at `:162`, asserts `_read_form().backend == "fake"`); `Voyage/voyage/tui.py:283-285` (`_read_choice_field` — what the test actually exercises)
- **Area:** (a) TUI Pilot failure matrices — tests that assert without exercising the named interaction

## Description

The test named "operable" does not operate anything: it assigns the `Select.value` property in-process, awaits one Pilot pause, and asserts the form reader + GPU-warning line update. That path (`value` setter → `Select.Changed` → `on_select_changed` → `_refresh_plan_and_errors`/`_refresh_gpu_warning`) is real logic, but the *interaction* the name promises — opening the dropdown overlay, navigating options, committing a choice — never happens. Concretely uncovered:

- The `Select` overlay open/commit path (the exact layer behind the 2026-09-25 liveness pass: "blank dropdowns" — an overlay that opens but paints no value line; cf. `test_select_values_render_in_form_text`, which snapshots the *closed* render only).
- Keyboard operation of the dropdown (enter/space/arrows on a focused `Select`); no test in the file presses any of them — `grep 'press("enter")\|press("space")\|press("down")\|press("up")' tests/test_tui_app.py` returns zero hits (9 `press` calls total, all text or `ctrl+g`/`ctrl+x`/`b`/`tab`).
- A regression where the overlay fails to open (or commits a different option than highlighted) passes this test, because the test writes past the overlay entirely.

The sibling `test_backend_select_is_visible_with_default_and_affordance` (`:129-146`) honestly names what it checks (visibility + `▾` affordance). The `_is_operable` name overclaims the same way for the interaction half. (`test_gpu_warning_line_present` uses the same programmatic pattern but names itself "present" — honest, out of scope.)

## Rationale (non-overlap)

- 093 (click-True + startup budgets, cited in-file) is about click landing and worker-startup waits, not Select interaction.
- 113 (TuiProgress content) and 114 (checkbox help) cover other TUI surfaces; neither mentions the Select overlay.
- 088 folds the TUI trio structurally ("load-flaky") without naming any interaction gap.
- `grep -rln "Select" Voyage/issues/*.md` names no overlay-operation finding (113 mentions dropdowns only as photon layout context, if at all).
- **Overlaps with:** 130 (same file, fixed-sleep render/tick budgets — different test, same Pilot-fragility family; fix separately).

## Live evidence (verified 2026-09-30, host source read)

- `tests/test_tui_app.py:160-167`: `app.query_one("#field-backend", Select).value = "fake"` → `await pilot.pause()` → assert `_read_form()` + `#gpu-warning` — no overlay open, no key press, no `Select.Changed`-via-UI.
- `grep -c press tests/test_tui_app.py` → 9; none targets a `Select` (`enter`/`space`/`down`/`up` all absent).
- `voyage/tui.py:842-844` (`on_select_changed` handler) — the only layer the test reaches, shared with programmatic sets.

## Repro

```bash
sed -n '149,169p' Voyage/tests/test_tui_app.py
grep -n 'press(' Voyage/tests/test_tui_app.py
# A broken overlay still passes: stub Select.open to raise and rerun —
# test_backend_select_is_operable stays green because it never calls open.
```

## Fix candidates

1. Drive the real interaction in-Pilot: focus `#field-backend`, `pilot.press("enter")` (or `"space"`), navigate, commit, then assert `_read_form().backend` + warning line — the keyboard path the liveness pass cared about.
2. If overlay-driving proves too version-fragile, rename to `test_backend_select_value_propagates` and file the overlay path as an explicit gap (this file) so the name stops promising it.
3. Assert the overlay opens at all (`SelectOverlay` mounted / expanded state) as a middle ground.

## Refs

- `Voyage/tests/test_tui_app.py:129-169`; `Voyage/voyage/tui.py:283-285,842-844`; DESIGN §140 2026-09-25 liveness pass (blank dropdowns); `Voyage/issues/113_*`, `114_*` (adjacent TUI surfaces, no overlap).
