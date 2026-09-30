# 130 — TUI Pilot fixed-sleep-no-retry waits below the 093 window: SVG snapshot (0.3 s) + heartbeat tick (2.5 s)

- **Severity:** LOW-MEDIUM (tests — load-flaky Pilot waits; same class as the 093 flakes, different budgets)
- **File:line:** `Voyage/tests/test_tui_app.py:732-738` (`await pilot.pause(0.3)` before the SVG snapshot); `Voyage/tests/test_tui_app.py:930-934` (`await pilot.pause(2.5)` waiting for heartbeat ticks); contrast the poll-everywhere pattern in the same file: `_click_generate_when_ready` (`:42-58`, up to 100 click polls) and `_STARTUP_WAIT_ITERATIONS = 600` (`:61-65`, 30 s startup budget per the 093 lesson)
- **Area:** (a) TUI Pilot failure matrices — Pilot timing budgets

## Description

Two Pilot tests wait with a single fixed sleep and no poll/retry loop, while every other async wait in the same file polls until a condition holds:

1. `test_select_values_render_in_form_text` (`:730-738`) mounts the app, does one `await pilot.pause(0.3)`, then snapshots the SVG and asserts `"ltxv"`, `"qwen"`, `"fp8"` appear as visible text. Under a loaded box (the exact condition 093 documents — thread/event-loop scheduling starves Pilot budgets), 0.3 s may elapse before the three `Select` widgets finish their first paint, and the test fails with no retry — the assertions read a render that had not settled.
2. `test_run_head_ticks_elapsed_while_running` (`:918-934`) starts a slow generate, waits for `_generation_running`, then does one `await pilot.pause(2.5)` and asserts `"elapsed" in head`. The heartbeat ticks on a 1.0 s `set_interval` (`voyage/tui.py:943-949`). 2.5 s allows barely two ticks with zero scheduling slack; one starved interval and the head still shows the pre-tick text.

Both are the structural inverse of the file's own hardened pattern: the click helper polls 100× because "a single `pause()` is not enough under load" (`:44-49`), and startup gates poll 600× because "10 s was too tight when the box is shared" (`:61-65`). These two waits never got the same treatment.

## Rationale (non-overlap)

- The 093 lesson (cited in the file at `:43` and `:64`: click-True helper + 10 s→30 s startup budgets) covers worker-startup gates and click landing only. Neither the 0.3 s render-settle pause nor the 2.5 s heartbeat pause is a worker-startup wait, and neither appears in any `Voyage/issues/` file (`grep -rln "STARTUP_WAIT\|never became clickable\|run_test\|Pilot" Voyage/issues/` hits only 088/089/113/114 — 088 mentions the Pilot file as "load-flaky" fold rationale without naming a budget; 089 is cache/marker hygiene; 113/114 are progress-content/checkbox-help).
- 046 is augment decode, unrelated. No existing file names a fixed-sleep render/tick budget.
- **Overlaps with:** 131 (same file, Select-overlay interaction gap — different test, same Pilot-fragility family; fix separately).

## Live evidence (verified 2026-09-30, host source read; no container needed)

- `tests/test_tui_app.py:730-738`: single `await pilot.pause(0.3)` then `_form_svg_rows` + three substring asserts — no loop, no condition.
- `tests/test_tui_app.py:930-934`: `assert app._generation_running` then single `await pilot.pause(2.5)` then `"elapsed" in head` — no tick poll.
- `voyage/tui.py:943-949`: heartbeat is `set_interval(1.0, ...)` — 2.5 s buys ~2 ticks best case.
- `tests/test_tui_app.py:42-65`: the file's own precedent — poll-until-landed (100×) and poll-until-started (600×).

## Repro

```bash
sed -n '724,739p;918,942p' Voyage/tests/test_tui_app.py
grep -n "pause(0.3)\|pause(2.5)" Voyage/tests/test_tui_app.py
# Load-flake demonstration: run the two tests in a full-suite run on a
# shared box (alongside a GPU job) vs in isolation; the single-sleep
# waits fail under load and pass idle, while the polling tests hold.
```

## Fix candidates

1. Poll for the render condition: loop `save_screenshot` + substring check up to ~5 s (same shape as `_click_generate_when_ready`), asserting once the text lands.
2. Poll for the tick: loop until `"elapsed" in head` up to ~10 s instead of one `pause(2.5)`.
3. Alternatively pin both to the file's existing budget idiom (`_STARTUP_WAIT_ITERATIONS`-style poll with a named constant + comment citing this file).

## Refs

- `Voyage/tests/test_tui_app.py:42-65` (093 poll precedent); `Voyage/voyage/tui.py:943-949` (1.0 s interval); `Voyage/issues/088_*` (fold rationale, no budget named); AGENTS.md §11 (093-Pilot click-True + 30 s budgets).

## Progress log (2026-09-30, Group D pass)

- Premises re-verified live (host reads, tree as-read): single
  `pause(0.3)` at `:733`, single `pause(2.5)` at `:932` — both as filed.
- Both waits converted to poll-until-landed (file's own idiom):
  SVG render 25×0.2s (~5s budget, breaks early) and heartbeat 20×0.5s
  (~10s budget); the final asserts still fail loudly on a truly blank
  render / dead ticker.
- Adjacent hardening (same family, same file): `_click_generate_when_ready`
  now tolerates `OutOfBounds` per poll iteration. Cause (live, mid-pass):
  a concurrent agent added two checkboxes to `voyage/tui.py`, pushing the
  Generate button below the fold at test sizes — under box load the
  scheduled `scroll_visible` had not applied when `pilot.click` ran, and
  the raise (unlike a `False` miss) escaped the helper. 100-try budget and
  fail-loud `AssertionError` unchanged.

## Resolution (2026-09-30, Group D pass)

- Resolved: both fixed sleeps poll; click helper survives the raise.
- Files changed: `tests/test_tui_app.py` (SVG poll, heartbeat poll,
  helper `try/except OutOfBounds`). Gate evidence: full file 33/33
  in-container (34.29s); `ruff check` + `ruff format --check` clean.
  DESIGN proposals: none. Residuals: none in 130's scope (the helper is
  shared with the 093 lesson — noted, not changed beyond the tolerance).
