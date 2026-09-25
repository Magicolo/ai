# 093 — `VoyageApp._running` collides with Textual `App._running`; Pilot click-after-`scroll_visible` racy

- Status: open
- Severity: low (flaky-gate class; state-masking hazard)
- Area: TUI tests — `voyage/tui.py` (`VoyageApp`), `tests/test_tui_app.py`
- Rank rationale: found during batch-2 TUI verification (bisected, pre-existing,
  out of that batch's scope); a name collision that masks generation state plus
  a racy Pilot idiom used across the suite.

## Technical description

While verifying batch-2 TUI fixes, `test_tui_app.py::
test_base_exception_in_worker_restores_form` flaked twice mid-task (30 s hang),
then passed consistently (109/109, then 563/563 green under lower load).
Bisection showed timing, not the batch's code: it passed with HEAD-`tui.py` +
new `tui_state.py`, then passed with the full tree on retry. Two contributing
factors found while debugging:

1. `VoyageApp._running` collides with Textual `App`'s own `_running` flag (True
   from mount), which masks generation state in that test's first wait loop —
   the test waits on an attribute the framework also writes.
2. Pilot click-after-`scroll_visible` is racy across the suite (no settle
   guarantee between scroll and click).

## Why this is an issue

- A framework-colliding attribute name silently corrupts generation-state reads;
  any future code branching on `self._running` risks reading Textual's flag.
- Racy Pilot idioms produce load-dependent flakes that erode gate trust (this
  exact test hung for 30 s twice, then went green — the worst flake signature).

## Evidence

Batch-2 TUI track report (2026-09-25): bisection transcript (HEAD-tui.py +
new-tui_state.py pass; full-tree retry pass under lower load); `rg -n
"_running" voyage/tui.py` shows the collision; `rg -n "scroll_visible"
tests/test_tui_app.py` shows the racy sites.

## Reproduction

Run `tests/test_tui_app.py::test_base_exception_in_worker_restores_form`
repeatedly under load; observe intermittent ~30 s hangs. Inspect
`VoyageApp._running` vs `App._running` in the Textual runtime.

## Source references

- `voyage/tui.py` (`VoyageApp`, `_running` uses); `tests/test_tui_app.py`
  (scroll/click sites, the flaking test).

## Resolution candidates

1. Rename `VoyageApp._running` to `_generation_running` (or similar) everywhere.
2. Add a settle (e.g. `await pilot.pause(...)` or visibility assertion) between
   `scroll_visible` and click in Pilot tests, or a shared helper.
3. Consider a stress-repeat marker for the suite's timing-sensitive tests.

## Investigation / progress / resolution log

- 2026-09-25: filed by the orchestrator from the batch-2 TUI track's bisection
  notes (no code changed yet).
- Open: rename + harden Pilot idiom + tests.
