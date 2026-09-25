# 078 — TUI Stop path has no exception guard; corrupt `state.json` escapes the handler

- Status: resolved (fixed 2026-09-25, TUI track)
- Severity: low (exception out of button/key handler; no `■ …` feedback line)
- Area: TUI — `voyage/tui.py:990-1003` (was `:985-998` before concurrent
  edits shifted lines)
- Rank rationale: pass-2 TUI finding; every sibling handler guards, this one
  doesn't.

## Technical description

```python
def _request_stop(self) -> None:
    ...
    from voyage.persistence import read_state, write_state
    state = read_state(self._run_dir)   # raises StateError on corrupt JSON — no try
    state.status = "STOP_REQUESTED"
```

Contrast: `_start_generation` wraps view-switch in try/except (863-869),
`_generate_in_thread` catches `VoyageError` + `BaseException` (924-929),
`_refresh_plan_and_errors` guards `_read_form`. `read_state` on a corrupt file
raises `StateError: invalid state file …/state.json: Expecting value…` (probed
live by sweep). `_request_stop` only checks `state_path.exists()`, not
readability.

## Why this is an issue

A corrupt `state.json` turns the Stop button into an unhandled exception out of
the handler: no `■ …` feedback line, no stop, and the operator is left staring
at a running TUI with no indication the request failed. Every sibling handler
guards its fallible calls — this one path is the odd one out, and it guards
the control-plane action the operator reaches for when things already go wrong.

## Evidence

Corrupt-state probe (re-run 2026-09-25, `PYTHONPATH=Voyage`):

```
$ python3 -c "...write '{bad json' to <tmp>/state.json; read_state(tmp)..."
voyage.errors.StateError: invalid state file /tmp/.../state.json: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
```

`_request_stop` (`tui.py:990-1003`) checks `state_path.exists()` only, then
calls the raising `read_state` with no `try` — unlike `_start_generation`
(`:863-869`) and `_generate_in_thread` (`:924-929`).

## Reproduction

Corrupt `<run>/state.json`, press Stop/ctrl+x in run view → exception out of
handler, no feedback line.

## Source references

- Files/lines above.

## Resolution candidates

try/except `StateError` (and `OSError` on write) → `append_run_line("■ cannot
stop: state unreadable …")`.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; fixed stale
  lines (`:985-998` → `:990-1003`); re-ran corrupt-`state.json` probe
  (`StateError` confirmed, pasted above).
- Open: implement + corrupt-state Pilot test.
- 2026-09-25 (fix, TUI track): relevance re-verified live
  (`_request_stop` still called bare `read_state` after the
  `exists()` check). Implemented the candidate in `voyage/tui.py`:
  `read_state` wrapped in try/except `(StateError, OSError)` and
  `write_state` in try/except `OSError`, both surfacing
  `■ cannot stop: state unreadable/write failed (…)` feedback lines
  (function-level `StateError` import, matching file style). Tests:
  `test_tui.py::test_stop_with_corrupt_state_reports_feedback`
  (headless Pilot: run marked running + garbage `state.json` →
  `cannot stop` line, still running, no exception). Gates:
  `scripts/gates.sh` GREEN (ruff + format + mypy strict + 563 pytest).
