# 231 — `read_all_metric_events` silently drops torn lines; `parse_metric_lines` loudly counts them (two truths)

Severity: MEDIUM (track E-11).

## Technical description

`voyage/logrotate.py:445-470` (`read_all_metric_events`) `except ValueError: continue` —
torn lines vanish. `parse_metric_lines` (`:306-334`) returns `(events, torn)` with
non-empty non-object lines counted. Scoreboard/status/soak use the silent path; only
direct `parse_metric_lines` callers see the count.

## Rationale

§60 promises torn lines are "never silently dropped — callers surface the count"
(`:309-315` docstring). The most-used reader breaks that promise. A crash-torn
`metrics.jsonl` tail shows fewer commits with no `torn` signal.

## Live evidence

Live probe — `metrics.jsonl` with `{"event":"a"}\nnot-json\n{"event":"b"}` + rotated
sibling: `read_all_metric_events` → 3 events, 0 torn surfaced;
`parse_metric_lines(…, run_id="x")` → `(1 event, torn=2)`.

Repro: append `not-json\n` to `logs/metrics.jsonl` → `scoreboard_rows` stages silently
miss that commit; no `errors` cell mentions torn input.

## Source refs

`voyage/logrotate.py:306-334,445-470`; `voyage/scoreboard.py:67-99`.

## Online sources

- None beyond in-tree §60 contract (docstring at `logrotate.py:309-315`).

## Fix candidates

- Add `read_all_metric_events_counted()` returning `(events, torn)` (or a `torn`
  out-param), route scoreboard/status through it into the `errors` cell; keep the silent
  variant as a documented convenience.

## Log

- 2026-10-07: filed from read-only Track E sweep; no code touched.

## Evaluation (2026-10-07)
Live probe in-container: `read_all_metric_events` silently skips torn
lines (`except ValueError: continue`, no count) while
`parse_metric_lines` returns `(events, torn)`; no
`read_all_metric_events_counted` exists (`hasattr == False`).
`scoreboard._stages_by_segment` uses the silent loop,
`adopted_segment_ids` discards `_torn`, `run_status_summary` surfaces
torn only via `torn_metric_lines`, and `scoreboard_rows` errors cells
carry metric-shape issues but never torn history loss. Confirmed two
truths as filed. Fix adds the counted twin in `logrotate.py` and routes
scoreboard/status through it (minimal `scoreboard.py` routing hunks);
the silent variant stays as a documented convenience.

## Progress log
- `logrotate.py`: added `read_all_metric_events_counted(run_dir) ->
  (events, torn)` via `parse_metric_lines` per file (events match the
  silent variant exactly); updated `read_all_metric_events` docstring to
  name the silent convenience + counted twin (issue 231).
- `scoreboard.py`: imported the counted reader; `scoreboard_rows`
  appends `torn metric lines: N (...)` to every row's errors cell when
  torn > 0 (same wording as `validate_scoreboard`); `run_status_summary`
  sources `torn` from the counted reader (docstring updated).
- New tests in `tests/test_issue_231_counted_metrics.py` (5 tests):
  counted events equal silent events with torn == 2; empty run reads
  ([], 0); rows surface torn in errors; status torn matches counted;
  silent variant stays silent but documented.
- Verified: `ruff check` + `format --check` clean; `mypy` strict clean
  on `logrotate.py`/`scoreboard.py`; scoped pytest 113 passed.

## Resolution (2026-10-07)
Fixed as proposed: counted twin exists, scoreboard/status route through
it into errors cells (`scoreboard_rows` errors + `run_status_summary`
torn), silent variant kept and documented. No open items; `_stages_by_segment`
keeps its dict-only shape (rows carry the torn signal, stages stay pure).
