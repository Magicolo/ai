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
