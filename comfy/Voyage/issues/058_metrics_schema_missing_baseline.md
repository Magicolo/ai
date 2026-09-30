# 058 — `metrics.jsonl` schema misses the structured-logging baseline: epoch-float `ts`, no `level`/`service`/version, no line-size bound

**Severity:** MEDIUM

**File:line:** `voyage/supervisor.py:468-470` (`_log_metric`)
- **Description:** Every metric line is `{"ts": time.time(), "run_id": ..., **event}` — epoch float, no severity, no service/component, no schema version, unbounded line length.
- **Rationale:** The searched baseline is consistent across sources: every entry carries ISO-8601 UTC timestamp, a small fixed `level` set, `service`/`env`, machine-readable `event`, correlation id; human message secondary; cap line size (~16 KB) so one oversized entry can't clog the pipeline. Current `ts` is unreadable at 3am (`1790733010.79…`), unsortable lexically, and `run_id` is the only correlation field (no `segment_id` on several events, no `run_id` filtering in readers — see 062). No `level` means WARN/ERROR alerting can't be built on the log without string-matching `event` names.
- **Evidence (re-verified live 2026-09-30):**
```python
# voyage/supervisor.py:468-470 (live)
def _log_metric(self, event: dict[str, object]) -> None:
    line = json.dumps({"ts": time.time(), "run_id": self._config.run_id, **event})
```
Probe output shape: `{"ts": 1790733010.7966387, "run_id": "x", "event": "segment_committed"}`. Supervisor emits 14 distinct event names (`audio_swap_teardown_error`, `circuit_breaker_open`, `director_prefetch_hit/miss`, `drift_hold`, `inspect_skipped`, `resource_gauges`, `segment_commit_failed`, `segment_committed`, `segment_inspected`, `take_rendered`, `video_left_evicted`, `video_resumed`, `worker_restart`) with no `level` mapping documented anywhere.
- **Repro:** `grep -n '"ts":' voyage/supervisor.py`; `python3 -c "import json,time; print(json.dumps({'ts': time.time()}))"` — observe epoch vs `datetime.now(timezone.utc).isoformat()`.
- **Fix candidates:** Add `ts_iso` (keep epoch `ts` for compat) or migrate to ISO-8601 `Z`; add `level` per event (ERROR for `segment_commit_failed`/`circuit_breaker_open`, WARN for restarts/skips, INFO otherwise) + `service: supervisor|video|audio|director`; add `schema: 1`; enforce a max line size with truncation + `truncated: true`. Document the mapping in `docs/OPERATIONS.md` + `ARCHITECTURE.md`.
- **Refs:** structured-logging baseline (required fields `ts` ISO-8601, `level`, `service`, `env`, `event`, `trace_id`; "set a maximum log line size (e.g. 16KB)"; "rotate … to prevent disk exhaustion"); `docs/ARCHITECTURE.md` (commit pipeline stages that deserve levels).

## Progress log

- 2026-09-30: premise RE-VERIFIED live in-container (as-read
  2026-09-30): `supervisor._log_metric` still
  `{"ts": time.time(), "run_id": ...}` (epoch float, no
  `ts_iso`/`level`/`service`/`schema`), `cli._read_all_metric_events`
  still `except ValueError: continue` with no torn count and no
  `run_id` filtering, `scoreboard._stages_by_segment` same shape.
  Related 029/055/059 work confirmed landed (rotation spanning via
  `iter_metric_files`, gauge hardening) — not duplicated. TDD: new
  schema tests watched fail (`ImportError` — helpers did not exist),
  then fixed. `voyage/bench.py` checked for concurrent hunks first:
  another agent's `report_document` (issue 060) is in flight there —
  left untouched per contract.

## Resolution (BASELINE LANDED in owned files; wiring deferred)

- `voyage/logrotate.py`: new `METRICS_SCHEMA_VERSION = 1`,
  `MAX_METRIC_LINE_BYTES = 16 KiB`, `format_metric_line(run_id,
  event)` (stamps `ts` + `ts_iso` + `run_id` + `schema`, base wins
  over event keys so callers cannot spoof correlation/version;
  longest-string halving until the line fits, then `truncated:
  true`, minimal fallback stays valid JSON), and
  `parse_metric_lines(lines, run_id=None)` (returns
  `(events, torn_count)` — non-empty non-JSON-object lines count
  loudly, mismatched `run_id` skips without counting as torn,
  blanks ignored).
- Readers/emitter (`supervisor._log_metric`,
  `cli._read_all_metric_events`/`_last_commit_stages`,
  `scoreboard._stages_by_segment`) are out-of-contract files —
  left untouched.
- Tests: `test_058_metric_line_carries_schema_version_and_iso_ts`,
  `test_058_metric_line_caps_size_with_truncated_flag`,
  `test_058_parse_filters_run_id_and_counts_torn`.
- Evidence: new file 13/13 pass; related suites 99 passed;
  `ruff check .` + `format --check` + `mypy voyage` clean.
- DESIGN proposals (text only — not implemented): (a) wire
  `supervisor._log_metric` through `format_metric_line` (keep epoch
  `ts` for compat, add `ts_iso`/`schema`); (b) wire
  `_read_all_metric_events`/`_stages_by_segment` through
  `parse_metric_lines` with the run's `run_id` + surface
  `torn_count` in `status`/`scoreboard` (loud forensics instead of
  silent skips); (c) add `level`/`service` mapping per event name
  (14 supervisor events enumerated in the issue) + document in
  `docs/OPERATIONS.md` + `ARCHITECTURE.md`; `bench.py` needs no
  change (`summarize_gauges` consumes already-parsed dicts).
