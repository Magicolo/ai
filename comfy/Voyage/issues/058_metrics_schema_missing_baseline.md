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
