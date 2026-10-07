# 224 — `_log_metric` bypasses the metric schema (`schema`/`ts_iso`/`truncated`) (MEDIUM)

## Technical description
`Supervisor._log_metric` (`supervisor.py:1026-1028`) does
`json.dumps({"ts": time.time(), "run_id": ..., **event})`.
`logrotate.format_metric_line` stamps `ts_iso`, `schema=1`, and `truncated`
handling with a 16 KiB cap (`logrotate.py:256-303`). Every commit-path event
therefore ships schema-less, ISO-less, and uncapped, while the same file's
other writers (if any adopt `format_metric_line`) ship versioned lines.

## Rationale
`parse_metric_lines`/`read_all_metric_events` accept both (legacy
tolerance), but `scoreboard`/`status`/`soak` readers cannot distinguish v0
from v1, and one oversized `stages` dict can clog the pipeline the cap was
built for (issue 058). Two writers, two contracts, one file.

## Live evidence
```
format_metric_line keys: ['event', 'run_id', 'schema', 'segment_id', 'ts', 'ts_iso']
supervisor _log_metric keys: ['event', 'run_id', 'segment_id', 'ts']
```

## Repro
Compare key sets as above; `grep -rn "_log_metric" voyage/supervisor.py |
wc -l` (≈30 call sites, all unversioned).

## Source refs
- `Voyage/voyage/supervisor.py:1026-1028`
- `Voyage/voyage/logrotate.py:75-88,256-303`

## Online sources
- Event-stream schema stability doctrine (same RPC refs as #200).

## Fix candidates
1. Route `_log_metric` through `format_metric_line` + `append_line` (gets
   rotation/fsync/cap for free).
2. Migrate readers to require `schema` with legacy fallback.

## Log
- Track A sweep, 2026-10-07. Read-only; nothing fixed.
