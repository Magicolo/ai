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

## Progress (2026-10-07, group O)

- `voyage/supervisor.py`: `_log_metric` now routes through
  `format_metric_line(self._config.name, event)` + `append_line` (schema,
  `ts_iso`, 16 KiB truncation cap; base fields unspoofable); dead
  `import json` removed (the formatter owns serialization now).
- `voyage/logrotate.py`: `parse_metric_lines` /
  `read_all_metric_events` docstrings state the contract explicitly —
  new writes are v1-required, missing-`schema` lines read as v0 legacy
  (tolerance IS the fallback; no reader filters on schema, so no
  behavior change for any consumer).
- New `tests/test_issue_224_metric_schema.py` (4 tests: emitted lines
  carry schema/ts_iso/run_id with event keys intact, oversized event
  truncates within cap with `truncated: true`, caller ts/run_id cannot
  spoof the stamps, hand-written v0 lines still parse).
- Verified in-container: `ruff check` clean, `ruff format --check`
  clean, `mypy` strict clean on `supervisor.py` + `logrotate.py`, scoped
  pytest 4/4 green; metric-reader neighbors (`test_ledger_rotation_rank2`,
  `test_observability`, `test_observability_rank2`, `test_scoreboard`,
  `test_stage_a_telemetry`, ...) green.

## Resolution (2026-10-07)

FIXED (writer half + contract half). Confirmed the reviewer flag in the
record: `format_metric_line` had zero production callers, so the issue's
"two writers" framing overstated — there was one unversioned writer plus
an unused formatter; the single writer is now routed through it. Left
open (out of scope, other owners): `voyage/sfx_finalize.py:1195`
(`sfx_pass_completed`) and `voyage/media.py:2175`
(`finalize_completed`) still `append_line` raw schema-less lines.

## Evaluation (2026-10-07, group O)

Re-verified live; claim holds with the reviewer's framing correction:
- `Supervisor._log_metric` (`voyage/supervisor.py:1224-1226`; issue cites
  `:1026-1028`, shifted) still emits
  `json.dumps({"ts": ..., "run_id": ..., **event})` via `append_line` —
  no `ts_iso`, no `schema`, no 16 KiB cap, and event keys can spoof
  `ts`/`run_id`.
- Reviewer flag CONFIRMED: `format_metric_line` has zero production
  callers — grep over `voyage/` + `scripts/` finds only the definition
  (`logrotate.py:291`) and docstring refs; all other hits are
  `tests/test_ledger_rotation_rank2.py`. So "two writers, two contracts"
  overstates: there is ONE unversioned writer plus one unused versioned
  formatter. The fix direction stands anyway (route the one writer
  through the formatter).
- Blast radius checked: no test asserts exact metric key sets (readers
  `json.loads` then check individual keys); all readers
  (`parse_metric_lines`, `read_all_metric_events`, `scoreboard.py:80`,
  `last_commit_stages`) ignore unknown keys, so additive `ts_iso` /
  `schema` keys are safe.
- Out-of-scope note: `voyage/sfx_finalize.py:1195` and
  `voyage/media.py:2175` also `append_line` raw `json.dumps({"ts": ...})`
  lines (schema-less `sfx_pass_completed` / `finalize_completed`) — both
  files are outside this group’s scope and stay unversioned; recorded as
  open in the Resolution.
