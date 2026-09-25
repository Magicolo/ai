# 071 — `bench.timing_stats([])` crashes; `summarize_gauges` mixes `"unknown"` strings into numeric fields

- Status: open
- Severity: medium (helper contract holes; string-typed numerics flow into
  soak/CLI untyped)
- Area: bench helpers — `Voyage/voyage/bench.py:12-19,38-49`,
  `Voyage/tests/test_benchmark.py:109-118`
- Rank rationale: pass-2 finding; the only test is happy-path (`[2.0,1.0,3.0]`),
  never empty/single/unknown.

## Technical description

`sum(seconds)/len(seconds)` with no empty guard (`min`/`max` would also raise).
`summarize_gauges([])` returns `rss_delta_mb: "unknown"` (str) where the non-empty
path returns float. No caller guards: `voyage/cli.py:914-1021` passes events
straight through.

## Why this is an issue

An empty timing input crashes with `ZeroDivisionError` instead of a typed
error, and string `"unknown"` values in numeric gauge fields flow untyped into
soak/CLI trend math — so one run with missing gauges breaks aggregation for
the whole report. The only test is happy-path, so neither contract hole is
pinned.

## Evidence (probed by pass-2 sweep; re-run 2026-09-25, `PYTHONPATH=Voyage`)

`timing_stats([])` → `ZeroDivisionError: division by zero`;
`summarize_gauges([])` → all-`"unknown"` dict.

Re-run 2026-09-25 (`PYTHONPATH=Voyage`):

```
$ python3 -c "from voyage.bench import timing_stats; timing_stats([])"
ZeroDivisionError: division by zero
$ python3 -c "from voyage.bench import summarize_gauges; print(summarize_gauges([]))"
{'segments': 0, 'rss_first_mb': 'unknown', 'rss_last_mb': 'unknown', 'rss_delta_mb': 'unknown', 'disk_first_gib': 'unknown', 'disk_last_gib': 'unknown'}
```

## Reproduction

The two calls above.

## Source references

- Files/lines above.

## Resolution candidates

Raise `ValueError` on empty input (matching the module's documented style) or
return count-0 stats; type `summarize_gauges` deltas as `float | None` instead of
`str`; add empty + unknown-path tests. (Distinct from 041's benchmark-coverage
gaps, which are about GPU artifacts/markers.)

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`bench.py:12-19`, `:38-49`; `test_benchmark.py:109-118`
  happy-path only — all match); re-ran both probes (outputs pasted above).
- Open: implement + tests.
