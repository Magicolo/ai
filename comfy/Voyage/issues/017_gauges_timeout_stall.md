# 017 — Best-effort `_sample_gauges` uses full 600 s RPC timeout ×3 (stalls every commit)

- Status: resolved 2026-09-25 (supervisor track; TOML knob deferred, see log)
- Severity: major (commit latency — up to ~30 min added)
- Area: correctness/perf — observability on the critical path
- Rank rationale: optional gauges can dominate commit latency 3× over; one sick
  worker stalls an otherwise healthy run.

## Technical description

```python
# voyage/supervisor.py:342-374
for name, worker in (("video",...), ("audio",...), ("director",...)):
    try:
        health = worker.call("health", {})  # no timeout= → default 600 s each
    except VoyageError:
        continue
```

The docstring claims "Never fails the commit" — true for errors, false for
latency: a sick-but-alive worker (slow health, swapped GPU) blocks the commit tail
up to 600 s per worker = 1800 s per segment. Same for `_embed_texts` (novelty
path, full timeout before token-set fallback). Gauges are observability, not
correctness — they must not dominate commit latency.

## Why this is an issue

Observability with a worst case of 600 s per worker — up to 30 minutes per
segment — lets optional gauges dominate the correctness path they merely
observe. A sick-but-alive worker with a slow `health` (exactly the state a
swapped GPU produces) stalls an otherwise healthy run on every commit, while
the docstring's "never fails the commit" promise covers errors but not
latency. In an infinite loop the per-segment landmine compounds into missed
cadence targets and misleading soak numbers that look like model regression.
Every commit issued while any worker is sluggish pays the delay.

## Evidence

Source quotes above; default timeout `rpc_timeout_seconds=600` (`config`).

Re-verified 2026-09-25:

```
$ rg -n 'worker.call\("health"' voyage/supervisor.py
365:                    health = worker.call("health", {})
$ rg -n "rpc_timeout_seconds" voyage/config.py | head -2
167:    rpc_timeout_seconds: float = 600.0
```

No `timeout=` kwarg — the full 600 s default applies to each of the 3 gauges.

## Reproduction

Stub one worker's `health` to sleep 600 s → `commit_one_segment` wall time +600 s
despite gauges being optional.

## Source references

- `voyage/supervisor.py:342-374`; novelty `_embed_texts` path nearby.

## Resolution candidates

1. `worker.call("health", {}, timeout=5.0)` (and short timeout for `embed` with
   fast fallback), or run gauges concurrently with timeouts.
2. Sample gauges every K segments instead of every commit; keep blanket
   `except Exception: pass` but bound the wait.
3. Test: slow-health stub must not delay commit beyond ~seconds.

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- Open: implement + test.
- 2026-09-25 (repair pass): added `## Why this is an issue`; Evidence enriched
  (`health` call at 365, no `timeout=`); refs verified current.
- 2026-09-25 (RESOLVED, supervisor track): implemented candidates 1 + 2
  (constant form). Health probes now carry `timeout=GAUGE_TIMEOUT_SECONDS`
  (5 s — worst case ~15 s per commit, not ~30 min) and the inner handler
  catches `Exception` so a sick worker (or timeout-kwarg-less test double)
  degrades to missing fields, never a failed commit
  (`voyage/supervisor.py:449-481`); novelty `_embed_texts` carries
  `timeout=EMBED_TIMEOUT_SECONDS` (60 s — generous so a healthy director
  never degrades to token-set fallback, a wedged one never stalls the
  commit; `voyage/supervisor.py:672`). Cadence: `RESOURCE_GAUGE_INTERVAL_SEGMENTS`
  (`voyage/supervisor.py:126`, default 1) skips sampling on off-cadence
  segments — kept at 1 because `test_benchmark.py` pins per-segment gauges;
  raise the constant to thin out probe traffic. A TOML knob needs
  `config.py` (out of scope) — deferred to the config-owning track. Tests
  (`Voyage/tests/test_commit_hardening.py`):
  `test_gauge_probes_carry_short_timeouts` (three 5.0 timeouts asserted,
  sick-worker gauges still log) and `test_embed_call_carries_bounded_timeout`
  (60.0 asserted). Gates: full `Voyage/scripts/gates.sh` green (626 passed).
