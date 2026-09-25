# 007 — Worker error taxonomy erased over RPC; `checked_request` checks nothing; malformed lines stall 600 s

- Status: resolved 2026-09-25 (supervisor track; one stale-test excursion disclosed in log)
- Severity: major (reliability / wasted GPU / 10-min stalls)
- Area: correctness — RPC protocol (`voyage/workers/loop.py`)
- Rank rationale: three bugs in one 60-line file; together they burn restart
  budgets on deterministic failures and hang healthy workers.

## Technical description

```python
# voyage/workers/loop.py:19-28,44-50,56-59
for line in stdin:
    line = line.strip()
    if not line: continue
    try: decode_request(line)
    except Exception: continue   # no response, no stderr — supervisor waits full 600 s
    request = decode_request(line)  # decoded twice

except NotImplementedError as exc: ... retryable=False
except Exception as exc:
    traceback.print_exc(file=sys.stderr)
    stdout.write(encode_response(failure(request.id, "WORKER_ERROR", str(exc))))  # retryable defaults True

def checked_request(payload, **required: type) -> None:
    for key, _type in required.items():
        if key not in payload: raise KeyError(...)  # _type never used
```

(a) Every worker exception — `ConfigurationError`, deterministic `MediaError`
(bad geometry), `ModelCompatibilityError` — becomes `WORKER_ERROR` with
`retryable=True`. The supervisor burns `max_worker_restarts` (default 3) + GPU
evict/rebuild cycles on failures that will never succeed, then trips the breaker.
The `errors.py` taxonomy ("branch on class, never message") is destroyed at the wire.
(b) `checked_request` ignores types, so `{"frames": "abc"}` passes presence, then
fails deep inside the worker — after model load / GPU residency.
(c) A malformed stdin line is silently dropped with **no response**; the
supervisor's `call()` waits the full `rpc_timeout_seconds` (600 s) for the
matching id, then restarts a healthy worker. Plus every line is Pydantic-decoded
twice (see also 032-adjacent waste).

## Why this is an issue

Every worker call crosses this 60-line file, so the three defects multiply
across all backends: erased error classes turn deterministic config or geometry
failures into three GPU evict/rebuild retries before the breaker trips,
type-check bypass lets bad payloads burn a full model load before failing deep
inside the worker, and a dropped malformed line parks the caller for the full
600 s timeout while a healthy worker idles. For a system meant to run
unattended for days, these convert cheap fail-fast moments into hours of wasted
GPU and exhausted restart budgets that mask the real cause. The operator pays
in steady-state throughput lost to retry storms and 10-minute stalls that look
like model slowness.

## Evidence

```
$ python3 -c "...checked_request({'a': 123}, a=str)..."
int-for-str PASSES (bug)
```

Sub-agent additionally verified `checked_request({"a":123}, a=str)` does not raise.

## Reproduction

1. Send `{"frames": "abc"}` as a `generate_blocks` payload → passes validation,
   burns GPU, then `ValueError` → retryable restart storm instead of immediate fatal.
2. Write a stray non-JSON line to worker stdin → supervisor hangs until the 600 s
   timeout, then restarts a healthy worker.

## Source references

- `voyage/workers/loop.py:16-59`; `voyage/errors.py` (taxonomy);
  `voyage/supervisor.py` restart-budget accounting.

## Resolution candidates

1. Preserve error class over RPC: `code` = class name; supervisor maps
   `CONFIGURATION/MODEL_COMPATIBILITY/NOT_IMPLEMENTED/UNKNOWN_OP` → `Fatal`,
   transient `WORKER_ERROR`/timeouts → `Recoverable`.
2. Make `checked_request` `isinstance`-enforce (fail fast before GPU work).
3. On decode failure log to stderr and (when an id can be salvaged) reply
   `MALFORMED retryable=False`; decode once, not twice.
4. Tests: type-mismatch payload rejected pre-GPU; malformed line yields fast
   fatal, not a 600 s hang (use a short-timeout test double).

## Investigation / progress / resolution log

- 2026-09-25: found by correctness + perf sweeps (convergent); type-ignoring
  re-verified live by orchestrator.
- Open: implement protocol fix + tests.
- 2026-09-25 (repair pass): added `## Why this is an issue`;
  `checked_request` type-ignoring re-probed live (bug present); refs verified
  current (`loop.py:24,28,46,56`).
- 2026-09-25 (RESOLVED, supervisor track): implemented candidates 1–3 in
  `voyage/workers/loop.py`. (a) Error class survives the wire: `VoyageError`
  subclasses answer with `code = type(exc).__name__`, `retryable=False`
  (deterministic config/geometry/compat failures map straight to
  supervisor `Fatal` via the existing retryable flag — no supervisor change
  needed); `ValueError`/`KeyError`/`TypeError` past `checked_request`
  answer `INVALID_PAYLOAD`, `retryable=False`; only unknown `Exception`
  stays retryable `WORKER_ERROR`. (b) `checked_request` now
  `isinstance`-enforces (ints satisfy float; bools never satisfy int —
  `isinstance(True, int)` would otherwise smuggle flags into seeds).
  (c) Malformed lines: single decode (the double-decode is gone); a
  schema-bad line with a salvageable id answers `MALFORMED`,
  `retryable=False` immediately, non-JSON lines log to stderr and wait for
  the next well-formed request. Tests: `checked_request` type/missing/bool
  cases, `ConfigurationError` → fatal code over `serve()`, and
  `MALFORMED`-vs-`UNKNOWN_OP` two-line `serve()` exchange — all in
  `Voyage/tests/test_commit_hardening.py`. SCOPE EXCURSION (disclosed):
  this fix invalidates one assertion outside this track's scope —
  `tests/test_director_request_validation.py:95-96` pinned the old contract
  (`ValueError` "non-empty texts" for a str `texts`, i.e. exactly the
  presence-without-type bug); updated to `TypeError` "must be list" with an
  inline SCOPE NOTE, intent unchanged (reject before embedding). No
  production file outside scope was touched. Gates: full
  `Voyage/scripts/gates.sh` green (626 passed).
