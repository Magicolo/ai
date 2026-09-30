# 007 — Worker error taxonomy erased over RPC; `checked_request` checks nothing; malformed lines stall 600 s

- Status: resolved in live tree (error codes + typed `checked_request` + MALFORMED reply)
- Severity: HIGH (reliability — wasted GPU / 10-min stalls; resolved, record only)
- Group: correctness/RPC — Rank: 1/5 (critical pattern, fixed)
- Area: correctness — RPC protocol (`voyage/workers/loop.py`)
- Rank rationale: three bugs in one 60-line file; together they burned restart
  budgets on deterministic failures and hung healthy workers.

## Technical description

Pre-fix (`voyage/workers/loop.py:19-28,44-50,56-59` at pass 1):

```python
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
(bad geometry), `ModelCompatibilityError` — became `WORKER_ERROR` with
`retryable=True`. The supervisor burned `max_worker_restarts` (default 3) + GPU
evict/rebuild cycles on failures that would never succeed, then tripped the
breaker. The `errors.py` taxonomy ("branch on class, never message") was destroyed
at the wire.
(b) `checked_request` ignored types, so `{"frames": "abc"}` passed presence, then
failed deep inside the worker — after model load / GPU residency.
(c) A malformed stdin line was silently dropped with **no response**; the
supervisor's `call()` waited the full `rpc_timeout_seconds` (600 s) for the
matching id, then restarted a healthy worker. Plus every line was Pydantic-decoded
twice.

Live state (re-verified 2026-09-30): `voyage/workers/loop.py:50-167` — error class
preserved as the wire `code` (`type(exc).__name__`, `retryable=False`;
`INVALID_PAYLOAD` for `ValueError/KeyError/TypeError`; `MALFORMED` with salvaged
id for torn lines; decode-once), and `checked_request` enforces `isinstance`
(`:144-167`, bool-never-int, int-satisfies-float).

## Why this is an issue

Every worker call crosses this file, so the three defects multiplied across all
backends: erased error classes turned deterministic config/geometry failures into
three GPU evict/rebuild retries before the breaker tripped, type-check bypass let
bad payloads burn a full model load before failing deep inside the worker, and a
dropped malformed line parked the caller for the full 600 s timeout while a
healthy worker idled. For a system meant to run unattended for days, these
converted cheap fail-fast moments into hours of wasted GPU and exhausted restart
budgets masking the real cause.

## Evidence

Live verification 2026-09-30 (in-container, `voyage:latest`):

```
$ docker run ... python3 -c "from voyage.workers.loop import checked_request; checked_request({'a': 123}, a=str)"
007-probe: TypeError (fixed): payload field 'a' must be str, got int
```

Pass-1 probe: `checked_request({'a': 123}, a=str)` did NOT raise (int-for-str
PASSED — bug). Current `loop.py:154-167` enforces the type.

## Reproduction

1. Send `{"frames": "abc"}` as a `generate_blocks` payload → pre-fix passed
   validation, burned GPU, then `ValueError` → retryable restart storm; now
   `INVALID_PAYLOAD` fatal pre-GPU.
2. Write a stray non-JSON line to worker stdin → pre-fix supervisor hung until the
   600 s timeout, then restarted a healthy worker; now stderr log + `MALFORMED`
   (when an id is salvageable), no stall.

## Source references

- `voyage/workers/loop.py:34-47` (wire codes), `:50-130` (serve/dispatch),
  `:133-141` (id salvage), `:144-167` (`checked_request`), `:170-183`
  (benchmark-count guard); `voyage/errors.py` (taxonomy);
  `voyage/rpc.py:304-307` (supervisor code→error mapping).

## Resolution candidates

1. (Landed) Preserve error class over RPC: `code` = class name; supervisor maps
   deterministic codes → `Fatal`, transient `WORKER_ERROR`/timeouts → `Recoverable`.
2. (Landed) Make `checked_request` `isinstance`-enforce (fail fast before GPU work).
3. (Landed) On decode failure log to stderr and (when an id can be salvaged) reply
   `MALFORMED retryable=False`; decode once, not twice.
4. Tests: type-mismatch payload rejected pre-GPU; malformed line yields fast
   fatal, not a 600 s hang (short-timeout test double).

## Online references

- Fail-fast validation at trust boundaries (validate inputs before expensive work):
  Pydantic validation docs https://docs.pydantic.dev/latest/
- JSON-RPC-style error codes with retryability as a first-class field:
  https://www.jsonrpc.org/specification
- Retry budgets / circuit breakers (why deterministic failures must not consume
  restart budget): https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker

## Investigation / progress / resolution log

- 2026-09-25: found by correctness + perf sweeps (convergent); type-ignoring
  re-verified live by orchestrator.
- Resolution batch 3: wire codes + typed `checked_request` + MALFORMED landed.
- 2026-09-30: re-verified live (`TypeError` probe + code present); reconstructed
  from archived pass-1 text (commit `b5d7dda`). Status → resolved.
