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

## Progress log

- 2026-09-30 (supervisor-side track, scope: `voyage/rpc.py`, new test
  file only — `voyage/workers/loop.py`, `voyage/errors.py`, supervisor and
  existing tests untouched): re-verified every premise against live code.
  Worker-side (a)(b)(c) no longer hold — `voyage/workers/loop.py:50-167`
  carries the landed fix (wire codes, typed `checked_request`, MALFORMED,
  decode-once) and is pinned by `tests/test_commit_hardening.py:251-301`
  (`test_checked_request_enforces_types`,
  `test_worker_error_class_survives_rpc`,
  `test_malformed_line_answers_malformed_fast`). No fix invented there.
  One premise DID still hold in-scope: `SubprocessWorker.call`
  (`voyage/rpc.py:334` pre-fix) performed no request checks, so a
  malformed supervisor-side request escaped the taxonomy as a raw
  `pydantic_core.ValidationError` (verified live — `call(123, {})` raised
  `ValidationError: 1 validation error for WorkerRequest`, not a
  `VoyageError`). The `retryable=False -> Fatal` mapping
  (`rpc.py:368-377`) existed but had zero pins.
- TDD: wrote `tests/test_rpc_paths_hardening.py` first; the 2 request-
  validation tests failed as predicted (raw `ValidationError`), the 3
  mapping tests passed immediately (characterization of landed behavior).
- Fix (`voyage/rpc.py`, `call()` entry): `isinstance` fail-fast — non-str
  `op` / non-dict `payload` raise `FatalWorkerError` before any pipe use
  or counter increment (deterministic caller bug: never retries, never
  burns restart budget). The op vocabulary stays unchecked on purpose:
  unknown ops remain the worker's `UNKNOWN_OP` fatal (sibling check —
  `sfx_finalize.py:332` calls op `generate_sfx`, which is not in `OPS`,
  so a supervisor-side allow-list would break the SFX pass).
- `voyage/errors.py`: no change needed — `FatalWorkerError` /
  `RecoverableWorkerError` (`errors.py:19-24`) already exist as the
  mapping targets.
- Post-fix: new file 13/13 green (incl. all pre-existing pins);
  `test_rpc_timeout.py`, `test_rpc_start.py`, `test_paths.py`
  (25 passed, 2 failed — both 015 collateral, unrelated to this fix),
  `test_commit_hardening.py` green. `ruff check` + `ruff format --check`
  + `mypy strict` green on all touched files (in-container).

## Resolution

- Resolved (supervisor side). `SubprocessWorker.call` now enforces the
  error-class taxonomy on the wire in both directions: malformed requests
  fail fast as `FatalWorkerError` (no raw escapes), and every failure
  response maps `retryable=False -> FatalWorkerError` /
  `retryable=True -> RecoverableWorkerError` (pinned, incl. the
  fail-safe `ok=False`-without-detail -> `Fatal UNKNOWN` branch).
- Files changed: `voyage/rpc.py` (request validation + docstring);
  `tests/test_rpc_paths_hardening.py` (new, 5 RPC tests: 2 fail-first +
  3 characterization). `voyage/errors.py` intentionally unchanged.
- Follow-ups for owning tracks (out of scope, not applied): none for
  007 — no caller passes non-str ops or non-dict payloads (all 9
  `.call(` sites verified), so the new guard is dormant in production.
