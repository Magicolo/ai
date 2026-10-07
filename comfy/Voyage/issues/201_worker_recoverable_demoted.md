# 201 — Worker-side `RecoverableWorkerError` is demoted to `Fatal` over the wire (HIGH)

## Technical description
`voyage/workers/loop.py:103-110` maps **every** `VoyageError` to
`retryable=False`. A handler raising
`RecoverableWorkerError("transient oom, restart me")` serializes as
`{"code":"RecoverableWorkerError","retryable":false}` → supervisor
`rpc.py:445-450` raises `FatalWorkerError`, burning zero restart budget.
Only *unexpected* `Exception` (`ERROR_WORKER`, default `retryable=True`,
`loop.py:127`) is restartable. `grep` shows zero worker
`raise RecoverableWorkerError` — the taxonomy hole is why the class is
unused.

## Rationale
The explicit recoverable class is unusable from workers. A worker that
correctly signals transience gets the opposite treatment (immediate `FAILED`
instead of one restart).

## Live evidence
```
docker run --rm -v $PWD:/app -w /app voyage:latest python3 -c "
import io, sys, json
from voyage.workers.loop import serve
from voyage.errors import RecoverableWorkerError
# stub stdin with one JSONL line, handler raises RecoverableWorkerError
"
output:
{"id":"req-000001","ok":false,...,"error":{"code":"RecoverableWorkerError",
"message":"transient gpu oom, please restart me","retryable":false}}
code= RecoverableWorkerError retryable= False
```

## Repro
`serve({"gen": lambda p: (_ for _ in ()).throw(RecoverableWorkerError("x"))})`
with one JSONL line on stubbed stdin; parse the stdout line.

## Source refs
- `Voyage/voyage/workers/loop.py:97-127`
- `Voyage/voyage/rpc.py:441-450`
- `Voyage/voyage/errors.py:19-24`

## Online sources
- Same JSONL/RPC framing refs as #200:
  https://raw.githubusercontent.com/earendil-works/pi/5fd446ca1843682e8da3fec4ceb71c42f56fbace/packages/coding-agent/docs/rpc.md ;
  https://0x8f701.github.io/rpi/user-guide/rpc-json.html

## Fix candidates
1. Branch in `loop.py`: `except RecoverableWorkerError → retryable=True`
   before generic `VoyageError → retryable=False`.
2. Or introduce an `ERROR_RECOVERABLE` code preserving the class.
3. Add a wire test pinning
   `RecoverableWorkerError → retryable True → supervisor Recoverable`.

## Log
- Track A sweep, 2026-10-07. Read-only; nothing fixed.
