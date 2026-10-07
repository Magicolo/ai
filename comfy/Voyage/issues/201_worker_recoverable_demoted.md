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

## Evaluation (2026-10-07)
Re-verified live against the current tree before fixing — every
load-bearing claim still holds: `voyage/workers/loop.py:103-110` maps
every `VoyageError` to `retryable=False` (and `RecoverableWorkerError`
subclasses `VoyageError` via `WorkerError`, so it hits that arm);
`voyage/rpc.py:441-450` routes `retryable=True` to
`RecoverableWorkerError` (restart path) and `False` to
`FatalWorkerError`; `Supervisor._call_with_restart`
(`voyage/supervisor.py:1042-1146`) already restarts on
`RecoverableWorkerError` under the per-worker budget, so no supervisor
change was needed — the loop.py wire fix completes the chain. The
"zero worker `raise RecoverableWorkerError`" observation is unchanged
(only `rpc.py` raises it supervisor-side today), which is consistent
with the class being unusable from workers pre-fix. No staleness found;
no adjustment to the fix direction.

## Progress
- Added `except RecoverableWorkerError → retryable=True` before the
  generic `VoyageError → retryable=False` arm in `voyage/workers/loop.py`
  (import extended; module docstring taxonomy updated). `FatalWorkerError`
  and plain `VoyageError` still serialize `retryable=False`.
- New `tests/test_issue_201_225_recoverable_embed.py` pins: recoverable
  wire code + `retryable=True`; fatal/generic still `False`; full chain
  (exact `serve` output bytes fed into a stub supervisor-side worker)
  raising supervisor-side `RecoverableWorkerError`, i.e. a restart, not
  `FatalWorkerError`.
- Scoped verify in-container (`voyage:latest`, bind mount): `ruff check`
  + `ruff format --check` clean on all touched files; `mypy`
  `voyage/workers/loop.py voyage/supervisor.py` clean; pytest 61 passed
  (7 new + neighbors `test_commit_hardening` / `test_rpc_paths_hardening`
  / `test_media_robustness_rank2` / `test_crash_matrix`).
- Changes left uncommitted for orchestrator review, per task scope.

## Resolution (2026-10-07)
Fixed as per candidate 1 + 3 (branch before the generic arm; wire test
pinning `RecoverableWorkerError → retryable True → supervisor
Recoverable`). No `errors.py` taxonomy change was needed — the class
already existed with the right meaning; only the wire demoted it.
