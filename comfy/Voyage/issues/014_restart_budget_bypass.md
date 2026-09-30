# 014 — `worker.restart()` inside `_call_with_restart` bypasses the restart budget and circuit breaker

- Severity: HIGH
- Group: correctness/failure-policy — Rank: 1/5
- File:line: `voyage/supervisor.py:472-526` (`_call_with_restart`), `voyage/rpc.py:182` (`restart`), `voyage/rpc.py:147` (`start` replays `init`)

## Technical description

Live handler (`voyage/supervisor.py:483-521`, re-verified 2026-09-30):

```python
budget = self._config.voyage.max_worker_restarts
while True:
    try:
        return worker.call(op, dict(payload))
    except RecoverableWorkerError as exc:
        used = self._restarts.get(worker_name, 0)
        if used >= budget:
            ... circuit_breaker_open ... raise FatalWorkerError(...)
        self._restarts[worker_name] = used + 1
        self._log_metric({... "worker_restart" ...})
        worker.restart()                    # <-- UNGUARDED
        if restart_hook is not None:
            restart_hook(segment_id)
```

`worker.restart()` (`voyage/rpc.py:177-180`):

```python
def restart(self) -> None:
    self.stop()
    self.start()   # <-- replays init via call(); can raise RecoverableWorkerError
```

`start()` (`voyage/rpc.py:133-147`) ends with `self.call(self._init_op, ...)` — a full RPC round-trip that raises `RecoverableWorkerError` on timeout / broken pipe / error response (model load OOM, missing weights, wedged binary).

That second failure escapes the `except RecoverableWorkerError` block **without** consuming budget and **without** tripping the circuit breaker: it propagates raw out of `_call_with_restart`, bypassing the `used >= budget` gate and the `circuit_breaker_open` metric. Consequences:

- A worker whose `init` always fails (e.g. bad weights) produces one `worker_restart` metric then a raw `RecoverableWorkerError` instead of the documented `FatalWorkerError(circuit breaker open ...)`. Callers branching on class (`run_segments` maps `Recoverable` vs `Fatal` differently) mis-handle it.
- The `restart_hook` (video resume, which itself calls `_call_with_restart` and *does* share budget) never runs for this failure, so resume accounting diverges from the metric log.
- Nested `_call_with_restart` for resume *does* bound its own retries, but the outer `restart()` failure is outside any loop — one bad `init` defeats the whole budget.

## Why it matters

Phase 6 slice B promises "at most N restarts per worker per run, then FAILED". This path violates the promise: the budget counts `call` failures but not `restart` failures, so the run can exit with an unclassified `RecoverableWorkerError` instead of resting at FAILED with a `circuit_breaker_open` event. Observability (logs/metrics) and status (`FAILED` vs propagate) both lie.

## Live evidence

```
$ grep -n "def _call_with_restart\|worker.restart()\|def restart\|self.call(self._init_op" comfy/Voyage/voyage/supervisor.py comfy/Voyage/voyage/rpc.py
supervisor.py:465:    def _call_with_restart(
supervisor.py:516:                worker.restart()
rpc.py:177:    def restart(self) -> None:
rpc.py:147:            self.call(self._init_op, dict(self._init_payload))
```

Excerpt (`supervisor.py:504-518`):

```
self._restarts[worker_name] = used + 1
self._log_metric({...})
worker.restart()
if restart_hook is not None:
    restart_hook(segment_id)
# Loop: the retried call runs at the top. A FatalWorkerError
# from the hook ... propagates without further retries.
```

The comment covers the hook's `Fatal` path but not `restart()`'s own `Recoverable` path. Host import of `voyage.supervisor` fails (`No module named 'pydantic'` — container-only), so evidence is source-inspection per allowance.

## Repro steps

1. Stub a worker whose `call(op)` raises `RecoverableWorkerError` once, and whose `start()` (init replay) always raises `RecoverableWorkerError("init boom")`.
2. Call `_call_with_restart(worker, "video", "000000", "generate_blocks", {})` with `max_worker_restarts=3`.
3. Observed: one `worker_restart` metric, then raw `RecoverableWorkerError("init boom")` escapes; `_restarts["video"] == 1`, no `circuit_breaker_open` event, no `FatalWorkerError`.
4. Expected: budget consumed for the restart failure, eventually `FatalWorkerError(circuit breaker open ...)`.

## Fix candidates

1. (Preferred) Wrap `worker.restart()` in the budget loop:
   ```python
   try:
       worker.restart()
   except RecoverableWorkerError as exc:
       # counts as another consumed restart; re-enter budget check
       continue  # or explicit used+=1 + gate + metric
   ```
   Simplest correct form: move the whole `restart + hook` sequence into a helper that raises `FatalWorkerError` when budget is exhausted, sharing the same `circuit_breaker_open` metric.
2. Make `restart()` itself never raise `Recoverable` (map init failures to `Fatal`)? Wrong — transient OOMs during init *are* recoverable; the budget must count them, not reclassify.
3. Regression test: counting fake worker (call fails N times, start fails M times); assert total restarts ≤ budget and terminal error is `FatalWorkerError` with a `circuit_breaker_open` metric.

## References

- `voyage/supervisor.py:483-521` (budget gate + unguarded `restart()`), `voyage/rpc.py:177-180` (`restart` = `stop` + `start`), `:133-147` (`start` replays `init` via `call`).
- Phase 6 slice B (restart budget + circuit breaker); `voyage/supervisor.py:539-555` (`_resume_video_worker` — the hook path that *does* share budget, contrast case).
- Python exception chaining in loops — unhandled exceptions in `except` blocks propagate without re-entering the loop: https://docs.python.org/3/tutorial/errors.html#handling-exceptions

## Progress log

- 2026-09-30: premise re-verified against live `voyage/supervisor.py` as read (`_call_with_restart` at `:489-545`, budget gate `:506-527`, bare `worker.restart()` at `:540`, hook at `:541-542`). Premise held — batch-1 (b40b052) touched `start_workers` and the commit CAS only, so this stacks cleanly with 012/099 (read both regions first per task guidance; no interaction).
- 2026-09-30: failing tests first in `tests/test_supervisor_hardening.py` (new file): `test_failed_restart_consumes_budget_and_trips_breaker`, `test_failed_restart_then_success_recovers`, and `test_failed_restart_hook_routes_through_budget` all failed pre-fix (raw `RecoverableWorkerError` escaped where `FatalWorkerError` was expected). Watched fail in-container via `scripts/test.sh`.
- 2026-09-30: fix implemented (see Resolution). Verification — all three pass post-fix; existing budget tests unaffected (`test_restart_budget_exhaustion_opens_circuit_breaker` still counts exactly 2 restarts at budget 2 — the success path counts nothing new; `test_resume_failures_consume_the_same_budget` green, confirming the nested resume path still shares the budget). Related suites green (see 013 log for the one foreign failure). `ruff check` + `ruff format --check` + `mypy` (strict) green on touched files.

## Resolution

`worker.restart()` and `restart_hook` failures now route through the restart-budget accounting, inline in `voyage/supervisor.py:_call_with_restart` (`:549-650`, restart guarded at `:604-650`):

- Both calls sit inside `try/except RecoverableWorkerError`. A failure re-reads the counter, re-enters the same `used >= budget` gate with the same `circuit_breaker_open` metric shape, and otherwise consumes one more attempt (`used + 1`), emits `worker_restart_failed` (worker/op/segment/attempt/budget/reason, `:638`), and `continue`s the loop.
- `FatalWorkerError` from the hook still propagates without further retries (not `Recoverable`) — the pre-existing comment contract is preserved, now enforced by the except type rather than by accident.
- Terminal behavior with budget 3 and an always-failing init: `worker_restart` x2, `worker_restart_failed` x1, `circuit_breaker_open` x1, `_restarts == 3`, terminal `FatalWorkerError("circuit breaker open …")` — the Phase 6 slice B promise ("at most N restarts, then FAILED") holds on the restart path too. No helper was extracted (kept strictly inside the owned function); the gate block reads twice by design.

Half left open: none — exhaustion on every path (call, restart, hook) now converges on the breaker.
