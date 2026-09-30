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
