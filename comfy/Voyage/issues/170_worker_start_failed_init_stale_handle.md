# 170 — `SubprocessWorker.start()` failing init leaves a live-but-uninitialized worker as the current handle

- Severity: LOW-MEDIUM (correctness / resource state — one free extra attempt on a poisoned handle plus a zombie/log-fd window; amplifies 014's budget bypass with a handle-liveness dimension)
- Area: worker RPC lifecycle — start/init below the 012/137 windows (012 covered partial-init strandings at `start_workers` scope; 137 the timeout-desync aftermath; 014 the *budget* bypass of `restart()` failures — this file is the *handle* the failure leaves behind)
- Files (as-read 2026-09-30):
  - `voyage/rpc.py:138-152` (`start`: `_proc` assigned at `:143`, init RPC at `:152` can raise)
  - `voyage/rpc.py:182-185` (`restart` = `stop` + `start`, no handle hygiene on `start` failure)
  - `voyage/rpc.py:161-180` (`stop`: `_proc=None` first, then reap; `_close_log_file` in `finally`)
  - Consumer: `voyage/supervisor.py:489-528` (`_call_with_restart`: `worker.restart()` at `:523` unguarded — 014)

## Technical description

`start()` publishes the process handle *before* the init handshake (`rpc.py:138-152`):

```python
def start(self) -> None:
    rotate_log(self._log_path)
    self._close_log_file()
    log_file = self._log_path.open("a", encoding="utf-8")
    self._log_file = log_file
    self._proc = subprocess.Popen(...)   # :143 — handle is live from here
    if self._init_op is not None:
        self.call(self._init_op, dict(self._init_payload))   # :152 — may raise
```

When the init call raises (`RecoverableWorkerError`: timeout, broken pipe, model-load OOM reported retryable; `FatalWorkerError`: desync id-mismatch per 137), `start()` propagates with:

1. `self._proc` still pointing at the failed-init process (dead or alive-but-sessionless — whichever the init failure left);
2. `self._log_file` still open (no close on the failure path — `_close_log_file` runs only at the *next* `start()`/in `stop()`'s `finally`);
3. `self._counter` already incremented (cosmetic — ids stay unique).

The next user of the handle talks to a worker whose session state is unknown: a dead process yields `BrokenPipeError → Recoverable` (self-healing via another budget-consuming restart), but a *live* process that failed init mid-handshake (e.g. slow model load timed out the init RPC while the load then completed) serves subsequent ops with a half-built session — `handle_generate_blocks` raises `RuntimeError("not initialized")` from inside the worker, which the loop maps to an error response and another restart cycle. Either way the run pays at least one extra confused attempt that the budget never counted (014's bypass) *on a handle that should have been fenced*.

`restart()` (`:182-185`) compounds it: `stop()` already ran, so a `start()` failure inside `restart()` cannot be cleaned up by re-calling `stop()` later without also killing whatever the half-started process became — and `stop()` on this handle *will* `wait()`/reap it, converting the leaked zombie into a reaped one only if someone calls it. In `_call_with_restart` nobody does: the `Recoverable` from the failed init propagates raw out of `worker.restart()` (014), and the supervisor's next retry — if the caller catches and retries outside the budget loop — reuses the poisoned handle.

## Why this is an issue

- It turns 014's accounting gap into a state gap: not just "one restart not counted" but "the next attempt runs against an uninitialized session". The failure mode is confusing mid-handshake errors (`not initialized`, half-loaded pipelines) instead of a clean reconnect.
- The leaked log fd + unreaped proc window lasts until the next `start()`/`stop()` on that handle — in the `_call_with_restart` escape path, that is the run's remaining lifetime (the worker object is reused, not replaced).
- The fix is small and local (fence in `start()` itself), unlike 014's loop restructuring — the two can land independently.

## Live evidence

CPU-only probe 2026-09-30 in `voyage:latest` (no GPU, no models — init fails at process spawn, which is exactly the path):

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from pathlib import Path
from voyage.rpc import SubprocessWorker
import tempfile
with tempfile.TemporaryDirectory() as td:
    w = SubprocessWorker('nonexistent_module_xyz_abc', Path(td), Path(td)/'w.log',
                         init_op='init', init_payload={}, timeout=20.0)
    try:
        w.start()
        print('start returned cleanly (unexpected)')
    except Exception as exc:
        print('start raised:', type(exc).__name__, str(exc)[:100])
    print('_proc is None:', w._proc is None)
    print('running:', w.running)
    print('log_file left open:', w._log_file is not None)
    w.stop()
    print('after stop _proc None:', w._proc is None)
"
start raised: RecoverableWorkerError worker nonexistent_module_xyz_abc closed stdout
_proc is None: False
running: False
log_file left open: True
after stop _proc None: True
```

- `start()` raised the init failure correctly (Recoverable), but `_proc is None: False` — the dead handle stays mounted as current — and the log fd stays open (`True`) until an explicit `stop()`.
- In the supervisor path no explicit `stop()` follows (the exception escapes `restart()` at `supervisor.py:523`), so both linger.

## Minimal repro

1. Stub a worker whose `init` op always raises a retryable error (or use a module that exits at once, as above).
2. `worker.restart()` (or `start()` directly) → `RecoverableWorkerError` escapes.
3. Assert `worker._proc is None` → fails (stale handle); assert the supervisor-side log fd closed → fails.
4. Follow with `worker.call("health", {})` → talks to the dead/half-init process (EOF/BrokenPipe → another `Recoverable`) instead of failing fast with "not running".

## Fix candidates

1. (Preferred) Fence in `start()`: wrap the init call so failure tears down what `start()` built:
   ```python
   if self._init_op is not None:
       try:
           self.call(self._init_op, dict(self._init_payload))
       except Exception:
           self.stop()   # reaps the proc, clears _proc, closes the log fd
           raise
   ```
   After this, a failed `start()`/`restart()` leaves the handle exactly as before the call (`_proc None`, fd closed) — callers can distinguish "not running" (`Fatal`, fail fast) from retryable staleness, and 014's loop fix stacks cleanly on top.
2. Alternative: clear-and-close *before* raising without a full `stop()` (avoid the 10 s grace wait on a wedged child): `self._proc` kill + `_close_log_file`. Cheaper but duplicates `stop()`'s reap logic — prefer reusing it.
3. Test: failing-init stub → `start()` raises AND `_proc is None` AND log fd closed AND `running is False`; succeeding retry afterwards works (no ghost of the failed attempt). Extend 014's counting-fake test to assert handle hygiene per attempt.

## References

- In-tree: `voyage/rpc.py:138-185,192-196,261-315`; `voyage/supervisor.py:442-460` (`start/stop_workers`), `:489-528` (budget loop).
- Neighbor issues — not a duplicate of 014 (014 is the *budget* bypass: the failed `restart()` escapes without consuming budget or tripping the circuit breaker; this is the *handle* it leaves: live-but-uninitialized `_proc` + open log fd) nor 012 (012 is `start_workers()` stranding *sibling* workers on partial init; this is one handle's self-inconsistency after its *own* init fails) nor 137 (137 is stale-bytes poisoning the *next* call on a *live* pipe; this is a dead/half-init process retained as current).
- External: `subprocess.Popen` resource discipline — unreaped children stay zombies until `wait()`; the `stop()` reap exists precisely for this (`rpc.py:172-178`, issue 058).

## Investigation log

- 2026-09-30: filed by the 168-177 tails sweep; probe run live in `voyage:latest` CPU-only per task brief (never `-e PYTHONPATH` alone — bind mount + image interpreter); code citations are as-read values (concurrent uncommitted edits noted in `voyage/rpc.py` among others).

## Progress log

- 2026-09-30: relevance/integrity check — still live. `voyage/rpc.py` `start()` assigned
  `self._proc` before the init RPC with no failure path: reproduced failing-test-first
  with the issue's own probe shape (nonexistent module, CPU-only, in-container) —
  `tests/test_rpc_start.py::test_failed_init_leaves_no_stale_handle` failed with
  `_proc` still mounted and the log fd open after `start()` raised.
- 2026-09-30: implemented the issue's preferred fence (candidate 1, reuse `stop()`),
  re-ran new + adjacent suites green, `ruff check` + `ruff format --check` + `mypy`
  (strict, in-container) clean on all touched files.

## Resolution

- Verdict: fixed.
- What changed (`voyage/rpc.py` only):
  - `voyage/rpc.py:163-194` — `start()` wraps spawn + init handshake in
    `try/except (VoyageError, OSError)` (narrow on purpose: `call()` only raises the
    worker taxonomy, `Popen` only raises `OSError`); on failure it calls `self.stop()`
    (reaps the child, clears `_proc`, closes the log fd) and re-raises. Covers both
    `Popen` failure and init-RPC failure. `stop()` stays safe/idempotent, so calling
    it after a failed `start()` is a no-op.
- Tests (`tests/test_rpc_start.py`, new, CPU-only, no GPU/network):
  - `test_failed_init_leaves_no_stale_handle` — the issue's probe as a test
    (nonexistent module → init raises Recoverable): asserts `_proc is None`,
    `running is False`, `_log_file is None` (all failed pre-fix).
  - `test_retry_after_failed_init_starts_clean` — init stubbed to fail once then
    succeed: asserts the fence after attempt 1 and a fresh mounted handle after
    attempt 2 (`_proc`/`_log_file` live, log not closed), then `stop()` rests clean.
- Gates (in-container, `voyage:latest`): `ruff check` + `ruff format --check` +
  `mypy` clean on `voyage/rpc.py` + both new test files; pytest
  `tests/test_rpc_timeout.py tests/test_rpc_start.py tests/test_commit_hardening.py
  tests/test_failure_policy.py` → 35 passed; plus `test_observability.py
  test_unit.py test_integration.py` → 69 passed total, no regressions.
- Supervisor-side contract (no supervisor change needed): after a failed `start()`
  (or `restart()`, which delegates) the handle rests exactly as before the call —
  `_proc None`, log fd closed, `running False`. Callers can distinguish "not
  running" (`call()` raises Fatal fast) from retryable staleness, retry `start()`
  with no ghost of the failed attempt, and stack issue 014's budget fix on top
  without handle-state surprises. `stop()` after a failed `start()` remains safe.
