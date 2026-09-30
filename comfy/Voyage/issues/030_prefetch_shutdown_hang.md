# 030 — Director prefetch has no timeout override and `shutdown(wait=False)` cannot stop a running prefetch (non-daemon thread joins interpreter exit)

- Severity: MEDIUM
- Group: correctness/concurrency — Rank: 3/5
- File:line: `voyage/supervisor.py:766-772` (prefetch `_call`), `voyage/supervisor.py:460` (`stop_workers` shutdown)

## Technical description

Live code (re-verified 2026-09-30):

```python
# voyage/supervisor.py:759-765
def _call() -> dict[str, Any] | None:
    try:
        return self._director.call("decide", payload)  # <-- NO timeout= override
    except VoyageError:
        return None
self._prefetch_future = executor.submit(_call)

# voyage/supervisor.py:444-453
def stop_workers(self) -> None:
    ...
    executor, self._prefetch_executor = self._prefetch_executor, None
    ...
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
```

Two compounding facts:

1. **No timeout override.** `SubprocessWorker.call(op, payload, timeout=None)` defaults to `self._timeout` = `config.voyage.rpc_timeout_seconds` = 600 s (`supervisor.py:317-323`, `config.py:453`). The prefetch LLM call (`decide` on Qwen3-8B CPU, ~5 min/segment observed on CPU E2E) can therefore occupy the single prefetch thread for up to 10 minutes. `EMBED_TIMEOUT_SECONDS` (60 s) is *not* used here — only `_embed_texts` passes it.
2. **`shutdown(wait=False, cancel_futures=True)` does not stop a *running* future.** Per `concurrent.futures` docs, `cancel_futures=True` cancels only *pending* (not-yet-started) futures; the running `_call` continues to completion. `ThreadPoolExecutor` threads are **non-daemon** (`threading.Thread(daemon=False)` default), so interpreter exit **joins** them: a `voyage run` that finishes / SIGINTs while a prefetch is in-flight hangs at exit until the director RPC completes (up to 600 s), or until the worker is killed underneath it (which then raises inside the orphaned thread, silently dropped).

Additionally the prefetch shares the director worker's `_call_lock` with the commit path (`rpc.py:127-131`): a long prefetch serializes the next commit's synchronous `decide` behind it (the prefetch-miss path still pays the wait via lock contention, even though the result is discarded).

## Why it matters

- Shutdown latency: the common case (finite `--segments N`, or SIGINT pause) pays up to minutes of unexplained hang at exit with no log line (the future's `VoyageError → None` is swallowed, and `shutdown(wait=False)` returns immediately while the non-daemon thread keeps the process alive).
- Operator impact: orchestration (systemd, CI) sees the process linger after "done", may SIGKILL it, which then looks like a crash (013 window) instead of a clean rest at PAUSED.
- The prefetch is explicitly best-effort ("must never break a commit" — `supervisor.py:785`); a best-effort thread must never gate process exit.

## Live evidence

```
$ grep -n "def _call\|_director.call\|shutdown(wait" comfy/Voyage/voyage/supervisor.py
759:        def _call() -> dict[str, Any] | None:
761:                return self._director.call("decide", payload)
765:        self._prefetch_future = executor.submit(_call)
453:            executor.shutdown(wait=False, cancel_futures=True)
```

- No `timeout=` at `:761` (contrast `_embed_texts` at `:801` which passes `timeout=EMBED_TIMEOUT_SECONDS`).
- `ThreadPoolExecutor` non-daemon default: `python3 -c "import concurrent.futures, threading; e=concurrent.futures.ThreadPoolExecutor(1); f=e.submit(lambda: threading.current_thread().daemon); print(f.result()); e.shutdown(wait=False)"` → `False` (verified stdlib behavior; prefetch thread inherits it).
- `cancel_futures` semantics: "If True, ... pending futures are cancelled. ... Running futures are unaffected." (`concurrent.futures.Executor.shutdown` docs).

## Repro steps

1. Start a run with the qwen director (slow `decide`, e.g. CPU backend ~minutes).
2. Let a commit submit a prefetch, then finish the batch (`--segments 1`) or SIGINT immediately.
3. Observed: `stop_workers()` returns, but the process does not exit until the director RPC completes (watch `ps` + director-worker log); `shutdown(wait=False)` did not stop it.
4. Variant: submit prefetch then issue a synchronous `decide` on the commit path → it blocks on `_call_lock` behind the prefetch even when the prefetch will be a miss.

## Fix candidates

1. (Preferred) Bound the prefetch: `self._director.call("decide", payload, timeout=PREFETCH_TIMEOUT_SECONDS)` with a small budget (e.g. 60–120 s, mirroring `EMBED_TIMEOUT_SECONDS` rationale — prefetch is speculative, a miss just decides synchronously). On timeout the future resolves to `None` quickly and the thread frees.
2. Make exit deterministic: after `shutdown(wait=False, ...)`, also `self._prefetch_future.cancel()` (best-effort for pending) AND ensure the executor thread is daemonized (custom `ThreadFactory` with `daemon=True`) or explicitly `shutdown(wait=True, timeout=...)` with a short join in `stop_workers`. Daemon + short-timeout call is the standard best-effort-prefetch pattern.
3. Reduce lock contention: give prefetch its own lightweight director handle, or skip prefetch submission when the commit path is about to need the director (the `_take_prefetch` miss path already exists — submission gating is cheaper than lock waiting).
4. Regression test: slow-director fake (`decide` sleeps 30 s); `stop_workers()` then assert process-joinable within ~5 s and future resolved to miss, not hang.

## References

- `voyage/supervisor.py:709-765` (prefetch submit), `:767-792` (take/miss), `:444-453` (`stop_workers`), `:317-323` (director timeout = 600 s default), `voyage/rpc.py:127-131` (shared `_call_lock`), `:256-277` (`timeout=None` → worker default).
- Python `Executor.shutdown(wait, cancel_futures)` — "cancel_futures ... only pending": https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.shutdown
- Python `threading.Thread.daemon` default `False`, non-daemon threads join at exit: https://docs.python.org/3/library/threading.html#threading.Thread.daemon

## Progress log (resolution 2026-09-30)

- As-read re-verified: `_call` at `supervisor.py:956` carried no `timeout=` override (contrast `_embed_texts`, which passes `EMBED_TIMEOUT_SECONDS`); `stop_workers` at `supervisor.py:555-564` dropped the future without cancel/drain after `shutdown(wait=False, cancel_futures=True)`.
- TDD: wrote `tests/test_prefetch_shutdown.py` first — collection failed with `ImportError: PREFETCH_TIMEOUT_SECONDS` (constant absent), proving the unbounded-timeout leg before any fix.
- Fix (`voyage/supervisor.py` only): new `PREFETCH_TIMEOUT_SECONDS = 60.0` (LLM-class budget mirroring `EMBED_TIMEOUT_SECONDS`, not the 600 s default) passed explicitly in `_call`; new `PREFETCH_SHUTDOWN_DRAIN_SECONDS = 2.0` with `future.cancel()` + `future.result(timeout=...)` best-effort drain in `stop_workers` (exceptions swallowed — a wedged prefetch must never fail shutdown).
- 137 resync preserved by construction: the prefetch still goes through `SubprocessWorker.call()`, so a prefetch timeout raises `RecoverableWorkerError` inside `_call` (caught to `None`) and the late line stays subject to the stale-line discard loop on the next synchronous decide. No `rpc.py` call-loop lines touched.
- Concurrent-agent note: another track's 104/095 hunks landed in `supervisor.py` mid-task (constants block, slice walk, checksum writer) — all in disjoint regions; re-read before each edit, no foreign hunk modified.

## Resolution

- Verdict: RESOLVED. Worst-case exit hold drops from 600 s to 60 s (the orphaned thread's own prefetch budget), and `stop_workers` itself returns in ~2 s worst case even with a wedged prefetch.
- Files changed: `voyage/supervisor.py` (2 constants + `_call` timeout + `stop_workers` cancel/drain); new `tests/test_prefetch_shutdown.py` (2 tests: bounded-timeout recording stub, wedged-prefetch shutdown timing).
- Test evidence (in-container, CPU-only): new module 2/2 green; scoped regressions `test_rpc_timeout` (137 resync), `test_generation_stack` (prefetch-hit), `test_supervisor_lifecycle`, `test_prefetch_summary` all green (126 passed across the 13-module scoped suite); `ruff check` + `ruff format --check` clean on touched files; `mypy voyage` clean (49 files).
- Residuals: threads stay non-daemon (no custom ThreadFactory — larger hunk, concurrent-edit risk); a wedged prefetch therefore still joins interpreter exit up to 60 s. If sub-5 s exit is ever required, daemonize the prefetch executor as a follow-up. Lock-contention candidate 3 (dedicated prefetch handle) untouched — out of scope.
