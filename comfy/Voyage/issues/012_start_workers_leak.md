# 012 — `run_segments` calls `start_workers()` before `try/finally`, orphaning GPU workers on init failure

- Severity: HIGH
- Group: correctness/lifecycle — Rank: 1/5
- File:line: `voyage/supervisor.py:622-630` (`run_segments`), `voyage/supervisor.py:442-450` (`start_workers`), `voyage/rpc.py:182-190` (`SubprocessWorker.start`)

## Technical description

Live code (re-verified 2026-09-30):

```python
# voyage/supervisor.py:435-442
def start_workers(self) -> None:
    self._logs.mkdir(parents=True, exist_ok=True)
    self._video.start()
    self._audio.start()
    self._director.start()
    self._workers_running = True
    ...

# voyage/supervisor.py:615-624
def run_segments(self, count: int | None) -> list[str]:
    ...
    self.start_workers()   # <-- OUTSIDE try
    try:
        ...
    finally:
        self.stop_workers()
```

`start_workers()` starts workers sequentially. Each `SubprocessWorker.start()` (`voyage/rpc.py:133-147`) replays `init` via `self.call(...)`, which can raise `RecoverableWorkerError` / `FatalWorkerError` (e.g. audio ACE-Step DiT OOM, director Qwen load failure, missing models).

If the second or third `start()` raises, the exception escapes **before** the `try` is entered, so `finally: stop_workers()` never runs. Workers already started (e.g. video) keep running, holding VRAM (video DiT ~6–14 GiB, ACE ~5 GiB) and file descriptors.

## Why it matters

- Orphaned GPU workers hold VRAM after the supervisor has exited with an error. The next `voyage run` then OOMs for no model reason (precedent: 4060 Ti 16 GiB has ~1–2 GiB headroom at fp8 full-res; one orphaned resident is fatal).
- `stop_workers()` also owns `_prefetch_executor` shutdown; skipping it leaks the executor thread.
- The failure mode is silent: the traceback shows the init error, not the leak. Operators retry and hit a *different* error (OOM), misattributing it to the model.

## Live evidence

Command + output (host, stdlib only):

```
$ grep -n "def run_segments\|def start_workers\|self.start_workers()" comfy/Voyage/voyage/supervisor.py
435:    def start_workers(self) -> None:
615:    def run_segments(self, count: int | None) -> list[str]:
623:        self.start_workers()
```

```
$ PYTHONPATH=Voyage python3 -c "from pathlib import Path; txt=Path('Voyage/voyage/supervisor.py').read_text(); print(txt[txt.find('def run_segments'):txt.find('def run_segments')+800])"
        self.start_workers()
        try:
            self._restarts = {}
```

`start()` replays init (`voyage/rpc.py:146-147`):

```python
if self._init_op is not None:
    self.call(self._init_op, dict(self._init_payload))
```

`call` raises `RecoverableWorkerError`/`FatalWorkerError` on timeout / broken pipe / error response — all reachable during model load. No `try` wraps the `self.start_workers()` call site.

Host cannot import `voyage.supervisor` directly (pydantic/numpy are container-only — `ModuleNotFoundError: No module named 'pydantic'`), so this is source-inspection + stdlib evidence, per task allowance. Container image `voyage:latest` carries the deps; behavior is identical.

## Repro steps

1. Configure a run whose audio backend init fails deterministically (e.g. point `models_dir` at an empty dir, or force ACE DiT OOM by holding VRAM with a second process).
2. Call `Supervisor(...).run_segments(1)`.
3. Observe: `start_workers` raises from `self._audio.start()`; `ps` / `nvidia-smi` still shows the video worker subprocess alive; `_workers_running` is `False` but `self._video.running` is `True`.
4. Retry the run on the same GPU → OOM even though the first run "exited".

## Fix candidates

1. (Preferred, minimal) Move `start_workers()` inside the `try`:
   ```python
   try:
       self.start_workers()
       ...
   finally:
       self.stop_workers()
   ```
   `stop_workers()` already tolerates non-started workers (`proc is None` → close log file only), so calling it after a partial start is safe.
2. (Defense in depth) Make `start_workers()` itself exception-safe: on failure, stop any workers already started before re-raising (reverse-order teardown). Protects direct `start_workers()` callers too (`commit_one_segment` requires it first).
3. Regression test (fake workers): stub `_video.start` OK + `_audio.start` raising; assert `run_segments` raises AND `_video.running is False` AND no executor left behind.

## References

- `voyage/supervisor.py:435-442` (sequential start), `:615-624` (call outside `try`), `:444-453` (`stop_workers` — safe on partial start, never reached).
- `voyage/rpc.py:133-147` (`start` replays `init` via `call`, the raising operation).
- Python `try/finally` — "If an exception occurs in `start_workers()` before the `try` statement is entered, the `finally` clause is never executed": https://docs.python.org/3/tutorial/errors.html#defining-clean-up-actions
- DESIGN §73 (supervisor owns lifecycle); §69 crash hook (`inject_worker_crash`) assumes `stop()` reaps — same invariant violated here.

## Progress log

- 2026-09-30: re-verified live before touching anything — `run_segments`
  still called `self.start_workers()` outside the `try` (pre-fix
  `supervisor.py:630`), and `start_workers` still started the three workers
  sequentially with no unwind (pre-fix `:442-449`). Premise holds; no
  prior fix, no concurrent edits in these regions (`git diff` shows only
  the hunks below; the tree's other uncommitted changes are `voyage/rpc.py`
  + `tests/test_rpc_{start,timeout}.py`, another agent's scope, untouched).
- 2026-09-30 (TDD red): new `tests/test_supervisor_lifecycle.py` with
  `test_start_workers_failure_stops_already_started` and
  `test_run_segments_partial_start_stops_workers` (recording start/stop
  doubles, no subprocesses; audio start raises `FatalWorkerError`). Both
  failed as predicted (`stopped == []` — the orphan leak, live).
- 2026-09-30 (green): implemented BOTH fix candidates — exception-safe
  `start_workers` (reverse-order unwind, original error wins) AND
  `start_workers()` moved inside the `try` in `run_segments` (defense in
  depth; `stop_workers` tolerates non-started workers, double-stop is
  safe). All 4 new tests pass.
- 2026-09-30: red-proof — `git stash push -- voyage/supervisor.py` →
  all 4 new tests fail; `git stash pop` → all 4 pass. Fix verified
  load-bearing, not vacuous.
- 2026-09-30: regression `test_failure_policy + test_crash_matrix +
  test_commit_hardening + test_commit_split + test_state_integrity` green
  (62 passed with the new file); full `./scripts/gates.sh` green —
  ruff + format + mypy strict clean, 1101 passed / 3 skipped /
  1 deselected (gpu), coverage 77% (floor 65).

## Resolution

Fixed (both fix candidates applied, defense in depth):

- `voyage/supervisor.py:442-465` — `start_workers()` is now
  exception-safe: starts are tracked in a `started` list and any failure
  stops already-started workers in reverse order before re-raising
  (unwind `stop()` errors are swallowed so the original start error wins;
  `_workers_running` stays `False`).
- `voyage/supervisor.py:647-648` — `run_segments()` now calls
  `self.start_workers()` INSIDE the `try`, so `finally:
  self.stop_workers()` always runs even when the start itself raises.
- Tests: `tests/test_supervisor_lifecycle.py`
  (`test_start_workers_failure_stops_already_started`,
  `test_run_segments_partial_start_stops_workers` — fake backends on
  `tmp_path`, no GPU/network).
- Gates: `ruff check` + `ruff format --check` + `mypy` (strict) clean on
  both touched files; full `Voyage/scripts/gates.sh` green (1101 passed,
  3 skipped, 1 deselected).
