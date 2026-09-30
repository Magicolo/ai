# 056 — Worker logs rotate only on worker `start()` — never during a long run

**Severity:** HIGH (unbounded growth on the "infinite" path)

**File:line:** `voyage/rpc.py:138-141` (`SubprocessWorker.start`); `voyage/logrotate.py:1-14` (module claim); `voyage/supervisor.py:296-320` (worker log paths `video-worker.log`, `audio-worker.log`, `director-worker.log`); `voyage/supervisor.py:468-470` (metrics path rotates per commit via `append_line`)
- **Description:** The module docstring promises "Daily (time-based) log rotation … `metrics.jsonl` and the three worker logs". `metrics.jsonl` rotates per commit via `append_line()` (`supervisor.py:463` → `logrotate.append_line` → `rotate_log`). Worker logs call `rotate_log()` exactly once in `SubprocessWorker.start()` while the stderr-redirected handle stays open for the whole worker lifetime. A healthy infinite run that never restarts a worker never rotates those files. Related structure-sweep item 11a (workers/loop stdout quarantine: C-level `printf` bypasses `contextlib.redirect_stdout`, `voyage/workers/loop.py:91-96`) compounds this — the very stream most likely to flood is the one with no mid-run rotation.
- **Rationale:** The runbook's core promise is "operationally indefinite" (DESIGN §3.6). Time-based rotation that only fires on restart is size-unbounded within a day *and* across days for stable workers. Best practice is size-or-time rotation at a line boundary with the writer reopening (logrotate / Kubernetes `containerLogMaxSize` pattern).
- **Evidence (re-verified live 2026-09-30):**
```python
# voyage/rpc.py:138-141 (live)
def start(self) -> None:
    rotate_log(self._log_path)
    self._close_log_file()
    log_file = self._log_path.open("a", encoding="utf-8")
```
`grep -rn rotate_log voyage/` → `logrotate.py` def, `rpc.py:139`, plus `append_line` internal (`logrotate.py:99`, called from `supervisor.py:470`). No other `rotate_log(self._log_path)` call site. The open `stderr=log_file` handle (`rpc.py:143-148`) is never reopened mid-run.
- **Repro:** `grep -rn "rotate_log" voyage/ --include='*.py'`; start a worker, `ls -l logs/*-worker.log` after 24h of steady commits with zero restarts — file is never renamed, no `-YYYY-MM-DD` sibling appears (contrast `metrics.jsonl`, which does).
- **Fix candidates:** Rotate-and-reopen on a cadence (per-commit `rotate_log` + reopen when rotated, or size check + reopen); or document that worker logs are restart-rotated only and add a `run.sh`/cron reopen hook. Never rotate under an open handle without reopening (would keep writing to the renamed inode).
- **Refs:** `voyage/logrotate.py:1-14`; K8s logging best practices (kubelet `containerLogMaxSize 10Mi / MaxFiles 5` defaults); `voyage/workers/loop.py:91-96` (stdout-quarantine note, structure sweep item 11a).

**Overlaps with:** 057 (rotation mtime/size/fsync — sibling rotation defect, different half; not a duplicate).

## Progress log (2026-09-30, supervisor-track resolution)

- Premise re-verified against current live code (in-container):
  `rotate_log` called only in `SubprocessWorker.start`
  (`Voyage/voyage/rpc.py`) plus `append_line` internals; no mid-run worker-log
  rotation; the stderr handle stays open for the worker lifetime — premise
  holds, unbounded growth on the infinite path confirmed.
- TDD: new `Voyage/tests/test_supervisor_av_align.py` —
  `test_rotate_open_log_truncates_in_place` failed first with `ImportError`
  (helper absent); `test_worker_logs_rotate_mid_run_keeps_bounded`
  (pre-bloated 4 KiB worker logs, 1 KiB trigger via monkeypatch, 2-segment
  fake run) failed first (live logs stayed 4096, no siblings), both pass
  after the fix.
- Fix (owned files only; `rpc.py` untouched): new `logrotate.py`
  `WORKER_LOG_FILENAMES` / `MAX_WORKER_LOG_BYTES` (10 MiB, K8s
  `containerLogMaxSize` parity) / `rotate_open_log` (copytruncate: archive
  to a dated sibling via `_unique_rotated`, then truncate live in place so
  the open `O_APPEND` writer continues at offset 0 — never rename under an
  open handle) / `rotate_worker_logs` (per-dir tick, never raises;
  `max_bytes=None` reads the current constant at call time as the
  monkeypatch seam); `Supervisor._rotate_worker_logs` wired once per
  committed segment in `_commit_segment` and `_adopt_unaccounted_segment`.
  `metrics.jsonl` path unchanged (per-append open/close, rename-safe).
- Evidence: new suite 8 passed; related suites as in 003 (38 + 40
  passed). `ruff check` + `ruff format --check` + `mypy strict` clean on
  touched files.

## Resolution

- Status: resolved this pass (mid-run copytruncate rotation per committed
  segment; retention via existing `_prune_siblings`). No `DESIGN.md` edit
  made here — as-built proposal is in the agent report.
