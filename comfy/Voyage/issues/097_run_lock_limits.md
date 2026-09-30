# 097 — Run lock: `flock` `OSError` misreported as contention on exotic FS, racy holder-pid read, stale lockfile never unlinked

- Severity: LOW
- Area: correctness / supervisor / concurrency
- File: `voyage/supervisor.py:352-390` (`_held_run_lock`, `_read_lock_holder` — sweep said `:344-381`, drifted +8)

## Description

Live code (re-verified 2026-09-30):

```python
# voyage/supervisor.py:352-390
lock_path = self._run_dir / "state.json.lock"
lock_path.parent.mkdir(parents=True, exist_ok=True)
lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
try:
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:                       # (1) ANY OSError → "locked by pid X"
        holder = self._read_lock_holder(lock_path)
        raise FatalWorkerError(
            f"run {self._run_dir} is locked by pid {holder}; "
            "refusing a second concurrent writer"
        ) from exc
    os.lseek(lock_fd, 0, os.SEEK_SET)
    os.ftruncate(lock_fd, 0)
    os.write(lock_fd, str(os.getpid()).encode("utf-8"))
    yield
finally:
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError:
        pass
    os.close(lock_fd)

def _read_lock_holder(self, lock_path: Path) -> str:
    try:
        return lock_path.read_text(encoding="utf-8").strip() or "unknown"   # (2) racy, no fd
    except OSError:
        return "unknown"
# (3) lockfile is never unlinked — stale pid persists after clean exit
```

Three small gaps:

1. **Over-broad `except OSError`.** `flock(LOCK_EX|LOCK_NB)` raises `BlockingIOError` (`EAGAIN`/`EWOULDBLOCK`, subclasses of `OSError`) on genuine contention — but also `ENOSYS`/`EOPNOTSUPP` on filesystems without flock (some NFS/CIFS mounts, exotic overlays), `EBADF`, etc. All are reported as "locked by pid X; refusing a second concurrent writer", misleading the operator when the real problem is an unsupported filesystem. The `from exc` chain preserves the cause, but the message lies.
2. **Racy holder read.** `_read_lock_holder` re-opens the path and reads it *without* holding the lock, racing the holder's `ftruncate`+`write` (can read a torn/empty file → "unknown", or a half-written pid). It also reads the *path*, not the *fd*, so a replaced lockfile (unlinked + recreated between `os.open` and `read_text`) reports the wrong pid.
3. **Stale pid never unlinked.** The lockfile persists after clean exit (lock is released via `LOCK_UN` + `close`, but the file + last pid remain). A later "is it locked?" inspection (`cat state.json.lock`) shows a dead pid with no way to distinguish stale from live (the lock itself is dead-with-process by design, documented — but the file suggests otherwise). No `unlink` on clean exit, no `fstat`-based liveness check.

## Rationale

- Low severity: the lock *works* on the supported filesystems (local ext4/overlay, the bind mounts in `scripts/run.sh`); contention messaging is correct in the common case. These are diagnostics + portability gaps, not a correctness hole in the critical section.
- Still worth filing: NFS-backed run dirs are plausible for multi-day voyages, and a misleading "second writer" error sends operators hunting a phantom process instead of checking mount options.

## Live evidence

```
$ grep -n "def _held_run_lock\|def _read_lock_holder\|flock\|read_text" comfy/Voyage/voyage/supervisor.py | head
352:    def _held_run_lock(self) -> Iterator[None]:
360:        lock_path = self._run_dir / "state.json.lock"
365:                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
383:    def _read_lock_holder(self, lock_path: Path) -> str:
386:            return lock_path.read_text(encoding="utf-8").strip() or "unknown"
```

- `except OSError` at `:366` with no `errno` discrimination (verified by reading `:363-372`).
- `_read_lock_holder` at `:383-388` uses `lock_path.read_text` (path-based, no fd, no lock) — races `:373-375` (`lseek`+`ftruncate`+`write`).
- No `unlink` of `state.json.lock` anywhere in the file (`grep -n "unlink.*lock\|lock.*unlink"` → zero hits).

Stdlib fact checks (no container needed): `BlockingIOError` subclasses `OSError` (`issubclass(BlockingIOError, OSError) == True`); `fcntl.flock` on `ENOSYS` filesystems raises `OSError: [Errno 38] Function not implemented` — indistinguishable from contention in the current handler.

## Repro

1. Exotic FS: place a run on an `flock`-less mount (or stub `fcntl.flock` to raise `OSError(ENOSYS)`) → second `commit_one_segment` reports "locked by pid unknown" instead of "filesystem does not support locking".
2. Race: instrument `_read_lock_holder` to read during the holder's `ftruncate`-before-`write` window → returns `""` → "unknown" even though a live holder exists.
3. Stale file: run one commit cleanly, `cat state.json.lock` → shows a pid with no live holder; no API distinguishes it from contention without attempting the lock.

## Fix candidates

1. Discriminate `errno`: only `EAGAIN`/`EWOULDBLOCK` → contention message; other `OSError` → re-raise as `FatalWorkerError(f"cannot lock {lock_path}: {exc}")` preserving `errno`. Two-line change, keeps the fast-fail.
2. Read the holder from the *fd* (`os.pread` after failed lock, or `fstat`), or best-effort with a comment acknowledging the race (it is diagnostic-only — the lock decision itself is already made by `flock`). At minimum document that "unknown" includes torn reads.
3. Optionally `unlink` the lockfile on clean exit when no contention is possible (or document "stale pid file is expected; the lock is the flock, not the file" in the docstring — the current docstring already says "dies with the process", extend it to "the pid file is advisory only").
4. Tests: stub `flock` raising `ENOSYS` → assert non-contention message; torn-file fixture → "unknown" without crash.

## Refs

- Overlaps with 099 (the lock hardened here is the lock 099 shows the control plane bypasses) — ownership stays here (lock diagnostics/portability).
- `voyage/supervisor.py:352-390` (full lock + holder quoted above).
- Python `fcntl.flock`: https://docs.python.org/3/library/fcntl.html#fcntl.flock ; `errno.EAGAIN` vs `ENOSYS`: https://docs.python.org/3/library/errno.html
- `BlockingIOError` subclasses `OSError`: https://docs.python.org/3/library/exceptions.html#BlockingIOError
- DESIGN §31 (atomic writes) + issue 004 (single-writer lock — this function is its implementation).

## Progress log

- 2026-09-30 (Group B): re-verified live against current tree before any fix.
  Static: `voyage/supervisor.py:437-443` discriminates `errno` (`EWOULDBLOCK`/
  `EAGAIN` → contention message, everything else re-raised raw); `:475-504`
  `_read_lock_holder` parses the pid and liveness-checks via `os.kill(pid, 0)`
  (`ProcessLookupError` → `"unknown"`, `PermissionError` → names the pid,
  torn/empty/non-numeric/non-positive → `"unknown"`); `:461-473` unlinks the
  lockfile on clean exit only while it still names this process (pid-guarded,
  best-effort, never raises). Behavioral probes in `voyage:latest`
  (`docker run --rm -v $PWD:/app -w /app voyage:latest`, CPU-only):
  `OSError(ENOSYS)` from stubbed `fcntl.flock` propagates raw
  (`errno == ENOSYS`, no `FatalWorkerError`, no "locked by pid"); stale pid
  `42424242` reads back `"unknown"`; lockfile absent after a clean-exit
  commit. All three legs of this issue are already covered — no code written.

## Resolution

- Verdict: FOLDED into 004. No code, no tests: batch-2's 004 fix
  (`voyage/supervisor.py:437-473` + `:475-504`) covers leg 1 (EWOULDBLOCK-only
  contention message, exotic-FS errors propagate raw), leg 2 (liveness-checked
  holder read — a dead pid reports `"unknown"` instead of naming a phantom),
  and leg 3 (pid-guarded unlink on clean exit — no stale pid file survives a
  clean commit). Residual race honestly documented in the 004 code comment
  (contender arriving between the guard read and `unlink` splits onto a fresh
  inode — the lock itself stays correct). No DESIGN change (004's §31/§72
  coverage stands).
