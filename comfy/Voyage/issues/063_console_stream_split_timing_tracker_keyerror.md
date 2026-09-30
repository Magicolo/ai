# 063 — Console breaks its own stream/timing contracts; spinner leaks on `BaseException`; tracker `KeyError` on unknown labels

**Severity:** MEDIUM

**File:line:** `voyage/console.py:149-150` (`error` hardcodes `sys.stderr`) vs `:121-138` (`line`/`styled` honor `self._stream`); `:159-198` (`stage`: `except Exception` only; plain failure path `:194-195` drops elapsed); `:361-373` (`ParallelDownloadTracker.succeed/fail` index `_tasks` directly); `docs/OPERATIONS.md:36-41`

**Overlaps with:** 028 (console contract breaks — same file, same stream-split root cause; recommend merging into one)
- **Description:** Merged docs-sweep item 9 + structure-sweep item 9 (same file, complementary halves — deduped here). (a) Every method honors the injected `stream` except `error()`, which prints to `sys.stderr` — tests/embeds capturing `stream` lose errors, and only the TUI's `redirect_stdout/redirect_stderr(buffer)` merge hides it. (b) The non-TTY `stage()` failure path prints `✗ {label} failed` with no elapsed seconds, while success prints `✓ {label} in {elapsed:.1f}s` and the rich path prints `failed after {elapsed:.1f}s` — the failure timing operators need most is the one dropped. (c) `stage()` stops the spinner/status only on `Exception`; `KeyboardInterrupt`/`CancelledError` (`BaseException`) skip `stop.set()`/`status.stop()` — the daemon ticker spins on and the rich `Status` is never stopped, corrupting terminal state. (d) `ParallelDownloadTracker.succeed/fail` index `self._tasks[label]` with no guard — a typo'd or double-reported label raises `KeyError` inside the completion path. (Re-verified live: `_started` now uses `.get(label, …)` — that half is fixed; `_tasks[label]` remains direct.)
- **Rationale:** Console is the only live monitor on long GPU stages; non-TTY output (pipes, CI, log files) is the machine-scraped surface. A display layer's contract is "same words, either sink" — one method breaking the sink rule plus unguarded indexing in the success path are exactly the faults that surface during the failure they're meant to report.
- **Evidence (re-verified live 2026-09-30):**
```python
# voyage/console.py:150 vs 126/138 (live)
print(f"✗ {message}", file=sys.stderr)   # error()
print(text, file=self._stream)           # line()/styled()
# voyage/console.py:194-198 (live: failure drops elapsed, success keeps it)
self.line(f"✗ {label} failed")           # no seconds
self.line(f"✓ {label} in {elapsed:.1f}s")
# voyage/console.py:180,194 (live)
except Exception:                        # BaseException skips teardown
# voyage/console.py:361-373 (live: _started guarded, _tasks not)
elapsed = time.monotonic() - self._started.get(label, time.monotonic())
self._progress.update(self._tasks[label], completed=1)
```
`ParallelDownloadTracker.fail` documents "stays on stdout so TUI capture keeps it" — the convention exists, `error()` violates it. Live probe: `VoyageConsole(no_color=True, stream=s).error('boom')` prints to real stderr; `s.getvalue()==''`.
- **Repro:** `python3 -c "from voyage.console import VoyageConsole; c=VoyageConsole(no_color=True); try:
 with c.stage('video'): raise RuntimeError('boom')
except RuntimeError: pass"` → `✗ video failed` (no seconds); `c.error('x')` lands on stderr. `grep -n "except Exception" voyage/console.py`.
- **Fix candidates:** Route `error()` through `self._stream` (keep a stderr mirror behind a flag if desired); add elapsed to the plain failure line; `except BaseException` for spinner teardown with re-raise; `.get()` + warning line for unknown tracker labels; document the stream contract in the module docstring.
- **Refs:** `voyage/console.py` (console-only contract); `docs/OPERATIONS.md:23-41` (TTY vs pipe parity promise); Textual validation pattern (validate-then-report discipline for tracker labels).
