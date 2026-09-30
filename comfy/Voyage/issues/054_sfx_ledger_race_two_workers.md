# 054 — SFX two-worker ledger appends race (concurrent `open("a")` + `fsync` without a lock)

**Severity:** MEDIUM

**File:line:** `voyage/sfx_finalize.py:181-196` (`append_sfx_window`), `255-365` (`render_sfx_bed`), esp. `316-365` (`_render_one` + `ThreadPoolExecutor.map`), `288-292` (dual sizes), `226-239` (order-sensitive validate)

**Description:**
With `num_workers=2`, `pool.map(_render_one, ...)` runs two threads sharing one `SubprocessWorker` pair (safe — distinct processes/locks) but one ledger file: both threads call `append_sfx_window` → `open("a")` → `write(json)` → `flush` → `os.fsync` concurrently. POSIX `O_APPEND` makes each `write(2)` atomic only up to `PIPE_BUF` for pipes; for regular files concurrent appends can interleave when the JSON line exceeds the filesystem's atomic-append guarantee (lines are ~200–400 bytes — usually safe, not guaranteed), and `existing` re-read staleness means a retry can double-append the same `window_id`. The stems list order is preserved by `map`, but ledger order is thread-race order — `validate_sfx_ledger` walks in file order for coverage (226-239), so a reordered ledger can false-positive a "coverage gap."

**Rationale:**
Takes-ledger philosophy is "immutable, versioned, fsynced" — concurrent unsynchronized appends violate the single-writer invariant (single writer per run, `fcntl` lock elsewhere in supervisor paths).

**Live evidence (current tree):**
```python
sfx_finalize.py:361-365: if num_workers==1: serial; else: with ThreadPoolExecutor(...) as pool: stems = list(pool.map(_render_one, ...))
sfx_finalize.py:358: append_sfx_window(ledger, logged, stored, sizes[slot])  # inside _render_one → from 2 threads
sfx_finalize.py:181-196: open("a") + write + flush + fsync, no threading.Lock
sfx_finalize.py:226-239: validate walks ledger in file order for gap detection (cursor vs start)
sfx_finalize.py:296: existing = {record["window_id"]: record ...}  # read once before pool; no re-read
```
`rg Lock voyage/sfx_finalize.py` → no hits.

**Repro (CPU):**
```python
import inspect
from voyage import sfx_finalize as s
src = inspect.getsource(s.render_sfx_bed)
assert "ThreadPoolExecutor" in src and "append_sfx_window" in src
assert "Lock" not in src  # no ledger mutex
```

**Fix candidates:**
- Collect `(window, stored, size)` in workers, append to ledger serially after `map` joins (or guard with a `threading.Lock`).
- Make `validate_sfx_ledger` order-insensitive (sort by `start`) so ledger order can never false-positive coverage.
- Alternatively shard ledgers per worker then merge deterministically.

**Refs:** `voyage/audio/planner.py:211-220` (`append_take` single-writer pattern); `voyage/rpc.py` shared-stream serialization precedent.

**Overlaps with:** 164 (state recovery omits SFX ledger — sibling SFX-ledger defect, different half; not a duplicate).
