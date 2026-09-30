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

## Progress log

- 2026-09-30: premise RE-VERIFIED live in-container (as-read 2026-09-30,
  tree drifts): `ThreadPoolExecutor` + `append_sfx_window` inside
  `_render_one` confirmed, `Lock` still absent, `append_sfx_window`
  still file-fsync only (no `fsync_dir`). Second half already
  RESOLVED before this pass: `validate_sfx_ledger` dedupes last-wins +
  sorts by `start` (`sfx_finalize.py:230-231`), covered by existing
  `test_153_validate_dedupes_duplicate_window_ids` — VERDICT recorded,
  no invented fix. TDD: 13 new tests in
  `tests/test_ledger_rotation_rank2.py` watched fail (11 failed, 2
  passed — the 2 passes are the already-resolved validate leg +
  the usually-atomic concurrent-append smoke), then fixed.

## Resolution (FIXED)

- `voyage/sfx_finalize.py::append_sfx_window` now calls
  `fsync_dir(ledger.parent)` after the file fsync (101 twin leg;
  imported from `voyage.atomic`).
- `voyage/sfx_finalize.py::render_sfx_bed::_render_one` no longer
  appends: it returns `(stem, logged|None, stored, size)` and the
  caller appends serially in plan order after the pool joins (both
  `num_workers=1` and `2` paths). Stems still land via parallel
  atomic replace (distinct files); the ledger is deterministic by
  construction — no `threading.Lock` needed. Docstring states the
  contract.
- Tests: `test_054_append_sfx_window_syncs_directory_entry`,
  `test_054_two_workers_append_in_window_order` (sleep-inverted
  workers: w0000 sleeps 0.3 s so w0001 finishes first — ledger still
  `[w0000, w0001]`), `test_054_concurrent_appends_all_parse`,
  `test_054_validate_is_order_insensitive` (regression for the
  already-landed half).
- Evidence: new file 13/13 pass; related suites
  `test_ledger_rotation_rank2 + test_sfx_finalize +
  test_observability + test_concept_integrity +
  test_worker_perf_rank2 +
  test_commit_side_integrity_095_101_104` = 99 passed, 1 skipped;
  `ruff check .` + `ruff format --check .` clean (183 files);
  `mypy voyage` clean (53 files).
- Residuals: none in owned files. `existing` dict still read once
  before the pool by design (each `window_id` rendered once per run;
  re-finalize duplicates dedupe last-wins in validate) — not a race.
