# 101 — Ledger/metadata durability gaps: file-fsync without fsync_dir, non-atomic concept group, unf synced metrics + rename

**Severity:** MEDIUM

**File:line (verified live 2026-09-30):**
- `voyage/audio/planner.py:211-220` (`append_take`: `flush` + `os.fsync`, no `fsync_dir`)
- `voyage/concepts.py:299-324` (`ConceptStore.append`: vector (atomic) → jsonl append (`flush` + `fsync`, lines 318-321, no `fsync_dir`) → `_write_index` (atomic))
- `voyage/concepts.py:221-268` (`_append_vector`: temp + fsync + rename + `fsync_dir`, lines 253-260 — the correct pattern, proving the jsonl side is the gap)
- `voyage/concepts.py:326-330` (`_write_index` via `atomic_write_json` — atomic alone)
- `voyage/logrotate.py:99-104` (`append_line`: bare `handle.write`, **no flush, no fsync at all**)
- `voyage/logrotate.py:70-96` (`rotate_log`: `path.rename(rotated)` at line 92, no `fsync_dir`; `_prune_siblings` unlinks without it either)
- `voyage/supervisor.py:461-463` (`_log_metric` → `append_line` on every commit, gauge, prefetch hit/miss — the highest-frequency writer)
- Contrast: `voyage/atomic.py:37-58` (`atomic_write_bytes`: file fsync + `os.replace` + `fsync_dir`, lines 43-48; docstring at `atomic.py:23-29` explains why the directory entry must be synced)

**Description:**
Three durability gaps of increasing severity, all in append paths the crash-recovery story depends on:

1. **`append_take` (`planner.py:211-220`) has file fsync but no `fsync_dir`.** The docstring claims "Durably append one take to the ledger (flush + fsync)". File fsync persists *content*; on ext4/xfs (default `auto_daemon` / ordered mode) a crash/power loss can still lose the *directory entry update* (file size change on an existing file is journaled via the inode, but the common strict reading of the durability contract — and the tree's own `atomic.py:23-29` — requires `fsync_dir` after any mutation whose loss matters). A lost ledger tail means rendered takes exist on disk but the planner doesn't know them: the next run re-renders 30-60 s of ACE-Step audio and, worse, `take_for_time` coverage regresses, potentially stalling the audio-ahead invariant. Same shape in `sfx_finalize.py:182-196` (identical "flush + fsync, takes pattern" comment, same missing `fsync_dir`).

2. **`ConceptStore.append` is three separate durable-ish writes, not one atomic group.** Order today: `_append_vector` (fully atomic, line 307) → jsonl append (file-fsync only, lines 317-321) → `_write_index` (atomic, line 323). Crash windows: (a) after jsonl fsync but before `_write_index` → on reload the record exists with `embedding_index >= 0` but `concept_index.json` lacks the key — index/jsonl disagree; (b) after `_append_vector` but before the jsonl write → orphan vector row (benign: nothing references it, but `stored_count` grows and the `_dangling_vector_records` guard at lines 206-219 / 281-287 now has a row nobody owns, which is at least confusing during forensics). The tree already built the fail-loud dangling-row detector (issue 059) for the *missing-row* direction; the *missing-index-key* direction has no detector. Note the asymmetry the code itself highlights: the `.npy` side got temp+fsync+rename+`fsync_dir` (lines 253-260) while the jsonl side it must stay consistent with did not.

3. **`append_line` (`logrotate.py:99-104`) has no flush/fsync at all, and rotation renames without `fsync_dir`.** Every `_log_metric` call (commit events, gauges, prefetch hits, `segment_committed` with stage timings — the observability surface `status`/`scoreboard`/`soak` read) goes through this function. A crash loses an unbounded tail of buffered metrics (stdio buffering: up to 8 KiB per handle), so post-crash forensics (`status` last-commit, soak averages) silently degrade. `rotate_log`'s `path.rename(rotated)` (line 92) without `fsync_dir` on the containing dir risks losing the rename itself on power loss — the live `metrics.jsonl` and its dated sibling can swap identities from the reader's perspective. Multi-writer interleaving: `path.open("a")` is `O_APPEND`, which is atomic per `write(2)` syscall, not per logical line — CPython's buffered writer may split one `handle.write(line + "\n")` across syscalls, so two processes appending (supervisor + a concurrent `benchmark`/`soak` on the same run dir) can interleave mid-line JSON fragments. Today the supervisor is effectively the single metrics writer per run, so this is latent — but nothing enforces it, and `iter_metric_files` readers assume one-JSON-per-line.

**Rationale:**
The tree's durability story (DESIGN §31, `atomic.py`) is careful and explicit — which makes these append paths the odd ones out: each carries a "durable" comment while skipping the directory half of durability, and `append_line` skips even the file half. Crash-recovery features (`validate_run`, orphan scans, recovery tapes) all assume the ledger/metrics they read survived; a power-loss test on a real voyage would find takes rendered but unledgered and metrics tails missing. These are cheap to fix (one `fsync_dir` import + call per site) relative to the forensics cost of discovering them after a crash.

**Live evidence (current tree):**
```
$ rg 'fsync_dir|os\.fsync|flush' voyage/audio/planner.py
211:    """Durably append one take to the ledger (flush + fsync)."""
219:        handle.write(json.dumps(take.to_dict()) + "\n")   # (via Read: flush at 219, fsync at 220)
220:        os.fsync(handle.fileno())
→ no fsync_dir anywhere in the file (grep-confirmed: only lines 212/219/220 match).

$ rg 'fsync|flush' voyage/logrotate.py
→ zero matches: append_line (99-104) is write-only; rotate_log (70-96) is rename-only.

$ rg 'fsync_dir' voyage/concepts.py
260:            fsync_dir(self._vectors_path.parent)   # vectors .npy only
→ the jsonl append (318-321) and nothing else; index goes through atomic_write_json.
```
Full live-read bodies: `planner.py:211-220`, `concepts.py:299-324`, `logrotate.py:70-104`, `atomic.py:23-58` (the in-tree statement of the correct contract).

**Repro (CPU, no GPU):**
1. `append_take` durability: append a take, `os.sync()`-free crash (kill -9 the writer between `os.fsync` return and any directory sync; deterministically: fault-inject by monkeypatching `os.fsync` to raise after the write, then show the ledger tail missing while the take file exists). Simpler static proof: `strace -e fsync,fdatasync,syncfs -f` around `append_take` shows no directory-fd sync.
2. Concept group tear: interpose a fault between the jsonl write (line 321) and `_write_index` (line 323) — e.g. `SIGKILL` via a patched `_write_index` that kills the process — then reload the store and compare `concept_index.json` keys against accepted records with `embedding_index >= 0`.
3. Metrics loss: call `append_line` in a loop, `SIGKILL` without close/flush, count surviving lines vs written (buffering eats the tail); concurrent-append interleave: two processes appending 10 KiB lines to one metrics file, then `json.loads` each line and count failures.

**Fix candidates:**
- Add `fsync_dir(ledger.parent)` after the file fsync in `append_take` (and the twin in `sfx_finalize.py:182-196`); same for the concepts jsonl append (`concepts.py:317-321`).
- Make the concept append group atomic: write jsonl-line + index through a single temp+rename (or journal the intent: write index first with the new key pointing at the *future* row, then append vector, then jsonl — ordering such that every crash state is either complete or fails-loud via the existing `_dangling_vector_records` guard; extend that guard to cover missing index keys too).
- `append_line`: add `flush` + file `fsync` (or document explicit best-effort loss with a comment — metrics are advisory, takes are not; the two deserve different contracts stated in code), `fsync_dir` after `rotate_log`'s rename, and a single `os.write` on an `O_APPEND` fd for one-syscall-per-line atomicity under multi-writer use.
- Tests: fault-injection crash-consistency tests (kill between group members, reload, assert no disagreement); `strace`-style assertion that each ledger append issues a directory sync; metrics round-trip under two-process append.

**Refs:**
- In-tree contract: `voyage/atomic.py:23-29` ("fsync on the file alone does not guarantee the rename survives a power loss on most filesystems — the containing directory entry must be synced too") and `:37-58` (the full pattern every append site should match).
- In-tree correct example: `voyage/concepts.py:253-260` (`.npy` temp + fsync + rename + `fsync_dir`).
- POSIX: `fsync` on a file vs `fsync` on the containing directory descriptor (`open(dir, O_RDONLY)` + `fsync`) — file data vs namespace durability; `O_APPEND` atomicity is per `write(2)` (see `open(2)`/`write(2)` man pages).
- Readers depending on these files: `voyage/audio/planner.py:197-208` (`load_takes`), `voyage/concepts.py:338-346` (`load_jsonl`), `voyage/logrotate.py:107-129` (`iter_metric_files`).

## Progress log

- 2026-09-30: premise CONFIRMED live — `rg fsync_dir|os.fsync` shows
  `planner.py:220`, `concepts.py:321`, `sfx_finalize.py:196` all file-fsync
  only; `logrotate.append_line`/`rotate_log` have zero sync calls. Contract
  limits this change to `planner.py` (+ `supervisor.py` commit-side half, which
  needs nothing: it delegates to `append_take`), so only the ledger leg is
  fixed here; the rest are recorded below as follow-ups for their owners.
  TDD: new test asserting `append_take` syncs the directory entry watched
  fail (planner had no `fsync_dir` reference), then fixed.

## Resolution (PARTIALLY FIXED — ledger leg; rest changed-hands)

- Fixed: `voyage/audio/planner.py::append_take` now calls
  `fsync_dir(ledger.parent)` after the file fsync (imported from
  `voyage.atomic`); docstring states the namespace-durability contract.
- NOT in scope (out-of-contract files, left untouched): `concepts.py` jsonl
  append + `_write_index` grouping, `logrotate.py` `append_line` flush/fsync +
  `rotate_log`/`_prune_siblings` `fsync_dir`, `sfx_finalize.py` ledger twin,
  multi-writer `O_APPEND` atomicity — same one-line `fsync_dir` pattern
  applies; recommend the owning passes take them.
- Tests: `test_append_take_syncs_directory_entry` (patched `fsync_dir`
  records exactly `[ledger.parent]`).
- Evidence: 21/21 new tests pass; audio/planner/rhythm/accounting suites
  (62) green; ruff + format + mypy clean on touched files.

## Progress log (remainders — 2026-09-30, owned files)

- Planner leg: VERDICT already-resolved (no invented fix) — live
  `planner.py::append_take` carries `fsync_dir(ledger.parent)` with
  the namespace-durability docstring from the prior pass.
- Concepts leg: CONFIRMED live — jsonl append `flush` + `os.fsync`
  only, no `fsync_dir`; three-write group (vector → jsonl → index)
  with the missing-index-key direction undetected
  (`validate_concepts` covered index→records only).
- Metrics leg: CONFIRMED live — `append_line` bare `write`, zero
  sync calls; `rotate_log` rename without `fsync_dir`;
  `_prune_siblings` unlink without dir sync.
- SFX twin leg: CONFIRMED live — `append_sfx_window` flush + fsync
  only, no `fsync_dir` (identical comment shape to the planner
  twin).
- `rotate_log`/`_prune_siblings` dir-sync leg: CONFIRMED live (see
  057 evidence). Multi-writer `O_APPEND` single-syscall atomicity:
  evaluated — supervisor is effectively the single metrics writer
  per run today, so the risky rewrite is deferred (documented
  below); the flush+fsync+dir-sync that matters for crash tails
  lands now.
- TDD: remainder tests in `tests/test_ledger_rotation_rank2.py`
  watched fail (missing `fsync_dir` refs, missing index-key error,
  missing sync calls), then fixed. `bench.py` untouched (concurrent
  060 hunks present — avoided per contract).

## Resolution (REMAINDERS FIXED — owned files only)

- `voyage/sfx_finalize.py::append_sfx_window`: `fsync_dir`
  after the file fsync (twin of `append_take`).
- `voyage/concepts.py::ConceptStore.append`: `fsync_dir` after the
  jsonl file fsync (both vector and jsonl sides now sync);
  docstring states the non-atomic group + fail-loud contract.
- `voyage/concepts.py::validate_concepts`: missing-index-key
  detector added (accepted ids ⊆ index keys, plus missing-index
  file with accepted records errors) — both crash directions now
  fail loud; docstring cites 059/101.
- `voyage/logrotate.py::append_line`: `flush` + `os.fsync` +
  `fsync_dir` (metrics tail durable); `rotate_log`: `fsync_dir`
  after rename; `_prune_siblings`: `fsync_dir` after any unlink
  batch (both best-effort, never raise housekeeping).
- Tests: `test_101_concepts_append_syncs_directory_entry` (≥2 dir
  syncs: vector + jsonl),
  `test_101_validate_concepts_reports_missing_index_key`,
  `test_057_append_line_flushes_and_syncs`,
  `test_057_rotate_syncs_directory_entry`,
  `test_057_prune_syncs_directory_entry`,
  `test_054_append_sfx_window_syncs_directory_entry`.
- Evidence: new file 13/13 pass; related suites 99 passed, 1
  skipped; `ruff check .` + `format --check .` (183 files) +
  `mypy voyage` (53 files) clean.
- DESIGN proposals (text only): (a) concept group atomicity —
  keep the three-write order + two-direction validator (landed)
  rather than a single temp+rename across two files (index is
  already atomic via `atomic_write_json`; a cross-file atomic
  group would need a journal); (b) metrics multi-writer
  single-`os.write(O_APPEND)` per line — adopt only if a second
  concurrent metrics writer ever lands (today single-writer, so
  the landed flush+fsync+dir-sync suffices).
