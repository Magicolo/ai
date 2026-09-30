# 122 — `write_tape_atomic` performs zero fsync: every ltxv/causvid recovery tape is rename-atomic but not durable

- **Severity:** Medium-Low (crash recovery — power loss can lose the latest segment's recovery anchor)
- **File:line:** `Voyage/voyage/workers/video_common.py:239-249` (`write_tape_atomic`: `write_text` + `replace`, no `flush`/`fsync`/`fsync_dir` — verified by source introspection: `'fsync' in src == False`); callers `Voyage/voyage/workers/video_ltxv.py:682-683`, `Voyage/voyage/workers/video_causvid.py:831-832` (once per committed segment)
- **Area:** workers-internals tail — shared recovery-tape writer (below pass-1 coverage)

## Description

`write_tape_atomic` writes the JSON tape to a `.tmp` sibling and renames — crash-safe against *process* crashes mid-write (the old tape survives), but with no `fsync` of the temp file before the rename and no `fsync_dir` after, an OS crash / power loss can leave `recovery.pt` missing, empty, or torn: the rename itself may not have reached stable storage, and even a persisted rename can point at unwritten data blocks. Every ltxv and causvid segment commits through this function, so the *latest* tape — exactly the one a restart needs — is the one at risk.

The tree knows the correct pattern and uses it two files away: `concepts._append_vector` does temp + `flush` + `os.fsync` + `os.replace` + `fsync_dir` (`concepts.py:251-260`), and `atomic.py:23-58` documents *why* the directory half matters. The tape writer simply never adopted it.

## Rationale

Recovery tapes are the resume path (DESIGN §27.1): `handle_rebuild`/`handle_resume` read the latest tape to adopt the tail anchor. A lost tape degrades to "re-render from seed" (fail-loud, per the clean-break design) — not corruption — but on a long voyage that is hours of GPU work discarded for want of two syscalls. The asymmetry is the point: the run's *most crash-exposed* write (last thing before the process may die) has the *weakest* durability of any persisted artifact in the video path.

## Evidence (verified live 2026-09-30)

- Source introspection in-container: `inspect.getsource(video_common.write_tape_atomic)` contains neither `fsync` nor `fdatasync` (probed `False`/`False`).
- Full body (`video_common.py:239-249`): `tape_tmp.write_text(...)` → `tape_tmp.replace(tape_path)` → return. No flush, no fsync, no dir sync.
- Contrast `concepts.py:253-260` (temp + fsync + replace + `fsync_dir`) and `atomic.py:43-48` (same) — the in-tree statement of the correct contract.
- Call frequency: once per segment in both `video_ltxv.py:683` and `video_causvid.py:832` (plus `build_recovery_tape` JSON content — small, single-block write, so the fix is cheap).

## Repro

Static (deterministic): `grep -n "fsync\|flush" Voyage/voyage/workers/video_common.py` → matches only unrelated code (zero hits inside `write_tape_atomic`). Dynamic: `strace -e fsync,fdatasync,syncfs -f` around any ltxv/causvid `generate_blocks` shows no file or directory sync for the `recovery.pt` write.

## Fix candidates

1. Mirror the `_append_vector` pattern inside `write_tape_atomic`: write + `flush` + `os.fsync` before `os.replace`, plus `fsync_dir(tape_path.parent)` after (import from `voyage.atomic` — check for import cycles first; `video_common` currently imports only `loop`, stdlib, numpy).
2. Keep the change inside `video_common` so both callers inherit it (no per-worker edits).
3. Test: fault-inject `os.fsync` to raise after write and assert the previous tape still loads (atomicity preserved); assert an `fsync_dir` call occurs per write (durability wired).

## Refs

- `Voyage/voyage/workers/video_common.py:239-249`; `Voyage/voyage/concepts.py:251-260`; `Voyage/voyage/atomic.py:23-58`; DESIGN §27.1.
- Adjacent, not overlapping: 101 (ledger/metrics/concepts *append* durability — this file is the recovery-*tape* write it never mentions); 129 (longlive `recovery.pt` bare `torch.save` — the worse sibling: no atomicity at all; fix together, keep both).

## Progress log

- 2026-09-30 (Group E2): evaluated live first. Premise CONFIRMED as-read: `write_tape_atomic` (`voyage/workers/video_common.py:269-279`) was `write_text` + `replace` with zero fsync (`'fsync' in src == False`, matching the filed evidence). No concurrent agent had touched the function (their pruning hunks are docstring/constants/derive additions elsewhere in the file — disjoint). TDD: `tests/test_e2_tape_fsync_122.py` written first, 2/3 failed pre-fix (`DID NOT RAISE OSError` on the fault-injection test, zero fsync calls observed), 3/3 green post-fix.

## Resolution

- Verdict: FIXED in `voyage/workers/video_common.py` (shared helper, so both callers — causvid `:843`, ltxv — inherit it with no per-worker edits, per fix candidate 1).
- Change: `write_tape_atomic` now opens the `.tmp` sibling explicitly, writes + `flush` + `os.fsync(fileno())` before `os.replace`, then `fsync_dir(tape_path.parent)` after — mirroring the `concepts._append_vector` contract the issue cites. A failed fsync raises before the rename, preserving the previous tape. Top-level `import os` + `from voyage.atomic import fsync_dir` added (cycle-free: `voyage.atomic` is stdlib-only; the module docstring's import-contract line updated to name it).
- Files changed: `voyage/workers/video_common.py` only (plus new `tests/test_e2_tape_fsync_122.py`).
- Test evidence (in-container `voyage:latest`, CPU-only): new file 3 passed (round-trip, fsync-observed via monkeypatched `os.fsync` + `fsync_dir`, fault-injection atomicity); existing suites unaffected (`test_causvid_worker`, `test_generate_blocks_request`, `test_longlive*` green — full-scope re-verified at the end of the pass). Ruff + format + mypy strict clean on both files.
- DESIGN proposal (quoted text only, for the DESIGN owner — §27.1 resume path): "Recovery tapes (`recovery.pt`, JSON content) are rename-atomic and durable: the temp file is fsynced before the rename and the directory fsynced after (`video_common.write_tape_atomic`), so the latest tape — the one a restart needs — survives OS crash / power loss, not just process crashes."
- Residuals: none in this file. The `.mp4` anchor durability next to the tape is 172's (closed by run-file pruning — see that file); the derive-target `derive_tail_from_segment_video` tmp+replace in the same module carries no fsync — flagged to the pruning owner as follow-up, not widened here.
