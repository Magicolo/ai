# 129 — LongLive `recovery.pt` uses bare `torch.save`: non-atomic direct overwrite, no fsync (the 122 gap's worse sibling)

- **Severity:** MEDIUM (crash recovery — a crash mid-save destroys the run's only resume anchor; worse than 122, which is at least rename-atomic)
- **File:line:** `Voyage/voyage/workers/video_longlive.py:919-920` (`recovery_path = output_path.with_name("recovery.pt")` → `torch.save(tape, str(recovery_path))` — direct overwrite, no temp file, no rename, no flush/fsync/fsync_dir)
- **Area:** workers-internals tail — longlive recovery-tape write path (below pass-1 coverage; 048 covered the decode fan-out, 123 covered tape *read* trust — neither touches this write site)

## Description

Every longlive segment commits its resume anchor with a direct `torch.save` onto the live `recovery.pt` path. There is no temp sibling, no `os.replace`, no `flush`, no `os.fsync`, no `fsync_dir`:

```python
# video_longlive.py:919-920
recovery_path = output_path.with_name("recovery.pt")
torch.save(tape, str(recovery_path))
```

A crash (SIGKILL, OOM-kill, power loss) anywhere inside the zip-pickle write leaves a **torn `recovery.pt` in place** — the previous good tape is already clobbered, there is no sibling to fall back to, and the reader (`_load_recovery_tape`, `:1173-1192`) fails closed on torn bytes, so the *next* restart cannot resume at all and the run re-renders from seed. The ltxv/causvid JSON tapes at least survive a process crash mid-write (temp + rename — 122's whole point is that only the *durability* half is missing); the longlive pt tape survives neither process crashes nor OS crashes.

The file knows the correct shape elsewhere: `video_common.write_tape_atomic` (`video_common.py:239-249`) does temp + rename for the JSON tapes, `concepts._append_vector` does temp + fsync + replace + `fsync_dir`, and `atomic.py:23-58` documents why the directory half matters. The longlive writer adopted none of it — plausibly because `torch.save` needs a path (not bytes), so the author wrote the one-liner instead of save-to-tmp + sync + rename.

## Rationale

The recovery tape is the resume path (DESIGN §27.1): `handle_rebuild`/`handle_resume` read the latest tape to adopt the tail anchor, and the evict-around-audio flow (`_with_audio_gpu` → evict → render take → rebuild from tape) exercises it on *every* segment with audio. The run's most crash-exposed write (last DiT-side artifact before the ~10 GiB VAE decode transient) has the weakest write contract of any persisted artifact in the video path. Standard practice is explicit: "implement atomic save with write-to-temp then rename so a crash never leaves a half-written file" (AI Engineering checkpoint lesson; PyTorch's own `torch.save` docs promise only a zip write to the given path, no atomicity).

## Evidence (verified live 2026-09-30, host reads)

- `grep -n "torch.save" voyage/workers/video_longlive.py` → exactly one hit: line 920 (the tape write).
- `grep -n "\.tmp\|fsync\|atomic\|replace" voyage/workers/video_longlive.py` → zero write-path hits (only an unrelated UMT5 comment at :10 and preamble comments at :770/:794).
- 122's scope line names only `video_common.py:239-249` with callers `video_ltxv.py:682-683` + `video_causvid.py:831-832` — longlive's `torch.save` site is never mentioned (`grep -n "longlive" issues/122_*` → zero content hits outside the title-adjacency note).
- Reader consequence: `_load_recovery_tape` (`:1183-1191`) rejects missing/wrong-suffix/non-dict but a torn zip raises the torch unpickling error *before* the profile check (`:1201-1203`/`1241-1243`) is ever reached — the failure surfaces as an opaque load error, not "re-render from seed".

## Repro

Static (deterministic): read `:907-920` — the tape dict is built, then saved directly onto the live path with no temp sibling in between. Dynamic: `kill -9` a longlive worker during the `torch.save` window (small but real — ~7 MB per DESIGN §22 log) → restart → `handle_resume` fails on the torn `recovery.pt` instead of falling back to the previous segment's tape (which no longer exists — it was overwritten in place).

## Fix candidates

1. Save-to-tmp + sync + rename inside `generate_blocks`: `torch.save(tape, tmp_path)` → file fsync → `os.replace(tmp, recovery_path)` → `fsync_dir(parent)` (mirror `write_tape_atomic` + the 122 fix, so both land together; keep the tmp name unique per attempt so two writers never stomp).
2. Keep the change at the single write site (`:919-920`) so `handle_resume`/`handle_rebuild` readers inherit it with no edits.
3. Test: fault-inject a crash between save and rename and assert the previous tape still loads (atomicity); assert an `fsync_dir` call per write (durability); assert a torn tape fails with the clean "re-render from seed" error, not an opaque traceback.

## Refs

 - `Voyage/voyage/workers/video_longlive.py:907-920,1173-1192,1195-1245`; `Voyage/voyage/workers/video_common.py:239-249`; `Voyage/voyage/atomic.py:23-58`; DESIGN §27.1.
 - Adjacent, not overlapping: 122 (JSON tape *durability* — this file is the pt tape's missing *atomicity + durability*); 123 (tape *read* verification — this file is the *write* that destroys the fallback); 101 (ledger/metrics append durability — never mentions `recovery.pt`); 013 (DONE-before-state — segment-level, not tape-level).

## Progress log

- 2026-09-30 (video-workers track): re-verified premise live first — CONFIRMED as-filed: `voyage/workers/video_longlive.py:963-964` was the single bare `torch.save(tape, str(recovery_path))` onto the live path (one `torch.save` hit repo-wide in the file, zero tmp/fsync/atomic/replace hits on the write path). TDD: new `tests/test_129_tape_atomic.py` (3 tests) failed pre-fix in-container (`voyage:latest`, CPU-only: `save_recovery_tape_atomic` missing), then green post-fix. Existing longlive suites initially caught a contract gap in the first helper shape (recording-double `save` fakes in `test_longlive_stages.py:76-77`, `test_longlive_offload_fusion.py:68-69`, `test_longlive_init_validation.py:106-107` never materialize a file — reopen-for-fsync raised `FileNotFoundError`); reworked the helper to save through one open `wb` handle (`torch.save` accepts file objects) + flush + fsync on that handle, which both the real writer and the recording doubles satisfy without touching their files. Full related set green post-fix (163 passed: 11 new + 152 existing incl. `test_longlive_stages`, `test_longlive_offload_fusion`, `test_longlive_init_validation`, `test_stage_a_telemetry`). Gates on touched files green in-container: `ruff check` + `ruff format --check` + `mypy` strict on `voyage/workers/video_longlive.py` + the new test (one `E402 noqa` added for the post-`sys.path` `voyage.atomic` import; `fsync_dir` imported directly because `video_common` does not explicitly re-export it for mypy).

## Resolution

- Verdict: FIXED (worker-side, issue scope only).
- Files changed: `voyage/workers/video_longlive.py` (new `save_recovery_tape_atomic` at `:151-170`: tmp sibling + single-handle save + flush + `os.fsync` + `os.replace` + `fsync_dir`; write site at `:987` routes through it) + new `tests/test_129_tape_atomic.py` (atomic write + fsync_dir call + crash-preserves-previous + write-site regression pin). No other files touched (122's JSON-tape gap kept separate as filed — `video_common.write_tape_atomic` untouched).
- Test evidence (in-container `voyage:latest`, CPU-only): new file 3 passed; related suites 163 passed total (see Progress log). Torn-tape reader behavior unchanged by design (fail-closed `_load_recovery_tape` inherits the atomic write with no edits).
- DESIGN proposal (quoted text only, for the DESIGN owner — §27.1): "LongLive `recovery.pt` commits via tmp-file + flush + fsync + atomic rename + directory fsync (`save_recovery_tape_atomic`), so a crash mid-save never clobbers the previous good tape — same contract as the JSON tapes (`write_tape_atomic`, issue 122)."
- Residuals: none in worker scope. Supervisor/reader halves need no handoff (readers inherit the fix with no edits, per fix candidate 2).
- Web rationale: PyTorch `torch.save` docs (zip write to the given path, no atomicity promise); AI Engineering checkpoint lesson ("atomic save with write-to-temp then rename so a crash never leaves a half-written file").
