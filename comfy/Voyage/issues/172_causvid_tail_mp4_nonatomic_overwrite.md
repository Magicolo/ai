# 172 — CausVid overwrites the crash-recovery anchor `video_tail.mp4` in place; LTXV promotes a fully-written chain file atomically

- Severity: LOW (crash recovery — torn-anchor window on SIGKILL/power loss; degrades to a silent fresh scene via 134, not corruption)
- Area: workers-internals tail — tail-anchor write path (below pass-1 coverage; 122 covered JSON *tape* atomicity, 129 the longlive *pt* non-atomicity — neither names the `.mp4` anchor writes)
- Files (as-read 2026-09-30):
  - `voyage/workers/video_causvid.py:808-811` (`_save_mp4(tail_window, tail_path, NATIVE_FPS)` direct onto the live anchor)
  - `voyage/workers/video_causvid.py:440-446` (`_save_mp4` → `video_common.save_mp4` → `imageio.mimsave` streaming write, no tmp/rename)
  - `voyage/workers/video_common.py:261-267` (`save_mp4`: fps guard only, no atomicity)
  - Contrast (the fixed pattern, same tree): `voyage/workers/video_ltxv.py:640-667` (per-block `_chain{index:02d}.mp4` fully written, then `chain_tails[-1].replace(tail_path)` + stale unlink; OOM path unlinks orphans at `:649-657`)

## Technical description

Every causvid segment finishes by encoding the tail window straight onto the live recovery anchor (`video_causvid.py:808-811`):

```python
tail_window = select_tail_window(video_frames, self._overlap_frames)
tail_path = output_path.parent / TAIL_FILENAME
_save_mp4(tail_window, tail_path, NATIVE_FPS)
tail_checksum = sha256_file(tail_path)
```

`imageio.mimsave` streams frames to that path for the duration of the encode. A crash (SIGKILL, OOM-kill, power loss) mid-encode leaves a **torn `video_tail.mp4` in place** — the previous good anchor is already clobbered, and the tape written two lines later (`:813-832`) either never lands (crash before it: anchor torn + tape stale-but-consistent with the *previous* anchor — recoverable) or lands referencing the torn file's checksum (`sha256_file` at `:812` hashes whatever bytes survived — consistent hash of corrupt content, so 123-style re-verification would *pass* on garbage).

The next resume then walks 134's silent path: `_materialize_resume_start` (`:701-738`) hits missing/unreadable/re-encode-mismatch → `print` to stderr → `None` → fresh scene, counted nowhere. The torn anchor thus converts a crash into an unlabeled scene restart one segment later.

LTXV — the sibling backend with the identical anchor contract — does not have this window: each block's tail is fully written to `_chain{index:02d}.mp4` (`:640-642`), and only the complete file is `replace()`d onto `video_tail.mp4` (`:662-667`; `replace` is an atomic rename on POSIX). A crash leaves either the previous anchor or an orphan chain file (which the OOM path explicitly unlinks at `:649-657`); never a torn live anchor. The segment `video.mp4` writes are non-atomic in both backends, but those are pre-`DONE` artifacts — a torn segment video fails commit validation loudly and is re-rendered, while the torn *anchor* fails one segment later and silently.

## Why this is an issue

- The tail mp4 is the resume path's anchor (DESIGN §5.4): the run's most crash-exposed video write has the weakest write contract of the video path — weaker than the JSON tape beside it (122: rename-atomic) and weaker than the sibling backend's identical file.
- The failure is both delayed and silent: the crash segment itself may commit fine (video.mp4 intact), and the damage surfaces next segment as 134's print-only fallback. Crash forensics (did the scene restart because of the kill, or was it always going to?) become impossible.
- The fix is a three-line mirror of the in-tree LTXV pattern (write tmp, fsync, rename) — no new dependency, no protocol change, slim-testable without a GPU.

## Live evidence

Host reads 2026-09-30 (no GPU needed; write-path shape is static):

```
$ sed -n '808,812p' voyage/workers/video_causvid.py
    808:        tail_window = select_tail_window(video_frames, self._overlap_frames)
    809:        tail_path = output_path.parent / TAIL_FILENAME
    810:        _save_mp4(tail_window, tail_path, NATIVE_FPS)
    811:        tail_checksum = sha256_file(tail_path)

$ sed -n '640,667p' voyage/workers/video_ltxv.py
    640:                pending_tail = output_path.parent / f"{output_path.stem}_chain{index:02d}.mp4"
    641:                _save_mp4(tail_clip, pending_tail, fps)
    642:                chain_tails.append(pending_tail)
    ...
    661:        tail_path = output_path.parent / TAIL_FILENAME
    662:        if len(chain_tails) == 1:
    663:            chain_tails[0].replace(tail_path)
    664:        else:
    665:            chain_tails[-1].replace(tail_path)
```

- Causvid: streaming encode directly onto the live anchor, then hashes the result — a torn file gets a valid checksum of invalid content.
- LTXV: encode-then-`replace` — the live anchor is only ever swapped for a complete file.
- `rg -n "replace|\.tmp|fsync" voyage/workers/video_causvid.py` around the tail write → no tmp/rename/fsync (the only `replace` hits in the file are unrelated path/manifest operations).

## Minimal repro

Static (deterministic): read `:808-811` — no temp sibling between the encode and the live anchor. Dynamic: `kill -9` a causvid worker during the tail `mimsave` (small window — 9 frames — but the segment-end encode contends with the VAE/GPU teardown, and power loss has no window minimum) → restart → `resume` adopts the tape → `_materialize_resume_start` prints `unreadable ... starting fresh` → segment commits VALID video on a new scene with `fresh_rollouts >= 1` and no supervisor-visible fallback event (134).

## Fix candidates

1. (Preferred) Mirror LTXV: `_save_mp4(tail_window, tail_path.with_suffix(".tmp"), fps)` → flush/fsync → `os.replace(tmp, tail_path)` → `fsync_dir(parent)` (pairs with 122's tape fix so anchor + tape share one durability contract; keep the tmp name unique per segment so a retry never stomps).
2. Keep the change inside `_save_mp4`'s causvid call site (or a `save_tail_atomic` helper in `video_common` next to `save_mp4`) so the segment `video.mp4` path is untouched (pre-DONE artifacts fail loudly already — no need to gild them).
3. Test (slim, no GPU): fault-inject `mimsave` to raise mid-write and assert the previous anchor still loads byte-identical (atomicity); assert an `fsync_dir` call per anchor write (durability); torn-anchor resume asserts the fallback metric once 134/169's metric lands.
4. Cross-check longlive: it carries no tail mp4 (tensor tape only — 129's site), so no third call site exists.

## References

- In-tree: `voyage/workers/video_causvid.py:701-738` (the silent consumer of the torn file), `:808-832`; `voyage/workers/video_ltxv.py:633-683`; `voyage/workers/video_common.py:239-267` (`write_tape_atomic` + `save_mp4` side by side — the atomicity gap is visible in one screen); DESIGN §5.4 (tail anchor + documented-approximation resume).
- Neighbor issues — not a duplicate of 122 (JSON *tape* durability — this file is the `.mp4` *anchor* the tape checksums), 129 (longlive `.pt` non-atomicity — different backend, different file), 134 (the silent *consumer* of the torn anchor — this file is the *producer* that tears it; 134 stays valid if anchors never tear, and vice versa), 101 (ledger/metrics append durability — never mentions anchors).
- External: same atomic-save rationale 129 cites (write-to-temp then rename so a crash never leaves a half-written file); `os.replace` atomicity: https://docs.python.org/3/library/os.html#os.replace

## Investigation log

- 2026-09-30: filed by the 168-177 tails sweep; re-verified live via Read/Grep (concurrent uncommitted edits noted in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py`, `voyage/config.py`, `voyage/persistence.py`, `voyage/rpc.py`, `voyage/supervisor.py` — citations are as-read values above).
