# 013 — DONE written before `state.json` advance leaves committed-but-unaccounted segment; retry overwrites via `mkdir exist_ok`

- Severity: HIGH
- Group: correctness/persistence — Rank: 1/5
- File:line: `voyage/supervisor.py:1786-1795` (DONE → state), `voyage/supervisor.py:1867` (segment `mkdir exist_ok`), `voyage/cli.py:876-885` (`validate_run` committed-vs-state check)

## Technical description

Live ordering in `_commit_segment` (re-verified 2026-09-30):

```python
# voyage/supervisor.py:1770-1778
atomic_write_json(segment / "sha256.json", {...})
# Single-step DONE (issue 058): one atomic write straight to DONE.
atomic_write_bytes(segment / paths.DONE_MARKER, b"")   # <-- durable commit mark

# voyage/supervisor.py:1780-1791 — 6. Supervisor-owned state advance
fresh = read_state(self._run_dir)
fresh.next_segment_number = number + 1
fresh.committed_segments += 1
fresh.timeline_frames += frames
...
write_state(self._run_dir, fresh)
```

DONE is durable **before** `state.json` advances. A crash (SIGKILL, OOM-killer, power loss) in that window leaves `segments/NNNNNN/DONE` on disk with `state.committed_segments` / `next_segment_number` / `timeline_frames` not yet incremented — a committed-but-unaccounted segment.

Retry path (`_commit_one_segment_locked`, `voyage/supervisor.py:1856-1859`):

```python
number = state.next_segment_number
segment_id = paths.format_segment_id(number)
segment = paths.segment_dir(self._run_dir, segment_id)
segment.mkdir(parents=True, exist_ok=True)   # <-- silently reuses the DONE dir
```

The retry recomputes the *same* `segment_id` (state never advanced), `mkdir exist_ok` succeeds, and the commit re-renders video/audio **over** the existing `video.mp4` / `audio.wav` / `metrics.json` without ever detecting the previous DONE. Media is silently overwritten; the first render is lost; `sha256.json` is recomputed over the second render so no checksum ever fires.

`validate_run` (`voyage/cli.py:853-856`) does check `len(committed) != state.committed_segments`, so the window is *detectable* after the fact — but the retry has already destroyed the evidence by overwriting.

## Why it matters

- Violates the module docstring contract (`voyage/supervisor.py:1-12`: "No state file may claim the segment is committed until artifacts are valid and durable" — here the inverse happens: artifacts claim committed while state does not).
- Silent media overwrite breaks provenance: `metrics.json` / `sha256.json` describe render #2 while an operator may have already previewed render #1.
- The window is small but the consequence is data loss, and kill-tests (SIGKILL → resume) are an explicit supported path (DESIGN §27.1, crash matrix).

## Live evidence

```
$ grep -n "DONE_MARKER\|fresh = read_state\|segment.mkdir" comfy/Voyage/voyage/supervisor.py
533:            if (segment / paths.DONE_MARKER).exists():
1778:            atomic_write_bytes(segment / paths.DONE_MARKER, b"")
1781:        fresh = read_state(self._run_dir)
1859:        segment.mkdir(parents=True, exist_ok=True)
```

Source excerpt around the window:

```
atomic_write_json(segment / "sha256.json", {...})
atomic_write_bytes(segment / paths.DONE_MARKER, b"")
# 6. Supervisor-owned state advance (single writer).
fresh = read_state(self._run_dir)
fresh.next_segment_number = number + 1
...
write_state(self._run_dir, fresh)
```

Host cannot import `voyage.supervisor` (pydantic container-only), so evidence is source-inspection per task allowance; ordering is unambiguous in the file.

## Repro steps

1. Init a fake-backend run; instrument (or breakpoint) between `atomic_write_bytes(... DONE ...)` and `write_state(...)` — or SIGKILL the supervisor in that window under load.
2. Observe `segments/000000/DONE` exists while `state.json` still has `committed_segments: 0`, `next_segment_number: 0`.
3. `voyage run` again → same `000000` re-rendered, first `video.mp4` overwritten, no error.
4. `voyage validate` before step 3 reports `state claims 0 segments, found 1 DONE`; after step 3 it reports VALID — the overwrite erased the inconsistency.

## Fix candidates

1. (Preferred) Advance state **before** DONE inside the same lock hold, or make the two atomic: write `state.json` first (it already is atomic via `atomic_write_json`), then DONE; on retry, if `segment/DONE` exists *and* state already advanced past it, skip (idempotent replay) instead of re-rendering. Requires a pre-check after `mkdir`: if `(segment / DONE).exists()` → either resume-advance state or fail loud, never blindly overwrite.
2. Minimal guard: after `mkdir`, `if (segment / DONE).exists(): raise FatalWorkerError(...)` — turns silent overwrite into a loud operator decision (manual `rm` or resume). Loses auto-heal but stops data loss.
3. Full transaction: write a `COMMITTING` marker with the render checksums before DONE; recovery replays the state advance from the marker without re-rendering.
4. Regression test: pre-create `segments/000000/DONE` + valid media with stale `state.json`; assert retry does NOT overwrite (mtime/content unchanged) and either heals state or raises.

## References

- `voyage/supervisor.py:1774-1791` (DONE-then-state window), `:1856-1859` (`mkdir exist_ok` reuse), `:1831-1839` (single-writer lock — held across both, but crash still splits them).
- `voyage/cli.py:837-856` (`validate_run` detects the divergence only if retry has not yet overwritten).
- `voyage/atomic.py:37-58` (both writes are individually atomic; the *pair* is not).
- SQLite-style "write-ahead log" rationale — the state advance needs to be replayable from durable markers: https://www.sqlite.org/wal.html
