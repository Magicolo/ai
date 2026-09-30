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

## Progress log

- 2026-09-30: premise re-verified against live `voyage/supervisor.py` as read (DONE write `:1803`, state advance `:1806-1829`, `mkdir exist_ok` with no DONE guard at `:1897`). Premise held — no batch-1 change touched this region (b40b052 only added the 099 CAS inside the state write-back, which the fix preserves).
- 2026-09-30: failing tests first in `tests/test_supervisor_hardening.py` (new file): `test_done_orphan_adopted_without_rerender` and `test_corrupt_orphan_refuses_without_overwrite` both failed pre-fix (retry silently re-rendered over DONE; no MediaError). Watched fail in-container via `scripts/test.sh`.
- 2026-09-30: fix implemented (see Resolution). Related-suite run surfaced one conflict: `test_crash_matrix.py::test_done_without_state_advance_heals_on_retry` plants a bare DONE dir (no media/metadata) and expects a clean re-commit — that setup cannot arise from the real crash window (DONE is written last, so a genuine orphan always carries full media + metadata), but the file is outside owned scope and cannot be edited. Added the artifact-free reclaim path so it heals as before; verified green post-fix.
- 2026-09-30: verification — all 15 hardening tests pass (13 fail-first incl. reclaim + 2 characterization guards); related suites green (`test_failure_policy`, `test_crash_matrix`, `test_commit_hardening`, `test_state_integrity` except one foreign failure below). `test_state_integrity.py::test_validate_detects_missing_recovery_tape` fails identically with and without this change (verified via stash baseline) — caused by the concurrent agent's uncommitted `paths.py` 015 work (`resolve_stored_path` raises where validate expects a reportable error), not this scope. `ruff check` + `ruff format --check` + `mypy` (strict) green on touched files.

## Resolution

Adopt-or-refuse replaces silent overwrite, supervisor-side, with no `--force` overwrite path:

- `voyage/supervisor.py:2126-2144` (`_commit_one_segment_locked`, inside the run lock): after `mkdir`, a present DONE diverts. Artifact-free dirs (exactly `{DONE}`) log `segment_reclaimed` (`:2138`) and fall through to a fresh render — nothing exists to overwrite. Anything else routes to adoption.
- `voyage/supervisor.py:1867-1952` (`_adopt_unaccounted_segment`, new): verifies `sha256.json` over the existing `video.mp4`/`audio.wav` (byte-identical media required), reads the frame count from the orphan's `metrics.json` and concepts/phase from its `world_state.json` (pydantic-validated via the already-imported `SegmentWorldState`), then advances `next_segment_number` / `committed_segments` / `timeline_frames` / `decision_index` and emits `segment_adopted` (`:1946`). Any unreadable/mismatched artifact raises `MediaError` ("refusing to re-render over it") — media untouched, state untouched.
- `voyage/supervisor.py:1848-1866` (`_write_state_preserving_control_plane`, new): the batch-1 issue-099 CAS extracted verbatim; `_commit_segment` now calls it and adoption shares it, so stop/pause is honored identically on both paths. `audio_buffer_seconds` keeps its pre-crash value on adoption — the takes ledger stays the planning truth (see `_ensure_audio_coverage` docstring), so worst case is an extra take render, never silence. Gauge sampling stays on the render path only (adoption is metadata-only).

Behavior: valid orphans heal without re-render (media bytes and mtime verified untouched); corrupt orphans fail loud; bare-DONE dirs reclaim. Half left open: none in the 013 window itself — the residual TOCTOU belongs to 004/016 (fd-passing design), documented there.
