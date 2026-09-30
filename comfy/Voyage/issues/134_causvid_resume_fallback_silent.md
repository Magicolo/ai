# 134 — Causvid resume-anchor fallback is print-only: missing/unreadable tail silently restarts the scene, `fresh_rollouts` ignored supervisor-side

- **Severity:** LOW (continuity observability — narrow reachability, but the failure mode is a silent scene restart on the path that runs when things already went wrong)
- **File:line:**
  - `Voyage/voyage/workers/video_causvid.py:701-738` (`_materialize_resume_start`: three `print(..., file=sys.stderr)` fallbacks → `None`; `self._pending_tail_path = None` cleared at `:711` before any validation)
  - `Voyage/voyage/workers/video_causvid.py:740-749` (`_rollout_start`: `None` → `(None, True)` fresh)
  - `Voyage/voyage/workers/video_causvid.py:784-802,857` (`fresh_rollouts` counted per rollout, returned in the result — the only record the fallback ever fired)
  - Supervisor/backends ignore it: `grep -rn "fresh_rollouts\|prompt_changed" voyage/supervisor.py voyage/backends.py voyage/cli.py voyage/media.py` → zero hits (workers return both; only `tests/test_causvid_worker.py:467-468` asserts them)
- **Area:** workers-internals tail — causvid resume path (below pass-1 coverage; 123 covered tape *checksum/shape* trust, 064-era work covered short-*anchor* loudness — this file is the *unreadable-anchor* fallback silence)

## Description

When the adopted tail cannot be used, causvid restarts the scene without telling anyone above the worker:

1. Tail path missing (`:712-717`), `mimread`/window/re-encode raising (`:725-730`), or re-encode shape mismatch (`:731-737`) → `print` to stderr → return `None`.
2. The pending anchor is cleared at `:711` *before* validation, so a transient read failure (short disk stall, momentary mount hiccup) permanently discards the anchor for the rest of the segment — every subsequent rollout in the same `generate_blocks` call also starts fresh, with no retry of the read.
3. The caller (`_rollout_start`, `:746-748`) maps `None` to a fresh rollout and the segment commits with `fresh_rollouts >= 1` — but the supervisor never reads `fresh_rollouts` (grep above) and the result carries no `start_latents_from`-mismatch warning (the `START_FROM_RESUME` label at `:896` was already attached at `resume_from_tape` time, before materialization failed). The segment VALIDates (fresh frames are valid frames), the tape for the *next* segment is built from the fresh chain, and the voyage continues on a new scene with the continuity break recorded nowhere except worker stderr.

Contrast the sibling: longlive's resume replays the tail forward eagerly inside `resume_from_tape` (failures raise through the restart budget); causvid's lazy materialization defers the failure into `generate_blocks`, where the `print`-and-fresh pattern converts it from a retried error into an accepted degradation.

## Rationale

Resume is the path that runs when things already went wrong (crash, OOM-evict, audio-swap rebuild). A fresh start after a failed resume is a legitimate last resort — but it must be *visible*: the operator reviewing a voyage with an unexpected scene cut currently finds `resumed: True` + `start_latents_from: committed_tail_reencode` on the resume op and a VALID segment, with the actual fresh restart buried in a worker log line. DESIGN §5.4 promises the resume path is "a documented approximation" with qualify-track A/B before trusting long resumed runs — silent fallback defeats exactly that trust accounting.

## Evidence (verified live 2026-09-30, host reads)

- `sed -n '701,738p' voyage/workers/video_causvid.py` — three `print` fallbacks, one shared `return None`; `:711` clears `_pending_tail_path` before the `:712` existence check.
- `grep -rn "fresh_rollouts\|prompt_changed" Voyage/voyage/supervisor.py Voyage/voyage/backends.py Voyage/voyage/cli.py Voyage/voyage/media.py` → empty (both fields are write-only telemetry; `prompt_changed` has the same fate — DESIGN `:525-526` advertises it "for the continuation-quality rule", but no rule reads it).
- Overlap check: 123 is tape *content* trust (sha/shape re-verification — this file is *readability* fallback, a different branch: 123's checksum never runs because materialization fails before any hash comparison); 064-era tail-length loudness is the *short-anchor* detector (`validate_tail_length` analogues — this file is the *missing/unreadable* anchor, which returns before length is ever measured).

## Repro

1. Commit a causvid segment, then delete (or single-byte-corrupt past decodability) its `video_tail.mp4`.
2. `rebuild`/`resume` from that segment's tape → `{"resumed": True, ...}` (parse checks path *existence* at tape-adopt time; the file vanishes between adopt and materialize, or `mimread` raises).
3. `generate_blocks` commits VALID video with `fresh_rollouts >= 1` — supervisor logs no resume-fallback event, metrics show a normal commit, and only worker stderr carries the one-line `starting fresh` notice.

## Fix candidates

1. Fail loud at the boundary: return a structured `resume_fallback: {reason, tail_path}` field in the `generate_blocks` result (alongside `fresh_rollouts`) and have the supervisor log a `video_resume_fallback` metric (warn-level, not fatal — generation stays crash-free, but the break is counted).
2. Don't clear the anchor until it materializes: move `self._pending_tail_path = None` to after a successful re-encode (or retry the read once), so a transient failure doesn't poison the whole segment's remaining rollouts.
3. Test: missing-tail / corrupt-tail / shape-mismatch resumes → assert the result carries the fallback reason and the supervisor emits exactly one fallback metric; happy path asserts zero.

## Refs

 - `Voyage/voyage/workers/video_causvid.py:209-225,271-370,701-749,784-868,870-897`; `Voyage/DESIGN.md` §5.4 (resume as documented approximation).
 - Adjacent, not overlapping: 123 (tape checksum/shape re-verification — pre-read content checks); 064-era short-anchor loudness (length checks — post-read); 006 (supervisor trusts worker *frame counts* — this file is the worker's *own* silent downgrade before counts exist).

## Progress log

- 2026-09-30 (video-workers track): re-verified premise live first — CONFIRMED as-filed: `voyage/workers/video_causvid.py:731-768` cleared `self._pending_tail_path = None` before validation with three print-only fallbacks, and `grep -rn "fresh_rollouts\|prompt_changed" voyage/supervisor.py voyage/backends.py voyage/cli.py voyage/media.py` still returns zero hits (both fields write-only). TDD: new `tests/test_134_resume_fallback.py` (4 tests) failed pre-fix in-container (`voyage:latest`, CPU-only: `resume_fallback` key missing), then green post-fix. Existing `tests/test_causvid_worker.py:566-579` (`test_missing_resume_anchor_falls_back_to_fresh`, asserts the anchor ends `None`) still passes — the fix preserves that end-state and only moves the clear to after the outcome is recorded.

## Resolution

- Verdict: FIXED worker-side; supervisor-read half RESIDUAL (below).
- Files changed: `voyage/workers/video_causvid.py` (`_materialize_resume_start` at `:731` now returns `(start_latents, fallback)` with `fallback` None on success else `{"reason", "tail_path"}` (`missing` / `unreadable` / `shape_mismatch`); the pending anchor clears only after the outcome is recorded, never before validation; `_rollout_start` at `:778` threads the triple through; `generate_blocks` at `:824-835` keeps the first fallback and returns it as `"resume_fallback"` at `:898` alongside the existing `fresh_rollouts`; stderr prints kept verbatim as the worker log) + new `tests/test_134_resume_fallback.py` (missing / unreadable / shape-mismatch → reason + tail + stderr line + fresh; no-pending happy path → `resume_fallback is None`).
- Test evidence (in-container `voyage:latest`, CPU-only): new file 4 passed; related suites 163 passed total (incl. `test_causvid_worker`, `test_tail_derive`, `test_stage_a_telemetry`). Gates on touched files green: `ruff check` + `ruff format --check` + `mypy` strict.
- DESIGN proposal (quoted text only, for the DESIGN owner — §5.4): "Causvid `generate_blocks` returns `resume_fallback` (`{reason: missing|unreadable|shape_mismatch, tail_path}`, None on the conditioned path) alongside `fresh_rollouts`, and the supervisor logs one warn-level `video_resume_fallback` metric when it is set — shared with 169's metric, fed from both backends."
- Residuals (supervisor owner, precise): wire the read half — `voyage/supervisor.py` (plus `voyage/backends.py` / `voyage/cli.py` / `voyage/media.py`, all zero hits today) never reads `fresh_rollouts`, `prompt_changed`, or the new `resume_fallback`; the `START_FROM_RESUME` label is still attached at `resume_from_tape` time (`video_causvid.py:941-945`) before materialization can fail, so a fallback segment still looks resumed-but-normal above the worker. Suggested: log `video_resume_fallback` from `result["resume_fallback"]` (warn-level, not fatal) exactly as this issue's fix candidate 1 proposes.
