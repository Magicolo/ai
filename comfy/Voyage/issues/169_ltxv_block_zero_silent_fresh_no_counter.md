# 169 — LTXV block-0 silent fresh restart: missing/unreadable tail renders 121 fresh frames with no counter, no log, no metric

- Severity: LOW (continuity observability — same failure class as 134, in the sibling backend, with even less telemetry: not even a write-only counter exists)
- Area: workers-internals tail — ltxv resume/conditioning path (below pass-1 coverage; 123 covered tape *content* trust, 064-era work the *short-anchor* detector — this file is the *missing/unreadable-anchor* fallback observability, which ltxv lacks entirely)
- Files (as-read 2026-09-30):
  - `voyage/workers/video_ltxv.py:596-605` (block-0 tail adoption: `None`/cut/missing → `conditioning_source = None`, no log)
  - `voyage/workers/video_ltxv.py:688-710` (result dict: `frames/fps/.../prompt_changed/...`, no fresh-blocks field)
  - `voyage/workers/video_ltxv.py:712-719` (`resume_from_tape`: adopts tail path, no GPU work, no validation beyond parse)
  - Contrast: `voyage/workers/video_causvid.py:784-795,857` (`fresh_rollouts` counted per rollout and returned)

## Technical description

LTXV block 0 adopts the resident tail with a silent three-way fallback (`video_ltxv.py:600-605`):

```python
if index == 0:
    tail_candidate = resident_tail
    if tail_candidate is None or scene_cuts[0] or not Path(tail_candidate).exists():
        conditioning_source = None
    else:
        conditioning_source = tail_candidate
```

When the tail is absent (fresh session), cut, or vanished between `resume_from_tape` and `generate_blocks`, the block renders a full 121-frame fresh clip (`novel = block`, `:626-627`) and the segment commits VALID video with `prefix_discarded_frames = 0`. Nothing records that a fresh start happened:

- No `fresh_blocks`/`resumed_fresh` counter in the result (`:688-710` — every count is frame accounting: `generated/conditioning/novel/committed/prefix_discarded`; none answers "was block 0 conditioned?").
- No stderr line (contrast causvid's three `print(..., file=sys.stderr)` fallbacks at `video_causvid.py:713/726/732`, and ltxv's own handoff-miss note at `video_ltxv.py:207`).
- `prompt_changed` (`:596,705`) tracks prompt text vs `_last_prompt`, not conditioning provenance — a same-prompt resume with a lost tail reports `prompt_changed: False` while rendering a fresh scene.
- The supervisor cannot distinguish either: `grep -rn "fresh_rollouts\|prompt_changed" voyage/supervisor.py voyage/backends.py voyage/cli.py voyage/media.py` → zero hits, and ltxv does not even emit the write-only fields causvid has.

Mid-segment has the same shape one layer down: when `_tail_clip_to_handoff_frames` returns `None`, the next block falls back to the chain mp4 (`:606-620`) — that path at least prints (`:207`). The block-0 path is the only fallback in either GPU worker with zero observability.

## Why this is an issue

Resume is the path that runs when things already went wrong (crash, OOM-evict, audio-swap rebuild). A fresh start after a failed resume is a legitimate last resort — but it must be *visible*: the operator reviewing a voyage with an unexpected scene cut currently finds `resumed: True` on the resume op (adopt time, before materialization), `conditioning_start_frame: 0`, and a VALID segment — with the actual fresh restart recorded nowhere. DESIGN §5.3 documents "fresh starts (no tail, missing tail, scene cut) commit all 121 frames" as *accounting* (frame counts stay truthful), but accounting without provenance defeats the qualify-track A/B the same section defers ("before trusting long resumed runs"): the A/B cannot even label which segments restarted.

## Live evidence

Host reads 2026-09-30 (no GPU needed):

```
$ rg -n "fresh" voyage/workers/video_ltxv.py
433:        continuity-critical tail), or None for a fresh text-to-video start.
572:        unless this is a fresh session, the tail file is gone, or the block
617:                    # Chain onto the previous block's freshly rendered tail video
839:    Probes are fresh text-to-video renders (scene cut, no resident tail), so
842:    Each probe commits a full 121-frame fresh segment (no prefix to drop).
```

- Zero `fresh_*` result fields (prose only). Contrast causvid: `fresh_rollouts = 0` (`:784`), `fresh_rollouts += 1 if was_fresh else 0` (`:795`), `"fresh_rollouts": fresh_rollouts` (`:857`).

```
$ rg -n "fresh_rollouts|prompt_changed" voyage/supervisor.py voyage/backends.py voyage/cli.py voyage/media.py
(no output, exit 1)
```

- Even causvid's write-only fields have no reader; ltxv's block-0 fallback has no field at all — strictly less observable than 134's site.

```
$ sed -n '600,605p' voyage/workers/video_ltxv.py
                if index == 0:
                    tail_candidate = resident_tail
                    if tail_candidate is None or scene_cuts[0] or not Path(tail_candidate).exists():
                        conditioning_source = None
```

- No log, no counter, no metric on the `None` branch (compare the `else` at `:613-620`, which is at least commented, and `:207`'s handoff print).

## Minimal repro

1. Commit an ltxv segment, then delete its `video_tail.mp4` (or point the tape at a missing path and `resume` — parse succeeds, `resume_from_tape` returns `{"resumed": True, ...}`).
2. `generate_blocks` with `scene_cuts=[False, ...]` → commits VALID 121-frame video with `conditioning_frames: 0`, `prefix_discarded_frames: 0`.
3. Observed: result carries no field distinguishing this from a conditioned 96-novel commit; supervisor logs a normal commit; only frame-count forensics (`conditioning_frames == 0` on a non-first segment) hints at the restart.
4. Note the forensics are ambiguous: a legitimate scene-cut block 0 also yields `conditioning_frames == 0` — provenance, not accounting, is the missing signal.

## Fix candidates

1. (Preferred) Mirror causvid: return `fresh_blocks` (count of blocks rendered with `conditioning_source is None`) in the `generate_blocks` result, and have the supervisor log a `video_resume_fallback` metric (warn-level, not fatal) when `fresh_blocks > 0` on a non-first segment — same metric 134's candidate-1 proposes, fed from both backends.
2. Log the fallback at the boundary: one stderr line on the block-0 `None` branch (mirrors `:207`'s handoff note and causvid's three prints) so worker logs alone can label restarts.
3. Distinguish scene-cut fresh (intended, operator-visible via `scene_cuts`) from missing-tail fresh (degraded): count them separately (`fresh_blocks_scene_cut` vs `fresh_blocks_missing_tail`) or record the tail path + reason on the missing branch.
4. Test: missing-tail / cut / fresh-session block-0 → assert the result carries the fresh reason and (once wired) the supervisor emits exactly one fallback metric; conditioned happy path asserts zero.

## References

- In-tree: `voyage/workers/video_ltxv.py:184-209,554-647,688-719`; `voyage/workers/video_causvid.py:701-749,784-868` (the counted precedent); `Voyage/DESIGN.md` §5.3 (fresh-start accounting + deferred qualify A/B) and §5.4 (resume as documented approximation).
- Neighbor issues — not a duplicate of 134 (134 is causvid's *unreadable-tail* fallback + the *write-only* `fresh_rollouts`/`prompt_changed` fields; this is ltxv's *block-0* fallback having *no field at all* — different backend, different branch, strictly less telemetry; fix the supervisor metric once, add the per-backend counters separately): 123 (tape content trust — pre-read checks; this file is *readability* fallback, which returns before any hash comparison), 064-era tail-length loudness (post-read length checks).
- External: same `record_prefetch_cancelled`-class rationale as 136/168 — degraded-but-accepted work must be counted, not just rendered.

 ## Investigation log

 - 2026-09-30: filed by the 168-177 tails sweep; re-verified live via Read/Grep (concurrent uncommitted edits noted in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py`, `voyage/config.py`, `voyage/persistence.py`, `voyage/rpc.py`, `voyage/supervisor.py` — citations are as-read values above).

## Progress log

- 2026-09-30 (Group E2): evaluated live first against the CURRENT tree (post-pruning `34c0a29`). Premise CONFIRMED as-read: `voyage/workers/video_ltxv.py:734-740` still takes the silent `conditioning_source = None` branch on block 0 (missing tail / scene cut / vanished path) with no counter, no log, no metric; the result dict (`:843+`) still carries frame accounting only (`generated/conditioning/novel/committed/prefix_discarded`, `prompt_changed`) — no `fresh_blocks`-class field; `resume_from_tape` still reports `resumed: True` at adopt time. Causvid's counted precedent (`fresh_rollouts` at `video_causvid.py:~796-807,868`) is intact. `video_ltxv.py` is explicitly out of this group's scope (concurrent-hot) and `supervisor.py` (the metric consumer) likewise — logged as residual with exact lines, no code touched. No supervisor-side workaround exists in this group's files: `bench.py`/`doctor.py` cannot observe a field the worker never emits.

## Resolution

- Verdict: RESIDUAL — fully verified, not ownable from this group's files.
- Files changed: none (this issue file only).
- Test evidence: live reads 2026-09-30 (cites above: `sed -n '734,740p'` branch, result-dict keys, causvid contrast); no test added — the pins (missing-tail/cut/fresh block-0 → fresh reason in result + one supervisor fallback metric; conditioned path → zero) belong to the ltxv/supervisor owners.
- DESIGN proposal (quoted text only, for the DESIGN owner — §5.3/§5.4): "Mirror causvid: `generate_blocks` returns `fresh_blocks` (blocks rendered with `conditioning_source is None`, split by scene-cut vs missing-tail reason) and the supervisor logs one warn-level `video_resume_fallback` metric when `fresh_blocks > 0` on a non-first segment — shared with 134's metric, fed from both backends — plus one stderr line on the block-0 fallback branch."
- Residuals (for the ltxv + supervisor owners, precise): (1) `voyage/workers/video_ltxv.py:734-740`: count + log the `None` branch; result dict `:843+`: add the fresh-reason field; (2) supervisor: `video_resume_fallback` metric wiring (shared with 134 candidate 1); (3) scene-cut fresh vs missing-tail fresh distinguished (candidate 3).
