# 104 — Audio slice loop has no iteration bound and `AudioTake.from_dict` validates nothing: degenerate takes spawn thousands of ffmpeg slices that trip the 0.6 s A/V gate

**Severity:** MEDIUM

**File:line (verified live 2026-09-30):**
- `voyage/supervisor.py:1251-1280` (slice walk: `while cursor < end - 1e-6:` at line 1259, no iteration cap; `piece = min(serving.covers_until(), end) - cursor` at 1263; `slice_take(...)` per piece at 1264-1274; `cursor += piece` at 1279 — sweep said `:1243-1272/:1251/:1255/:1257-1267/:1271`, drifted +8)
- `voyage/audio/planner.py:74-86` (`AudioTake.from_dict`: bare `float(...)`/`int(...)` casts, no finite/range/positivity checks)
- `voyage/audio/planner.py:118-125` (`take_for_time`: `covers_from <= video_time < covers_until()`)
- `voyage/media.py:160-198` (`slice_take`: `-t max(duration_seconds, 0.1)` floor at line 184)
- `voyage/media.py:26` (`AV_ALIGNMENT_TOLERANCE_SECONDS = 0.6`) + `:39-58` (`check_av_alignment` raising `MediaError` past it)
- `voyage/supervisor.py:1730-1734` (commit enforces video-vs-expected *and* video-vs-audio alignment — the gate this issue trips; sweep said `:1722-1728`)
- `voyage/cli.py:876-924` (`validate_run` enforces the same 0.6 s budget — a poisoned segment fails `validate` too; sweep said `:766-807`)

**Description:**
The slice walk covers `[video_time, video_time + duration)` by repeatedly asking the planner for the serving take and cutting one ffmpeg slice per take piece. Two missing guards compose:

1. **No validation at the ledger boundary.** `AudioTake.from_dict` (lines 73-86) accepts anything JSON can express: `duration: nan`, `duration: inf`, negative durations, `covers_from: nan`, near-zero durations. `load_takes` (`planner.py:197-208`) feeds these straight into the planner. A hand-edited, corrupt, or (forward-compat) foreign-version ledger line thus becomes a live take. Note the contrast: `EmptyAceStep1.5LatentAudio.seconds` rejects sub-1.0 s durations at the render layer, and `beat._require_finite` guards the beat math — but the *read* path for persisted takes has no equivalent, so garbage that could never be *rendered* can still be *served*.

2. **No iteration bound on the walk.** Line 1259 loops `while cursor < end - 1e-6` with progress strictly dependent on `piece > 0`. Failure modes by take shape:
   - **Near-zero duration (e.g. 1 ms):** `take_for_time` keeps serving successive 1 ms slivers; each iteration spawns a full ffmpeg process (`slice_take` → `run_capture`, lines 175-195) that renders `max(0.001, 0.1) = 0.1 s` of audio per line 184. A 4 s segment over 1 ms takes = **~4000 ffmpeg spawns** producing ~400 s of slice audio for 4 s of video; assembly then joins them (with crossfades) into an `audio.wav` orders of magnitude longer than `video.mp4`, and the commit's `check_av_alignment` (line 1726) raises `MediaError` past the 0.6 s budget — after minutes of wasted GPU-adjacent CPU work. Wall-clock DoS first, correctness failure second.
   - **`piece <= 0` (stagnant cursor):** if a take ever satisfies `take_for_time(cursor)` while `min(covers_until, end) - cursor <= 0` (degenerate/NaN-adjacent coverage arithmetic), `cursor += piece` never advances and the loop spawns ffmpeg **forever** — unbounded process creation with no counter, no deadline, no log throttling. (Pure-NaN takes return `None` from `take_for_time` since NaN comparisons are `False` — that path raises the clean `MediaError` gap message at line 1254. The unbounded path is the *positive-but-tiny or stagnant* piece, which sails past the `None` check.)
   - **Negative durations** currently fail at `take_for_time` (empty coverage → `None` → gap `MediaError`), but only by accident of comparison semantics, not by validation — same for `inf` (serves everything, one giant slice, then `slice_take -t inf` → ffmpeg behavior undefined).

**Rationale:**
This loop shells out to ffmpeg once per iteration with zero bound — it is the only unbounded-spawn loop on the commit path, and its input (ledger takes) is the least-validated struct on that path. The 0.6 s A/V gate (issues 003/retrofit at lines 1730-1734) will *catch* the resulting monster audio, but catching is not containing: by gate time the run has already spawned thousands of processes and written thousands of slice files, and the raised `MediaError` rests the run `FAILED` for what is really corrupt-input, not a media failure. Validate-then-walk is cheaper than walk-then-reject by orders of magnitude.

**Live evidence (current tree, host stdlib — ledger + walk arithmetic, no torch needed):**
```
$ python3 -c "<slice arithmetic over 1ms takes>"
1ms-take iterations for 4s segment: 4000 ffmpeg spawns;
each renders max(0.001,0.1)=0.1s => ~ 400.0 s audio for 4s video
```
Live-read chain:
```
planner.py:74-86:   covers_from=float(raw["covers_from"]), duration=float(raw["duration"]), ...
                    # no isfinite, no > 0, no covers_from >= 0 — any JSON float survives
supervisor.py:1259: while cursor < end - 1e-6:      # no index cap, no piece floor
supervisor.py:1263: piece = min(serving.covers_until(), end) - cursor
media.py:184:      f"{max(duration_seconds, 0.1):.6f}",   # 0.1 s floor AMPLIFIES sub-0.1 pieces
supervisor.py:1734: av_drift = check_av_alignment(...)   # 0.6 s gate trips only after the damage
```
`rg 'finite|<= 0|> 0' voyage/audio/planner.py` → no validation of take geometry anywhere in the file (the `<=` hits, if any, are unrelated ahead-window comparisons).

**Repro (CPU, no GPU):**
1. `init` a fake-backend run; append a crafted ledger line to `audio/takes.jsonl`: `{"take_id":"take_0000","path":"<real take wav>","caption":"x","seed":0,"covers_from":0.0,"duration":0.001,"segment_index":0}`.
2. Commit a 4 s segment: observe thousands of `slice_*.wav` files, one ffmpeg spawn each, then `MediaError: segment … A/V alignment drift … exceeds 0.6s` at the commit gate (or `validate` failing the same way on the corpse).
3. Stagnant variant: craft overlapping takes whose `covers_until` arithmetic yields `piece <= 0` at some cursor (e.g. zero-duration take exactly at the cursor with a second take covering it — `take_for_time` returns the max-`segment_index` match); observe the loop never terminating (kill the run; count `slice_*.wav` growth).

**Fix candidates:**
- Validate `AudioTake` at the boundary in `from_dict` (finite, `duration > 0`, `covers_from >= 0`, finite `bpm` when present) raising `StateError`/`MediaError` on violation — corrupt ledger fails loud at load, never inside the walk. Mirror `beat._require_finite` (in-tree precedent).
- Cap the walk: `max_slices` derived from segment geometry (e.g. `ceil(duration / min_piece) + num_takes + slack`, or a flat few-hundred constant with a comment) raising `MediaError` on breach; plus `if piece < 0.05: raise MediaError(...)` (tiny slivers are never legitimate music coverage — the smallest real take is ≥1 s by the render-layer floor).
- Consider clamping `slice_take`'s floor upward awareness: the `max(duration, 0.1)` floor exists for ffmpeg's sake, but any caller passing `< 0.05` is a bug — assert at the `media.py` boundary too (defense in depth; `slice_take` is also called from the SFX path).
- Tests: ledger-matrix unit test (nan/inf/negative/zero/tiny durations → load-time rejection); walk test with 1 ms takes asserting bounded iterations + `MediaError` before the Nth ffmpeg spawn (mock `run_capture` and count calls); stagnant-piece test asserting termination.

**Refs:**
- Overlaps with 100 (NaN/inf validation family — `beat._require_finite` precedent cited by both; 100 owns the RPC-deadline leg, this issue owns the ledger/walk leg).
- In-tree: `voyage/audio/planner.py:74-86` (unvalidated boundary), `:197-208` (`load_takes`), `voyage/supervisor.py:1251-1280` (the walk), `voyage/media.py:160-198` (the 0.1 s floor), `voyage/media.py:26-58` + `voyage/supervisor.py:1730-1734` + `voyage/cli.py:876-924` (the 0.6 s gate that catches-but-doesn't-contain), `voyage/audio/beat.py:24-33` (`_require_finite` precedent).
- `ffmpeg -t` with absurd/NaN durations is undefined behavior at the tool boundary — another reason the floor must be guarded by validated inputs, not trusted to the callee.
