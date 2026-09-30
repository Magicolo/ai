# 189 — Single-slice audio paths skip every duration/output check the multi-slice paths enforce

- Severity: LOW-MEDIUM
- Area: media audio assembly — verification asymmetry
- Files (as-read 2026-09-30):
  - `voyage/media.py:160-198` (`slice_take` — no dest size/probe check)
  - `voyage/media.py:252-268` (`assemble_segment_audio` single-slice branch — ffmpeg copy, no output check)
  - `voyage/media.py:269-271` (multi-slice branch — probes every slice, rejects `d <= 0`)
  - `voyage/media.py:598-600` (`build_final_audio` single-slice window — `slices[0].replace(window_path)`)
  - `voyage/augment.py:169-174,211-212` (sibling precedent: decode/encode both verify non-empty outputs)

## Description

Three single-input fast paths bypass the verification the neighboring multi-input
paths perform:

1. `slice_take` (`:160-198`) returns `dest` on ffmpeg exit 0 with no size or probe
   check. A zero-byte-but-exit-0 ffmpeg output (truncated take, full disk on the
   tmp fs — the exact condition 102's preflight gap enables) becomes a "slice".
2. `assemble_segment_audio` with one slice (`:252-268`) re-encodes copy-through
   with no output verification, while the multi-slice path probes every input
   (`:269-271`, `MediaError` on `d <= 0`) before blending.
3. `build_final_audio` with one slice per window (`:598-600`) renames the slice
   straight to `window_path` — no probe, no floor — while multi-slice windows go
   through `assemble_segment_audio` with the fade-absorption compensation and its
   checks (`:602-638`).

Net: the most common shape (one take serving the whole window — the steady state
after the planner warms up) is the least verified. A degenerate slice propagates
as a valid window into the pairwise `_blend_pair` reduction, where
`_audio_duration_seconds` finally fails loud — far from the slice that caused it,
with the window/segment identity lost in a `final_blend_NN.wav` temp name.

## Rationale

Verification belongs at the narrowest primitive (`slice_take`), not three call
levels up in `_blend_pair`. The augment siblings in the same tree already set the
pattern: `ffmpeg_decode_chunk` rejects empty frames, `ffmpeg_encode_chunk`
rejects empty outputs. The audio path's silence on the happy path means the
steady-state (single-slice) windows never get even the cheap `st_size == 0`
check the spike-era chunk code has.

## Live evidence (verified live 2026-09-30, host reads)

- `sed -n '160,198p' voyage/media.py` — `slice_take` ends `if proc.returncode != 0:
  raise …; return dest`; no `dest.stat()`, no probe between.
- `sed -n '252,271p' voyage/media.py` — single-slice copy vs `durations = [probe…
  ]` + `if any(d <= 0 …)` two branches later in the same function.
- `sed -n '598,600p' voyage/media.py` — `slices[0].replace(window_path)` with no
  intervening check; contrast `:638` (`assemble_segment_audio(…)` for multi).
- `rg -n "st_size == 0|produced empty" voyage/augment.py` → the two sibling
  checks this path lacks.

## Repro

Static (deterministic, CPU-only): place a valid take, stub `run_capture` to write
a 0-byte `dest` with returncode 0, call `slice_take` → returns the empty path
without raising; feed it as a single-slice window through `build_final_audio` →
`window_path` is the empty file, failure surfaces only later in `_blend_pair`'s
`_audio_duration_seconds` against a `final_blend_NN.wav` intermediate.

## Fix candidates

1. (Preferred) Verify in `slice_take`: after exit-0, reject `dest` missing or
   `st_size == 0` (mirrors `ffmpeg_encode_chunk:211-212`), and optionally probe
   `duration > 0` — one check covers all three single-slice paths at the source.
2. Add the same `st_size` guard to the `assemble_segment_audio` single-slice
   branch for defense in depth (cheap, matches the file's own multi-slice
   strictness).
3. Tests: 0-byte-exit-0 slice → `MediaError` naming the take/start/duration;
   single-slice window over a degenerate slice fails at slice time, not blend.

## Refs

- In-tree: `voyage/media.py:160-198,228-312,598-600`; `voyage/augment.py:169-174,
  211-212` (precedent); `voyage/media.py:441-446` (`_audio_duration_seconds`,
  where the failure lands today).
- Not-a-duplicate: 102 is concat quoting/RAM/preflight-fs (staging location,
  not per-slice verification); 043 is whole-file RAM at publish; 095 is
  fade-absorption length compensation (multi-slice correctness, not single-slice
  verification); 050 is video double-encode. None names the single-slice
  verification gap.
