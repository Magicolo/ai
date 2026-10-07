# 251 — Upscale poller always decodes from frame 0 (O(N²) H.264 decode tax)

Severity: MEDIUM (track D-07).

## Technical description

`_default_decode_fn` hardcodes `fps=None`, forcing the from-start
`select=between(n,start,end)` path for every chunk.

## Rationale

Chunk at `start=186,count=32` decodes 218 frames to keep 32. An 8-chunk 232f segment
decodes ~4× the segment. The fast-seek branch (`-ss start/rate` + rebased select) would
fix it but trades keyframe-approximate seek risk on H.264 (GOP-dependent window shift) —
the exact hazard the `select`-exact path was written to avoid. So the safe path is also
the quadratic path, with no middle option (exact seek after `-i`, or segment-accurate
trim).

## Live evidence

```
$ grep -n "fps=None" voyage/augment_upscale_poller.py
154: return ffmpeg_decode_chunk(source_video, dest_dir, start_frame, frame_count, fps=None)
$ python3 - <<'EOF'
# redundant decodes for 232f/32f tiling (prefix sums)
wins=[32,32,32,32,32,32,32,15]; s=0; tot=0
for c in wins: tot+=s+c; s+=c-1
print('decoded frames for 232f:', tot, 'ratio:', round(tot/232,2))
EOF
decoded frames for 232f: 1063 ratio: 4.58
```

Repro: count `ffmpeg` argv shapes during an 8-chunk upscale poll — all lack `-ss`, all
start at 0.

## Source refs

`voyage/augment_upscale_poller.py:148-154`; `voyage/augment.py:333-414` (fast-seek `fps`
branch exists but is never taken by this caller).

## Online sources

- ffmpeg seeking semantics (`-ss` before `-i` = fast/keyframe-approximate; after `-i` =
  accurate/slow).
- ComfyUI-VideoHelperSuite batch experience.

## Fix candidates

- Accurate `-ss` after `-i`; or one full decode per segment shared by chunks (decode once
  to PNGs, slice); or document + gate fast-seek behind exactness tests.

## Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.

## Evaluation (2026-10-07, group N)

Re-verified live against current code before fixing — claim CONFIRMED:

- `_default_decode_fn` (`voyage/augment_upscale_poller.py:148-154`)
  still hardcodes `fps=None`, forcing the from-start
  `between(n,start,end)` path for every chunk. The fast-seek branch
  (`voyage/augment.py:363-381`) is input-`-ss` (keyframe-approximate) —
  the exact hazard the issue notes — and no caller passes `fps`.
- The O(N^2) arithmetic was rechecked by hand: windows
  32,32,32,32,32,32,32,15 over 232f decode prefix sums = 1063 frames
  (4.58x). Computed, not profiled — but it is exact integer math on
  the window layout, not a timing estimate, so it stands.
- Fix adopted: per-segment shared decode (decode `(0, total)` ONCE,
  slice windows by file copy/hardlink + renumber) behind opt-in
  `shared_segment_decode` (default False — existing stub-based tests
  pin per-chunk calls). Engages only with >= 2 missing chunks (a lone
  chunk gains nothing). Byte-identical by construction (same decoder
  output bytes, copies). Wired at the interleaved-finalize and
  background call sites with `_accepts_keyword` guards; the parallel
  driver is deliberately EXCLUDED (per-chunk work-stealing granularity
  — a whole-segment decode per single-chunk task would decode MORE).

## Progress log (2026-10-07, group N)

- Implemented in `voyage/augment_upscale_poller.py`: new
  `_slice_shared_frames` helper + opt-in `shared_segment_decode`
  param on `upscale_poll_once` (default False — all existing
  stub-based tests pin per-chunk calls and pass unchanged). Engages
  only with 2+ missing chunks; staging carries `.partial` (next
  prune sweeps crash leftovers) and is removed per source in
  `finally`; short shared decodes fail loud like short chunk
  decodes. Wired (guarded `_accepts_keyword`) at the
  interleaved-finalize segment loop (`augment_finalize.py`) and the
  background segment loop (`augment_background.py`); joint units
  (4f, single-chunk) and the parallel driver (per-chunk stealing)
  intentionally excluded.
- New tests in `Voyage/tests/test_group_n_perf.py`: one decode +
  byte-correct slices (window contents pinned), default shape
  unchanged, lone-chunk stays per-chunk, slice bounds fail loud.

## Resolution (2026-10-07)

- Verdict: RESOLVED. Production upscale polling decodes each source
  once per pass instead of once per chunk (~4.6x fewer decoded
  frames on 232f/32f tilings by window arithmetic); slices are file
  copies of the same decoder output, so renders are byte-identical
  (pinned by content assertions).
- Files changed: `voyage/augment_upscale_poller.py`,
  `voyage/augment_finalize.py`, `voyage/augment_background.py`,
  `Voyage/tests/test_group_n_perf.py` (new).
- Verification: ruff check + format clean on touched files; mypy
  strict clean on touched modules; scoped pytest 318 passed /
  5 skipped plus 153 passed neighbors. No GPU workloads. No commits.
- Left open: nothing in-scope. The accurate-`-ss`-after-`-i`
  alternative was not implemented (shared decode dominates it:
  after-`-i` seek still decodes from the previous keyframe).
