# 288 — CausVid validates overlap against the wrong block size at init, and skips the fps-first boundary its siblings have

Severity: MEDIUM (pass-2 worker-tails sweep; two related boundary gaps, one entry).

## Technical description

(a) `handle_init` validates `overlap_frames` against the module constant
`NUM_FRAME_PER_BLOCK` (3), but `CausvidSession.__init__` re-validates against the YAML
config's `num_frame_per_block`. If the pinned config ever diverges from 3, init passes
and the failure lands after minutes of model load. (b) `handle_generate_blocks` checks
`_SESSION is None` before any payload validation, while ltxv/ltx25 deliberately
`validate_fps` first so bad requests fail fast and stay CPU-testable without a GPU
session.

## Rationale

Fail-fast ordering is an explicit, documented convention in this tree ("Reject a bad frame
rate before the session check so the boundary fails fast (and stays CPU-testable without
a GPU session)" — `video_ltxv.py:953-955`, `video_ltx25.py:1139-1141`). CausVid is the
odd one out on both counts.

## Live evidence

```
$ grep -n "validate_fps\|_SESSION is None" voyage/workers/video_causvid.py | head -3
1052:    if _SESSION is None:          # handle_generate_blocks — no validate_fps before it
$ sed -n '527,557p' voyage/workers/video_causvid.py
  block = int(getattr(config, "num_frame_per_block", NUM_FRAME_PER_BLOCK))
  validate_overlap_frames(overlap_frames, block)     # session: config value
$ sed -n '987,988p' voyage/workers/video_causvid.py
  overlap_frames = int(payload.get("overlap_frames", DEFAULT_OVERLAP_FRAMES))
  validate_overlap_frames(overlap_frames, NUM_FRAME_PER_BLOCK)   # init: constant
```

Repro: (a) point `VOYAGE_CAUSVID_DIR` at a config with `num_frame_per_block: 6`, init
with `overlap_frames: 3` → init passes, session raises post-load. (b) With no session,
`generate_blocks` with `fps: -1` → `RuntimeError("not initialized")` on causvid vs
`ValueError` on ltxv/ltx25.

## Source refs

`voyage/workers/video_causvid.py:972-1006` (init), `:527-557` (session), `:1051-1053`
(generate); contrast `video_ltxv.py:953-956`, `video_ltx25.py:1141-1143`.

## Online sources

- None (in-tree fail-fast convention is the anchor).

## Fix candidates

- (a) Load the config (cheap, no torch) in `handle_init` and validate against its block
  size — or drop the init-time check and document fail-after-load; (b) add the two-line
  `validate_fps` pre-check to causvid's `handle_generate_blocks`.

## Log

- 2026-10-07: filed from read-only pass-2 worker-tails sweep; no code touched.
