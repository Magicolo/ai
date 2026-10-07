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
