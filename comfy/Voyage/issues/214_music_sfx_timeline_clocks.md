# 214 — Music timeline (frames/fps) vs SFX bounds (probed seconds) diverge per segment (HIGH)

## Technical description
Music mix walks `_segment_timeline` (manifest `frames / fps`,
sample-exact; `voyage/media_audio.py:518-546`). SFX bounds walk
`segment_sfx_bounds` (probed container `duration`, fallback
`nb_frames / fps`; `voyage/sfx_finalize.py:248-313`). Rescale
`_scale_bounds_to_timeline` (`sfx_finalize.py:316-338`) only scales
uniformly.

## Rationale
Container durations round to ms; H.264 timestamps/B-frames add jitter.
Worst-case 0.5 ms/segment × 256 ≈ 128 ms systematic skew, plus per-file VFR
jitter the uniform rescale cannot correct. SFX captions (`"; ".join`
junction windows) then misalign with the music mix's segment boundaries,
and the SFX `duration=longest` dub inherits the skew. Both timelines claim
to be the same timeline but are computed from different clocks.

## Live evidence
```
$ python3 -c "print('256-seg worst drift ms:', 256*0.5)"
256-seg worst drift ms: 128.0
$ grep -n "def _segment_timeline" voyage/media_audio.py; grep -n "def segment_sfx_bounds" voyage/sfx_finalize.py
media_audio.py:518 / sfx_finalize.py:248
```

## Repro
Synthesize 256 manifests with `frames=232, fps=24` but container durations
`frames/fps ± 0.0004`; compare `_segment_timeline` ends vs summed probed
durations — gap grows linearly.

## Source refs
- `Voyage/voyage/media_audio.py:518-546`
- `Voyage/voyage/sfx_finalize.py:248-338`

## Online sources
- ffmpeg concat-filter vs demuxer guidance (normalize fps/geometry before
  joining; mismatched rates give VFR output).
- In-tree `FPS_MATCH_TOLERANCE = 0.5` admission that container rates lie.

## Fix candidates
1. Derive SFX bounds from the same `frames/fps` manifest walk (single
   timeline owner); keep probe only as fallback.
2. Add a per-segment drift metric.

## Log
- Track D sweep, 2026-10-07. Read-only; nothing fixed.
