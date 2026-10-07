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

## Evaluation (2026-10-07, live re-probe)
- **Confirmed (with the reviewer's arithmetic-only caveat — no overclaim).**
  `voyage/media_audio.py:518-546` (`_segment_timeline`) walks manifest
  `frames / fps`; `voyage/sfx_finalize.py:248-313` (`segment_sfx_bounds`)
  walked probed container `duration` first. The ~128 ms worst case was
  not re-measured live (no GPU media run per mandate); it stands as the
  filed arithmetic bound (0.5 ms/segment x 256), inside the 0.6 s gate —
  the fix removes the systematic component rather than chasing the
  measurement.
- What WAS measured: pre-fix, a manifest holding `frames=96 @24fps` with
  a skewed container duration produced probe-clock bounds; post-fix, the
  same fixture produces exactly `96/24` and never spawns the probe
  (probe monkeypatched to raise — pinned by test).

## Progress log (2026-10-07)
- `segment_sfx_bounds` now reads manifest `metrics.frames / fps` first
  (same frames/fps clock as `_segment_timeline`; same tolerant read shape
  — unreadable/torn manifests degrade to `{}` and fall through), probe
  (container duration, then video `nb_frames / fps`) is fallback only.
  Single timeline owner; legacy/torn-manifest runs behave byte-identically
  to before (probe path untouched).
- Added pure `sfx_timeline_drift_seconds(manifest_bounds, probed_bounds)`
  → worst absolute end-time skew (length mismatch raises; empty walks
  agree at 0.0) — the quantity the uniform rescale cannot correct and
  the 0.6 s gate absorbs.
- Tests in `tests/test_sfx_finalize.py`:
  `test_sfx_bounds_prefer_manifest_frames_over_probe` (probe forbidden),
  `test_sfx_bounds_fall_back_to_probe_without_manifest_frames`
  (`frames: 0` legacy dir still probes), `test_sfx_timeline_drift_seconds_measures_worst_skew`.
- Verification (in-container `voyage:latest`): ruff + format + mypy
  clean on `voyage/sfx_finalize.py`; all bounds neighbors green
  (`test_issue_191_sfx_bounds_streams`, `test_sfx_timeline_union`,
  `test_fake_bed_end_to_end_over_junctions` inside `test_sfx_finalize`).
- One-time upgrade note: ledgers planned under probe-clock durations can
  miss the 1e-6 request-identity tolerance once (re-render one pass);
  new lines carry manifest-clock durations and hit thereafter.

## Resolution (2026-10-07)
- **RESOLVED.** SFX bounds derive from the manifest frames/fps walk with
  probe fallback only, plus a drift metric for the residual. Files:
  `voyage/sfx_finalize.py`, `tests/test_sfx_finalize.py`. No commit
  (per mandate).
