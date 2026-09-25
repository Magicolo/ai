# 063 — Audio RPC numeric/enum fields unvalidated (`bpm=0`, `sample_rate=0`, negative duration clamped, free-form `task_type`)

- Status: open
- Severity: medium (silent wrong-length/failed renders; ffmpeg errors escape as
  untyped `RuntimeError`)
- Area: workers/audio contract — `voyage/workers/audio_acestep.py:80-99`,
  `voyage/audio/acestep.py:114-125`
- Rank rationale: pass-2 worker finding; explicit `bpm=0` bypasses the
  energy-map clamp; negative durations render 1 s instead of rejecting.

## Technical description

```python
# workers/audio_acestep.py:80-99
bpm=int(bpm_raw) if bpm_raw is not None else None,   # 0 passes
task_type=str(payload.get("task_type","text2music")), # any string
src_audio=payload.get("reference_audio"),             # no type check
# sample_rate/channels only type-checked as int; duration_seconds only as float
# audio/acestep.py:114-125
duration=max(MIN_DURATION_SECONDS, duration_seconds), # negative → silent 1.0 s
bpm=bpm if bpm is not None else bpm_for_energy(energy),
```

`bpm_for_energy` clamps energy to 60..140 BPM (probed: `0.5→110`, `-10→60`,
`10→140`, floor `1.0`), but an explicit `"bpm":0` bypasses it;
`duration_seconds:-5` becomes `max(1.0,-5)=1.0` — 1 s of music for a negative
request instead of rejection.

## Why this is an issue

Invalid audio parameters fail late and expensively: a negative duration silently
renders 1 s of music, an explicit `bpm=0` bypasses the energy-map clamp, and
zero sample rates/channels reach ACE/ffmpeg as untyped `RuntimeError`s — after
GPU time is spent. The worker boundary is the cheapest place to reject these,
and every silent coercion here produces a wrong-length artifact instead of an
error.

## Evidence

Live `bpm_for_energy` probes (re-run 2026-09-25, `PYTHONPATH=Voyage`):

```
$ PYTHONPATH=Voyage python3 -c "from voyage.audio.acestep import bpm_for_energy; ..."
0.5 110
-10 60
10 140
```

The clamp works for derived BPM but an explicit `"bpm":0` bypasses it
(`acestep.py:117`); source quotes in Technical description above.

## Reproduction (no GPU run needed — all fail at the boundary)

`generate_audio` with `{"bpm":0}`, `{"sample_rate":0}`, `{"channels":0}`,
`{"duration_seconds":-3}`, `{"task_type":"not-a-task"}`,
`{"reference_audio":12345}` — all pass `checked_request` and reach ACE/ffmpeg.

## Source references

- Files/lines above.

## Resolution candidates

Validate `bpm in 1..300` (or `None`), `sample_rate>0`, `channels in (1,2)`,
`duration_seconds>0` (reject, don't clamp, at the worker boundary),
`task_type in ("text2music","repaint",...)`, `reference_audio` as
str-existing-path-or-None.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 worker sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`audio_acestep.py:80-99`, `acestep.py:114-125` — both
  match); re-ran `bpm_for_energy` probe (`0.5→110`, `-10→60`, `10→140`,
  clamp confirmed, explicit-bpm bypass stands).
- Open: validate + tests (each field).
