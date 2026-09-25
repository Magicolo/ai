# 003 — A/V alignment never enforced at commit or `validate` — only at `finalize`

- Status: resolved (fixed 2026-09-25)
- Severity: critical (commit-soundness gate weaker than finalize)
- Area: correctness — segment validation
- Rank rationale: `validate` is documented as the commit gate (§70) but passes
  segments that `finalize` later aborts; drifts accumulate silently across long runs.

## Technical description

Commit checks video vs expected duration only; audio is never compared:

```python
# voyage/supervisor.py:1173-1180
if abs(float(video_info["duration"]) - duration) > AV_ALIGNMENT_TOLERANCE_SECONDS:
    raise MediaError(f"segment {segment_id} A/V duration drift")
```

`voyage/cli.py:524-553,555-604` (`_check_segment_metrics`) only asserts durations
are `> 0`. But `voyage/media.py:220-261` (`_verify_segment`, finalize-only) does:

```python
drift = abs(video_duration - audio_duration)
if drift > AV_ALIGNMENT_TOLERANCE_SECONDS: raise MediaError(...)
```

So a segment with 2 s video + 10 s audio (wrong slice, stale take, `slice_take`
clamped `max(duration,0.1)` artifact) commits cleanly and passes `validate_run`
(VALID), then `finalize_run` aborts — possibly hundreds of segments later.

## Why this is an issue

- Late failure is the most expensive kind: GPU time already spent, run supposedly
  healthy, user discovers at export.
- `validate` giving VALID for a finalize-fatal run destroys trust in the gate.
- Misaligned segments also poison beat-grid/take planning downstream (see 013/027
  in spirit: `_ensure_audio_coverage` assumes sane durations).

## Evidence

- Source quotes above; `test_state_integrity.py:test_finalize_rejects_av_misalignment`
  proves the asymmetry: commit 1 seg, replace `audio.wav` with 10 s synth, rewrite
  checksums → `finalize` raises `Alignment` while `validate_run` still reports `[]`
  (both durations > 0).

## Reproduction

1. Fake-backend run, commit 1 segment.
2. Replace `segments/000000/audio.wav` with a 10 s sine synth, rewrite `sha256.json`.
3. `voyage validate --run <run>` → VALID (bug). `voyage finalize` → Alignment error.

## Source references

- `voyage/supervisor.py:1173-1180`, `voyage/cli.py:524-553,555-604`,
  `voyage/media.py:220-261`.

## Resolution candidates

1. Enforce `abs(video_dur - audio_dur) <= 0.6` in `commit_one_segment`
   (post-`validate_audio`) AND in `cli._check_segment_metrics` (stored
   `metrics.video/audio.duration` suffice — no extra probe needed).
2. Record the measured drift in the `segment_committed` metric event so scoreboard/
   soak can trend it (see 049).
3. Add tests: misaligned-by-construction segment must fail commit; rewritten-audio
   segment must fail `validate_run`.

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep; existing test name cited as proof of the
  finalize-side check.
- Open: implement + tests.
- 2026-09-25 (repair pass): refs verified current (`supervisor.py:1179`;
  `cli.py:524,555+`; `media.py:220-261`); `## Why this is an issue` already
  present, no change.
- 2026-09-25 (consumer-side resolution, media/validate half — FIXED):
  `voyage/media.py` gained `av_drift_seconds` + `check_av_alignment`
  (same 0.6 s budget, returns the drift; `_verify_segment` now delegates
  to it, message unchanged); `cli._check_segment_metrics` enforces the
  drift over stored `metrics.video/audio.duration` (no probe) with a
  `run_dir` parameter for the tape resolution; new tests in
  `tests/test_av_alignment_consumer.py` (helper units + rewritten-audio
  segment fails `validate_run`, clean run passes). The commit-path half
  (supervisor.py step 4 + `av_drift_seconds` in the `segment_committed`
  metric event) is the supervisor track's: call
  `check_av_alignment(float(video_info["duration"]),
  float(audio_info["duration"]), segment_id)` next to the existing
  expected-duration check — signature and hook documented in the
  helper's docstring.
- 2026-09-25 (review): hook consumed by the orchestrator — commit step 4 now
  calls `check_av_alignment` (`supervisor.py`, next to the expected-duration
  check) and records `"av_drift_seconds"` in the `segment_committed` event.
  New `test_commit_rejects_av_drifted_audio` pins the commit path
  (monkeypatched `validate_audio` +10s → `MediaError` "alignment drift").
  Issue fully resolved on all three gates (commit/validate/finalize).
