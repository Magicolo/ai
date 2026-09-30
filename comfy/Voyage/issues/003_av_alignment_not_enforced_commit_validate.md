# 003 — A/V alignment never enforced at commit or `validate` — only at `finalize`

- Status: resolved in live tree (`check_av_alignment` at commit + validate + finalize)
- Severity: HIGH (commit-soundness gate; resolved, record only)
- Group: correctness/validation — Rank: 1/5 (critical pattern, fixed)
- Area: correctness — segment validation
- Rank rationale: `validate` is documented as the commit gate (§70) but passed
  segments that `finalize` later aborted; drifts accumulated silently across long runs.

## Technical description

Pre-fix, commit checked video vs expected duration only; audio was never compared:

```python
# voyage/supervisor.py:1173-1180 (pass 1)
if abs(float(video_info["duration"]) - duration) > AV_ALIGNMENT_TOLERANCE_SECONDS:
    raise MediaError(f"segment {segment_id} A/V duration drift")
```

`voyage/cli.py:524-553,555-604` (`_check_segment_metrics`) only asserted durations
were `> 0`. But `voyage/media.py:220-261` (`_verify_segment`, finalize-only) did:

```python
drift = abs(video_duration - audio_duration)
if drift > AV_ALIGNMENT_TOLERANCE_SECONDS: raise MediaError(...)
```

So a segment with 2 s video + 10 s audio (wrong slice, stale take, `slice_take`
clamped artifact) committed cleanly and passed `validate_run` (VALID), then
`finalize_run` aborted — possibly hundreds of segments later.

Live state (re-verified 2026-09-30): one shared gate
`voyage/media.py:26-56` (`AV_ALIGNMENT_TOLERANCE_SECONDS = 0.6`,
`av_drift_seconds`, `check_av_alignment`) enforced at all three points —
commit (`voyage/supervisor.py:1730-1734`), validate
(`voyage/cli.py:842-845`), finalize verify (`voyage/media.py:359`).

## Why this is an issue

- Late failure is the most expensive kind: GPU time already spent, run supposedly
  healthy, user discovers at export.
- `validate` giving VALID for a finalize-fatal run destroys trust in the gate.
- Misaligned segments also poison beat-grid/take planning downstream
  (`_ensure_audio_coverage` assumes sane durations).

## Evidence

Live verification 2026-09-30:

```
$ rg -n "AV_ALIGNMENT_TOLERANCE|check_av_alignment" voyage/supervisor.py voyage/cli.py voyage/media.py
voyage/supervisor.py:53:     AV_ALIGNMENT_TOLERANCE_SECONDS,
voyage/supervisor.py:55:     check_av_alignment,
voyage/supervisor.py:1722:         if abs(float(video_info["duration"]) - duration) > AV_ALIGNMENT_TOLERANCE_SECONDS:
voyage/supervisor.py:1726:         av_drift = check_av_alignment(
voyage/cli.py:40: from voyage.media import AV_ALIGNMENT_TOLERANCE_SECONDS, check_free_space, finalize_run
voyage/cli.py:801:     if video_duration > 0 and audio_duration > 0:
voyage/cli.py:803:         if drift > AV_ALIGNMENT_TOLERANCE_SECONDS:
voyage/media.py:26: AV_ALIGNMENT_TOLERANCE_SECONDS = 0.6
voyage/media.py:52:     drift = av_drift_seconds(video_duration, audio_duration)
voyage/media.py:359:     check_av_alignment(video_duration, audio_duration, name)
```

Pass-1 proof of the asymmetry: `test_state_integrity.py:
test_finalize_rejects_av_misalignment` — commit 1 seg, replace `audio.wav` with a
10 s synth, rewrite checksums → `finalize` raised `Alignment` while `validate_run`
still reported `[]` (both durations > 0).

## Reproduction

1. Fake-backend run, commit 1 segment.
2. Replace `segments/000000/audio.wav` with a 10 s sine synth, rewrite `sha256.json`.
3. Pre-fix: `voyage validate --run <run>` → VALID (bug); `voyage finalize` →
   Alignment error. Post-fix: both reject with the measured drift.

## Source references

- `voyage/media.py:26-56` (shared gate), `:324-360` (`_verify_segment`);
  `voyage/supervisor.py:1722-1726` (commit); `voyage/cli.py:766-806` (validate).

## Resolution candidates

1. (Landed) Enforce `abs(video_dur - audio_dur) <= 0.6` in `commit_one_segment`
   (post-`validate_audio`) AND in `cli._check_segment_metrics` (stored
   `metrics.video/audio.duration` suffice — no extra probe needed).
2. Record the measured drift in the `segment_committed` metric event so
   scoreboard/soak can trend it (see 049).
3. Tests: misaligned-by-construction segment must fail commit; rewritten-audio
   segment must fail `validate_run`.

## Online references

- FFmpeg audio/video sync and duration probing semantics (`-shortest`, stream
  durations vs container duration):
  https://ffmpeg.org/ffmpeg.html
- PCM WAV vs AAC/MP4 container duration rounding (why a tolerance, not exact
  equality, is the correct gate): https://ffmpeg.org/ffmpeg-formats.html

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep; existing finalize-side test cited as proof.
- Phase 6 slice A + resolution batch 3: shared `check_av_alignment` + commit-side
  enforcement landed.
- 2026-09-30: re-verified live (gate present at all three call sites);
  reconstructed from archived pass-1 text (commit `b5d7dda`). Status → resolved.
