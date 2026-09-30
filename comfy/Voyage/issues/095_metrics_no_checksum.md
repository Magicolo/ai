# 095 — `sha256.json` covers only `video.mp4` + `audio.wav`; metrics/transition/prompt/audio_state/world_state never hashed

- Severity: MEDIUM (raised from LOW-MEDIUM: corrupt metadata is trusted by validate/resume with no diagnostic — real integrity defect, no workaround)
- Area: correctness / persistence / integrity
- File: `voyage/supervisor.py:1778-1781` (checksum write), `voyage/cli.py:781-800` (`_check_segment_checksums`)

## Description

Live write (`voyage/supervisor.py:1778-1781`, re-verified 2026-09-30):

```python
atomic_write_json(
    segment / "sha256.json",
    {"video.mp4": sha256_file(video_out), "audio.wav": sha256_file(audio_out)},
)
```

Live check (`voyage/cli.py:781-800`):

```python
for artifact in ("video.mp4", "audio.wav"):
    recorded = expected.get(artifact)
    ...
    elif sha256_file(target) != recorded:
        errors.append(...)
```

The segment also durably stores `metrics.json`, `transition.json`, `prompt_plan.json`, `audio_state.json`, `world_state.json` (written just above at `supervisor.py:1750-1777` via `atomic_write_json`). None of them is hashed or re-verified. A bit-flip / torn write that survives `atomic_write_json` (or a post-commit edit, manual or malicious) in any of those files is invisible to `validate_run`:

- `metrics.json` drives `_check_segment_metrics` (frame counts, durations, A/V drift, recovery tape) — corrupt it and validate trusts corrupt values.
- `transition.json` / `prompt_plan.json` are the director provenance (DESIGN §74/§75); silent mutation breaks audit without tripping any gate.
- `world_state.json` / `audio_state.json` feed resume and finalize decisions.

The checksum file's name (`sha256.json`) promises segment integrity; it delivers media-only integrity.

## Rationale

- Integrity asymmetry: the two largest files (media) are protected while the files that *interpret* them (durations, frames, tapes) are not. An attacker / bit-rot that wants to change verdicts targets exactly the unhashed files.
- Recovery (`_latest_recovery_tape` reads `metrics.json` `recovery_tape`) trusts unhashed data to rebuild GPU state — a corrupt tape pointer burns restart budget on doomed resumes.
- Cost of fixing is one dict extension + loop generalization; no format break if done additively (old `sha256.json` files simply lack the new keys — treat missing as "not covered", or backfill on next validate).

## Live evidence

```
$ grep -n "sha256.json" comfy/Voyage/voyage/supervisor.py comfy/Voyage/voyage/cli.py
supervisor.py:1771:                segment / "sha256.json",
cli.py:745:    checksums_path = segment / "sha256.json"
cli.py:754:    for artifact in ("video.mp4", "audio.wav"):
```

Write excerpt (live):

```
atomic_write_json(
    segment / "sha256.json",
    {"video.mp4": sha256_file(video_out), "audio.wav": sha256_file(audio_out)},
)
```

Five `atomic_write_json` calls precede it (`world_state.json`, `transition.json`, `prompt_plan.json`, `audio_state.json`, `metrics.json` at `:1750-1777`) — none appears in the checksum dict.

## Repro

1. Commit a fake-backend segment; record `sha256.json`.
2. Flip one byte in `segments/000000/metrics.json` (e.g. `"frames": 29` → `"frames": 39`) keeping valid JSON.
3. `voyage validate` → still `VALID` (frame-sum check uses the corrupt value consistently; no checksum mismatch because metrics is unhashed). `timeline_frames` accounting is now wrong with no diagnostic.
4. Same for `transition.json` (change `destination_concept`) → validate passes; provenance silently altered.

## Fix candidates

1. (Preferred, additive) Extend the dict: hash all six files (`video.mp4`, `audio.wav`, `metrics.json`, `transition.json`, `prompt_plan.json`, `audio_state.json`, `world_state.json`); generalize `_check_segment_checksums` to iterate `expected` keys (not a hardcoded pair) so old files (missing keys → "not covered" warning, not error) and new files both validate.
2. Document the promise: `sha256.json` = full segment manifest; `validate` recomputes all entries (DESIGN §70). Backfill script for old runs (optional — or accept "legacy files cover media only" with a version marker).
3. Regression test: corrupt each JSON artifact byte-preservingly → assert `checksum mismatch for <name>`; corrupt media → existing behavior unchanged.

## Refs

- Not a duplicate of 098 (residue detection vs integrity coverage — complementary, both touch `validate_run`).
- `voyage/supervisor.py:1750-1781` (five metadata writes + two-entry checksum), `voyage/cli.py:781-800` (two-entry check).
- DESIGN §70 (validate recomputes checksums) vs §74/§75 (plan persistence) — the hashed set should equal the persisted set.
- `voyage/hashing.py:19-25` (`sha256_file` helper — reuse for JSON files, chunked, constant memory).
