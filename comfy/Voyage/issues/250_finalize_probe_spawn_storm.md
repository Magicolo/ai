# 250 — Redundant finalize/prewarm work: ffprobe spawn storm + weights/joint re-hash per pass, per commit (~10³ spawns, minutes)

Severity: MEDIUM (track D-06).

## Technical description

Same files probed repeatedly across stages: `build_final_audio` (tail
`probed_take_seconds` + per-window `_audio_duration_seconds`), `assemble_segment_audio`
(per-slice `probe`), SFX (per-stem probe on hit **and** render paths + bed/pair/mix
probes), `validate`/`_verify_segment` (probe + sha), `presented_frames` (full decode).

## Rationale

256-seg estimate: music ~512 + SFX ~500–1000 + validate 256 ≈ 1500 spawns. At 30–80
ms/spawn → 45–120 s of pure process-spawn overhead before any pixel/audio render, all
re-reading bytes already measured. Durations threaded through
`_blend_pair(first_seconds, second_seconds)` prove the codebase knows how to avoid
re-probes — it just doesn't do it at the finalize-workflow level.

## Live evidence

```
$ grep -c "probe(\|_audio_duration_seconds\|probed_take_seconds" voyage/sfx_finalize.py voyage/media_audio.py
sfx_finalize.py: ~20 sites; media_audio.py: ~10 sites
$ python3 -c "print('1500 spawns * 0.05s =', 1500*0.05/60, 'min floor')"
1500 spawns * 0.05s = 1.25 min floor
```

Repro: wrap `run_capture` with a counter during a 3-segment fake finalize; count probes
per segment (scales linearly; project to 256).

## Source refs

`voyage/media_audio.py:391,555,590,955`;
`voyage/sfx_finalize.py:659,794,830,900,984,1020,1152,1182,1291,1386`.

## Online sources

- ffmpeg engineering handbook (concat demuxer stream-copy vs filter re-encode cost
  discipline — same "don't pay per-input repeatedly" principle).

## Fix candidates

- Single probe-plan per finalize (dict path → info) threaded through music/SFX/validate;
  cache `_audio_duration_seconds` by (path, mtime, size); reuse slice/window durations
  already in hand.

## Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.

## Consolidated from 254_weights_and_joints_rehashed_per_pass_per_commit (2026-10-07)

Severity: MEDIUM (track D-09).

### Technical description

`weights_key_for` = `sha256_file(interp) + sha256_file(esrgan)` (86 MB RIFE + 2.5 MB
SRVGG disk reads). Called by `run_durable_model_pass` (every finalize) and
`resolve_background_plan` (**every commit's** `prewarm_once`). `ensure_joint_units`
additionally `sha256_file` every segment video per call, and `_poll_to_completion`
re-resolves per pass.

### Rationale

Weights never change mid-run; segment SHAs only change on re-render (detectable via
manifest checksum already in hand). 256 commits → 256× ~90 MB re-reads + 256× seg-0
ffprobe + per-poll re-hashes of GBs of segment video. All on the critical path before
any chunk renders.

### Live evidence

```
$ grep -n "weights_key_for\|sha256_file" voyage/augment_finalize.py voyage/augment_background.py voyage/augment_joints.py | head -20
```

(weights_key_for def + 2 call sites; joint `_checksum` per-pair dict is per-call only.)

Repro: counter on `hashing.sha256_file` during 5 sequential `prewarm_once` calls with
unchanged models — 10+ full-file hashes, zero ledger effect.

### Source refs

`voyage/augment_finalize.py:40-61`; `voyage/augment_background.py:311-316,430+`;
`voyage/augment_joints.py:242-247,436-440`.

### Online sources

- Standard content-addressing practice (hash once, key by mtime/size thereafter).
- In-tree `source_key` already carries the segment checksum — re-hashing the bytes
  re-derives it.

### Fix candidates

- Memoize `weights_key_for` by (dev,ino,mtime,size); pass segment `source_key`s into
  joint planning instead of re-hashing; compute `BackgroundPlan` once per config+models.

### Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.
- 2026-10-07: consolidated into 250 (same redundant-work-per-finalize/commit class).
