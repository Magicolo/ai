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

## Evaluation (2026-10-07, group N)

Re-verified live against current code before fixing — claim CONFIRMED
with corrections:

- `weights_key_for` (`voyage/augment_finalize.py:40`) hashes the 86 MB
  RIFE + 2.5 MB SRVGG legs on EVERY call: once per finalize (`:733`)
  and once per commit via `resolve_background_plan`
  (`voyage/augment_background.py:333`). No memo anywhere.
- Joint shas memoize PER CALL only: `ensure_joint_units`
  (`voyage/augment_joints.py:249-253`) and `assemble_joint_timeline`
  (`:445-449`) each build a fresh dict; `prune_orphan_plan_dirs`
  (`voyage/augment_drain.py:443-447`) re-hashes every joint video on
  every finalize start.
- `_audio_duration_seconds` (`voyage/media_audio.py:588`) has NO cache;
  it is the single definition re-exported by `media.py` and imported by
  `sfx_finalize.py`/`mastering.py`. Hit path (`sfx_finalize.py:1053`)
  and render path (`:1089`) each probe once by necessity (fresh files);
  the waste is repeats on UNCHANGED files across retries/passes
  (orphan adoption `:967`, windows `:955`, mix gates `:1221`/`:1251`).
  The 152 fold work already threads some durations; the finalize level
  does not.
- The "1500 spawns / 45-120 s" figures are computed-not-profiled
  projections (reviewer flag stands): spawn overhead is real but
  decode-bound probes dominate wall time, so the per-spawn floor
  understates the true cost. Recorded as estimates.
- Full "single probe-plan threaded through music/SFX/validate" would
  change signatures in `cli_finalize.py`/`supervisor.py`/validators —
  out of scope for this track. Adopted instead: process-wide
  stat-keyed caches (same dedupe, no signature churn) + memoized
  `weights_key_for` + process-wide joint-sha cache. Byte-identical
  outputs (counts/durations/keys only, no pixel/audio path changes).

## Progress log (2026-10-07, group N)

- Evaluated live (see Evaluation above); implemented in
  `voyage/media_audio.py` (`_AUDIO_DURATION_CACHE` shared by
  `_audio_duration_seconds` and `probed_take_seconds`, positive
  results only, identity-keyed), `voyage/augment_finalize.py`
  (`weights_key_for` memoized per file identity),
  `voyage/augment_joints.py` (new `joint_video_sha` process cache;
  both per-call dicts and the `_joint_ts_piece` param route through
  it), `voyage/augment_background.py` (`probe_segment_source`
  memoized per file identity).
- `sfx_finalize.py` needed no edit: its hit/render probes hit fresh
  files (stat differs, correct miss) or now hit the shared cache.
- New tests in `Voyage/tests/test_group_n_perf.py` (duration shared
  cache + invalidation on rewrite, take-seconds sharing, weights
  memo + invalidation, joint-sha memo + invalidation).
- Live proof in-container (real ffmpeg, no GPU): 4.0 s sine wav —
  duration + take-seconds + duration = 4.0/4.0/4.0 with exactly
  1 probe. Full scoped suite green (see Resolution).

## Resolution (2026-10-07)

- Verdict: RESOLVED (cache form). Unchanged files probe once per
  process no matter how many stages/passes ask; weights hash once
  per identity (per-finalize + per-commit re-reads gone); joint
  videos hash once per process across planning/assembly/GC;
  seg-0 geometry probes once per identity per commit.
- Files changed: `voyage/media_audio.py`,
  `voyage/augment_finalize.py`, `voyage/augment_joints.py`,
  `voyage/augment_background.py`,
  `Voyage/tests/test_group_n_perf.py` (new).
- Verification: ruff check + format clean on touched files; mypy
  strict clean on touched modules; scoped pytest 318 passed /
  5 skipped plus 153 passed (finalize/sfx/boundary/mix-cache
  neighbors); live 1-probe proof above. No GPU workloads.
  No commits.
- Left open: a true single probe-plan object threaded through
  music/SFX/validate signatures (would touch `cli_finalize.py` /
  supervisor / validators — out of scope); the process-wide caches
  achieve the same dedupe without signature churn. Per-pass poller
  ledger re-scans stay (ledger truth must be re-read as chunks land).
