# 060 — Benchmark `measured=0` (and negative `warmup`) crashes all 5 workers with `ZeroDivisionError`

- Status: resolved (fixed 2026-09-25)
- Severity: medium (any RPC caller can crash the worker; `checked_request` does
  not cover these fields)
- Area: workers — `handle_benchmark` in all five worker modules
- Rank rationale: pass-2 worker-internals finding; pure input-validation hole,
  CPU-reproducible.

## Technical description

All five `handle_benchmark` handlers share the unguarded shape (`director.py:
340-341,357` shown; identical in the GPU workers plus `max(peaks)`/`sum(peaks)/
len(peaks)` on the same empty list):

```python
warmup = int(payload.get("warmup", 1))
measured = int(payload.get("measured", 3))
...
mean = sum(walls) / len(walls)
```

None validates `measured > 0` or `warmup >= 0`. `{"warmup":-2,"measured":1}` →
`range(-1)` empty → same crash.

## Why this is an issue

Any RPC caller can crash a worker process with a two-field payload: `benchmark`
is exposed on all five workers and `checked_request` does not cover these
fields, so a typo or hostile caller turns a measurement op into an untyped
`ZeroDivisionError`. It also poisons any harness that computes counts
dynamically — a zero-length measurement set crashes instead of returning an
empty report.

## Evidence (code experiment, host CPU-only, verified by orchestrator 2026-09-25)

```
$ PYTHONPATH=Voyage python3 -c "from voyage.workers import director as d; d.handle_benchmark({'warmup':0,'measured':0})"
    mean = sum(walls) / len(walls)
           ~~~~~~~~~~~^~~~~~~~~~~~
ZeroDivisionError: division by zero
```

Director reproduces on host; video/audio paths are identical (need CUDA only for
the render itself, not the validation hole). Sites: `video_longlive.py:972-973,
1011,1020-1021`; `video_ltxv.py:742-743,771,781-782`; `video_causvid.py:996-997,
1025,1036-1037`; `audio_acestep.py:118-119,141,150-151`; `director.py:340-341,357`.

## Reproduction

Send `{"warmup":0,"measured":0}` to any `benchmark` op.

## Source references

- Files/lines above.

## Resolution candidates

Validate at the top of each `handle_benchmark`: `if warmup < 0 or measured <= 0:
raise ValueError(...)`; or return a 400-style error before the loop. Unify via
the shared helper from 019's `video_common.py` when it exists.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 worker sweep; director case reproduced live.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  all cited lines live (`director.py:340-341,357`, `video_longlive.py:972-973,
  1011,1020-1021`, `video_ltxv.py:742-743,771,781-782`,
  `video_causvid.py:996-997,1025,1036-1037`, `audio_acestep.py:118-119,141,
  150-151` — all match); re-ran director probe (`ZeroDivisionError` confirmed).
- Open: validate + test per worker.
- 2026-09-25: FIXED. Shared `validate_benchmark_counts(warmup, measured)`
  in `voyage/workers/loop.py:62` (`warmup < 0 or measured <= 0` →
  `ValueError`); called first in all five `handle_benchmark` handlers —
  `director.py:374`, `audio_acestep.py:152`, `video_longlive.py:1036`,
  `video_ltxv.py:768`, `video_causvid.py:1005` — before any session/stack
  check or state mutation, so bad counts fail fast on CPU with a typed
  error instead of `ZeroDivisionError` (and `warmup=0, measured>0` stays
  valid per the existing stage-shape tests). Reordered `audio_acestep`
  (validation before `_require_stack`), `video_ltxv`/`video_causvid`
  (validation before `_SESSION` check and, for ltxv, before the tail-state
  save). Tests: `Voyage/tests/test_benchmark_counts.py` (7 tests: helper
  accept/reject, per-worker rejection, stack-not-loaded/session-untouched
  ordering). Gates: `ruff check` clean, `ruff format --check voyage tests`
  clean, `mypy voyage` strict clean (40 files), `pytest` 472 passed.
