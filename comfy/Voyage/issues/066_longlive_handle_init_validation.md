# 066 — `video_longlive.handle_init` accepts any `latent_shape`/`local_attn_size`/`sink_size` (CausVid validates, LongLive doesn't)

- Status: resolved (fixed 2026-09-25)
- Severity: medium (late `IndexError` / negative-size CUDA alloc far from the bad
  input)
- Area: LongLive worker RPC validation — `voyage/workers/video_longlive.py:873,
  880-892`
- Rank rationale: pass-2 worker finding; the sister worker has the validators to
  copy.

## Technical description

```python
checked_request(payload, models_dir=str, device=str, latent_shape=list)
...
latent_shape = [int(v) for v in raw_shape]
...
"local_attn_size": int(payload.get("local_attn_size", 8)),
"sink_size": int(payload.get("sink_size", 8)),
```

No length/positivity check; downstream uses `latent_shape[1]`
(`build_longlive_config:194-195`), `shape[2..4]` (`begin_sequence:505-509`,
`append_block:535-536`), and `local_attn_size*frame_seq_length`
(`_install_pos_only_caches:248-254`, where `-1` is a sentinel and `0`/negative
yields zero/negative `torch.zeros` dims). Contrast
`voyage/workers/video_causvid.py:167-174,114-124` (`validate_latent_shape`
requires 5 positive ints; `validate_overlap_frames` positive/divisible).

## Why this is an issue

Bad shapes and sizes pass the RPC boundary and explode deep inside session
build or forward — an `IndexError` on `shape[1]` or a negative-size CUDA alloc
far from the caller's mistake, with no hint which field was wrong. The sister
worker already carries the exact validators to copy, so this is a known-good
pattern left unapplied rather than a design question.

## Evidence

```
$ rg -n "validate_" Voyage/voyage/workers/video_longlive.py
(no hits — only list() copies; re-run 2026-09-25)
$ PYTHONPATH=Voyage python3 -c "from voyage.workers.video_causvid import validate_latent_shape; validate_latent_shape([1,2])"
ValueError: CausVid latent shape must be 5 positive ints [B, T, C, H, W] (got [1, 2])
```

LongLive has no equivalent; `init {"latent_shape":[1,2]}` passes
`checked_request` then fails later inside session build.

## Reproduction

`init {"latent_shape":[1,2],"local_attn_size":0}` passes the boundary, fails
later inside session build/forward.

## Source references

- Files/lines above.

## Resolution candidates

Port a `validate_latent_shape`-equivalent + `local_attn_size>0` (allow the `-1`
sentinel explicitly if intended) + `sink_size>=0` + capacity-floor check
(`sink+block<=local`, already documented in `build_longlive_config` docstring)
into `handle_init`.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 worker sweep with live contrast probe.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`video_longlive.py:873` `checked_request`, `:880-892`
  shape/size handling, `:194-195`/`:505-509`/`:535-536`/`:248-254` consumers,
  `video_causvid.py:114-124`/`168-176` validators — all match); re-ran contrast
  probe (CausVid rejects `[1,2]`; LongLive has no `validate_*` hit).
- Open: implement + tests.
- 2026-09-25: FIXED in `voyage/workers/video_longlive.py`: ported
  `validate_latent_shape` (`:45`, 5 positive ints, causvid mirror),
  `validate_local_attn_size` (`:56`, positive or the -1 full-context
  sentinel), `validate_sink_size` (`:64`, >= 0), and
  `validate_attn_capacity` (`:70`, sink + one block must fit local,
  skipped for -1) using the new `NUM_FRAME_PER_BLOCK = 8` constant
  (`:41`, also reused by `build_longlive_config` at `:243` so the floor
  and the config share one source). `handle_init` (`:931-947`) parses
  and validates all four before the CUDA check (`import torch` moved
  below the pure checks, mirroring causvid's fail-fast ordering), so bad
  dims raise `ValueError` on CPU instead of `IndexError`/negative CUDA
  allocs deep in session build. Tests:
  `Voyage/tests/test_longlive_init_validation.py` (shape/attn/capacity
  units + `handle_init` rejection without GPU). Gates: `ruff check`
  clean, `ruff format --check voyage tests` clean, `mypy voyage` strict
  clean (40 files), `pytest` 472 passed.
