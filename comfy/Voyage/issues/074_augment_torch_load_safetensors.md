# 074 — Augment worker `torch.load`s `.safetensors` paths with no format gate and no hash check

- Severity: MEDIUM
- File: `Voyage/voyage/workers/augment_worker.py:301-330` (`_load_rrdb_net`, `_load_film_net` → `torch.load(..., weights_only=True)`), `:54-65` (`_LOAD_ERRORS` taxonomy), `:77-90` (`_require_weights` = exists + non-empty only)
- Area: workers — finalize augmentation loaders

## Description

Both loaders call `torch.load` unconditionally, yet one of the two registered weight files is `film_net_fp16.safetensors` (registry `FILM_FILE`, `voyage/model_registry.py:287`) — a safetensors file, not a pickle. The module docstring (`:61-64`) acknowledges the mismatch ("a safetensors file passed to torch.load… propagate as worker errors"). Consequences: (a) a swapped-in pickle at the FILM path *will* be unpickled (only `weights_only=True` stands between the attacker and code exec — good but single-layer); (b) `UnpicklingError` is deliberately outside `_LOAD_ERRORS`, so a corrupt/mismatched file surfaces as a generic worker error rather than the actionable `ModelCompatibilityError`; (c) no sha/size check precedes the load — `_require_weights` accepts any non-empty file.

## Rationale

Per PyTorch docs, `torch.load` "uses an unpickler under the hood. Never load data from an untrusted source"; safetensors exists precisely so weight loads never invoke the pickle machine (HF diffusers guide). Format-branching + pre-verify is the standard control.

## Live evidence

Re-verified 2026-09-30 live:

```
Voyage/voyage/workers/augment_worker.py:54-65:
_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError, OSError, ValueError, TypeError, EOFError,
)
"""torch.load / load_state_dict failures meaning "weights unusable here".
Exotic failures outside this tuple (e.g. unpickling errors from a
safetensors file passed to torch.load) propagate as worker errors instead.
"""
Voyage/voyage/workers/augment_worker.py:77-90 (_require_weights):
    path = Path(weights)
    if not path.exists() or path.stat().st_size == 0: raise NotImplementedError(...)
    return path     # exists + non-empty only
Voyage/voyage/workers/augment_worker.py:301-307 (_load_rrdb_net):
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
Voyage/voyage/workers/augment_worker.py:317-323 (_load_film_net):
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
Voyage/voyage/model_registry.py:284-287:
FILM_HF_REPO = "Comfy-Org/frame_interpolation"
FILM_HF_REVISION = "219da3c9d8c357ceaf457fc1d5932c6e861b8dee"
FILM_FILE = "film_net_fp16.safetensors"
```

`verify_film_models`/`verify_realesrgan_models` check only `*_MIN_BYTES`. Credit: both `torch.load` calls do pass `weights_only=True`.

## Repro

Point `interpolate_pair`/`upscale_frames` at the real FILM `.safetensors` → `torch.load` failure path (worker error, not compatibility error); swap in a same-size non-weight blob → accepted by `_require_weights`, fails deep in `torch.load`.

## Fix candidates

1. Branch on suffix: `safetensors.torch.load_file` for `.safetensors`, `torch.load(weights_only=True)` for `.pth`.
2. Sha-verify against manifest/recorded hash before load (extends issue 071 to the augment pair).
3. Include `UnpicklingError` (or map it) in `_LOAD_ERRORS` → `ModelCompatibilityError`.

## Refs

- Overlaps with 071 (pre-load hash verification — fix candidate 2 extends 071 to the augment pair).
- `torch.load` docs ("uses an unpickler… Never load data from an untrusted source… `weights_only`") — https://docs.pytorch.org/docs/stable/generated/torch.load
- "safetensors is a secure alternative to pickle" — https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors
- Safetensors format rationale — https://github.com/huggingface/safetensors

## Progress log

- 2026-09-30: premise re-verified against live
  `Voyage/voyage/workers/augment_worker.py` (both loaders unconditional
  `torch.load(weights_only=True)`; `_LOAD_ERRORS` five-tuple without
  `UnpicklingError`; `_require_weights` exists + non-empty only) and
  `Voyage/voyage/model_registry.py` (071 already records `checkpoint_sha256`
  + `checkpoint_file` for film/realesrgan and checks them in `verify_model`;
  the remaining gap is worker load-time, mirroring `video_longlive` /
  `video_causvid` which `verify_checkpoint_against_manifest` before
  `torch.load`). Dockerfile.video:157-168 already documents the intended
  split (FILM via safetensors, ESRGAN via torch.load weights_only) with
  `safetensors==0.8.0` in the video image — the worker just never branched.
- 2026-09-30: TDD — six failing-first tests in
  `tests/test_augment_weight_loading.py` (`UnpicklingError` in
  `_LOAD_ERRORS`, `.safetensors` via safe loader, `.pth` via weights-only,
  corrupt pickle → compatibility, size floor, manifest mismatch; 5 failed +
  1 missing-helper error pre-fix). All green post-fix (10/10 with the 072
  file in the same run).
- 2026-09-30: gates (scoped) — ruff + format + mypy strict clean on
  `voyage/workers/augment_worker.py` + `tests/test_augment_weight_loading.py`
  (one `import-not-found` ignore for `safetensors.torch`, absent from the slim
  image by design — the video image carries it); scoped suite 107 passed /
  3 skipped (same set as the 072 log). Full `gates.sh` is the orchestrator's
  job. Two ruff findings fixed on sight (TRY004 non-dict decode → TypeError,
  TRY203 bare re-raise removed).
- NOTE (issue 166): the vendored RRDBNet/FilmNetMini architectures
  structurally do NOT match the pinned Real-ESRGAN/FILM weights
  (`strict=True` raises `ModelCompatibilityError` by construction — module
  docstring :9-19 states it). This fix does NOT claim the pinned weights now
  load: the format branch + pre-verify gates are correct and testable on
  synthetic blobs, but end-to-end loading of the pinned weights awaits the
  upstream FILM port + SRVGG loader (166 follow-up).

## Resolution

- Fix candidate 1 applied (format branch): new `_load_state_dict` helper —
  `.safetensors` (case-insensitive suffix) decodes via
  `safetensors.torch.load_file` (never the pickle machine; decode failures
  normalize to `ValueError`, missing package raises `ImportError` unchanged
  as an environment issue), every other suffix via
  `torch.load(weights_only=True)`; non-dict payloads raise `TypeError`.
  Both `_load_rrdb_net` / `_load_film_net` route through it (their
  `import torch` preamble is gone — the helper owns loader imports).
- Fix candidate 2 applied (pre-verify, torch-free, before any loader):
  `_verify_weights_size` enforces the registry floor (`FILM_MIN_BYTES` /
  `REALESRGAN_ANIME_MIN_BYTES`, lazily imported so the worker top level
  stays torch-free) and `_verify_weights_manifest` walks up to the nearest
  `manifest.json` and checks the 071 shapes (`checkpoint_sha256` +
  `checkpoint_file`, plus `checkpoint_shas` dicts) for exactly this file —
  mismatch raises `ModelCompatibilityError` before any decoder runs; no
  manifest / no entry / unreadable manifest passes through (ingest-time
  constants + `verify_model` stay the closed gates there, torn manifests are
  `validate_run` territory).
- Fix candidate 3 applied (taxonomy): `pickle.UnpicklingError` joins
  `_LOAD_ERRORS`, so corrupt pickles map to `ModelCompatibilityError` at the
  loader boundary instead of surfacing as generic worker errors; the stale
  docstring claiming safetensors-via-torch-load is rewritten to describe the
  branch.
- Residuals: pinned-weight end-to-end stays red by construction (166 —
  `strict=True` shape mismatch); the worker smoke tests needing real torch
  still skip in slim (3 skips, by design); `voyage/audio/mmaudio_sfx.py` and
  the registry need no change for this issue (its `.pth` vocoder weight
  already loads `weights_only=True`; vocoder snapshot hashing is 072/071
  territory).
