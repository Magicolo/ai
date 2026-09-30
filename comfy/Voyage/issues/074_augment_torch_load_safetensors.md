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
