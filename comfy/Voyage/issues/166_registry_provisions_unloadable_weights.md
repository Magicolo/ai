# 166 — Registry provisions FILM + Real-ESRGAN weights that `augment_worker` structurally cannot load; default `generate` downloads both anyway

- **Severity:** MEDIUM (wasted fetch + dead advertised floor today; becomes HIGH the moment finalize wires weights→loader — every default CUDA run would hard-fail on weights it just downloaded)
- **File:line:** `Voyage/voyage/workers/augment_worker.py:9-19` (vendored stand-ins docstring), `:301-313` (`_load_rrdb_net`, `strict=True`), `:317-330` (`_load_film_net`, `strict=True`); `Voyage/voyage/model_registry.py:284-309` (FILM/REALESRGAN pins), `:943-972` (spec file/check lists); `Voyage/voyage/models_ensure.py:93-105` (default-on ensure), `:181` ("the CLI never passes it today, so CUDA runs ensure them")
- **Area:** supply-chain / registry / worker — weight/loader contract

## Description

Both augment loaders `strict=True`-load into vendored spike architectures the module docstring explicitly says do not match the pinned weights:

- RRDBNet is the classic x4 (23 blocks × 64 feats, `RRDB_NUM_FEATURES = 64` / `RRDB_NUM_BLOCKS = 23`) while `RealESRGAN_x4plus_anime_6B.pth` is the compact SRVGG-6B variant ("needs its own loader — follow-up, not this spike", `augment_worker.py:11-14`).
- `FilmNetMini` is a flow-blender stand-in ("Official `film_net` weights will NOT load here (shape mismatch raises ModelCompatibilityError); the full upstream FILM port is follow-up", `:15-19`).

So the registry's `film` (66 MB, `Comfy-Org/frame_interpolation@219da3c9`, `film_net_fp16.safetensors`, `FILM_MIN_BYTES = 60_000_000`) + `realesrgan-anime` (18 MB, `amd/realesrgan-x4plus-anime-6b@b14ff5f8`, `RealESRGAN_x4plus_anime_6B.pth`, `REALESRGAN_ANIME_MIN_BYTES = 15_000_000`) specs (`model_registry.py:284-330`, specs near `:943-972`) advertise weights that unconditionally raise `ModelCompatibilityError` at use (`_load_rrdb_net:307-313`, `_load_film_net:323-330`, both `model.load_state_dict(state, strict=True)` wrapped into `ModelCompatibilityError`).

Meanwhile `required_specs(..., augment_enabled=True)` is the default and `ensure_models` downloads both on every CUDA `generate` missing them (`models_ensure.py:93-105` extends `film` + `realesrgan-anime` when `augment_enabled`; `:181` docstring: "the CLI never passes it today, so CUDA runs ensure them").

No production caller consumes them yet: `rg` over `Voyage/voyage/` for `FILM_REPO_PATH|REALESRGAN_SUBDIR|film_net|RealESRGAN` hits only `model_registry.py` + `workers/augment_worker.py` (+ tests and two `test_longlive.py` filename assertions); `voyage/augment.py` and `supervisor.py` never import the worker (verified 2026-09-30 — caller grep returns empty outside `workers/augment_worker.py`).

## Rationale (non-overlap)

- Today the cost is wasted fetch (~84 MB per fresh provision) plus a dead advertised floor — the "augmentation floors" feature shipped in `d7b4086` is not executable end-to-end.
- Severity is MEDIUM rather than HIGH only because nothing calls the loaders in production yet (no runtime failure path exists); it becomes HIGH the moment finalize wires weights→loader before the upstream port lands.
- 074 (`augment torch.load on .safetensors`) covers the loader's `torch.load` mechanics, not the weight/loader shape contract; 065/091 cover operator docs, not executability; 046/047 cover chunking/reload perf, not compatibility. No overlap.

## Live evidence (verified 2026-09-30)

```
$ sed -n '9,19p' Voyage/voyage/workers/augment_worker.py
Architectures are vendored minimal inline — do NOT import Comfy nodes
(the worker images carry no ComfyUI tree):
- RRDBNet: ... the compact `RealESRGAN_x4plus_anime_6B` SRVGG variant needs its own loader —
  follow-up, not this spike).
- FilmNetMini: ... Official `film_net` weights will NOT load here (shape mismatch raises ModelCompatibilityError);
  the full upstream FILM port is follow-up.
$ sed -n '301,330p' Voyage/voyage/workers/augment_worker.py
def _load_rrdb_net(...): ... state = torch.load(..., weights_only=True); model.load_state_dict(state, strict=True)
  except _LOAD_ERRORS as exc: raise ModelCompatibilityError(... compact anime_6B needs its own loader ...)
def _load_film_net(...): ... strict=True ... raise ModelCompatibilityError(... spike stand-in ... full upstream FILM port is follow-up)
$ sed -n '284,309p' Voyage/voyage/model_registry.py  # FILM/REALESRGAN pins + Track-C header
$ sed -n '93,105p;181,187p' Voyage/voyage/models_ensure.py  # default-on extend + "CLI never passes it today"
$ rg -ln "FILM_REPO_PATH|REALESRGAN_SUBDIR|film_net|RealESRGAN" Voyage/voyage/
Voyage/voyage/model_registry.py / Voyage/voyage/workers/augment_worker.py   # no caller
$ rg -n "augment_worker|_load_rrdb|_load_film|upscale_frames|interpolate_pair" Voyage/voyage/ --include="*.py" | grep -v "workers/augment_worker.py"
(empty — no production importer)
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c \
  "import voyage.model_registry as r; print([(f.repo_id,f.revision,f.filename) for f in r.MODEL_SPECS['film'].files + r.MODEL_SPECS['realesrgan-anime'].files])"
[('Comfy-Org/frame_interpolation', '219da3c9...', 'frame_interpolation/film_net_fp16.safetensors'),
 ('amd/realesrgan-x4plus-anime-6b', 'b14ff5f8...', 'RealESRGAN_x4plus_anime_6B.pth')]
```

## Repro (offline, CPU)

Provision the two specs (`voyage models download film/realesrgan-anime`), then `augment_worker.upscale_frames([tensor], <realesrgan .pth>)` / `interpolate_pair(a, b, <film .safetensors>)` → `ModelCompatibilityError` by construction (strict shape mismatch), while `models_ensure.required_specs` includes both specs for any CUDA backend by default.

```bash
sed -n '9,19p;301,330p' Voyage/voyage/workers/augment_worker.py  # contract states mismatch
sed -n '93,105p' Voyage/voyage/models_ensure.py  # default-on ensure
```

## Fix candidates

1. Land the full upstream FILM port + SRVGG anime-6B loader, then re-verify both pinned weights load.
2. Until then, flip `augment_enabled` default to opt-in (or gate ensure on loader readiness) so default runs stop fetching unloadable weights.
3. Add a contract test: every `FileSpec` filename under `film`/`realesrgan-anime` must round-trip through its worker loader on synthetic tensors (fails today — pins the follow-up).

## Refs

- `Voyage/voyage/workers/augment_worker.py:9-19,301-330`; `Voyage/voyage/model_registry.py:284-330,943-972`; `Voyage/voyage/models_ensure.py:93-105,181`
- safetensors-as-pickle-alternative rationale (https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors); ESRGAN RRDBNet architecture family (https://github.com/xinntao/Real-ESRGAN)

(End of file)
