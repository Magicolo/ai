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

## Progress log

- 2026-09-30 (Group E2): evaluated live first. Premise CONFIRMED as-read on every leg: `augment_worker.py:1-12` still declares the spike quarantine (RRDBNet = x4 family, anime_6B SRVGG "needs its own loader — follow-up"; `FilmNetMini` shape-mismatch → `ModelCompatibilityError`); `_load_rrdb_net`/`_load_film_net` still `strict=True` (074 landed only the loader *mechanics* — suffix branch, size/manifest pre-checks — not the shape contract); `model_registry.py` still pins `film` + `realesrgan-anime` weights; `models_ensure.py:83` defaults `augment_enabled=True` with the film+realesrgan extend at `:105-113`. `model_registry.py` + `models_ensure.py` are out of this group's scope and the port itself (full FILM + SRVGG loader) is a GPU-box task — so this is logged as a residual with exact state, no code touched. No caller consumes the weights in production (caller grep still empty outside the worker + tests), so severity stays MEDIUM.

## Resolution

- Verdict: RESIDUAL — fully verified, not ownable from this group's files (registry specs, ensure defaults, and the upstream port are all outside `workers/*` + `audio/*` + `vision/*` + `bench.py` + `doctor.py`).
- Files changed: none (this issue file only).
- Test evidence: live reads 2026-09-30 (cites above); no test added — the contract test (fix candidate 3: every `film`/`realesrgan-anime` FileSpec round-trips through its loader) must fail until the port lands, and a knowingly-failing test is not a gate asset. Existing `test_augment_weight_loading` pins the current `ModelCompatibilityError` behavior.
- DESIGN proposal (quoted text only, for the DESIGN owner — augment track): "Until the full upstream FILM port + SRVGG anime-6B loader land, `augment_enabled` should default opt-in-gated (or ensure gated on loader readiness) so default CUDA runs stop fetching weights the spike loaders structurally reject; the weight↔loader round-trip contract test pins the port's completion."
- Residuals (for the registry/augment owners, precise): (1) land the FILM port + SRVGG loader, re-verify both pinned weights load; (2) `voyage/models_ensure.py:83,105-113`: flip/gate the `augment_enabled` default meanwhile; (3) add the FileSpec round-trip contract test with the port.

## Progress log (2026-09-30, this pass — IMPLEMENTED, ESRGAN leg)

- TDD red first (in-container `voyage-video:latest`, CPU-only, models
  volume mounted ro, no host pip): new
  `tests/test_augment_contract_166.py` written before the loader — the
  provisioned ESRGAN leg failed as documented (`_load_rrdb_net` on the
  real `RealESRGAN_x4plus_anime_6B.pth` → `ModelCompatibilityError`
  "needs its own loader"), while the mapping + stub-rejection legs
  passed. `voyage:latest` is slim (no torch/safetensors — verified
  `find_spec` None for both), so torch legs can only execute in
  `voyage-video` (torch 2.8.0+cu128, CUDA unavailable → CPU); the
  committed provisioned legs skip loudly there-absent instead of failing.
- Weight forensics (same container, read-only): the pinned pth unwraps
  one level (`params_ema` → 192 tensors) into an upstream RRDB layout —
  `conv_first`, `body.0-5` × (`rdb1/2/3` × `conv1-5`, 32ch in / 64ch
  out, growth 32), `conv_body`, `conv_up1/up2/hr`, `conv_last` — i.e.
  xinntao `RRDBNet(num_feat=64, num_block=6, num_grow_ch=32)` naming,
  NOT the vendored classic (`conv_up_first/...`, `body.N.blocks.M`).
  Activation math (LeakyReLU 0.2, 0.2 residual scaling, two x2 nearest
  stages) is identical either way. FILM probed too: 82 safetensors keys
  under `extract`/`fuse`/`predict_flow` — a full upstream port, out of
  scope here.
- Green after: `voyage/workers/augment_worker.py` gained
  `_build_upstream_rrdb_net(num_blocks)` (upstream names, depth measured
  from the state dict), `_unwrap_esrgan_state` (single-key
  `params_ema`/`params` strip, flat states pass through), and
  `_upstream_block_count` (contiguous `body.0..N` + upconv-head check);
  `_load_rrdb_net` builds upstream when detected, vendored classic
  otherwise (byte-identical fallback — `_build_rrdb_net()` signature
  untouched), `strict=True` both ways; module docstring + RRDBNet bullet
  updated (ESRGAN leg DONE, FILM stand-in unchanged). New error names
  both layouts instead of the stale "needs its own loader".
- Gates (in-container): `ruff check` + `ruff format --check` + `mypy`
  strict clean on `voyage/workers/augment_worker.py` +
  `tests/test_augment_contract_166.py`; `voyage-video` driver green
  (strict load → `_UpstreamRRDBNet`, block count 6, synthetic
  `upscale_frames` scale 2 on 8×8 CPU → `(3, 16, 16)` finite in
  [0.12, 0.80]; FILM still `ModelCompatibilityError` with the port
  noted); slim `voyage:latest` pytest green:
  `test_augment_contract_166 + weight_loading + augment_models +
  rhythm + beat_ties` = 52 passed, 2 skipped (provisioned legs skip by
  design — proven in `voyage-video`), neighbors (`runner + config +
  plan + preset_fanout + registry_pins + worker_perf_rank2`) = 145
  passed, 4 skipped. One test-shape fix on the way: bicubic downscale
  (scale 2 of the native x4 pass) can ring ±0.05 past the [0, 1] clamp,
  so the contract bounds overshoot (`[-0.1, 1.1]`) instead of the exact
  range (native scale-4 clamp behavior untouched).

## Resolution (2026-09-30, this pass)

- Verdict: ESRGAN leg IMPLEMENTED (the weight IS provisionable —
  `~/.cache/voyage-models/realesrgan/RealESRGAN_x4plus_anime_6B.pth`,
  17,938,799 bytes — and now loads strict end to end); FILM leg remains
  an explicit remainder (see contract below). Files changed:
  `voyage/workers/augment_worker.py` (loader only, no registry/ensure
  touch), `tests/test_augment_contract_166.py` (new, 5 tests), this
  issue file. Test evidence: TDD red→green above; per-file gates green.
- DESIGN proposal (quoted text only, for the DESIGN owner — §§56-57, to
  replace the "not executable end-to-end" note): "The Real-ESRGAN
  anime-6B weight loads strict into the upstream-named RRDB builder at
  its measured depth (6 body blocks) and upscales end to end; the
  augmentation floor is executable for upscale. FILM stays fail-loud
  until the full upstream port lands (contract below) — default CUDA
  runs still fetch a FILM weight no loader accepts."
- Residuals (explicit remainder — the FILM port contract): port the
  upstream frame-interpolation net that owns the pinned
  `film_net_fp16.safetensors` (82 keys: `extract`/`fuse`/`predict_flow`
  prefixes, fp16) so that `_load_film_net` decodes via safetensors and
  `load_state_dict(strict=True)` succeeds; then flip
  `test_film_provisioned_weights_fail_loud_with_port_note` from the
  `ModelCompatibilityError`-with-"port" assertion to a strict-load +
  synthetic `interpolate_pair` round-trip. The port must keep: no Comfy
  imports (worker images carry no ComfyUI tree), fp16-on-CUDA /
  fp32-elsewhere, the OOM-halving call shape `(B, 2, C, H, W)` +
  `moment`, `FILM_MIN_BYTES` pre-check, and torch-free
  `NotImplementedError` when weights are absent. `models_ensure.py`
  default-gating (candidate 2) stays with the registry owner.
