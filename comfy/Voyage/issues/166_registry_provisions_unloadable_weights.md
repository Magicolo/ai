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

## Progress log (2026-09-30, this pass — IMPLEMENTED, FILM leg)

- TDD red first (ephemeral pytest in `voyage-video:latest`, CPU-only,
  models volume mounted ro, no host pip): the two new FILM contract tests
  failed as designed (`_load_film_net` on the real
  `film_net_fp16.safetensors` → `ModelCompatibilityError` from the
  `FilmNetMini` stand-in, full 82-key unexpected-keys list in the error),
  4 passed. `voyage-video` carries no pytest (`pip install pytest` runs
  ephemeral per invocation — container overlay only, host untouched);
  slim `voyage:latest` legs skip-by-design (no torch/safetensors there).
- Weight forensics: all 82 safetensors keys dumped live (`extract` 16 /
  `fuse` 26 / `predict_flow` 40, every tensor fp16) and matched 1:1
  against the host Comfy checkout's upstream port
  (`Comfy/comfy_extras/frame_interpolation_models/film_net.py`, read-only
  reference, never imported — worker images carry no ComfyUI tree). Every
  in-channel (fuse 1930/2442/1162/522/202, predictors 1920/896/384/128)
  re-derives from the ctor math, so the port is fully determined — no
  invented shapes, no hash invented (none recorded).
- Green after: `_build_film_net()` returns a plain-torch upstream FILM
  port (`_FilmConv` even-pad + LeakyReLU 0.2, subtree extractor,
  cross-level feature pyramid, 4-predictor coarse-to-fine flow, fusion
  decoder) whose submodule nesting reproduces the pinned key layout
  exactly, so `_load_film_net` `strict=True` succeeds on the provisioned
  68,882,302-byte file (safetensors branch + `FILM_MIN_BYTES` floor +
  manifest check all kept). Call shape stays the worker's own
  `(B, 2, C, H, W)` pairs + float moment, so `interpolate_pair` /
  `interpolate_triplet` / `_run_stacked` OOM-halving are untouched;
  precision stays fp16-CUDA / fp32-CPU via `_prepare_model` (the warp
  upcasts to float32 and back, as upstream). Pyramid depth clamps to the
  input-feasible count (upstream runs 7 levels, needing sides >=64px);
  sides below `FILM_MIN_SIDE` (8) fail loud with `ValueError`.
  `FILM_MINI_CHANNELS` + `_FilmNetMini` deleted (zero other references
  in-tree — verified by grep before removal).
- Evidence (provisioned weights, CPU fp32): strict load, 82 keys with the
  exact extract/fuse/predict_flow prefixes, all fp16 on disk; synthetic
  16px gradient pair at moment 0.5 → mid (3, 16, 16), all finite,
  [0.065, 0.914], 127 ms including load; endpoints deliberately NOT
  asserted near inputs (t0-vs-before mean-abs 0.31 — the fuse head is a
  learned blend, not an identity — so the tests pin shape/finite/loose
  range only); triplet (B=2) both mids finite; random-init 16px →
  (1, 3, 16, 16) (keeps the untouched `test_augment_runner` smoke green);
  4px pair → clear `ValueError` (new floor test, torch-only, no weights).
- Gates: slim `voyage:latest` — `ruff check` + `ruff format --check` +
  `mypy` strict clean on both touched files; pytest 125 passed, 7
  skipped (provisioned/torch legs skip loudly). `voyage-video` torch
  legs: 165 passed (contract 7 + runner/weight_loading/models/config/
  plan + rhythm/beat-ties + media-unified + e2-157_193 neighbors). The
  `ruff format` reflow stayed inside the new hunks (diff hunks verified
  own-scope only). CUDA fp16 executes by construction only
  (`_prepare_model` halves; no GPU on this box).

## Resolution (2026-09-30, this pass — FILM leg)

- Verdict: FULL PORT (both 166 legs DONE — ESRGAN batch 10 + FILM this
  pass; the "not executable end-to-end" premise is retired for upscale
  AND interpolate). Files changed: `voyage/workers/augment_worker.py`
  (FILM_* constants + ported builder + loader/docstring updates; RRDB
  leg untouched), `tests/test_augment_contract_166.py` (fail-loud test
  flipped to strict-load + synthetic round-trip + min-side floor), this
  issue file. Test evidence: TDD red→green above; per-file gates green
  in both images.
- DESIGN proposal (quoted text only, for the DESIGN owner — §§56-57, to
  replace the "not executable end-to-end" note): "Both pinned augment
  weights now strict-load and run end to end — Real-ESRGAN anime-6B
  upscales and FILM interpolates (fp16 on CUDA, fp32 on CPU, OOM-halving
  preserved) — so the augmentation floors are executable; the remaining
  step is wiring finalize weights→loader, plus a live-GPU fp16 numeric
  and quality eyeball the next time a CUDA box is available."
- Residuals: (1) `models_ensure.py` default-gating (candidate 2) stays
  with the registry owner — default CUDA runs fetch both weights, which
  now load; (2) no production caller wires weights→loader yet
  (`voyage/augment.py` / `supervisor.py` never import the worker —
  orchestration track); (3) live CUDA fp16 numeric + quality eyeball
  open (CPU-only box here).

## Progress log (2026-10-01, batch 13 — resolution leg WIRED, inference handed off)

- Premises re-verified live FIRST (in-container `voyage:latest`,
  CPU-only, no host pip): SRVGG loader leg + full FILM port both hold
  (`_build_upstream_rrdb_net` + `_build_film_net` strict-load the pinned
  weights per the batch-10/11 logs); caller grep still empty — no
  production module resolves registry paths into
  `upscale_frames`/`interpolate_pair` (the remaining step named in the
  FILM resolution). `voyage/supervisor.py` untouched throughout (foreign
  hunks this pass); no longlive2 line touched.
- TDD red first: new `tests/test_issue_166_resolve_weights.py` failed at
  collection (`ImportError: cannot import name 'AugmentWeights' from
  'voyage.augment'`), green after the implementation (6 passed,
  1 skipped — the provisioned load-through skips in slim: no torch, no
  `/models`; proven shape only, same skip contract as
  `test_augment_contract_166.py`).
- Wired: `voyage/augment.py` appended `AugmentWeights` (frozen dataclass:
  `film: Path | None`, `realesrgan: Path | None`) + `resolve_augment_weights`
  (`models_dir: Path | str -> AugmentWeights`, torch-free, lazy registry
  imports so module scope stays stdlib-only). A leg resolves only when
  its file exists at the registry-relative path with >= the registry
  floor bytes (`FILM_MIN_BYTES` / `REALESRGAN_ANIME_MIN_BYTES`) —
  missing/zero-byte/truncated read as not provisioned (mirrors
  `augment_worker._require_weights` torn-download rule); wrong
  `models_dir` type raises `TypeError`. Mid-pass fix: Real-ESRGAN pins
  moved to canonical `voyage.registry_realesrgan` by a concurrent split
  (same 082 pattern as `registry_film`) — verified live by grep, imports
  re-pointed at the canonical home (the `registry_records` re-export
  still works but is their in-flight surface). Own diff is purely
  additive (`git diff`: 0 removed lines in `voyage/augment.py`).
- Gates (in-container `voyage:latest`, CPU-only): `ruff check` +
  `ruff format --check` + `mypy` strict clean on both touched files
  (one `ruff --fix` + reflow cycle, own file only; the `pytest.skip`
  narrowing the host LSP flags is mypy-clean in-container).
  Neighbors green: 117 passed, 8 skipped across the 166/augment/pins/
  152/media-unified set. 6 failures in `test_augment_models.py` +
  `test_augment_config.py` collection are FOREIGN (all one root:
  `ImportError: cannot import name 'warn_if_deprecated_backend' from
  'voyage.config'` — another group's in-flight config/cli split,
  `config.py` carries ~100 of their uncommitted lines, zero mine).

## Resolution (2026-10-01, batch 13)

- Verdict: RESOLUTION LEG WIRED — any production caller can now map
  `config.video.models_dir` to loader-ready paths with graceful skip.
  Files changed: `voyage/augment.py` (+61, append-only),
  `tests/test_issue_166_resolve_weights.py` (new, 7 tests), this issue
  file. Test evidence: TDD red→green above; per-file gates green;
  neighbors green modulo the foreign config-split breakage (theirs, not
  this leg — this leg imports nothing from `config`/`cli`).
- DESIGN proposal (quoted text only, for the DESIGN owner — §§56-57, to
  extend the "both weights strict-load" note): "Finalize resolves its
  model pass through `resolve_augment_weights(config.video.models_dir)`:
  each leg is a loader-ready path or None when the weights are absent,
  and absent legs keep the ffmpeg fallback (default-off unless
  provisioned) — the registry-to-loader seam is production code, the
  chunk-scale inference call is the remaining slice below."
- Residuals (exact inference handoff — NOT forced, per the GPU/hot-file
  rule): the model-pass chunk worker does not exist yet — writing it
  needs GPU semantics this box cannot prove (fp16-on-CUDA numerics +
  quality eyeball, cuda:0/cuda:1 device pairing, chunk-scale OOM-halving
  behavior past ~200 frames) and touches `finalize_run` in
  `voyage/media.py` (hot file; the 152 residual explicitly warns
  against media-side topology attempts alone). Exact next slice for the
  augment/GPU owner: (1) add the chunk worker (calls
  `resolve_augment_weights`, passes non-None legs to
  `augment_worker.upscale_frames` / `interpolate_pair` with
  `device=chunk.device`, keeps the ffmpeg encode for None legs);
  (2) thread it through `run_augment_chunks` behind an opt-in knob
  (ffmpeg stays the default — `finalize_run`'s current contract is
  unchanged by this pass); (3) prove on idle-CUDA with the provisioned
  `~/.cache/voyage-models/{frame_interpolation,realesrgan}` weights
  (both present on the host 2026-10-01) incl. the fp16 numeric + eyeball
  the FILM resolution left open.

## Progress log (2026-10-01, this pass — inference wiring evaluated, NOT forced)

- Premises re-verified live FIRST (host greps, no edits):
  `resolve_augment_weights` + `AugmentWeights` referenced only in
  `voyage/augment.py` + `tests/test_issue_166_resolve_weights.py`
  (the batch-13 seam holds); no production caller wires
  weights→loader (`voyage/augment.py` / `supervisor.py` never import
  the worker — only docstring/comment mentions in
  `cli_observe.py:318`, `workers/__init__.py:13`,
  `video_ltxv.py:150`); `finalize_run` (`voyage/media.py:1311`) +
  `FinalizeOptions` + `AugmentConfig` carry no model-pass knob
  (floors-only) — wiring needs a new knob plus a chunk worker that
  does not exist yet.
- CPU-provability assessment: the only fully CPU-testable slice on
  this box (slim `voyage:latest`, no torch, no `/models`) would be an
  opt-in knob whose absent-weights leg falls back to ffmpeg — i.e.
  scaffolding with zero observable behavior change (knob accepted,
  same bytes out), while the model legs need torch + provisioned
  weights (`voyage-video` CPU could load per batch-10/11, but
  chunk-scale OOM-halving past ~200 frames + fp16-on-CUDA numerics +
  quality eyeball need idle-CUDA per the FILM resolution). Per the
  brief (do NOT force hot `finalize_run` changes that cannot be
  proven) + the 152 no-media-topology-alone warning + YAGNI (a
  fallback-only knob nobody calls is dead plumbing), no `media.py` /
  `config.py` / CLI change made. TDD N/A — no behavior changed, so no
  failing test written (a knowingly-no-op knob test is not a gate
  asset).
- No code change: `voyage/` + `tests/` untouched by this leg (own diff
  for the whole pass is exactly the 8 slow-marker lines in the two
  089 files, verified `git diff`).

## Resolution (2026-10-01, this pass)

- Verdict: BLOCKED (CPU-only box) — recorded with exact handoff,
  nothing forced. Files changed: none (this issue file only).
  DESIGN proposals: none (the batch-13 §§56-57 "both weights
  strict-load" note stands; no new DESIGN claim owed).
- Residuals (exact inference handoff for the augment/GPU owner —
  unchanged from batch 13, still current): (1) add the chunk worker
  (calls `resolve_augment_weights`, passes non-None legs to
  `augment_worker.upscale_frames` / `interpolate_pair` with
  `device=chunk.device`, keeps the ffmpeg encode for None legs);
  (2) thread it through `run_augment_chunks` behind an opt-in knob
  (ffmpeg stays the default — `finalize_run`'s current contract is
  unchanged until then); (3) prove on idle-CUDA with the provisioned
  host weights incl. the fp16 numeric + eyeball the FILM resolution
  left open.

## Progress log (2026-10-01, this pass — inference wiring evaluated, NOT forced)

- Premises re-verified live FIRST (host greps, no edits):
  `resolve_augment_weights` + `AugmentWeights` referenced only in
  `voyage/augment.py:402,417` + `tests/test_issue_166_resolve_weights.py`
  (the batch-13 seam holds); no production caller wires weights→loader
  (`voyage/augment.py` / `voyage/supervisor.py` never import the worker
  — only docstring/comment mentions in `cli_observe.py:318`,
  `workers/__init__.py:13`, `video_ltxv.py:150`); `finalize_run`
  (`voyage/media.py:1311`) + `FinalizeOptions` (`voyage/media.py:1110`)
  + `AugmentConfig` (`voyage/config.py:403`) carry no model-pass knob
  (floors-only) — wiring needs a new knob plus a chunk worker that does
  not exist yet. `voyage/augment.py` untouched by this pass (own diff
  for 166 is this issue file only).
- CPU-provability assessment: the only fully CPU-testable slice on this
  box (slim `voyage:latest`, no torch, no `/models`) would be an opt-in
  knob whose absent-weights leg falls back to ffmpeg — i.e. scaffolding
  with zero observable behavior change (knob accepted, same bytes out),
  while the model legs need torch + provisioned weights (chunk-scale
  OOM-halving past ~200 frames + fp16-on-CUDA numerics + quality eyeball
  need idle-CUDA per the FILM resolution). Per the brief (do NOT force
  hot `finalize_run` changes that cannot be proven) + the 152
  no-media-topology-alone warning + YAGNI (a fallback-only knob nobody
  calls is dead plumbing; `voyage/media.py` is foreign here anyway), no
  `media.py` / `config.py` / CLI change made. TDD N/A — no behavior
  changed, so no failing test written (a knowingly-no-op knob test is not
  a gate asset). No `augment.py` seam slice landed: `resolve_augment_weights`
  already is the CPU-provable seam, and any further `augment.py`-only
  helper without a `media.py` caller would be dead code.
- No code change: `voyage/` + `tests/` untouched by this leg (own diff
  for the whole pass is `scripts/gates.sh` + 6 test-marker files for
  089, verified `git diff --name-only` — concurrent hunks in
  `issues/031,035,036,081,082,088,152`, `registry_records.py`,
  `test_generation_stack.py`, `test_prefetch_shutdown.py` left intact).

## Resolution (2026-10-01, this pass — re-evaluated)

- Verdict: BLOCKED (CPU-only box) — recorded with exact handoff,
  nothing forced. Files changed: none (this issue file only).
  DESIGN proposals: none (the batch-13 §§56-57 "both weights
  strict-load" note stands; no new DESIGN claim owed).
- Residuals (exact inference handoff for the augment/GPU owner —
  unchanged from batch 13, still current): (1) add the chunk worker
  (calls `resolve_augment_weights`, passes non-None legs to
  `augment_worker.upscale_frames` / `interpolate_pair` with
  `device=chunk.device`, keeps the ffmpeg encode for None legs);
  (2) thread it through `run_augment_chunks` behind an opt-in knob
  (ffmpeg stays the default — `finalize_run`'s current contract is
  unchanged until then); (3) prove on idle-CUDA with the provisioned
  host weights incl. the fp16 numeric + eyeball the FILM resolution
  left open.

## Progress log (2026-10-01, record-only maintenance pass — inference evaluated, NOT forced)

- Premises re-verified live FIRST (host greps, no edits):
  `resolve_augment_weights` + `AugmentWeights` referenced only in
  `voyage/augment.py:402,417` (+ `return AugmentWeights` `:454`) +
  `tests/test_issue_166_resolve_weights.py` (the batch-13 seam holds);
  no production caller wires weights→loader (`voyage/augment.py` /
  `supervisor.py` never import the worker — only docstring/comment
  mentions in `cli_observe.py:318`, `workers/__init__.py:13`,
  `video_ltxv.py:150`); `finalize_run` (`voyage/media.py:1311`) +
  `FinalizeOptions` (`voyage/media.py:1110`) + `AugmentConfig`
  (`voyage/config.py:403`) carry no model-pass knob (floors-only) —
  wiring needs a new knob plus a chunk worker that does not exist yet.
  `voyage/augment.py` untouched by this pass (own diff for 166 is this
  issue file only).
- CPU-provability assessment: the only fully CPU-testable slice on this
  box (slim `voyage:latest`, no torch, no `/models`) would be an opt-in
  knob whose absent-weights leg falls back to ffmpeg — i.e. scaffolding
  with zero observable behavior change (knob accepted, same bytes out),
  while the model legs need torch + provisioned weights (chunk-scale
  OOM-halving past ~200 frames + fp16-on-CUDA numerics + quality eyeball
  need idle-CUDA per the FILM resolution). Per the brief (do NOT force
  hot `finalize_run` changes that cannot be proven) + the 152
  no-media-topology-alone warning + YAGNI (a fallback-only knob nobody
  calls is dead plumbing; `voyage/media.py` is hot/foreign here), no
  `media.py` / `config.py` / CLI change made. TDD N/A — no behavior
  changed, so no failing test written (a knowingly-no-op knob test is
  not a gate asset). No `augment.py` seam slice landed:
  `resolve_augment_weights` already is the CPU-provable seam, and any
  further `augment.py`-only helper without a `media.py` caller would be
  dead code.
- No code change: `voyage/` + `tests/` untouched by this leg.

## Resolution (2026-10-01, record-only maintenance pass)

- Verdict: BLOCKED (CPU-only box) — recorded with exact handoff,
  nothing forced. Files changed: none (this issue file only).
  DESIGN proposals: none (the batch-13 §§56-57 "both weights
  strict-load" note stands; no new DESIGN claim owed).
- Residuals (exact inference handoff for the augment/GPU owner —
  unchanged from batch 13, still current): (1) add the chunk worker
  (calls `resolve_augment_weights`, passes non-None legs to
  `augment_worker.upscale_frames` / `interpolate_pair` with
  `device=chunk.device`, keeps the ffmpeg encode for None legs);
  (2) thread it through `run_augment_chunks` behind an opt-in knob
  (ffmpeg stays the default — `finalize_run`'s current contract is
  unchanged until then); (3) prove on idle-CUDA with the provisioned
  host weights incl. the fp16 numeric + eyeball the FILM resolution
  left open.
