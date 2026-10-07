# 303 — SFX.md + ARCHITECTURE.md 2-GPU pairing is inverted vs code

Severity: MEDIUM-HIGH (pass-2 DESIGN-docs drift sweep).

## Technical description

Both docs say video-augment runs on `cuda:0` and MMAudio SFX on `cuda:1`. Code does the
opposite: model pass pinned to `cuda:1` (2060), SFX on `cuda:0` (4060).

## Rationale

Operator following the docs co-locates the wrong stacks (FILM-era 5.9 GiB thinking vs
RIFE-era 0.65 GiB reality); OOM triage goes to the wrong card.

## Live evidence

- Docs side: `docs/SFX.md:138-140` — "Video-augment chunks run on `cuda:0` while the
  MMAudio SFX stack … renders on `cuda:1`"; `docs/ARCHITECTURE.md:73-75` — identical
  sentence, cites `voyage/augment.py:14-18`.
- Code side: `voyage/augment.py:19-22` — "model-pass chunks are pinned to cuda:1 via
  `model_pass_devices` while the MMAudio SFX stack … renders on cuda:0";
  `voyage/augment.py:514-532` (`model_pass_devices` returns `(AUGMENT_DEVICE_SECONDARY,)`
  = cuda:1 on 2-GPU), `voyage/augment.py:534-571` (`upscale_pass_devices`/
  `interp_pass_devices` same); `voyage/config.py:158,176,199+` — all CUDA rows pair
  `sfx_device="cuda:0"`.
- Command: `grep -n "cuda:0.*cuda:1\|Video-augment chunks" docs/SFX.md docs/ARCHITECTURE.md`
  → both hit; `sed -n '19,22p' voyage/augment.py` → opposite assignment.

Repro: read the two doc paragraphs, then `model_pass_devices(devices=("cuda:0","cuda:1"))`
— returns `('cuda:1',)`.

## Source refs

`docs/SFX.md:138-140`; `docs/ARCHITECTURE.md:73-75`; `voyage/augment.py:19-22,514-571`;
`voyage/config.py:158,176,199+`.

## Online sources

- None (code is the authority; docs contradict it).

## Fix candidates

- Swap the device names in both doc paragraphs (augment→cuda:1, SFX→cuda:0); fix the
  `augment.py:14-18` line ref (actual contract now at `augment.py:19-22,514+`); same
  inversion appears in `docs/SFX.md:146` run.sh sentence — update together.

## Evaluation (2026-10-07)

Claim CONFIRMED live. Code side: `voyage/augment.py:18-26` module
docstring pins model-pass to cuda:1 / SFX to cuda:0;
`model_pass_devices` (`:514-531`), `upscale_pass_devices` (`:534-550`),
`interp_pass_devices` (`:553-571`) all return
`(AUGMENT_DEVICE_SECONDARY,)` = cuda:1 on 2-GPU; every registry CUDA
row pairs `sfx_device="cuda:0"` (`voyage/config.py:158,176,199+`,
ltxv/causvid/ltx25/ltx23); SFX worker default is cuda:0
(`voyage/workers/sfx_mmaudio.py:51` — filed ref `:40` had drifted).
Docs side: `docs/SFX.md:138-140` and `docs/ARCHITECTURE.md:73-75`
both had the pairing backwards, with stale `augment.py:14-18` refs.
The `SFX.md:146` run.sh sentence was stale a second way: it cited
`scripts/run.sh:69-70` with a `ltxv|causvid|acestep|mmaudio` list and
`[video]`/`[audio]`/`[sfx]` section syntax — `run.sh` actually sniffs
the stored manifest backend (video/audio/sfx + director), selects
`voyage-ltx:latest` for ltx25/ltx23 and `voyage-video:latest` otherwise
(`Voyage/scripts/run.sh:106-196`), and the `_require_cuda_stack` ref
had moved to `voyage/cli_planning.py:183`.

## Progress log

- 2026-10-07: rewrote both CUDA-pairing paragraphs (devices swapped,
  refs corrected to `augment.py:18-26` + `:514-571` +
  `config.py:158,176,199+` + `sfx_mmaudio.py:51`); rewrote the run.sh
  sentence (manifest sniff, ltx image split, llama-director CUDA need,
  explicit-env-wins; ref `cli_planning.py:183`). No code touched.

## Resolution (2026-10-07)

RESOLVED docs-only. Files: `Voyage/docs/SFX.md`, `Voyage/docs/ARCHITECTURE.md`.
Verify: `grep -n "Video-augment chunks" Voyage/docs/SFX.md Voyage/docs/ARCHITECTURE.md`
→ cuda:1/cuda:0 everywhere; code refs match live lines above.
