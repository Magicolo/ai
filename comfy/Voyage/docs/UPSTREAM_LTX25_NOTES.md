# UPSTREAM_LTX25_NOTES — LTX-2.5 integration notes

Pins live in code in `voyage/registry_ltx25.py` (the single source of
truth); this file mirrors them for humans. Revisit every note if a pin
moves.

Status: worker landed. `voyage/workers/video_ltx25.py` serves the
`ltx25` backend (in-process pinned ComfyUI, Mode-A 1216x704@24 quality
path, 121/25/96 accounting with 25-frame frozen-prefix continuation, `recovery.pt` §5.3 JSON tape with profile `ltxv25`
— see `recovery-profiles` line in BACKENDS.md; `generate --backend
ltx25`, `models download/verify ltx25`).

## Upstream pins

| Artifact | URL | Pinned revision |
|----------|-----|-----------------|
| Execution stack | [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) | `2f35f4a08176d993cded35dac3332be4f7287f41` |
| GGUF loader | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` (+ 31-line gemma4 patch vendored at `worker/patches-ltx25-gemma4.patch`) |
| Q3 DiT (UNGATED) | [Abiray/LTX-2.5-Distilled-GGUF](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF) | `7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0` (short `7b0c2025`) |
| Gemma4 TE (GATED) | [elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF) | `2a18e836d286eb0570ec9f013c3591eb8c614d57` (short `2a18e836`; local sha256 matches elix3r SHA256SUMS) |
| VAEs + upscaler (GATED) | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | `5e6e71018ee1756ed329b697a7b4aedc934dfce9` (short `5e6e7101`) |

Gated downloads need an HF token with the Lightricks license accepted
(the experiments used the second line of `comfy/hugging-face-token`;
never put tokens in code). Worker image is `voyage-ltx`
(`worker/Dockerfile.ltx`, py3.11 + torch 2.14/cu130 — a separate image
because that stack would break `voyage-video`'s frozen pins).

## What to download (registry + CLI wired)

From Abiray (community GGUF quant): only
`LTX-2.5-Distilled-Q3_K_M.gguf` (12,923,897,280 B). Q3 is the ONLY rung
— no Q2K fallback (OOM is a clean failure, not a ladder).

From elix3r (gated): only
`gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf` (5,956,930,560 B).

From Lightricks/LTX-2.5 (gated): `vae/ltx-2.5-video-vae-conv-bf16.safetensors`
(1,452,269,922 B), `vae/ltx-2.5-audio-vae-bf16.safetensors` (364,866,540 B),
and `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`
(995,778,752 B, shared with the `ltx23` backend — not duplicated).

Skipped: Q5/Q4/Q2K DiTs, UD-Q4/Q3 TEs (deleted during consolidation),
the diffusion video VAE (viable but conv stays default), E5 int2/qint2
trio (deferred experiment, still in the volume, out of spec).

## Mode-A accounting (DESIGN §5.3)

Every segment renders a 121-frame Mode-A clip: stage-1 608x352x121
distilled 8-sigma euler_ancestral, 2x latent upscale, stage-2
1216x704x121 3-step refine. Fresh blocks commit all 121; continued
blocks pin the 25-frame tail prefix (`LTXVImgToVideoInplace`,
strength 1.0) and commit 96 novel (`_LTX_NOVEL_BLOCK_FRAMES * blocks`
plans the steady-state 96). `video_tail.mp4` + §5.3 JSON tape anchor
recovery. Native 1216x704@24 (segment geometry is fixed; finalize
floors are a minimum, never a downscale target). Peak ~14.8 GiB —
locks `segment_frames=96` (Phase-0 Spike A: 14933 MiB; handover seam
ratios 0.93x/1.02x vs the 3x qual gate, Spike B).

## License implications

Check each weight repo for its exact license text before
redistributing weights or images (the registry records the pointers;
the run manifest records the selection). Lightricks weights carry the
Lightricks community license (gated, account-bound); the Gemma TE
carries the Gemma license via elix3r; the Abiray GGUF carries that
repo's stated terms.
