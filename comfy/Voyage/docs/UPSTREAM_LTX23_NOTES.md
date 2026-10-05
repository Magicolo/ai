# UPSTREAM_LTX23_NOTES — LTX-2.3 integration notes

Pins live in code in `voyage/registry_ltx23.py` (the single source of
truth); this file mirrors them for humans. Revisit every note if a pin
moves.

Status: worker landed. `voyage/workers/video_ltx23.py` serves the
`ltx23` backend (same Mode-A 1216x704@24 recipe and 121/25/96
accounting as `ltx25`, with the LTX-2.3 DualCLIP text encoder and the
unsloth distilled VAEs; `generate --backend ltx23`, `models
download/verify ltx23`).

## Upstream pins

| Artifact | URL | Pinned revision |
|----------|-----|-----------------|
| Q3 DiT + connectors + VAEs | [unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF) | `96e8ed4925ead3db9ff4d0084f165ef6a74f28d0` (short `96e8ed49`) |
| Gemma3 TE backbone | [unsloth/gemma-3-12b-it-qat-GGUF](https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF) | `858acec7ec0541a46c39985c95d3b52d8f3ab183` (short `858acec7`) |
| Spatial upscaler (shared) | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | `5e6e71018ee1756ed329b697a7b4aedc934dfce9` (short `5e6e7101`; gated — same file as `ltx25`, referenced not duplicated) |

Same execution stack as `ltx25` (ComfyUI `2f35f4a` + ComfyUI-GGUF
`6ea2651`, `voyage-ltx` image); the unsloth repos needed no auth at pin
time.

## What to download (registry + CLI wired)

From unsloth/LTX-2.3-GGUF: `distilled/ltx-2.3-22b-distilled-Q3_K_M.gguf`
(10,770,199,584 B), `text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors`
(2,312,144,712 B), `vae/ltx-2.3-22b-distilled_video_vae.safetensors`
(1,452,256,522 B), `vae/ltx-2.3-22b-distilled_audio_vae.safetensors`
(364,853,140 B).

From unsloth/gemma-3-12b-it-qat-GGUF: `gemma-3-12b-it-qat-Q2_K.gguf`
(4,768,221,568 B).

The text encoder is a DualCLIP pair (QAT backbone + connectors,
`DualCLIPLoaderGGUF` type `ltxv`): ComfyUI COMBO validation uses
non-recursive `get_filename_list`, so the worker registers the backbone
dir and the `text_encoders/` dir as two `clip` paths and passes BARE
filenames. Skipped: Q5/Q4/Q2 DiTs and the bit-identical dev video VAE
(deleted during consolidation).

## Mode-A accounting

Identical to `ltx25` (121-frame Mode-A clip, 25-frame frozen-prefix
continuation, 96 novel committed, §5.3 tape with
profile `ltx23`). The 2.3 family is the only one with a viable 2-GPU
split (Gemma3 Q2_K encodes on the 6 GB card), but the worker runs
everything on cuda:0 by default like `ltx25`. A dedicated Mode-A 121f
VRAM probe for the 2.3 stack is still open (the ltx25 probe does not
transfer — different DiT/TE/VAE residency).

## License implications

Check each weight repo for its exact license text before
redistributing weights or images (the registry records the pointers;
the run manifest records the selection). The unsloth GGUFs redistribute
the original model weights — the LTX DiT carries the Lightricks
community license and the Gemma3 backbone carries the Gemma license;
the shared upscaler is the gated Lightricks file (account-bound).
