# MODELS — exact IDs, revisions, links

All pins live in code in `voyage/model_registry.py` (the single source of
truth); this file mirrors them for humans. Most repos are **ungated** —
no token required — except the LTX-2.5 text encoder, VAEs and upscaler
(see below), which need a token with access. There is no `models` CLI
verb (two-verb CLI: `configure` + `generate` only): `configure`
verifies — and downloads when missing — only the stacks the effective
config needs (programmatic entry
`model_registry.download_model(models_dir, "<spec>")`, with `<spec>` the
parenthesized name on each section header below); `configure
--no-download` verifies without fetching (verify-only).

## Video — LTXV 2B distilled (`ltxv-2b`, ~7 GB, default)

| Artifact  | Repo / file | Revision |
|-----------|-------------|----------|
| DiT (`ltxv-2b-0.9.8-distilled.safetensors`, ~6.3 GB) + spatial upscaler (`ltxv-spatial-upscaler-0.9.8.safetensors`) | [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) | `8984fa25007f376c1a299016d0957a37a2f797bb` |
| Text encoder/tokenizer (`text_encoder/*`, `tokenizer/*`) | [PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) | `b89adadeccd9ead2adcb9fa2825d3fabec48d404` |

Code (not weights): [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
at `4b2d053057623ddd4d0a1d3e9cd28890e9ef487f`, installed (`--no-deps`,
`[inference]` extra) in `worker/Dockerfile.video`.

## Video — CausVid DMD + Wan2.1-1.3B base (`causvid`, ~28 GB)

| Artifact  | Repo / file | Revision |
|-----------|-------------|----------|
| DMD checkpoint (`autoregressive_checkpoint/model.pt`, ~10.6 GB) | [tianweiy/CausVid](https://huggingface.co/tianweiy/CausVid) | `b545eb2728fc9d1515023a270b847f7b24b3aa89` |
| Base DiT (`diffusion_pytorch_model.safetensors`, ~5.7 GB) + VAE (`Wan2.1_VAE.pth`) + T5 (`models_t5_umt5-xxl-enc-bf16.pth`) + `google/umt5-xxl/` tokenizer | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | `37ec512624d61f7aa208f7ea8140a131f93afc9a` |

Code (not weights): [tianweiy/CausVid](https://github.com/tianweiy/CausVid)
at `adb6a5ecd07666b4d0290042915c8406e6d5ce22`, cloned in
`worker/Dockerfile.video` (leaf deps only — never the upstream
`requirements.txt`). The DMD checkpoint is CC BY-NC-SA 4.0
(non-commercial, share-alike); the Wan2.1 base is Apache 2.0. Native
geometry 832×480 @ 16 fps; full notes:
`docs/UPSTREAM_CAUSVID_NOTES.md`.

## Video — LTX-2.5 Q3 + Gemma4 TE + VAEs (`ltx25`, ~38 GB, joint A/V)

| Artifact | Repo / file | Revision |
|----------|-------------|----------|
| DiT (`LTX-2.5-Distilled-Q3_K_M.gguf`, ~12.9 GB) | [Abiray/LTX-2.5-Distilled-GGUF](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF) | `7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0` |
| Text encoder (`gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf`, ~6.0 GB, **gated**) | [elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF) | `2a18e836d286eb0570ec9f013c3591eb8c614d57` |
| Video VAE (`vae/ltx-2.5-video-vae-conv-bf16.safetensors`, ~1.5 GB, **gated**) + audio VAE (`vae/ltx-2.5-audio-vae-bf16.safetensors`, **gated**) + spatial upscaler (`latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`, **gated**) | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | `5e6e71018ee1756ed329b697a7b4aedc934dfce9` |

Driven in-process through pinned ComfyUI (`2f35f4a`) + ComfyUI-GGUF
(`6ea2651`) + gemma4 patch — see `LTX2.md` experiments E1-E4/S21-S30.
Generates joint audio (no ACE-Step/MMAudio for this backend).

## Video — LTX-2.3 Q3 + Gemma3 TE + VAEs (`ltx23`, ~20 GB, joint A/V)

| Artifact | Repo / file | Revision |
|----------|-------------|----------|
| DiT (`distilled/ltx-2.3-22b-distilled-Q3_K_M.gguf`, ~10.8 GB) + connectors (`text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors`, ~2.3 GB) + video/audio VAEs (`vae/*`, ~1.8 GB) | [unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF) | `96e8ed4925ead3db9ff4d0084f165ef6a74f28d0` |
| Text encoder backbone (`gemma-3-12b-it-qat-Q2_K.gguf`, ~4.8 GB) | [unsloth/gemma-3-12b-it-qat-GGUF](https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF) | `858acec7ec0541a46c39985c95d3b52d8f3ab183` |

Same pinned ComfyUI stack as ltx25; the Mode-A spatial upscaler is
shared from the ltx25 volume (not duplicated). Joint audio, like ltx25.

## Finalize augmentation — FILM interpolation (`film`, ~66 MB)

| Artifact | Repo / file | Revision |
|----------|-------------|----------|
| FILM fp16 (`frame_interpolation/film_net_fp16.safetensors`) | [Comfy-Org/frame_interpolation](https://huggingface.co/Comfy-Org/frame_interpolation) | `219da3c9d8c357ceaf457fc1d5932c6e861b8dee` |

Torch-native load (state dict via `safetensors`, no retraining code). The
repack bundles google-research/frame-interpolation (Apache 2.0) and
hzwer/Practical-RIFE (MIT), hence the `mit-and-apache-2.0` tag. Weights
land in `<models>/frame_interpolation/` (ComfyUI layout); leaf deps in
`worker/Dockerfile.video`.

## Finalize augmentation — RIFE interpolation (`rife`, ~87 MB)

| File under `~/.cache/voyage-models/...` | Origin | Size | License |
|---|---|---|---|
| `frame_interpolation/rife_v4.25_heavy.safetensors` | `Comfy-Org/frame_interpolation` (same pinned revision as FILM) | 86.7 MB | MIT |

RIFE v4.25-heavy fp16 is the DEFAULT interp backend (`interp_backend = "rife"`): the 2026-10-06 probe measured it sharpest and closest-to-FILM of all variants (0.05-0.10 s/pair vs FILM 0.85 s/pair at 2048x1152, identical 0.65 GiB peak, clean line-art eyeball). FILM stays for hero/archival renders via `--interp-backend film`. The repack ships the same way as the FILM file (single-file safetensors, floor-checked + sha-verified by the registry).

## Finalize augmentation — realesr-animevideov3 (`realesrgan-anime`, ~2.5 MB)

| Artifact | Repo / file | Revision |
|----------|-------------|----------|
| Anime upscaler (`realesr-animevideov3.pth`, native 4x SRVGGNetCompact XS: 16 conv, 64 feat) | [nateraw/real-esrgan](https://huggingface.co/nateraw/real-esrgan) (mirror of the [xinntao/Real-ESRGAN v0.2.5.0 release asset](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth)) | `44ad8adf6069185b86df22349b12f255821c86ab` |

BSD-3-Clause (c) 2021 Xintao Wang. Torch-native `torch.load`
(weights-only) of the SRVGGNetCompact generator (key-sniffed, not the
RRDB layouts — the file wraps params one level deep under `params`;
the old 18 MB `RealESRGAN_x4plus_anime_6B` RRDB weights are superseded,
~11-13x slower on the 2060); weights land in `<models>/realesrgan/`.

## Director — Qwen3-8B + MiniLM (`director-qwen8b`, ~16 GB)

| Artifact | Repo | Revision |
|----------|------|----------|
| LLM | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d9c46dd6066d109eabc6255188de91218` |
| Novelty embeddings | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

The director worker resolves both ids to its single `/models` copy
(`Qwen3-8B/`, `all-MiniLM-L6-v2/`) and fetches a missing snapshot into
the volume on demand — a deleted volume re-downloads automatically
instead of silently falling back. Any other known registry repo id maps
the same way; unknown ids and local directory paths pass through to
hub/cache behavior untouched.

## Director — Qwen3-4B-AWQ GPU decider (`director-qwen4b-awq`, ~2.6 GB)

| Artifact | Repo | Revision |
|----------|------|----------|
| 4-bit AWQ decider | [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ) | `74d4bd2bd4bff9cafc9345221320bffb08b406a3` |
| Novelty embeddings | shared with the Qwen3-8B stack above (MiniLM `1110a243…`) | — |

Apache 2.0 ([license](https://huggingface.co/Qwen/Qwen3-4B-AWQ/blob/main/LICENSE)).
Default placement is cuda:1 (the second GPU) via `VOYAGE_DIRECTOR_PYTHON`;
`--director-device cpu` opts back into the Qwen3-8B CPU path above.
Full row (subdir, allow-list, size floor): `MODEL_SPECS["director-qwen4b-awq"]`.

## Director — Qwen3.5-4B GGUF sidecar (`director-qwen35-gguf`, ~3 GB)

| Artifact | Repo | Revision |
|----------|------|----------|
| Q4_K_M sidecar GGUF | [bartowski/Qwen_Qwen3.5-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF) (`Qwen_Qwen3.5-4B-Q4_K_M.gguf` — the sibling list is authoritative; the unprefixed spelling does not exist) | `4168f45a16a1290d65a4ec0fa312ae917a4c15d6` |

Apache-2.0. Served by the loopback `llama-server` sidecar when the
director backend is `llama` (DESIGN §140 llama entry); `voyage generate`
ensures this file instead of the AWQ stack. Same /models resolution as
the director pair above (`Qwen3.5-4B-GGUF/`).
Full row (file, size floor): `MODEL_SPECS["director-qwen35-gguf"]`.

## Inspector — Qwen3.5-9B VLM (`inspector-qwen35`, ~19 GB, optional)

| Artifact | Repo | Revision |
|----------|------|----------|
| VLM | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

The allow-list must include `chat_template.jinja`. Only needed when
`[experimental] visual_inspector = true`. Same /models resolution as the
director pair above (`Qwen3.5-9B/`). Note: the inspector loads with
`trust_remote_code=False` (transformers 5.17.0 ships native qwen3_5
modeling — verified live 2026-09-29, issue 056); no remote code executes.
The pin + allow-list remain as availability guards.

## Audio — ACE-Step 1.5 (`audio-acestep`)

| Artifact | Repo | Revision |
|----------|------|----------|
| Main model | [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) | `19671f406d603126926c1b7e2adc169acbcade22` |
| Planner LM (0.6 B) | [ACE-Step/acestep-5Hz-lm-0.6B](https://huggingface.co/ACE-Step/acestep-5Hz-lm-0.6B) | `148d8ea0225bdab342ee1ae3a354275ccd60ca80` |

Code: [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) at
`ca1e85fe9430179831e6bc6be790c332190a3866`. Checkpoints must land in
`<models>/acestep/checkpoints/` matching upstream `MAIN_MODEL_COMPONENTS`
(includes the gate-only 1.7 B LM).

## SFX — MMAudio 44 kHz (`sfx-mmaudio`, ~13 GB)

| Artifact | Repo | Revision |
|----------|------|----------|
| small_44k (157 M, 601 MB) | [hkchengrex/MMAudio](https://huggingface.co/hkchengrex/MMAudio) | `eb13a1a98fdbec91753775c57b074ccdfc60587c` |
| medium_44k (621 M, 2.4 GB) | [hkchengrex/MMAudio](https://huggingface.co/hkchengrex/MMAudio) | `eb13a1a98fdbec91753775c57b074ccdfc60587c` |
| large_44k_v2 (1.03 B, 3.9 GB, recommended) | [hkchengrex/MMAudio](https://huggingface.co/hkchengrex/MMAudio) | `eb13a1a98fdbec91753775c57b074ccdfc60587c` |
| VAE v1-44 (1.2 GB) + synchformer (907 MB) | [hkchengrex/MMAudio](https://huggingface.co/hkchengrex/MMAudio) | `eb13a1a98fdbec91753775c57b074ccdfc60587c` |
| 44 kHz BigVGAN vocoder | [nvidia/bigvgan_v2_44khz_128band_512x](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x) | `95a9d1dcb12906c03edd938d77b9333d6ded7dfb` |
| DFN5B CLIP tower | [apple/DFN5B-CLIP-ViT-H-14-384](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384) | `01b771ed0d1395ca5ffdd279897d665ebe00dfd2` |

Code: [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio) at
`974010a026c731054592d8f777218bd9d85a6c24`. Layout under
`<models>/mmaudio/` (`weights/`, `ext_weights/`, `vocoder/`,
`clip/`). Ladder results 2026-09-29: small fits the 6 GB 2060
(4.6 GiB peak), medium OOMs it, large needs the 4060 (6.2 GiB peak).

## Mastering — SonicMaster (spec pinned by Track A, venv by Track D)

| Artifact | Repo / file | Revision |
|----------|-------------|----------|
| Mastering DiT (`sonicmaster/model.safetensors`, ~3.45 GB, public, Apache-2.0) | [nateraw/sonicmaster](https://huggingface.co/nateraw/sonicmaster) (mirror of [amaai-lab/SonicMaster](https://huggingface.co/amaai-lab/SonicMaster)) | `a1765134e0808a8fb45e1a92874327e01d07cd75` |
| Mastering VAE (`sonicmaster/vae.safetensors`, **gated** — needs a token with access, Stability AI Community License) | [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) | `9612befb10cc9992a2f82558bbb4735efaba3297` |

IDs above are the live-probe winners (`output/sonicmaster-ab/METRICS.md`):
isolated `sonicmaster-probe` container (pytorch 2.4.0 cu124 + transformers
4.44.0 + diffusers 0.30.0), single 30 s chunk, 44.1 kHz stereo, Euler 10
steps, guidance 1.0 (the probe additionally used the `google/flan-t5-large`
text encoder — not part of the spec) — outputs render
~2–4 LU quieter with ~-6 dB peaks (headroom a normalize-then-limit stage
reclaims downstream). Full row: `MODEL_SPECS["audio-sonicmaster"]`
(`voyage/registry_mastering.py`, pinned by Track A).

Track D carries only the execution environment: the isolated
`/opt/venvs/mastering` stack (torch 2.4.0/cu124 + transformers 4.44.0 +
diffusers 0.30.0 + soundfile/safetensors/hub — the probe's exact core)
baked into the `voyage-ltx` and `voyage-video` worker images and spawned
via `VOYAGE_MASTERING_PYTHON` (unset on the slim image and on hosts falls
back to the supervisor interpreter). The registry spec + ensure wiring
are Track A's (`audio-sonicmaster`, ensured by default, `--no-master` /
`--no-audio` opt out).

Provisioning: sign in with a token that can access the gated VAE repo, then `configure` + `generate`
verifies — and downloads when missing — only the stacks the effective
config needs (`--no-download` verifies without fetching, failing loud
offline instead of hanging on the hub).

## License notes

- 4x-UltraSharp weights are a Zoomy concern, not Voyage's (Voyage pins the
  Real-ESRGAN anime 6B mirror above instead).
- FILM repack is MIT + Apache 2.0; Real-ESRGAN anime is BSD-3-Clause —
  both permissive, unlike the non-commercial stacks below.
- MMAudio weights are CC-BY-NC-4.0 (non-commercial) — same class as the
  CausVid DMD checkpoint; check before redistributing models or images.
- The Wan 2.2 base weights carry their own license; check the repo page
  before redistributing models.
