# MODELS — exact IDs, revisions, links

All pins live in code in `voyage/model_registry.py` (the single source of
truth); this file mirrors them for humans. Every repo is **ungated** —
no token required. Verify local files with `voyage models verify`.

## Video — LongLive 2.0 (`models download longlive2-bf16`, ~48 GB)

| Artifact  | Repo / file | Revision |
|-----------|-------------|----------|
| Generator | [Efficient-Large-Model/LongLive-2.0-5B](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B) | `8521079b863720a57c1a8d9b19c8d9e6ccb04c0f` |
| Base VAE/text weights | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) (`wan_models/` subdir) | pinned snapshot in registry |

Code (not weights): [NVlabs/LongLive](https://github.com/NVlabs/LongLive)
at `6b36d20ec6f7958d29d11a704dfa64611a9f2572`, cloned in
`worker/Dockerfile.video`. Runtime patches on top: see
`docs/UPSTREAM_LONG_LIVE_PATCHES.md`.

## Video — LTXV 2B distilled (`models download ltxv-2b`, ~7 GB)

| Artifact  | Repo / file | Revision |
|-----------|-------------|----------|
| DiT (`ltxv-2b-0.9.8-distilled.safetensors`, ~6.3 GB) + spatial upscaler (`ltxv-spatial-upscaler-0.9.8.safetensors`) | [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) | `8984fa25007f376c1a299016d0957a37a2f797bb` |
| Text encoder/tokenizer (`text_encoder/*`, `tokenizer/*`) | [PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) | `b89adadeccd9ead2adcb9fa2825d3fabec48d404` |

Code (not weights): [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
at `4b2d053057623ddd4d0a1d3e9cd28890e9ef487f`, installed (`--no-deps`,
`[inference]` extra) in `worker/Dockerfile.video`.

## Video — CausVid DMD + Wan2.1-1.3B base (`models download causvid`, ~28 GB)

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

## Director — Qwen3-8B + MiniLM (`models download director-qwen8b`, ~16 GB)

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

## Inspector — Qwen3.5-9B VLM (`models download inspector-qwen35`, ~19 GB, optional)

| Artifact | Repo | Revision |
|----------|------|----------|
| VLM | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

The allow-list must include `chat_template.jinja`. Only needed when
`[experimental] visual_inspector = true`. Note: the inspector loads with
`trust_remote_code=True` (the model ships custom modeling/processor code;
the Qwen3 text path stays `False`) — a compromised revision is RCE in the
director container. The pin + allow-list mitigate availability, not
execution; vendoring + hash-pinning the modeling files is the follow-up.

## Audio — ACE-Step 1.5 (`models download audio-acestep`)

| Artifact | Repo | Revision |
|----------|------|----------|
| Main model | [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) | `19671f406d603126926c1b7e2adc169acbcade22` |
| Planner LM (0.6 B) | [ACE-Step/acestep-5Hz-lm-0.6B](https://huggingface.co/ACE-Step/acestep-5Hz-lm-0.6B) | `148d8ea0225bdab342ee1ae3a354275ccd60ca80` |

Code: [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) at
`ca1e85fe9430179831e6bc6be790c332190a3866`. Checkpoints must land in
`<models>/acestep/checkpoints/` matching upstream `MAIN_MODEL_COMPONENTS`
(includes the gate-only 1.7 B LM).

## License notes

- 4x-UltraSharp / RealESRGAN weights are a Zoomy concern, not Voyage's.
- The Wan 2.2 base weights carry their own license; check the repo page
  before redistributing models.
