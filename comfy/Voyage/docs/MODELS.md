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

## Director — Qwen3-8B + MiniLM (`models download director-qwen8b`, ~16 GB)

| Artifact | Repo | Revision |
|----------|------|----------|
| LLM | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d9c46dd6066d109eabc6255188de91218` |
| Novelty embeddings | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

## Inspector — Qwen3.5-9B VLM (`models download inspector-qwen35`, ~19 GB, optional)

| Artifact | Repo | Revision |
|----------|------|----------|
| VLM | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

The allow-list must include `chat_template.jinja`. Only needed when
`[experimental] visual_inspector = true`.

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
