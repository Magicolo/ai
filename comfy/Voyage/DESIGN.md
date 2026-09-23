# Voyage — Autonomous Infinite Audiovisual Voyage

## Design Specification for Coding Agents

**Status:** Implementation specification / architecture baseline  
**Research snapshot:** 2026-09-21  
**Primary video backend:** LongLive 2.0 / Wan2.2-TI2V-5B  
**Primary audio backend:** ACE-Step 1.5  
**Primary interface:** Python CLI + supervised worker processes  
**Primary operating target:** Linux, NVIDIA CUDA, 64 GB system RAM, one 16 GB VRAM GPU + one 8 GB VRAM GPU  
**Target output:** one finalized 16:9 video file, approximately 768×432 at 24 fps, with evolving music and environmental audio  

---

# 1. Purpose

This document defines the architecture, implementation boundaries, operational behavior, failure model, model integrations, development phases, standards, tests, and documentation requirements for **`voyage`**, a local command-line application that generates an effectively indefinite audiovisual experience.

The intended use is highly autonomous generative art:

- The human supplies **styling constraints**.
- The machine decides **what is being depicted**.
- The machine gradually changes what is depicted over time.
- The video should feel like a continuous voyage rather than a playlist of unrelated clips.
- The soundtrack should be a major creative component and may evolve more aggressively than the visuals.
- Semantic concepts should generally not be revisited after they leave the current world state.
- Generation must be safe to leave running for many hours or days.
- A crash must not destroy the entire run or require restarting from the beginning.
- Intermediate artifacts are expected and desirable.
- The final video is assembled only when the user requests finalization.

The project is intentionally **not** a general-purpose video-generation framework. It is a purpose-built autonomous audiovisual director around a streaming autoregressive video backend and an independent generative music backend.

---

# 2. Core product concept

The system should be understood as three interacting systems:

1. **Renderer** — generates audiovisual material.
2. **Director** — decides what the material should gradually become.
3. **Archivist** — persists enough state to make long-running generation safe and resumable.

The critical conceptual separation is:

```text
Human
  │
  │ immutable style charter
  ▼
Director / world model
  │
  ├── next visual destination
  ├── transition mechanism
  ├── audiovisual mood
  └── novelty constraints
  │
  ├───────────────────────────┐
  ▼                           ▼
Video renderer              Audio renderer
LongLive 2.0                ACE-Step 1.5
  │                           │
  ▼                           ▼
video segment               audio segment
  │                           │
  └──────────────┬────────────┘
                 ▼
             segment commit
                 │
                 ▼
        persistent voyage state
                 │
                 └──────► next decision
```

Do **not** collapse these responsibilities into one Python function or one model prompt.

---

# 3. User requirements

The implementation must preserve the following product requirements.

## 3.1 Style is human-owned and persistent

The user supplies a style charter such as:

> pastel neon line-art, slow motion, peaceful, dreamlike

The style charter is immutable for the duration of a run unless the user explicitly changes the run configuration.

The director may evolve subject matter, setting, materials, scale, atmosphere, camera context, and narrative implications, but it may not silently replace the style charter.

Examples of style constraints:

- visual medium / rendering technique;
- color behavior;
- movement speed;
- pacing;
- mood;
- texture;
- degree of surrealism;
- visual complexity;
- camera behavior;
- transition smoothness;
- audio aesthetic.

Examples of things the user should **not** need to provide:

- recurring character;
- specific object;
- scene list;
- environment list;
- story outline;
- subject taxonomy;
- sequence of prompts.

## 3.2 Content is autonomous

The system owns the choice of:

- subjects;
- objects;
- environments;
- phenomena;
- transformations;
- material changes;
- scale changes;
- camera contexts;
- conceptual destinations.

The user should be able to start a run with almost no semantic instructions beyond style.

## 3.3 Slow/moderate visual evolution

The visual journey should evolve gradually.

A desirable example:

```text
mechanical orchard
    ↓
metallic fruit begins behaving like translucent glass
    ↓
glass reflections deepen into pools
    ↓
orchard becomes partially submerged
    ↓
architecture begins resembling coral
    ↓
coral structures become a luminous underwater ecosystem
```

An undesirable example:

```text
orchard → spaceship → dragon → medieval castle → football stadium
```

The second example changes concepts too quickly and produces a collage rather than a voyage.

## 3.4 Concepts should not deliberately repeat

The default policy is **no deliberate conceptual revisiting**.

A transformed motif may resemble something earlier, but the director should not intentionally return to an old canonical concept.

The system therefore requires persistent semantic novelty memory.

## 3.5 Audio evolves aggressively

Music is not background filler.

Audio should be capable of making substantial stylistic transitions while the visuals move more slowly.

Examples:

```text
ambient piano
    → granular synth texture
    → restrained electronic pulse
    → glass harmonics
    → organic percussion
    → deep drone
```

Environmental sounds and short sound events may respond to the current world.

## 3.6 Infinite means operationally indefinite

`voyage run` should not have a hard-coded maximum duration.

The logical generation loop should continue until one of these occurs:

- user requests stop;
- user requests pause;
- unrecoverable configuration failure;
- hardware failure that cannot be recovered;
- disk space policy triggers a safe stop;
- an explicit optional duration limit is reached;
- an implementation-defined resource budget is exhausted.

## 3.7 Resumability is a first-class feature

A restart must not require beginning from segment zero.

The system must be able to recover from:

- supervisor crash;
- video worker crash;
- audio worker crash;
- director worker crash;
- CUDA process failure;
- OOM;
- power loss;
- SIGINT;
- partially written files;
- incomplete state writes;
- corrupted intermediate output;
- disk filling during generation.

---

# 4. Non-goals for V1

The following are explicitly out of scope for the first implementation.

- Training or fine-tuning models.
- A hosted web service.
- Cloud inference.
- Distributed multi-GPU tensor parallelism.
- Kubernetes.
- Redis, Kafka, RabbitMQ, or another message broker.
- A continuously appended single MP4 during generation.
- ComfyUI as a runtime dependency.
- Automatic model downloads during `voyage run`.
- A complicated vector database.
- Fully autonomous visual feedback control before the basic generator is stable.
- Automatic style changes.
- Automatic migration between different video backends inside an active run.
- Real-time streaming to the network.
- Perfect byte-for-byte reproducibility on real GPU inference.

V1 should be intentionally small and boring.

---

# 5. Current technology choices

## 5.1 Primary video backend: LongLive 2.0

Use the current LongLive repository:

- GitHub: https://github.com/NVlabs/LongLive
- Documentation: https://nvlabs.github.io/LongLive/LongLive2/docs/
- Paper: https://arxiv.org/abs/2605.18739

The current LongLive 2.0 implementation is built around:

- **Wan2.2-TI2V-5B**
- causal/autoregressive temporal generation;
- per-block prompt conditioning;
- local attention with a rolling KV cache;
- attention sink mechanisms;
- multi-shot sink support;
- relative RoPE support for very long / infinite generation;
- optional NVFP4 weight and KV-cache quantization;
- optional TorchAO FP8 PTQ;
- streaming / asynchronous VAE decoding;
- optional LoRA support.

LongLive 2.0's current public model table lists:

| Model | Parameters | Published upstream throughput | Notes |
|---|---:|---:|---|
| LongLive-1.3B | 1.3B | 20.7 FPS | LongLive 1.0 architecture; fallback only |
| LongLive-2.0-5B | 5B | 24.8 FPS | BF16 baseline; multi-shot |
| LongLive-2.0-5B-NVFP4-S4 | 5B | 29.7 FPS | 4-step NVFP4 |
| LongLive-2.0-5B-NVFP4-S2 | 5B | 45.7 FPS | 2-step NVFP4 |

These throughput values are upstream reference measurements and are **not** predictions for the user's GPUs. Every hardware profile must be benchmarked locally.

### Why LongLive 2.0

It directly addresses the difficult part of this project: turning a short-video diffusion architecture into an autoregressive, long-running generator rather than repeatedly starting unrelated clips.

The upstream implementation's `CausalDiffusionInferencePipeline` performs these operations for each temporal block:

1. selects the prompt for the block;
2. invalidates cross-attention cache when the prompt changes;
3. denoises the new block;
4. writes the clean latent result to the output sequence;
5. reruns the block at timestep zero to update the causal KV cache;
6. rolls the local cache when required;
7. optionally preserves sink/pinned scene context;
8. optionally decodes the block through a streaming VAE.

This is the fundamental mechanism `voyage` should reuse.

### Why not LongLive 1.0

LongLive 1.0 is historically important and remains a fallback reference implementation, but LongLive 2.0 is the current upstream branch and was specifically extended with improved quantization, KV-cache handling, multi-shot support, and infinite-generation support.

LongLive 1.0 checkpoint:

- https://huggingface.co/Efficient-Large-Model/LongLive-1.3B

Its model license is different from LongLive 2.0; record it separately when using it.

---

## 5.2 Video base model: Wan2.2-TI2V-5B

Model:

- Hugging Face: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- Code: https://github.com/Wan-Video/Wan2.2

Current LongLive 2.0 uses `Wan2.2-TI2V-5B` as the causal video backbone.

Important upstream configuration values:

```yaml
model_name: Wan2.2-TI2V-5B
fps: 24
temporal_compression_ratio: 4
spatial_compression_ratio: 16
num_heads: 24
head_dim: 128
num_transformer_blocks: 30
num_frame_per_block: 8
local_attn_size: 32
sink_size: 8
```

The current LongLive 2.0 reference latent shape is commonly:

```text
[B, F, C, H, W]
[1, 128, 48, 44, 80]
```

with:

- `C = 48`
- `H = 44`
- `W = 80`
- 4× temporal compression
- 16× spatial compression in the underlying VAE/model definition
- 24 fps output.

Therefore an 8-latent-frame causal block corresponds to approximately:

```text
8 latent frames × 4 video frames/latent = 32 video frames
32 / 24 fps = 1.333... seconds
```

The exact pixel dimensions associated with the reference shape are approximately 704×1280.

### Resolution policy

Do **not** assume that a 768×432 latent shape is equally validated by the upstream LongLive release.

The user's desired final output is approximately 768×432, but V1 should validate a hardware-specific rendering profile.

Recommended strategy:

1. Initially test the upstream reference spatial shape.
2. Test a lower-memory latent shape at the same aspect ratio.
3. Measure peak VRAM, generation speed, quality, and cache behavior.
4. Select the highest stable profile that fits the 16 GB GPU.
5. Convert the final video to exactly 768×432 only at finalization unless direct generation at that size is demonstrated to be stable and visually preferable.

A target-resolution candidate can be represented as:

```text
pixel dimensions: 768×432
latent dimensions: derived from the exact LongLive/Wan VAE compression contract
```

The implementation must calculate and validate these dimensions rather than hard-coding an assumed compression ratio into unrelated modules.

---

## 5.3 LongLive 2.0 checkpoint choices

### Preferred rapid-iteration checkpoint

**LongLive-2.0-5B-NVFP4-S2**

- https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2

Use for development benchmarking because it is the lowest-step current LongLive 2.0 checkpoint.

The model card describes it as a 2-step NVFP4 model.

### Preferred quality/reference checkpoint

**LongLive-2.0-5B-NVFP4-S4**

- https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4

Use when S2 produces insufficient visual quality.

### BF16 checkpoint

**LongLive-2.0-5B**

- https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B

Use as the correctness reference when debugging quantization or a numerical discrepancy.

### Important integration rule

Never infer the correct quantization backend from the model name alone.

The current LongLive code distinguishes between Transformer Engine and FourOverSix NVFP4 checkpoints.

The worker must:

- inspect the checkpoint;
- determine whether it is a pre-materialized NVFP4 state dict;
- select the matching upstream loader path;
- validate the configured backend against the checkpoint type;
- fail loudly on an incompatible combination.

Do not silently cast a quantized checkpoint to BF16.

---

## 5.4 LongLive model licensing

The current LongLive 2.0 repository code is released under Apache 2.0.

The LongLive 2.0 model cards currently identify the model license as **NVIDIA Open Model License** / NVIDIA Open Model Agreement rather than Apache 2.0.

References:

- https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
- https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

For every run, the exact model identifier, revision, local checkpoint hash if practical, and license URL must be captured in `run_manifest.json`.

Do not make a general legal claim such as “the model is Apache licensed” merely because the GitHub repository is Apache licensed.

---

# 6. Primary audio backend: ACE-Step 1.5

Repository:

- https://github.com/ace-step/ACE-Step-1.5

Model:

- https://huggingface.co/ACE-Step/Ace-Step1.5

ACE-Step 1.5 is selected because it provides local text-to-music generation with:

- 2B and 4B DiT variants;
- a 0.6B / 1.7B / 4B planning LM family;
- 10-second through 10-minute generation;
- repaint/continuation-oriented functionality;
- reference audio;
- metadata controls;
- 8-step turbo variants;
- consumer-GPU support;
- a project license of MIT.

### V1 hardware target

Use the 8 GB GPU for audio.

Initial model profile:

```text
DiT: acestep-v15-turbo
Planner LM: acestep-5Hz-lm-0.6B
Backend: PyTorch or the lightest supported local backend
```

The upstream ACE-Step documentation suggests:

- 6–8 GB: 2B turbo + 0.6B LM
- 8–16 GB: 2B turbo/sft + 0.6B / 1.7B LM

Therefore the 8 GB GPU should start with the 0.6B planner, not the 1.7B planner.

The 4B XL variants are intentionally not a V1 dependency because the 8 GB GPU is insufficient for the intended simple and robust deployment profile.

### Why ACE-Step instead of video-model audio

Keeping music generation separate means:

- video and audio can evolve at different rates;
- audio generation can continue even if video generation is paused;
- the project can replace the audio backend without redesigning video generation;
- the 8 GB GPU gets a well-defined job;
- long-run audio can be regenerated without rerunning expensive video inference.

---

# 7. Optional SFX backend

V1 should define an abstract SFX interface but must not make a 16 GB AudioGen model a hard dependency.

Candidate:

**AudioGen Medium**

- https://huggingface.co/facebook/audiogen-medium

AudioGen Medium is a 1.5B text-to-sound model and uses a CC-BY-NC-4.0 model license. Upstream guidance historically places it around a 16 GB VRAM class for comfortable inference.

It therefore should not compete with the 8 GB ACE-Step worker in V1.

V1 audio requirement:

- music generation is required;
- environmental/event sound generation is optional;
- procedural ambience may be used as a temporary fallback for SFX if the generative SFX backend is unavailable.

The SFX provider abstraction must make future model replacement possible without touching the supervisor or timeline code.

---

# 8. Director model

Primary director model:

**Qwen/Qwen3-8B**

- https://huggingface.co/Qwen/Qwen3-8B

The director does not need to be a video model.

It performs:

- conceptual planning;
- transition design;
- prompt generation;
- novelty reasoning;
- audio planning;
- policy enforcement against its own previous suggestions.

Run it on CPU or a low-priority process using system RAM when practical so the 16 GB and 8 GB GPUs remain dedicated to rendering.

The director must communicate through a typed JSON schema, never by parsing arbitrary prose.

Optional later visual inspector:

**Qwen/Qwen3.5-9B**

- https://huggingface.co/Qwen/Qwen3.5-9B

It is a multimodal-capable model and can be evaluated later for frame/segment inspection. Do not make it a V1 requirement.

---

# 9. Optional semantic embedding model

For novelty memory, use a lightweight sentence embedding model rather than asking the director LLM to judge every pair of concepts itself.

Candidate:

**sentence-transformers/all-MiniLM-L6-v2**

- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

The embedding model should run on CPU.

Its role is narrow:

```text
canonical concept text → vector → cosine similarity
```

It is not an artistic evaluator.

The final acceptance/rejection decision remains deterministic policy code.

---

# 10. Experimental backends and references

These are not V1 dependencies but are important implementation references.

## LTXV

- https://github.com/Lightricks/ComfyUI-LTXVideo

The current `LTXVLoopingSampler` source demonstrates:

- overlapping temporal tiles;
- temporal continuation;
- per-temporal-tile prompt conditioning;
- optional long-term context latents;
- AdaIN normalization for accumulated statistics drift;
- spatial tiling.

Its `MultiPromptProvider` maps prompts to temporal tiles.

This is a useful conceptual reference for prompt evolution and long-video conditioning.

Current source examined:

- `looping_sampler.py`

The current loop sampler is designed around video latents and explicitly rejects audio-visual nested latents, which is one reason it is not the primary V1 runtime.

## FramePack

- https://github.com/lllyasviel/FramePack

FramePack uses constant-sized historical context and progressive next-frame-section prediction.

It is a useful fallback if LongLive 2.0 cannot run stably on the target GPU.

## MAGI-1

- https://github.com/SandAI-org/MAGI-1

MAGI-1 is another autoregressive video model with chunk-wise generation and prompt control. Its architecture is useful as a future backend target but is not needed for V1.

## SkyReels V2

- https://github.com/SkyworkAI/SkyReels-V2

SkyReels V2 is explicitly an infinite-length diffusion-forcing video architecture and is useful as an alternative benchmark.

## LongLive-RAG

- https://github.com/qixinhu11/LongLive-RAG

This may become useful if long-term world memory later requires retrieval beyond the simple concept index.

---

# 11. System architecture

## 11.1 Process model

The application consists of one supervisor and multiple long-lived worker processes.

```text
┌──────────────────────────────────────────────────────────┐
│                    voyage CLI / supervisor               │
│                                                          │
│  Run state        Segment scheduler      Finalizer       │
│  World director   Failure handling       Status UI       │
└─────────────┬────────────────┬───────────────────────────┘
              │                │
       JSONL RPC / UDS   JSONL RPC / UDS
              │                │
      ┌───────▼──────┐   ┌─────▼────────┐
      │ video worker │   │ audio worker │
      │ LongLive 2.0 │   │ ACE-Step 1.5 │
      │ GPU 0 16 GB  │   │ GPU 1  8 GB  │
      └──────────────┘   └──────────────┘
              ▲
              │
       director process
       Qwen3-8B on CPU
```

The director can either be a subprocess like the render workers or a Python module hosted by the supervisor. The recommended V1 implementation is a subprocess so a broken LLM invocation cannot corrupt the supervisor event loop.

## 11.2 Why workers instead of importing everything into one process

Separate processes provide:

- CUDA-context isolation;
- easier recovery from OOM;
- simpler dependency isolation;
- independent Python environments;
- reduced package conflicts;
- the ability to restart a failed model without restarting the entire run;
- clean GPU ownership;
- simpler memory accounting.

Do not use multiprocessing shared GPU tensors in V1.

Use filesystem checkpointing and explicit RPC instead.

---

# 12. Python environments

Prefer three environments.

## Supervisor environment

Recommended:

```text
Python 3.12
```

Contains:

- CLI;
- Pydantic models;
- supervisor;
- state management;
- ffmpeg orchestration;
- director RPC;
- tests;
- metrics;
- no heavy GPU model dependencies.

Use `uv` for reproducible environment management.

## LongLive environment

Follow the exact current LongLive 2.0 requirements for the chosen inference path.

The current upstream BF16 path is documented around:

- Python 3.10;
- PyTorch 2.8.0;
- CUDA 12.8;
- TorchVision 0.23.0;
- FlashAttention where applicable.

The current NVFP4 path uses a distinct, newer environment and CUDA extension toolchain. Follow the version matrix in the LongLive 2.0 documentation rather than mixing the BF16 and NVFP4 dependency sets casually.

Do not install LongLive dependencies into the supervisor virtual environment.

## ACE-Step environment

Use the versions specified by ACE-Step 1.5, currently Python 3.11–3.12.

Keep its environment independent as well.

---

# 13. Repository layout

Recommended repository structure:

```text
voyage/
├── pyproject.toml
├── README.md
├── DESIGN.md
├── LICENSE
├── NOTICE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── .gitignore
│
├── src/
│   └── voyage/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── errors.py
│       ├── logging.py
│       ├── supervisor.py
│       ├── rpc.py
│       │
│       ├── director/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── schema.py
│       │   ├── prompts.py
│       │   ├── evolution.py
│       │   ├── policy.py
│       │   ├── novelty.py
│       │   └── embeddings.py
│       │
│       ├── video/
│       │   ├── __init__.py
│       │   ├── protocol.py
│       │   ├── longlive.py
│       │   ├── streaming.py
│       │   ├── recovery.py
│       │   └── resolution.py
│       │
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── protocol.py
│       │   ├── acestep.py
│       │   ├── music_state.py
│       │   ├── sfx.py
│       │   └── mix.py
│       │
│       ├── run/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── scheduler.py
│       │   ├── segment.py
│       │   ├── checkpoints.py
│       │   └── lifecycle.py
│       │
│       ├── state/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── store.py
│       │   ├── manifest.py
│       │   ├── atomic.py
│       │   └── migrations.py
│       │
│       ├── media/
│       │   ├── __init__.py
│       │   ├── ffmpeg.py
│       │   ├── validate.py
│       │   ├── concat.py
│       │   ├── audio.py
│       │   └── timeline.py
│       │
│       └── doctor/
│           ├── __init__.py
│           ├── command.py
│           ├── cuda.py
│           ├── gpu.py
│           ├── models.py
│           └── benchmark.py
│
├── workers/
│   ├── video_worker.py
│   ├── audio_worker.py
│   └── director_worker.py
│
├── configs/
│   ├── default.toml
│   ├── hardware/
│   │   └── generic-16gb-8gb.toml
│   └── examples/
│       └── peaceful-neon.toml
│
├── scripts/
│   ├── bootstrap.sh
│   ├── doctor.sh
│   └── download_models.sh
│
├── docs/
│   ├── INSTALL.md
│   ├── MODELS.md
│   ├── BACKENDS.md
│   ├── ARCHITECTURE.md
│   ├── STATE_AND_RECOVERY.md
│   ├── PROMPTING.md
│   ├── AUDIO.md
│   ├── OPERATIONS.md
│   ├── TROUBLESHOOTING.md
│   ├── LICENSES.md
│   └── BENCHMARKING.md
│
└── tests/
    ├── unit/
    ├── integration/
    ├── crash/
    ├── fake_backends/
    └── fixtures/
```

Avoid creating more modules than the responsibilities justify.

---

# 14. Configuration design

Use **TOML** for the `voyage`-owned configuration.

Reasoning:

- easy to hand edit;
- standard library support is mature;
- compact syntax;
- better fit for a CLI application than embedding a giant YAML schema;
- backend-specific configuration may remain in native formats if required.

The supervisor should not attempt to normalize every LongLive config field. Instead, define an explicit `LongLiveProfile` in the voyage configuration and translate it into the upstream configuration contract.

Example:

```toml
schema_version = 1

[run]
output_dir = "./runs"
name = "neon-voyage"
seed = 913827

[style]
prompt = "pastel neon line-art, peaceful, slow cinematic motion, dreamlike, subtle surrealism"

# Controller targets, not direct diffusion parameters.
motion_energy_min = 0.20
motion_energy_max = 0.35
visual_complexity_min = 0.30
visual_complexity_max = 0.50
semantic_drift_min = 0.12
semantic_drift_max = 0.25
surrealism = 0.70
transition_smoothness = 0.90

[voyage]
allow_concept_revisit = false
major_transition_min_seconds = 30
major_transition_max_seconds = 120
world_decision_interval_seconds = 16
blocks_per_prompt_stage = 3

[video]
backend = "longlive2"
fps = 24
render_width = 1280
render_height = 704
final_width = 768
final_height = 432
num_frame_per_block = 8
local_attn_size = 32
sink_size = 8
use_relative_rope = true
multi_shot_sink = true
multi_shot_rope_offset = 8.0
sampling_steps = 2
kv_quant = true
streaming_vae = true
async_vae = true

[video.longlive]
repo = "https://github.com/NVlabs/LongLive"
repo_revision = "<pin a commit at implementation time>"
checkpoint = "<local path>"
model_id = "Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2"

[audio]
backend = "acestep"
segment_seconds = 45
crossfade_seconds = 2.0
sample_rate = 48000
channels = 2

audio.acestep]
dit_model = "acestep-v15-turbo"
lm_model = "acestep-5Hz-lm-0.6B"

[director]
model_id = "Qwen/Qwen3-8B"
provider = "local"
temperature = 0.85
max_tokens = 2500

[novelty]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
reject_cosine_similarity = 0.84
```

The example thresholds are starting points, not scientific constants. Tune them experimentally.

The implementation must validate all numeric controller ranges and reject impossible configurations before model startup.

---

# 15. Style charter model

The style charter is represented by a dedicated immutable object:

```python
class StyleSpec(BaseModel):
    prompt: str
    motion_energy_min: float
    motion_energy_max: float
    visual_complexity_min: float
    visual_complexity_max: float
    semantic_drift_min: float
    semantic_drift_max: float
    surrealism: float
    transition_smoothness: float
```

Do not store style only as an arbitrary string.

The freeform prompt gives models descriptive language, while the numeric controller values give deterministic policy code something to reason about.

The user may later provide more style dimensions without changing the core architecture.

---

# 16. World-state model

The world state is mutable.

Recommended structure:

```python
class WorldState(BaseModel):
    concept_id: str
    canonical_concept: str
    scene_summary: str
    visual_subjects: list[str]
    environment: str
    materials: list[str]
    spatial_scale: str
    motion_description: str
    mood: str
    camera_behavior: str
    transition_phase: str
    destination_concept_id: str | None = None
    destination_concept: str | None = None
    age_seconds: float
```

The exact fields can evolve, but all persisted state must be versioned.

Do not put arbitrary non-schema JSON into `state.json`.

---

# 17. Transition-state model

The director should not simply replace the current prompt.

Represent a transition explicitly:

```python
class TransitionPlan(BaseModel):
    source_concept: str
    destination_concept: str
    mechanism: Literal[
        "material_metamorphosis",
        "environmental_transformation",
        "scale_shift",
        "geometric_transformation",
        "physical_rule_change",
        "lighting_transformation",
        "perceptual_transformation",
        "hybrid",
    ]
    transition_strength: float
    estimated_duration_seconds: float
    intermediate_stages: list[str]
    major_transition: bool
```

The transition mechanism is important because it gives the video model a narrative bridge.

For example:

```text
source: mechanical orchard
destination: submerged coral ecosystem
mechanism: environmental_transformation + material_metamorphosis
```

can produce much more coherent prompt staging than merely supplying two nouns.

---

# 18. Prompt-generation policy

Every video prompt must be constructed by code from three layers:

```text
STYLE_PREFIX
    + WORLD_STATE / TRANSITION DESCRIPTION
    + MOTION / CAMERA CONSTRAINTS
```

Do not allow the director to create the entire prompt string without policy wrapping.

For example:

```text
STYLE:
pastel neon line-art, peaceful, slow cinematic motion, dreamlike, restrained surrealism

WORLD:
a luminous fungal metropolis built inside a hollow crystal cavern

TRANSITION:
the rigid fungal architecture gradually becomes translucent mineral growth

MOTION:
very slow forward camera movement, gentle organic motion, no abrupt cuts
```

The code should combine these into a concise renderer prompt.

## 18.1 Style enforcement

Before a prompt reaches LongLive:

1. Trim whitespace.
2. Verify the generated prompt contains the required semantic style policy through code-level injection, not by trusting LLM compliance.
3. Prepend the immutable style prefix.
4. Append motion constraints derived from `StyleSpec`.
5. Reject prompts containing explicit attempts to override the style charter if a policy parser can identify them.

Do not rely entirely on negative prompts for style persistence.

## 18.2 Prompt staging

The director should generate a small number of semantic stages, each reused for multiple LongLive blocks.

Example:

```text
stage 0: unchanged world, establish atmosphere
stage 1: subtle material change
stage 2: environmental response
stage 3: structure transforms
stage 4: destination becomes dominant
```

With 8-latent-frame blocks (~1.33 s each), perhaps 3 blocks per stage initially produces ~4 seconds per stage.

The exact numbers should be configurable.

---

# 19. Director output schema

The director worker must return structured JSON matching a Pydantic schema.

Example:

```json
{
  "destination": {
    "canonical_name": "luminous coral archive",
    "summary": "an immense underwater archive made from translucent coral and softly glowing mineral shelves"
  },
  "transition": {
    "mechanism": "hybrid",
    "transition_strength": 0.24,
    "estimated_duration_seconds": 64,
    "intermediate_stages": [
      "crystal structures become porous",
      "water begins occupying spaces that previously held air",
      "rigid shelves become coral-like growth",
      "bioluminescent organisms appear gradually"
    ],
    "major_transition": false
  },
  "video": {
    "stages": [
      "...",
      "...",
      "...",
      "..."
    ]
  },
  "audio": {
    "music_caption": "slow ambient electronic composition with glass harmonics and watery resonance",
    "energy": 0.48,
    "tempo_bpm": 74,
    "texture": "airy, crystalline, submerged",
    "environment": [
      "distant cavern resonance",
      "soft underwater movement"
    ]
  },
  "novelty": {
    "why_new": "distinct environment and physical motif compared with known concepts"
  }
}
```

The worker must reject invalid JSON and retry generation rather than returning malformed data downstream.

---

# 20. Director prompt design

The director system prompt should clearly state:

- it is an autonomous audiovisual art director;
- the user owns the style charter;
- the director owns subject matter;
- the voyage should evolve continuously;
- transitions should be gradual;
- old canonical concepts are forbidden unless configuration explicitly allows revisits;
- the output must be machine-readable JSON;
- prompts must describe change mechanisms rather than abrupt substitution;
- music may evolve more strongly than visuals;
- no hidden changes to user-owned style constraints.

The director should receive:

```text
STYLE CHARTER
CURRENT WORLD
CURRENT TRANSITION
RECENT WORLD SUMMARY
FORBIDDEN CONCEPT SUMMARY
CURRENT AUDIO STATE
TARGET CONTROLLER METRICS
```

It should not receive an ever-growing raw transcript.

---

# 21. Concept novelty memory

The novelty system is independent of the director.

## 21.1 Persistent files

Recommended:

```text
concepts.jsonl
concept_vectors.npy
concept_index.json
```

Each concept record:

```json
{
  "id": "concept-000184",
  "canonical_name": "luminous coral archive",
  "summary": "...",
  "embedding_index": 183,
  "first_segment": 42,
  "created_at": "2026-09-21T23:11:42Z"
}
```

## 21.2 Similarity policy

For every proposed concept:

1. canonicalize the concept text;
2. embed it;
3. compare it against prior concepts;
4. reject if similarity exceeds the configured threshold;
5. ask the director for another concept;
6. repeat for a bounded number of attempts;
7. if all attempts fail, use a deterministic novelty fallback prompt.

A cosine threshold around 0.82–0.88 is a reasonable initial exploration range, but it is not a universal truth. Store the threshold in configuration and benchmark its qualitative behavior.

## 21.3 Motif return policy

The default should be:

```toml
allow_concept_revisit = false
```

A later configuration may allow transformed motifs, but the novelty system should still demand substantial semantic distance.

---

# 22. LongLive 2.0 streaming integration

This is the most technically important implementation section.

## 22.1 Do not call `pipeline.inference()` for every segment

The current upstream `CausalDiffusionInferencePipeline.inference()` is a finite-call API.

It initializes or resets causal caches for an inference call. Repeated calls therefore do not constitute one continuous causal generation stream.

`voyage` must extract the per-block generation loop into a persistent session abstraction.

Recommended class:

```python
class LongLiveStreamSession:
    async def append_blocks(
        self,
        prompts: Sequence[str],
        *,
        seed: int,
        scene_cut: Sequence[bool],
    ) -> AsyncIterator[VideoBlock]: ...

    async def recover(
        self,
        checkpoint: VideoRecoveryCheckpoint,
    ) -> None: ...

    async def close(self) -> None: ...
```

The exact implementation may be synchronous internally; the public worker RPC remains asynchronous.

## 22.2 Extract from upstream `_inference_inner`

The core logic should be based on the current upstream code in:

```text
pipeline/causal_diffusion_inference.py
```

Relevant concepts:

- `encode_prompt_blocks`
- `_initialize_kv_cache`
- `_initialize_crossattn_cache`
- `_initialize_sample_scheduler`
- per-block denoising loop
- clean timestep-zero cache update
- `_is_scene_cut`
- `_pin_current_chunk`
- streaming VAE logic
- KV rolling
- relative RoPE.

Do not copy the entire file blindly into `voyage`.

Create a minimal compatibility adapter around the smallest subset of LongLive that is required.

## 22.3 Session initialization

At worker startup:

1. Load the LongLive generator.
2. Load T5 text encoder.
3. Load VAE.
4. Load and validate checkpoint.
5. Apply quantization if configured.
6. Allocate KV cache.
7. Allocate cross-attention caches.
8. Configure local attention.
9. Configure sink.
10. Configure relative RoPE.
11. Warm up the GPU with a tiny validated inference.
12. Report readiness to supervisor.

## 22.4 Per-block generation

For each new block:

1. Determine current block index.
2. Select the prompt.
3. Encode it, or use a cached text embedding if repeated.
4. Invalidate cross-attention cache if the prompt differs from the previous block.
5. Create the block noise from the run RNG stream.
6. Denoise using the configured step count.
7. Store the resulting latent in the persistent recovery buffer.
8. Perform the clean timestep-zero forward pass to update the causal KV cache.
9. Apply scene-cut / sink logic if needed.
10. Optionally decode the block via streaming VAE.
11. Return the block to the supervisor.
12. Never advance the supervisor's committed segment index here.

The worker acknowledges that a block is **generated**, not **committed**.

---

# 23. LongLive relative RoPE

The upstream LongLive 2.0 code added relative RoPE support as part of its move toward infinite video generation.

In `wan_5b/modules/causal_model.py`, `use_relative_rope` changes how keys are stored and how RoPE is applied to the attended window.

The cache can therefore roll without requiring a monotonically growing absolute positional representation to remain inside the trained context range.

For `voyage`:

- make `use_relative_rope` a first-class configuration option;
- test it independently before enabling multi-hour production runs;
- record whether it was enabled in `run_manifest.json`;
- never silently change it during resume.

If the exact LongLive revision used by the project changes the mechanism, update this section and the adapter accordingly.

---

# 24. LongLive KV cache model

The current LongLive 2.0 causal self-attention code maintains a rolling cache.

Important concepts:

```text
persistent history
      │
      ▼
┌─────────────────────────────┐
│ global sink                 │
├─────────────────────────────┤
│ optional pinned scene sink  │
├─────────────────────────────┤
│ rolling local attention     │
└─────────────────────────────┘
      ▲
      │
 new block inserted here
```

`local_attn_size` controls how many temporal frames participate in the local rolling window.

`multi_shot_sink` can preserve special chunks across scene transitions.

Do not modify cache tensors from unrelated threads.

The video worker should be single-threaded with respect to GPU inference.

The only exception is the VAE decoding helper where upstream explicitly supports asynchronous/pipeline decoding.

---

# 25. Scene transitions and sink behavior

Not every semantic evolution is a “scene cut.”

Distinguish:

- **micro mutation** — subtle change inside the same scene;
- **medium transition** — gradual environmental/physical change;
- **major transition** — the current world gives way to a substantially different destination.

Only major transitions should normally activate scene-cut-specific sink behavior.

A major transition prompt should start with the LongLive scene-cut prefix expected by the current implementation if the selected sink logic depends on that prefix.

Do not mark every prompt change as a scene cut.

---

# 26. Text embedding caching

A practical optimization is to cache the encoded text embeddings for prompts that are repeated across multiple blocks.

The cache key must include:

- exact prompt text;
- model ID;
- checkpoint revision;
- text encoder revision;
- dtype;
- tokenizer revision.

Cache values may live in process memory during one run, but they should not be persisted as a required recovery artifact for V1.

On worker restart, regenerate embeddings from text.

---

# 27. Video recovery model

The video worker must persist a **recovery tail** at every segment commit.

Recommended content:

```text
recovery.pt
```

containing:

```text
- recent clean latent blocks needed to rebuild local cache;
- enough global/pinned sink information to recreate configured scene context;
- absolute logical block index;
- RNG state if required by implementation;
- model profile identifier;
- dtype.
```

Do not attempt to serialize the whole GPU KV cache in V1.

## 27.1 Rebuild strategy

On worker restart:

```text
load recovery.pt
    ↓
allocate empty KV cache
    ↓
replay recovery latent blocks using clean timestep=0
    ↓
restore scene/sink metadata
    ↓
set logical block index
    ↓
mark worker READY
```

The purpose is to restore the *causal context*, not to regenerate historical video.

## 27.2 Recovery correctness

Recovery must have a deterministic fake-backend test.

For the fake backend:

```text
run blocks 0..N
crash after N
restart
recover
run N+1..M
```

must yield exactly the same logical state and fake media sequence as a single uninterrupted run.

Real GPU inference need not be byte-for-byte deterministic across hardware, drivers, kernels, or quantization implementations.

---

# 28. Segment design

A segment is the transactional unit of the voyage.

Recommended initial logical size:

```text
16–32 seconds of video
```

At 24 fps and 8 latent frames per block:

```text
1 block ≈ 1.333 s
12 blocks ≈ 16 s
24 blocks ≈ 32 s
```

This is deliberately long enough to hide individual prompt-stage boundaries while short enough to checkpoint frequently.

The exact size is configurable.

---

# 29. Segment directory

Example:

```text
segments/
└── 000042/
    ├── video.mp4
    ├── audio.wav
    ├── world_state.json
    ├── transition.json
    ├── prompt_plan.json
    ├── audio_state.json
    ├── metrics.json
    ├── recovery.pt
    ├── sha256.json
    └── DONE
```

A segment must never be modified after `DONE` is created.

If an implementation requires a replacement, create a new attempt and update the manifest through a new transaction rather than mutating historical artifacts.

---

# 30. Transactional segment commit

A segment is committed in this order:

```text
1. Generate video
2. Generate audio
3. Validate media
4. Write metadata to .partial files
5. fsync metadata
6. Atomically rename metadata
7. Write recovery checkpoint to .partial
8. fsync checkpoint
9. Atomically rename checkpoint
10. Compute and persist checksums
11. Write DONE.partial
12. fsync
13. rename DONE
14. Atomically update state.json
15. Atomically update run manifest / committed index
```

The precise order can be simplified, but the invariant must remain:

> No state file may claim a segment is committed until the segment's required artifacts are valid and durably present.

---

# 31. Atomic file handling

Use:

```python
write temp
flush
os.fsync(fd)
os.replace(temp, destination)
```

where appropriate.

Never do:

```python
open("state.json", "w")
json.dump(...)
```

as the only persistence mechanism for critical state.

A partially written state file must not be able to destroy the previous valid state.

---

# 32. Run manifest

Each run gets:

```text
run_manifest.json
```

It must include:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "created_at": "...",
  "voyage_git_revision": "...",
  "config_sha256": "...",
  "style_spec": { },
  "hardware": { },
  "software": { },
  "models": {
    "video": {
      "repo": "NVlabs/LongLive",
      "revision": "...",
      "model_id": "...",
      "checkpoint_sha256": "...",
      "license": "...",
      "license_url": "..."
    },
    "audio": { },
    "director": { },
    "embedding": { }
  },
  "seed": 913827,
  "timeline": {
    "fps": 24,
    "final_width": 768,
    "final_height": 432
  },
  "committed_segments": 0
}
```

Capture exact model revisions whenever the source supports them.

If a model is downloaded by path only, record the path plus local file checksum.

---

# 33. Current world state file

`state.json` contains the current resumable director state.

It must include:

- current segment number;
- next segment number;
- world state;
- transition state;
- audio state;
- recent concepts;
- RNG metadata;
- controller metrics;
- worker checkpoint identifiers;
- lifecycle status.

Example lifecycle values:

```text
CREATED
STARTING
RUNNING
PAUSE_REQUESTED
PAUSED
STOP_REQUESTED
FINALIZING
COMPLETE
FAILED
PAUSED_DISK_FULL
```

Do not encode lifecycle status only in the process exit code.

---

# 34. Timeline model

Timeline correctness must be frame-based, not wall-clock-based.

The canonical video timeline is:

```text
absolute_frame_index
```

derived from exact decoded output.

Useful conversions:

```python
def latent_block_to_video_frames(latent_frames: int, temporal_ratio: int) -> int:
    return latent_frames * temporal_ratio


def frames_to_seconds(frames: int, fps: int) -> Fraction:
    return Fraction(frames, fps)
```

Do not use floating-point duration as the source of truth.

Every segment should record:

```text
start_frame
end_frame_exclusive
frame_count
fps
```

Audio should be mapped to the same logical timebase.

---

# 35. Audio timeline

The audio segmenter should be independent of video block size.

Recommended initial audio generation cadence:

```text
30–60 second music segments
```

with:

```text
~2 seconds crossfade
```

The director can issue a new music plan every 30–90 seconds without forcing a video world transition.

This allows the audio to feel alive while preserving visual continuity.

The audio worker should be capable of producing the next segment ahead of the video timeline when GPU capacity permits.

---

# 36. Audio state

Recommended model:

```python
class AudioState(BaseModel):
    music_caption: str
    genre: str
    energy: float
    tempo_bpm: float | None
    tonal_brightness: float
    harmonic_complexity: float
    instrumentation: list[str]
    ambience: list[str]
    active_events: list[str]
    previous_audio_summary: str
```

The director should modify audio state independently of visual transition strength.

---

# 37. ACE-Step continuation strategy

The audio worker should hide ACE-Step-specific continuation details behind:

```python
class AudioBackend(Protocol):
    async def generate_music(
        self,
        plan: MusicPlan,
        *,
        duration_seconds: float,
        reference_audio: Path | None,
        seed: int,
    ) -> AudioArtifact: ...
```

The V1 backend should implement the continuation workflow supported by the exact ACE-Step 1.5 release installed in the environment.

Do not guess an internal ACE-Step API from documentation if the installed release differs.

The adapter must include an integration test that:

1. generates a short piece;
2. verifies output duration;
3. verifies the audio file can be decoded by ffmpeg/libav;
4. verifies continuation/repaint mode if used by the implementation.

Keep ACE-Step API compatibility isolated to `audio/acestep.py`.

---

# 38. Audio mixing

Use WAV or another lossless PCM representation for intermediate audio.

A segment may contain:

```text
music.wav
ambience.wav
sfx.wav
```

which are mixed into:

```text
audio.wav
```

before segment commit.

Final MP4 encoding should be the first point where a lossy compressed audio codec is required, unless a lossless audio codec in the chosen container is intentionally preferred.

Recommended final AAC profile initially:

```text
AAC 256–320 kb/s stereo
```

The exact rate should be configurable.

---

# 39. Loudness policy

Do not normalize every segment independently to maximum amplitude.

Independent per-segment peak normalization causes audible pumping when the music changes.

Instead:

- leave generation levels untouched during creative generation;
- use a consistent final mix target;
- apply a single final loudness policy where practical;
- preserve headroom before the final limiter.

A later version may use measured LUFS-based targets, but this is secondary to continuity.

---

# 40. Supervisor scheduling

The supervisor should schedule at three conceptual rates.

## Fast loop

Video blocks.

```text
~1.33 seconds of video per block in the reference configuration
```

## Medium loop

World evolution prompt stages.

```text
~4–16 seconds
```

## Slow loop

New conceptual destination.

```text
~30–120+ seconds
```

Audio planning has its own cadence, typically:

```text
~30–60 seconds
```

Do not synchronize all clocks to one interval.

---

# 41. Segment planning algorithm

At the beginning of a new segment:

```text
load committed state
      ↓
observe current world
      ↓
decide whether current transition should continue
      │
      ├── yes → extend existing transition
      │
      └── no  → generate new destination
                   ↓
             novelty validation
                   ↓
             stage transition
                   ↓
             derive audio plan
```

The director should not invent a completely new destination at every segment.

The current transition should usually span several segments.

---

# 42. Transition phases

Use behavioral phases rather than a fixed story structure.

Suggested state machine:

```text
ESTABLISH
   ↓
DRIFT
   ↓
TRANSFORM
   ↓
DESTABILIZE
   ↓
EMERGE
   ↓
STABILIZE
   └──────────────→ new destination
```

These are not literal prompts.

They describe how strongly the director should push semantic change.

Examples:

### ESTABLISH

- preserve subject identity;
- low semantic change;
- introduce texture and camera movement.

### DRIFT

- introduce subtle abnormalities;
- change material behavior;
- change lighting/environment.

### TRANSFORM

- stronger environmental transformation;
- architecture/material/scale changes.

### DESTABILIZE

- more surreal relationships;
- still obey peaceful/smooth style constraints.

### EMERGE

- destination concept becomes recognizable.

### STABILIZE

- hold the new world long enough for the viewer to perceive it.

This gives the LLM a reusable temporal grammar.

---

# 43. Visual controller feedback metrics

These metrics are initially optional but the architecture should reserve fields for them.

Suggested normalized metrics:

```text
motion_energy
visual_complexity
semantic_change_rate
palette_distance
style_similarity
scene_boundary_strength
```

A future visual inspector may estimate these from frames.

The controller can then implement simple feedback:

```text
if motion_energy > target_max:
    reduce camera motion
    reduce action verbs

if semantic_change_rate < target_min:
    increase transition strength

if complexity > target_max:
    remove object density
    simplify environment

if style_similarity < target_min:
    strengthen immutable style prefix
```

Do not make the first version depend on a VLM for basic operation.

---

# 44. Visual inspector phase

When implemented, the inspector should operate asynchronously after segment commit.

It should:

1. sample a small number of frames;
2. produce a concise textual scene summary;
3. estimate visual metrics;
4. write `metrics.json`;
5. update the director's next-decision context.

It should **not** block video generation unless an explicit closed-loop mode is enabled.

This is important: an unavailable or slow inspector must not stop an otherwise healthy voyage.

---

# 45. Worker RPC protocol

Use JSON Lines over stdin/stdout or a UNIX domain socket.

The preferred V1 arrangement is:

- supervisor launches worker process;
- requests written to worker stdin;
- responses emitted to stdout;
- diagnostics emitted to stderr.

Do not mix diagnostics into stdout.

## Request

```json
{
  "id": "req-000042",
  "op": "generate_blocks",
  "payload": {
    "segment_id": "000042",
    "block_start": 312,
    "prompts": ["...", "..."],
    "scene_cut": [false, false],
    "seed": 918273
  }
}
```

## Success response

```json
{
  "id": "req-000042",
  "ok": true,
  "result": {
    "blocks_generated": 2,
    "artifacts": [
      "..."
    ],
    "next_block_index": 314
  }
}
```

## Error response

```json
{
  "id": "req-000042",
  "ok": false,
  "error": {
    "code": "VIDEO_OOM",
    "message": "CUDA out of memory",
    "retryable": true
  }
}
```

Every operation has a stable operation name and typed payload schema.

---

# 46. Worker operations

Minimum worker API:

```text
init
health
generate_blocks
checkpoint
resume
shutdown
```

Optional:

```text
benchmark
warmup
metrics
```

`generate_blocks` must be idempotent at the supervisor transaction level even if the worker itself internally retries.

The supervisor should never issue two concurrent GPU-generation requests to the same worker.

---

# 47. Supervisor state machine

The supervisor controls lifecycle.

```text
CREATED
   ↓
STARTING
   ↓
RUNNING
   ├───────────────┐
   │               │
pause             failure
   │               │
   ▼               ▼
PAUSE_REQUESTED   RECOVERING
   │               │
   ▼               └──────► RUNNING
PAUSED
   │
resume
   │
   └──────────────► RUNNING

RUNNING → STOP_REQUESTED → FINALIZING → COMPLETE
```

The supervisor should not mark a worker as failed merely because one media generation attempt failed.

Use error classification.

---

# 48. Error classes

Define explicit errors.

```python
class VoyageError(Exception): ...


class ConfigurationError(VoyageError): ...


class WorkerError(VoyageError): ...


class RecoverableWorkerError(WorkerError): ...


class FatalWorkerError(WorkerError): ...


class MediaError(VoyageError): ...


class StateError(VoyageError): ...


class ModelCompatibilityError(VoyageError): ...


class DiskSpaceError(VoyageError): ...
```

The supervisor should make restart decisions based on error class rather than string matching arbitrary exception messages.

---

# 49. OOM handling

A video OOM is recoverable if the process can be safely restarted.

Policy:

```text
OOM
 ↓
stop current request
 ↓
kill video worker
 ↓
validate last committed segment
 ↓
restart video worker
 ↓
replay recovery checkpoint
 ↓
retry with the same logical block
```

If repeated OOM occurs:

1. lower resolution profile;
2. lower local attention size;
3. enable KV quantization if not already enabled;
4. reduce VAE workload / move VAE where supported;
5. fall back to a lower-quality profile;
6. eventually fail the run with an actionable error.

Do not automatically skip video blocks after OOM.

Skipping causes timeline holes.

---

# 50. Audio failure handling

A temporary audio worker failure should not invalidate already generated video.

Policy:

```text
audio worker crash
    ↓
restart worker
    ↓
regenerate current audio segment from saved AudioPlan
```

The video segment may remain uncommitted until both required media streams exist.

A configuration flag may later permit video-only archival segments, but the default finalization policy should require a valid audio stream for every committed segment.

---

# 51. Director failure handling

If the director returns invalid JSON:

1. retry with a stricter prompt;
2. lower temperature;
3. validate against schema;
4. if still invalid, fall back to deterministic continuation of the current transition.

A bad LLM response must never corrupt persistent state.

The deterministic fallback can simply be:

```text
continue current stage
reduce semantic mutation
preserve style
```

This allows the rendering system to survive transient LLM issues.

---

# 52. SIGINT / termination handling

`voyage stop` and Ctrl-C must be graceful.

The supervisor should:

1. set a stop flag;
2. stop scheduling new blocks;
3. allow the currently executing GPU block to reach a safe boundary if practical;
4. request worker checkpoint;
5. complete any valid in-flight media transaction;
6. update state to `PAUSED` or `STOPPED`/equivalent;
7. exit cleanly.

Never abruptly `kill -9` the GPU worker unless graceful shutdown fails.

---

# 53. Disk space handling

Before every segment commit, estimate free space.

Maintain a configured minimum free-space reserve:

```toml
min_free_space_gib = 20
```

If the reserve would be crossed:

```text
RUNNING
  ↓
PAUSED_DISK_FULL
```

The supervisor must not allow a final metadata write to fail because the disk was filled by the previous media file.

---

# 54. Media validation

Every generated media artifact must be validated before commit.

For video:

```text
ffprobe succeeds
stream exists
codec is expected
frame rate is expected
frame count > 0
duration > 0
resolution matches expected backend output
```

For audio:

```text
ffprobe succeeds
sample rate is expected
channel count is expected
duration > 0
```

The validator should use the executable directly, not a shell pipeline.

---

# 55. ffmpeg process execution

Always invoke ffmpeg with an argument list.

Good:

```python
await asyncio.create_subprocess_exec(
    "ffmpeg",
    "-hide_banner",
    "-nostdin",
    "-i",
    str(input_path),
    ...,
)
```

Bad:

```python
subprocess.run(f"ffmpeg ... {user_string}", shell=True)
```

No shell interpolation.

Capture stderr and include the relevant last lines in `MediaError`.

---

# 56. Finalization

The final output is produced only by:

```bash
voyage finalize --run RUN_DIR --output final.mp4
```

Finalizer steps:

1. scan segment directories;
2. identify valid committed segments;
3. verify `DONE` markers;
4. verify checksums if enabled;
5. validate monotonic frame ranges;
6. validate audio/video duration alignment;
7. create a concat manifest;
8. attempt a stream-copy concat when codec/container compatibility is guaranteed;
9. otherwise perform exactly one final encode;
10. scale/pad to exact 768×432 if needed;
11. mux audio;
12. validate final media;
13. atomically publish final path.

Never mutate the source segment files during finalization.

---

# 57. Final video normalization

The final output target is:

```text
768×432
24 fps
16:9
```

Use a scale+pad operation that preserves the entire image where possible.

Conceptually:

```text
scale proportionally to fit within 768×432
pad remaining pixels symmetrically
```

Do not crop creative content unless the configuration explicitly requests cropping.

The finalizer must report the exact transform it applied.

---

# 58. CLI specification

The first stable command set should be:

```text
voyage init
voyage doctor
voyage models
voyage benchmark
voyage run
voyage status
voyage pause
voyage resume
voyage stop
voyage validate
voyage finalize
voyage inspect
```

## `voyage init`

Example:

```bash
voyage init \
  --output ./runs/my-voyage \
  --style "pastel neon line-art, peaceful, slow cinematic motion"
```

Creates:

```text
voyage.toml
run_manifest.json
state.json
concepts.jsonl
segments/
logs/
```

No model should start.

## `voyage doctor`

Checks:

- NVIDIA driver;
- CUDA runtime;
- available GPUs;
- compute capability;
- VRAM;
- PyTorch;
- FlashAttention/Triton availability;
- ffmpeg;
- model files;
- model/checkpoint compatibility;
- filesystem permissions;
- free space;
- worker Python interpreters;
- ACE-Step availability.

## `voyage models`

Subcommands:

```text
voyage models list
voyage models download
voyage models verify
voyage models info
```

Model downloads must be explicit.

## `voyage run`

Starts the autonomous voyage.

Optional:

```bash
voyage run --run ./runs/my-voyage
```

## `voyage status`

Display:

- runtime;
- segment count;
- exact timeline duration;
- current concept;
- current destination;
- transition phase;
- audio state;
- worker health;
- disk space;
- recent error.

## `voyage pause`

Request a safe pause.

## `voyage resume`

Resume from the last committed state.

## `voyage stop`

Safely stop generation.

Optional:

```bash
voyage stop --finalize
```

## `voyage validate`

Offline consistency check for a run.

## `voyage finalize`

Create one final MP4.

## `voyage benchmark`

Examples:

```text
voyage benchmark video
voyage benchmark audio
voyage benchmark end-to-end
```

---

# 59. `voyage status` output

The CLI should present human-readable information similar to:

```text
Voyage: neon-voyage
Status: RUNNING
Uptime: 03:17:42

Video
  Backend: LongLive 2.0 / NVFP4 S2
  Render: 1280×704
  Timeline: 00:47:21
  Segments: 177
  Current block: 8 / 24
  GPU: 16.2 GB / 16 GB [quantized]

World
  Current: luminous fungal metropolis
  Destination: crystalline ocean archive
  Phase: TRANSFORM
  Novelty: accepted

Audio
  Backend: ACE-Step 1.5
  Music: ambient electronic / glass harmonics
  Energy: 0.47
  Buffered audio: 92 s

Workers
  video: READY
  audio: READY
  director: READY

Storage
  Free: 814 GiB
```

Human-readable formatting can evolve without changing the machine-readable state files.

---

# 60. Logging

Use Python `logging` with structured fields.

Every log entry should include where useful:

```text
run_id
segment_id
block_index
worker
operation
attempt
elapsed_ms
```

Worker stdout is reserved for RPC responses.

Worker stderr contains logs.

Supervisor should capture worker stderr to:

```text
logs/video-worker.log
logs/audio-worker.log
logs/director-worker.log
```

Use log rotation.

Do not let logs grow without bound during multi-day runs.

---

# 61. Observability metrics

The supervisor should track at least:

```text
video_blocks_generated_total
audio_segments_generated_total
director_decisions_total
worker_restarts_total
video_oom_total
media_validation_failures_total
concept_rejections_total
segment_commit_total
segment_commit_failures_total
video_block_seconds
video_generation_seconds
audio_generation_seconds
director_generation_seconds
```

Also report gauges:

```text
current_timeline_seconds
audio_buffer_seconds
free_disk_bytes
video_vram_bytes
audio_vram_bytes
```

V1 can emit these to a JSONL metrics stream instead of Prometheus.

---

# 62. Randomness and RNG policy

Use separate random streams for independent concerns.

Recommended deterministic seed derivation:

```text
run_seed
  ├── video_seed(segment, block)
  ├── audio_seed(segment, audio_chunk)
  └── director_seed(decision_index)
```

Do not use one global `random.seed()` and then let unrelated code consume values opportunistically.

A change to director logging should not silently alter video randomness.

Use a stable hash-based seed derivation such as BLAKE2b or SHA-256 truncated to the backend-supported integer range.

Persist logical seeds in the segment metadata.

---

# 63. Determinism policy

The project should be **logically reproducible**, not necessarily bitwise reproducible.

Record:

- base run seed;
- per-segment seed;
- per-block seed;
- model revision;
- configuration hash;
- worker environment versions;
- GPU model;
- driver version;
- quantization mode.

Real diffusion outputs may vary due to:

- GPU architecture;
- CUDA kernels;
- fused kernels;
- attention implementation;
- quantization;
- compiler behavior.

This distinction must be documented.

---

# 64. Hardware probing

`voyage doctor` must not trust a user-written hardware profile blindly.

It should query runtime facts, including:

```bash
nvidia-smi
```

through a safe subprocess invocation, and/or use PyTorch CUDA APIs.

Report:

- GPU name;
- total memory;
- free memory;
- compute capability;
- driver version if available;
- CUDA version;
- PyTorch version.

The target hardware is currently described as:

```text
GPU 0: 16 GB VRAM
GPU 1: 8 GB VRAM
RAM: 64 GB
```

Do not hard-code these identities into application logic.

---

# 65. Hardware profile selection

The doctor should benchmark profiles rather than assuming one exact model configuration.

Suggested video profiles:

```text
longlive2-s2-low
longlive2-s2-native
longlive2-s4-low
longlive2-s4-native
```

Each profile defines:

- checkpoint;
- sampling steps;
- spatial shape;
- block size;
- attention size;
- KV quantization;
- VAE mode;
- relative RoPE;
- sink settings.

The selected profile becomes immutable for the run.

Do not switch from S2 to S4 halfway through a run because an arbitrary quality heuristic changed.

A future migration mechanism may permit it, but V1 should avoid mixing model behavior inside one timeline.

---

# 66. Performance goals

Because the user's local GPU models are unspecified in this document, V1 should establish goals rather than promise specific FPS values.

## Video

Primary development target:

```text
stable continuous inference
constant RAM growth
constant KV-cache size
no gradual throughput degradation
```

Secondary target:

```text
approach real-time generation if hardware permits
```

Upstream LongLive 2.0 reports real-time-class performance on substantially larger datacenter GPUs; those values are useful as architectural evidence but are not local hardware guarantees.

## Audio

ACE-Step's upstream goals are comfortably above real-time for many configurations, so audio generation should generally be schedulable ahead of the video timeline.

---

# 67. Memory constraints

The most important memory invariant is:

> RAM and VRAM usage must not increase linearly with voyage duration.

Video history should not remain as an ever-growing in-memory tensor.

Only retain:

- current KV cache;
- current recovery tail;
- small director state;
- small audio buffer;
- current segment artifacts.

The full video exists on disk, not in Python memory.

The full concept history exists on disk, not as one giant prompt.

---

# 68. Long-running stability test

A release candidate should run for at least:

```text
8–12 hours
```

and preferably a multi-day endurance test.

Measure:

- process RSS;
- GPU VRAM;
- generation latency;
- worker restart count;
- file count;
- disk throughput;
- concept rejection rate;
- audio buffer depth.

Plot memory against generated timeline duration.

Expected behavior:

```text
RSS: approximately flat
VRAM: approximately flat
KV cache: flat
Disk: linear with media duration
Concept files: sublinear / modest
```

A monotonic VRAM trend is a release blocker unless clearly explained by an intentionally growing cache with a fixed upper bound.

---

# 69. Crash-injection testing

Provide a test harness capable of intentionally killing workers at controlled points.

Examples:

```text
before video generation
mid video generation
after video file write
before audio write
before metadata rename
before checkpoint rename
before DONE
before state update
after state update
```

Every injected failure should leave a recoverable filesystem state.

The validator must identify incomplete artifacts without confusing them with committed segments.

---

# 70. State repair

`voyage validate` should be able to detect:

- orphan `.partial` files;
- missing `DONE` markers;
- checksum mismatches;
- invalid segment numbering;
- invalid frame ranges;
- missing recovery checkpoints;
- inconsistent `state.json` vs segment directories;
- impossible final durations.

It may offer a safe repair mode later.

V1 should be conservative:

```text
validate = read-only
repair = explicit command
```

Never silently mutate a run during validation.

---

# 71. Idempotence rules

A retry of a failed segment must not duplicate committed media.

The supervisor should identify work by:

```text
(run_id, segment_id, attempt_id)
```

A segment directory such as:

```text
segments/000042/
```

contains only the committed attempt.

Failed attempts can live under:

```text
segments/000042/attempts/0001/
```

if debugging requires them.

Once the successful attempt commits, the supervisor records its artifact hash.

---

# 72. Concurrency rules

V1 should deliberately limit concurrency.

Allowed:

- director can plan the next transition while audio or video is rendering;
- audio worker can render ahead;
- VAE decoding can use upstream-supported asynchronous mechanisms;
- filesystem validation can happen independently.

Disallowed in V1:

- two concurrent LongLive generations on the same GPU;
- speculative video branches;
- multiple directors competing to modify state;
- concurrent writes to `state.json`;
- concurrent mutations of concept history.

One component owns each piece of mutable state.

---

# 73. Ownership model

```text
supervisor     owns lifecycle + commit state
video worker   owns GPU video session + causal KV cache
 audio worker  owns audio generation process
 director      owns ephemeral creative proposals
 novelty store owns immutable concept history
 finalizer     owns final media assembly
```

The director proposes state transitions.

The supervisor validates and commits them.

This prevents an LLM from directly mutating the persistent run state.

---

# 74. Director proposal transaction

The director should never write `state.json` itself.

Instead:

```text
Director
  ↓
EvolutionDecision
  ↓
Pydantic validation
  ↓
novelty check
  ↓
style policy check
  ↓
supervisor accepts proposal
  ↓
persist decision
```

A decision should be immutable once associated with a committed segment.

---

# 75. Prompt history persistence

Persist prompt plans per segment.

Example:

```json
{
  "segment_id": "000042",
  "stages": [
    {
      "stage": 0,
      "block_start": 0,
      "block_end": 2,
      "prompt": "..."
    },
    {
      "stage": 1,
      "block_start": 3,
      "block_end": 5,
      "prompt": "..."
    }
  ]
}
```

This is valuable for debugging why a visual transition occurred.

---

# 76. Why not direct prompt interpolation by the supervisor

Do not linearly interpolate arbitrary text strings.

```text
"forest"
→ "coral reef"
```

has no useful textual interpolation semantics.

Instead interpolate **conceptual state**, then regenerate natural-language prompts from the evolving state.

The LLM is responsible for expressing the transition in words.

The supervisor is responsible for enforcing the timeline and policy.

---

# 77. Why not image-keyframe generation as the primary mechanism

Image-keyframe interpolation is useful but not necessary for V1.

It introduces:

- another image-generation backend;
- extra checkpoints;
- more VAE transfers;
- more state;
- more opportunities for visual resets.

LongLive's native causal continuity is the simpler foundation.

An image-keyframe backend can be added later as a recovery or transition enhancement if tests show the purely causal approach drifts too much.

---

# 78. Why not ComfyUI as the application core

ComfyUI remains valuable for experimentation, but the primary architecture does not need it.

The application is fundamentally a stateful scheduler with long-lived model processes, transactional persistence, and media assembly.

ComfyUI would become an unnecessary central dependency for:

- lifecycle management;
- infinite-loop state;
- crash recovery;
- RPC;
- segment persistence.

Instead:

```text
voyage core
   │
   ├── LongLive adapter
   ├── ACE-Step adapter
   └── optional ComfyUI backend later
```

This preserves the user's ability to experiment with LTXV and other node ecosystems without coupling the production daemon to a graph UI.

---

# 79. Development phases

Each phase must have explicit exit criteria.

## Phase 0 — Environment and hardware doctor

Implement:

- project skeleton;
- `voyage doctor`;
- worker environment detection;
- ffmpeg detection;
- GPU probing;
- model manifest support;
- explicit model verification.

Success criteria:

- both GPUs identified;
- CUDA works;
- LongLive environment can run a smoke test;
- ACE-Step environment can run a smoke test;
- ffmpeg validates test media;
- all model licenses and revisions are recorded.

Do not implement autonomous evolution yet.

---

## Phase 1 — Finite LongLive segment

Implement:

- video worker;
- LongLive checkpoint loading;
- finite block generation;
- one fixed prompt;
- one segment;
- video validation;
- state persistence;
- finalization.

Use no LLM.

Success criteria:

```text
voyage run
→ generates one video segment
→ survives supervisor restart
→ validate passes
→ finalize creates a valid MP4
```

---

## Phase 2 — Persistent LongLive stream

Refactor upstream per-block logic into:

```text
LongLiveStreamSession.append_blocks()
```

Implement:

- persistent KV cache;
- rolling local attention;
- relative RoPE;
- prompt changes per block;
- scene-cut support;
- recovery tail;
- cache reconstruction.

Success criteria:

- 5+ minute uninterrupted generation;
- no unbounded VRAM growth;
- restart and recovery works;
- prompt changes influence subsequent blocks without resetting earlier frames;
- timeline is monotonic.

---

## Phase 3 — Autonomous director

Implement:

- Qwen3-8B director worker;
- structured `EvolutionDecision`;
- style charter;
- transition-state machine;
- novelty embeddings;
- concept history;
- staged prompts;
- deterministic fallback transitions.

Success criteria:

- user supplies only style;
- concepts evolve autonomously;
- old concepts are rejected;
- visual transitions are gradual;
- style remains injected at every block.

---

## Phase 4 — Audio

Implement:

- ACE-Step worker;
- music plans;
- audio state;
- long audio segments;
- continuation/repaint path;
- crossfade;
- audio validation;
- final mux.

Success criteria:

- audio stays ahead or roughly synchronized with video;
- worker crash can recover without losing video;
- final file has continuous audio;
- music changes more aggressively than visuals without sounding random.

---

## Phase 5 — Closed-loop visual inspection

Implement:

- frame sampling;
- optional Qwen3.5-9B inspector;
- visual metrics;
- controller feedback;
- style-adherence feedback;
- motion/complexity regulation.

Success criteria:

- inspector failure does not stop voyage;
- measured metrics affect later prompts;
- visual complexity does not continually rise;
- motion remains within style targets.

---

## Phase 6 — Production hardening

Implement:

- crash injection;
- long-run memory checks;
- disk pressure tests;
- worker restart policies;
- log rotation;
- benchmark reports;
- finalizer robustness;
- operational docs.

Success criteria:

- overnight test succeeds;
- repeated worker crash/recovery succeeds;
- finalization succeeds after partial failures;
- no silent state corruption.

---

## Phase 7 — Alternative backends

Add optional adapters:

- LTXV;
- FramePack;
- MAGI;
- SkyReels;
- newer LongLive releases;
- ComfyUI backend.

The core supervisor should not change materially.

---

# 80. Implementation task decomposition for coding agents

Coding agents should implement work in small, reviewable tasks.

## Task group A — bootstrap

- Create `pyproject.toml`.
- Configure Python 3.12.
- Add Ruff.
- Add Pyright or mypy.
- Add Pytest.
- Add pre-commit if desired.
- Create module skeleton.
- Add CI for unit tests.

## Task group B — configuration

- Implement Pydantic config models.
- Implement TOML loader.
- Validate ranges.
- Compute config hash.
- Write config snapshot into each run.

## Task group C — state

- Implement `state.json`.
- Implement run manifest.
- Implement atomic writes.
- Implement segment transaction state.
- Implement checksum records.

## Task group D — RPC

- Implement typed JSONL protocol.
- Implement subprocess worker supervisor.
- Implement request IDs.
- Implement timeouts.
- Implement graceful shutdown.
- Implement worker restart.

## Task group E — LongLive adapter

- Pin LongLive commit.
- Document required dependency versions.
- Build model loader.
- Build stream session.
- Extract per-block causal generation.
- Add relative RoPE configuration.
- Add recovery replay.
- Add metrics.

## Task group F — director

- Implement Qwen3-8B worker.
- Build prompt templates.
- Implement JSON-schema output.
- Implement retry logic.
- Implement style policy.
- Implement transition policy.

## Task group G — novelty

- Add embedding model.
- Add concept store.
- Add similarity search.
- Add rejection loop.
- Add deterministic fallback.

## Task group H — ACE-Step

- Build separate worker.
- Validate installed API.
- Implement text-to-music.
- Implement continuation.
- Add crossfade.
- Add audio validation.

## Task group I — media

- ffmpeg wrapper;
- ffprobe wrapper;
- concat;
- mix;
- final encode;
- output validation.

## Task group J — CLI

- `init`;
- `doctor`;
- `models`;
- `run`;
- `status`;
- `pause`;
- `resume`;
- `stop`;
- `validate`;
- `finalize`;
- `benchmark`.

## Task group K — testing

- fake workers;
- crash injection;
- recovery tests;
- media fixtures;
- property tests;
- endurance harness.

---

# 81. Coding standards

Use modern Python.

## Required

- Python 3.12 for supervisor.
- Type annotations on public functions.
- Pydantic models for persistent/IPC schemas.
- `pathlib.Path` rather than stringly-typed paths.
- `asyncio` for process supervision and I/O.
- `subprocess` without shell execution.
- Context managers for file handles and process resources.
- Explicit exception taxonomy.
- Unit tests for every persistence invariant.

## Strongly recommended

- Ruff.
- Pyright or mypy.
- pytest.
- hypothesis for state-machine properties.
- coverage.
- pre-commit.

## Avoid

- module-level mutable state;
- hidden threads;
- background tasks without ownership;
- global random generators;
- unbounded queues;
- silent exception handling;
- broad `except Exception: pass`;
- magic numbers in video timing;
- subprocess shell strings;
- writing state directly over the previous valid state.

---

# 82. Type design rules

Backend-specific code should not leak into supervisor types.

Bad:

```python
segment.latent_tensor: torch.Tensor
```

Good:

```python
segment.recovery_artifact: ArtifactRef
```

The supervisor should not need to import PyTorch.

Similarly, `WorldState` should contain semantic state, not prompt embedding tensors.

---

# 83. Dependency boundaries

The supervisor package should be lightweight.

Do not import:

```python
torch
transformers
diffusers
```

into ordinary state/config modules.

The intended dependency graph is:

```text
state/config/media/cli
       │
       ▼
 supervisor
  │       │
  ▼       ▼
RPC     director client
  │
  ├────► LongLive worker
  └────► ACE-Step worker
```

GPU-specific packages remain in workers.

---

# 84. Model version pinning

For every model integration:

```text
provider/repository
model ID
revision / commit
license
local path
checksum if practical
```

Pin Git repositories to commits, not floating branches, for production runs.

During active development, a branch may be used intentionally, but the run manifest must record the exact resolved commit.

---

# 85. Model download policy

Never download models automatically from `voyage run`.

Downloads must be explicit:

```bash
voyage models download longlive2-s2
voyage models download acestep
voyage models download director
```

The user must be able to inspect what will be downloaded.

No model download should overwrite an existing model without an explicit flag.

## 85.1 Reference model download commands

These commands are concrete reference implementations, not requirements that the supervisor invoke them directly. The actual `voyage models download` implementation should use the Hugging Face CLI/library or ACE-Step's official downloader, record the resolved revision, and never hide network activity.

### LongLive 2.0 — NVFP4 S2

```bash
hf download Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2 \
  --local-dir ./models/LongLive-2.0-5B-NVFP4-S2
```

Official model card:

https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2

The current model card identifies this as the 2-step NVFP4 checkpoint and instructs `inference.sampling_steps: 2`. It also documents `model_quant`, FP4 KV-cache quantization, multi-shot sink, and streaming/asynchronous VAE options.

### LongLive 2.0 — NVFP4 S4

```bash
hf download Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4 \
  --local-dir ./models/LongLive-2.0-5B-NVFP4-S4
```

Official model card:

https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4

### LongLive 2.0 — BF16

```bash
hf download Efficient-Large-Model/LongLive-2.0-5B \
  --local-dir ./models/LongLive-2.0-5B
```

Official model card:

https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B

### Wan2.2-TI2V-5B base model

LongLive 2.0 expects the Wan2.2-TI2V-5B component directory layout under `wan_models/`. Follow the exact directory structure in the LongLive 2.0 README and/or helper scripts rather than inventing a new layout.

Official base model:

https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

### ACE-Step 1.5

The preferred route is the official ACE-Step installation/downloader because the repository bundles multiple compatible components:

```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync
uv run acestep-download --model acestep-v15-turbo
```

For the 8 GB target, the planner should resolve the compatible `acestep-5Hz-lm-0.6B` model as well. If using direct Hugging Face download for a fully local cache, the canonical repository is:

```bash
hf download ACE-Step/Ace-Step1.5 \
  --local-dir ./models/Ace-Step1.5
```

Official model card:

https://huggingface.co/ACE-Step/Ace-Step1.5

### Director — Qwen3-8B

```bash
hf download Qwen/Qwen3-8B \
  --local-dir ./models/Qwen3-8B
```

Official model card:

https://huggingface.co/Qwen/Qwen3-8B

### Embedding model

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir ./models/all-MiniLM-L6-v2
```

Official model card:

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

These commands should be included in `docs/MODELS.md` with the exact model revisions validated by the implementation.

---

# 86. Model license documentation

Maintain:

```text
docs/LICENSES.md
```

with a table like:

| Component | Model | License | Source |
|---|---|---|---|
| Video | LongLive 2.0 5B NVFP4 S2 | NVIDIA Open Model License | HF model card |
| Video base | Wan2.2-TI2V-5B | Apache-2.0 | HF model card |
| Video code | LongLive repo | Apache-2.0 | GitHub |
| Music | ACE-Step 1.5 | MIT | GitHub/HF |
| Director | Qwen3-8B | Apache-2.0 | HF |
| Embedding | all-MiniLM-L6-v2 | verify current card | HF |
| Optional SFX | AudioGen Medium | CC-BY-NC-4.0 | HF |

The table is informational and must point at the authoritative license page.

---

# 87. Reference documentation to include

## `README.md`

Must cover:

- what Voyage does;
- quick install;
- model prerequisites;
- basic `voyage init`;
- basic `voyage run`;
- `voyage status`;
- finalization.

## `docs/INSTALL.md`

Must cover:

- supervisor environment;
- LongLive environment;
- ACE-Step environment;
- CUDA requirements;
- ffmpeg;
- environment variables;
- model downloads.

## `docs/MODELS.md`

Must include exact model IDs and links.

## `docs/BACKENDS.md`

Must explain the adapter interface and experimental backends.

## `docs/ARCHITECTURE.md`

Must include diagrams and process ownership.

## `docs/STATE_AND_RECOVERY.md`

Must document every persistence invariant and crash scenario.

## `docs/PROMPTING.md`

Must explain:

- style charter;
- transition prompts;
- novelty system;
- staged prompt design;
- why style is injected by code.

## `docs/AUDIO.md`

Must explain:

- ACE-Step;
- audio state;
- music continuation;
- crossfade;
- final mix.

## `docs/OPERATIONS.md`

Must explain:

- start;
- pause;
- resume;
- stop;
- crash recovery;
- finalization;
- long-run monitoring.

## `docs/TROUBLESHOOTING.md`

Must cover:

- OOM;
- CUDA failures;
- model mismatch;
- ffmpeg errors;
- audio worker failure;
- disk full;
- corrupt checkpoint;
- worker environment problems.

## `docs/BENCHMARKING.md`

Must define:

- warm-up handling;
- block throughput;
- segment throughput;
- peak VRAM;
- CPU usage;
- VAE time;
- audio generation time.

---

# 88. Documentation for source modifications

When modifying LongLive source code, create:

```text
docs/UPSTREAM_LONG_LIVE_PATCHES.md
```

For every patch:

- upstream file;
- upstream revision;
- reason;
- behavioral difference;
- patch summary;
- test covering the change;
- whether the patch can be upstreamed.

Prefer a small patch layer rather than maintaining a large fork diff.

---

# 89. Git strategy

Keep the core project repository independent from the upstream LongLive repository.

Recommended:

```text
third_party/longlive/
```

or a submodule if the team accepts submodule maintenance.

A subtree/vendor copy may be preferable if the worker needs tightly controlled patches.

Whichever strategy is chosen, the run manifest must contain the resolved revision.

---

# 90. Upstream synchronization

Create an explicit process:

```text
upstream update
    ↓
run compatibility suite
    ↓
run GPU smoke
    ↓
run recovery suite
    ↓
run 10-minute endurance
    ↓
update docs and manifests
```

Never casually `git pull` the video backend inside an existing deployment.

---

# 91. LongLive source-level integration references

Coding agents should inspect these exact upstream files before implementing the adapter:

### `pipeline/causal_diffusion_inference.py`

Inspect:

- `CausalDiffusionInferencePipeline.__init__`
- `CausalDiffusionInferencePipeline.inference`
- `_inference_inner`
- `_initialize_kv_cache`
- `_initialize_crossattn_cache`
- `_initialize_sample_scheduler`
- `_set_all_modules_max_attention_size`
- `_set_all_modules_sink_size`
- `_set_all_modules_global_sink_size`

### `utils/prompt_conditioning.py`

Inspect:

- `encode_prompt_blocks`

This is the basis of per-block prompt control.

### `wan_5b/modules/causal_model.py`

Inspect:

- `CausalWanSelfAttention.forward`
- `causal_rope_apply`
- rolling-cache update logic;
- sink handling;
- pinned chunk handling;
- `use_relative_rope` path.

### `utils/wan_5b_wrapper.py`

Inspect:

- `WanTextEncoder`
- `WanVAEWrapper`
- `WanDiffusionWrapper`

### `utils/inference_utils.py`

Inspect:

- `load_generator_checkpoint`
- `setup_nvfp4_pipeline`
- `place_vae_for_streaming`
- `save_video`

Do not infer APIs from older LongLive 1.0 code if current 2.0 source differs.

---

# 92. Research references: LTXV implementation

Coding agents should inspect:

- https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/looping_sampler.py

Important symbols:

- `LTXVLoopingSampler`
- `_process_temporal_chunks`
- `_prepare_guider_for_chunk`
- `MultiPromptProvider`

The source demonstrates a useful alternative approach to long-video conditioning and prompt staging.

---

# 93. Research references: FramePack implementation

Inspect:

- https://github.com/lllyasviel/FramePack/blob/main/demo_gradio.py
- https://github.com/lllyasviel/FramePack/blob/main/demo_gradio_f1.py

Relevant implementation ideas:

- history latent buffers;
- progressive section generation;
- soft overlap appending;
- frequent intermediate MP4 output;
- low-memory model movement.

FramePack demonstrates why writing intermediate results regularly is useful for long-running generation.

---

# 94. Research references: MAGI-1 implementation

Inspect:

- https://github.com/SandAI-org/MAGI-1
- `inference/pipeline/video_generate.py`
- `inference/pipeline/prompt_process.py`

Relevant ideas:

- autoregressive chunk generation;
- prompt embeddings;
- chunk-level time scheduling;
- persistent latent representations.

---

# 95. Research references: SkyReels

Inspect:

- https://github.com/SkyworkAI/SkyReels-V2

The current README is useful for understanding:

- diffusion-forcing long generation;
- overlap history;
- asynchronous inference;
- long-duration frame counts;
- memory reductions.

Do not copy the SkyReels pipeline into Voyage without a clear backend interface.

---

# 96. Research references: ACE-Step

Inspect:

- https://github.com/ace-step/ACE-Step-1.5
- https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md
- https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md
- https://huggingface.co/ACE-Step/Ace-Step1.5

The ACE-Step documentation explicitly describes the separation between its planning LM and DiT executor.

Use that separation as an implementation analogy for the overall Voyage architecture:

```text
Director = creative planner
Renderer = executor
```

but keep Voyage's director separate from ACE-Step because the world model is audiovisual rather than purely musical.

---

# 97. CLI examples to document

## Minimal voyage

```bash
voyage init \
  --output ./runs/dream-voyage \
  --style "pastel neon line-art, peaceful, slow cinematic motion"

voyage run --run ./runs/dream-voyage
```

## Inspect progress

```bash
voyage status --run ./runs/dream-voyage
```

## Pause safely

```bash
voyage pause --run ./runs/dream-voyage
```

## Resume

```bash
voyage resume --run ./runs/dream-voyage
```

## Stop without finalizing

```bash
voyage stop --run ./runs/dream-voyage
```

## Finalize later

```bash
voyage finalize \
  --run ./runs/dream-voyage \
  --output ./dream-voyage-final.mp4
```

## Verify integrity

```bash
voyage validate --run ./runs/dream-voyage
```

---

# 98. Example default autonomous behavior

Given:

```text
pastel neon line-art, peaceful, slow cinematic motion
```

The director might choose:

```text
Concept 1:
a quiet floating train station inside a warm fog bank
```

then:

```text
Concept 2:
the station architecture develops translucent botanical structures
```

then:

```text
Concept 3:
those botanical structures become a suspended forest
```

then:

```text
Concept 4:
the forest gradually reveals enormous glass roots descending into a luminous ocean
```

then:

```text
Concept 5:
the ocean floor becomes an illuminated archival city
```

The system should not plan all five before starting.

Instead it should make local decisions while preserving a broad sense of direction.

This keeps stochastic emergence part of the artwork.

---

# 99. Preventing runaway novelty

A novelty system that only maximizes semantic difference can become chaotic.

Therefore novelty is constrained by:

```text
novelty
  ∩
transition plausibility
  ∩
style adherence
  ∩
peacefulness
```

A new concept may be very different semantically but still be rejected if the director cannot provide a believable transition from the current world.

This is one reason the director should output both:

```text
what comes next
```

and:

```text
how we get there
```

---

# 100. Preventing style collapse

Long autoregressive generation can drift away from the initial aesthetic.

The system should defend against this at several levels:

1. immutable style prefix in every prompt;
2. persistent numerical style controller targets;
3. optional visual inspector;
4. transition prompts that explicitly preserve the style;
5. a fixed negative prompt policy where useful;
6. scene transitions that retain camera/pacing constraints.

Do not solve style drift solely by increasing CFG.

Higher CFG is not guaranteed to improve long-run artistic consistency and may reduce organic emergence.

---

# 101. Prompt granularity policy

Prompt frequency is a major quality/performance tradeoff.

Too frequent:

```text
prompt changes every block
```

may create visible semantic instability.

Too infrequent:

```text
one prompt for 60 seconds
```

may cause stagnation.

Initial recommendation:

```text
one semantic stage ≈ 3 LongLive blocks
≈ 4 seconds in the reference temporal configuration
```

then gradually tune based on empirical video review.

---

# 102. Director temperature policy

Use a relatively high director temperature for creativity but deterministic policy validation afterward.

Suggested starting point:

```text
0.8–0.9
```

Do not expose director temperature as a critical visual-quality parameter to the end user in V1.

It can be a developer configuration knob.

---

# 103. Quality policy

The project should distinguish:

```text
development profile
production profile
```

Development profile:

- faster model;
- fewer steps;
- lower resolution;
- shorter audio;
- inspector disabled.

Production profile:

- chosen stable LongLive quality profile;
- stable resolution;
- full audio;
- visual inspector optionally enabled;
- stronger validation;
- persistent checksums.

Do not tune production settings before the development profile is operational.

---

# 104. Benchmarking protocol

Every benchmark must specify:

- exact model revision;
- exact checkpoint;
- resolution;
- block size;
- attention window;
- step count;
- quantization;
- VAE mode;
- GPU model;
- driver;
- CUDA;
- PyTorch;
- warm-up count;
- measured blocks.

Report:

```text
blocks/sec
video-fps-equivalent
seconds-generated / wall-second
peak-VRAM
average-VRAM
VAE percentage of wall time
text-encoding percentage
```

Exclude first-run compiler/model-load startup from steady-state FPS measurements.

---

# 105. Test matrix

## Unit

- config validation;
- schema serialization;
- atomic writer;
- manifest transitions;
- timeline math;
- seed derivation;
- prompt wrapper;
- style enforcement;
- concept similarity policy;
- worker RPC parsing;
- error classification.

## Integration

- fake video worker;
- fake audio worker;
- fake director;
- segment commit;
- finalization;
- crash recovery;
- worker restart.

## GPU smoke

- model load;
- single block;
- one segment;
- prompt switch;
- relative RoPE enabled;
- checkpoint replay.

## Endurance

- 8–12 hours;
- 100+ segments if hardware permits;
- worker restart during run;
- audio regeneration;
- director failures.

---

# 106. Property-based tests worth implementing

Using Hypothesis or equivalent:

## Segment continuity

For arbitrary valid segment lengths:

```text
concatenated frame ranges remain contiguous
```

## State commits

For arbitrary failure points:

```text
state never references an invalid committed segment
```

## Atomic writes

Simulate interruption before rename and verify:

```text
previous valid state remains readable
```

## Seed derivation

Verify:

```text
same logical coordinates → same seed
changed coordinates → statistically different seed
```

## Novelty

Verify rejected concepts never enter committed concept history.

---

# 107. Fake backend design

Fake workers are important.

A fake video worker should generate synthetic frames whose pixels encode block index.

For example:

```text
frame color/value = hash(segment_id, block_index)
```

This makes ordering bugs visually obvious.

A fake audio worker should generate a sine/chirp or silence track whose frequency encodes segment ID.

A fake director should produce deterministic transitions:

```text
A → B → C → D
```

The complete supervisor can therefore be tested without CUDA or model downloads.

---

# 108. Test fixtures

Keep tiny media fixtures checked into the repository only when licensing permits.

Otherwise generate fixtures procedurally.

For ffmpeg tests, prefer generated test patterns rather than external media.

Do not make CI depend on Hugging Face or model downloads.

---

# 109. CI requirements

Every normal CI run should execute:

```text
ruff
pyright/mypy
pytest unit
pytest integration
```

GPU tests should be a separate job/profile and should be explicitly marked.

No ordinary CI job should require a CUDA GPU.

---

# 110. Security and safety

The application is local, but it still runs external processes and consumes LLM-generated text.

Treat all generated text as untrusted input to:

- filesystem operations;
- ffmpeg arguments;
- subprocesses;
- logs.

Never use generated strings as shell commands.

Never allow an LLM-generated path to become an arbitrary filesystem path without validation.

Concept text is data, not code.

---

# 111. Filesystem safety

Run directories must be resolved to absolute paths after input parsing.

Prevent accidental writes outside the run directory.

Artifact paths should be generated by `Path` operations controlled by the supervisor.

Never concatenate arbitrary prompt text into filenames.

Use IDs for filenames.

---

# 112. Network policy

After model installation, generation should be capable of running with no external network access.

The supervisor should not unexpectedly contact:

- Hugging Face;
- GitHub;
- external LLM APIs;
- external media services.

Model downloads are explicit operations.

The local director should use a local model by default.

---

# 113. Why not a cloud LLM director

A cloud LLM director would violate the preferred local/offline architecture and introduce:

- network dependence;
- variable latency;
- recurring cost;
- credentials;
- non-deterministic API behavior.

The director interface should nonetheless be provider-neutral so a cloud provider can be added later without modifying the world-state model.

---

# 114. Optional future web UI

A later web UI should attach to the supervisor, not bypass it.

The UI could display:

- current frame;
- timeline;
- world state;
- current destination;
- audio waveform;
- worker health;
- controls for pause/resume/stop.

The CLI remains the canonical interface.

---

# 115. Optional future interactive steering

Although the initial product is autonomous, the architecture should reserve:

```text
voyage steer
```

for future use.

A steering command should append a human constraint to the director context, not directly mutate video prompts.

Example:

```bash
voyage steer --run RUN_DIR --note "make the next transition colder and more aquatic"
```

The director then incorporates it at the next safe decision boundary.

This preserves the same planner/renderer separation.

---

# 116. Future audio-reactive mode

Once a stable baseline exists, the project may allow the director to synchronize visual transitions with musical events.

For example:

```text
music energy rises
      ↓
visual motion increases slightly
      ↓
large transition begins
```

However, V1 must not make video dependent on audio generation completing first.

Audio-reactive control is advisory.

---

# 117. Future visual-memory retrieval

If simple concept history becomes insufficient, evaluate LongLive-RAG and related memory systems.

Potential architecture:

```text
long-term memory
      ↓
semantic retrieval
      ↓
current world + retrieved motifs
      ↓
director
```

Do not introduce retrieval infrastructure before the simple novelty store demonstrates a real limitation.

---

# 118. Future model migration strategy

The backend interface should eventually permit:

```python
class VideoBackend(Protocol):
    async def initialize(self, profile: VideoProfile) -> None: ...
    async def append_blocks(self, request: VideoBlockRequest) -> list[VideoBlock]: ...
    async def checkpoint(self) -> VideoRecoveryCheckpoint: ...
    async def recover(self, checkpoint: VideoRecoveryCheckpoint) -> None: ...
    async def health(self) -> BackendHealth: ...
    async def shutdown(self) -> None: ...
```

Possible implementations:

```text
LongLive2Backend
LTXVBackend
FramePackBackend
MAGIBackend
SkyReelsBackend
```

The supervisor must not depend on a concrete backend implementation.

---

# 119. Upgrade path to newer LongLive releases

LongLive 2.0 is the initial design target because it is currently the most directly aligned upstream architecture.

Future LongLive releases may add:

- better quantization;
- improved KV compression;
- stronger long-memory behavior;
- multi-shot improvements;
- new Wan-derived backbones;
- improved inference tooling.

Such upgrades should be integrated behind the same `VideoBackend` interface.

Never make the public CLI expose dozens of backend-specific flags.

---

# 120. Initial production profile recommendation

Once Phase 0 and Phase 2 pass their acceptance tests, the first production candidate should be approximately:

```text
Video:
  backend = LongLive 2.0
  model = LongLive-2.0-5B-NVFP4-S2
  steps = 2
  frame block = 8 latent frames
  local attention = 32
  sink size = 8
  relative RoPE = enabled and validated
  KV quantization = enabled if checkpoint/config support it
  streaming VAE = enabled if stable
  output FPS = 24

Audio:
  backend = ACE-Step 1.5
  DiT = acestep-v15-turbo
  LM = acestep-5Hz-lm-0.6B
  chunk = 45 s
  crossfade = 2 s

Director:
  model = Qwen3-8B
  CPU
  high creativity
  strict JSON schema

Novelty:
  embedding = all-MiniLM-L6-v2
  initial rejection threshold = 0.84

Final:
  768×432
  24 fps
  AAC stereo
```

This profile is a **starting point**, not a promise that every element will fit on every 16 GB/8 GB hardware combination.

Hardware probing and benchmark results must supersede this document's initial assumptions.

---

# 121. Acceptance criteria for the project

The implementation is considered V1-complete only when all of the following are true.

## Core

- `voyage init` creates a valid run.
- `voyage doctor` identifies the local environment.
- `voyage run` starts workers and produces video.
- video generation is truly causal across multiple segments;
- the style charter is present in every video prompt;
- the director can generate new content without user-provided subjects;
- concept history prevents deliberate repetition;
- ACE-Step produces evolving music;
- audio and video are synchronized at finalization;
- `voyage stop` is graceful.

## Recovery

- kill video worker → recover;
- kill audio worker → recover;
- kill director → recover;
- kill supervisor → recover;
- corrupt incomplete artifact → ignore or classify as failed;
- restart from last valid checkpoint;
- no previous committed segment lost.

## Long-run behavior

- no unbounded VRAM growth;
- no unbounded RAM growth;
- no cumulative timeline drift;
- no accumulating temporary files after successful commits;
- finalizer can assemble the full run.

## Quality

- visual changes are gradual rather than random hard cuts;
- music can evolve more rapidly than visuals;
- style remains broadly consistent;
- concepts show meaningful semantic diversity.

---

# 122. Definition of done for each pull request

A PR that modifies core runtime code should include:

- tests for changed behavior;
- documentation update if public behavior changed;
- error handling;
- type checks;
- no unrelated dependency additions;
- no hidden model/API changes;
- clear migration notes if state schemas changed.

A PR modifying a model adapter should additionally include:

- exact upstream revision;
- model name;
- environment requirements;
- license reference;
- smoke test;
- recovery test if checkpoint behavior changed.

---

# 123. Schema versioning

All persistent schemas must carry an explicit version.

For example:

```json
{
  "schema_version": 1
}
```

Do not infer versions from filenames.

When a schema changes incompatibly:

- implement a migration;
- test the migration;
- preserve the old file until the new file is validated;
- record the new version.

---

# 124. Migration philosophy

Prefer forward migrations.

Do not rewrite every old segment when a small schema field can be interpreted with defaults.

The system should remain able to inspect historical runs generated by earlier Voyage versions.

Media should never require migration.

Metadata may.

---

# 125. Source comments philosophy

Comment **why**, not what.

Good:

```python
# We rebuild the KV cache rather than serializing it because the cache is
# GPU-resident and tightly coupled to the exact worker process/model state.
```

Bad:

```python
# Loop over blocks.
for block in blocks:
```

LongLive source-derived code should preserve attribution comments where required by its license.

---

# 126. Upstream license compliance

When copying or adapting LongLive source:

- preserve required notices;
- keep the Apache 2.0 license text as required for redistributed derivative code;
- document modifications;
- do not remove NVIDIA / upstream attribution notices;
- distinguish repository source licensing from checkpoint licensing.

When redistributing Voyage, include `NOTICE` and `docs/LICENSES.md` as appropriate.

---

# 127. Operational runbook

The expected daily workflow should be:

```text
1. voyage doctor
2. voyage models verify
3. voyage init
4. voyage run
5. voyage status
6. pause/resume as needed
7. voyage stop
8. voyage validate
9. voyage finalize
```

The user should not need to open a Python debugger to operate the system.

---

# 128. What a coding agent should do first

A coding agent starting from this document and an empty repository should **not** immediately implement the whole system.

The recommended order is:

```text
1. Read this DESIGN.md completely.
2. Inspect current LongLive 2.0 source at the pinned upstream revision.
3. Inspect the current LongLive 2.0 docs.
4. Inspect the exact model card and license.
5. Inspect ACE-Step 1.5 API documentation.
6. Create the supervisor skeleton.
7. Implement state + CLI + fake workers.
8. Get end-to-end fake generation passing.
9. Integrate real LongLive in one finite segment.
10. Refactor LongLive into a persistent streaming session.
11. Add recovery.
12. Add autonomous director.
13. Add ACE-Step audio.
14. Add endurance tests.
```

Do not reverse this order.

---

# 129. Agent implementation rule: preserve working boundaries

When implementing a feature, ask:

> Which subsystem owns this behavior?

Examples:

```text
Prompt novelty        → director/novelty.py
Style enforcement     → director/policy.py
KV cache              → video/longlive.py
State commit          → state/store.py
Final MP4             → media/concat.py
Audio crossfade       → audio/mix.py
Worker restart        → supervisor.py
```

Do not fix a supervisor bug by modifying LongLive internals if the supervisor owns the behavior.

Do not fix a LongLive cache bug by adding prompt hacks to the director.

This separation is central to keeping the project maintainable.

---

# 130. Agent implementation rule: prefer observation over speculation

Before changing an upstream model behavior:

1. inspect the current upstream source;
2. reproduce the observed behavior with the smallest test;
3. identify the actual contract;
4. patch the narrowest layer possible;
5. add a regression test.

Do not assume an API from an older LongLive release remains identical.

---

# 131. Agent implementation rule: do not optimize prematurely

The project has many potential optimization points:

- KV quantization;
- Torch compile;
- VAE offload;
- FlashAttention;
- Triton;
- CPU pinned memory;
- asynchronous decode.

The correct order is:

```text
correctness
→ recovery
→ profiling
→ targeted optimization
```

A 20% faster generator that loses state after a crash is not an improvement for this project.

---

# 132. Agent implementation rule: isolate experimental flags

All experimental behavior must have explicit feature flags.

Example:

```toml
[experimental]
visual_inspector = false
comfyui_backend = false
longlive_cache_compression = false
```

Do not bury experimental behavior behind an undocumented environment variable.

Environment variables may be used for low-level worker compatibility but public application behavior should live in configuration.

---

# 133. Agent implementation rule: no hidden retries

Retries must be visible in logs and bounded.

Example:

```text
video request attempt 1/3
video request attempt 2/3
```

Infinite retry loops are forbidden.

After a bounded retry budget, escalate to supervisor recovery or fail safely.

---

# 134. Agent implementation rule: preserve failure evidence

When an attempt fails:

- preserve logs;
- preserve the error code;
- preserve relevant config;
- preserve the attempt seed;
- preserve enough metadata to diagnose the failure.

Do not automatically delete failed attempts during V1 unless they are known to be enormous temporary files.

A `cleanup` command can be implemented later.

---

# 135. Future research directions

The architecture should be able to accommodate future improvements such as:

- LongLive-RAG;
- adaptive memory retrieval;
- KV cache compression;
- better attention sinks;
- newer LongLive 2.x releases;
- improved Wan causal backbones;
- learned world-state embeddings;
- VLM-based visual evaluation;
- audio-semantic alignment;
- audio-reactive transition timing;
- procedural sound synthesis;
- transition-specific LoRAs;
- LTXV as a secondary high-quality backend;
- FramePack as a fallback;
- multi-backend A/B experimentation.

None of these should compromise the core supervisor/renderer/director separation.

---

# 136. Canonical online references

The following links are the primary technical references for implementation.

## LongLive 2.0

- https://github.com/NVlabs/LongLive
- https://nvlabs.github.io/LongLive/LongLive2/docs/
- https://arxiv.org/abs/2605.18739
- https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B
- https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2
- https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4

## Wan2.2

- https://github.com/Wan-Video/Wan2.2
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

## NVIDIA licensing

- https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
- https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

## ACE-Step 1.5

- https://github.com/ace-step/ACE-Step-1.5
- https://huggingface.co/ACE-Step/Ace-Step1.5
- https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md
- https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md

## Director

- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/Qwen/Qwen3.5-9B

## Embeddings

- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Alternative video backends

- https://github.com/Lightricks/ComfyUI-LTXVideo
- https://github.com/lllyasviel/FramePack
- https://github.com/SandAI-org/MAGI-1
- https://github.com/SkyworkAI/SkyReels-V2
- https://github.com/qixinhu11/LongLive-RAG

## Optional SFX

- https://huggingface.co/facebook/audiogen-medium

---

# 137. Final architectural summary

The intended V1 system is:

```text
                       STYLE CHARTER
                            │
                            ▼
                    ┌───────────────┐
                    │     Qwen3     │
                    │    Director   │
                    └───────┬───────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
      World State     Transition Plan    Audio State
           │                │                │
           └────────────┬───┴────────────────┘
                        ▼
                ┌───────────────┐
                │  Supervisor   │
                │ transactional │
                │ orchestration  │
                └──────┬────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
       ┌─────────────┐   ┌─────────────┐
       │ LongLive 2.0│   │ ACE-Step 1.5│
       │ video GPU   │   │ audio GPU   │
       └──────┬──────┘   └──────┬──────┘
              │                 │
              ▼                 ▼
          video.wav?          audio.wav
              │                 │
              └────────┬────────┘
                       ▼
                 segment commit
                       │
              ┌────────▼────────┐
              │ persistent disk │
              │    + recovery   │
              └────────┬────────┘
                       │
                       ▼
                 next segment
```

The key engineering idea is simple even though the renderer internals are sophisticated:

> **The video model maintains short-term physical/visual continuity. The director maintains long-term semantic continuity. The audio model maintains an independent musical continuity. The supervisor makes all three persistent and recoverable.**

That separation is what allows the system to run indefinitely without requiring an indefinitely growing model context, an indefinitely growing Python process, or a single fragile output container.

---

# 138. Explicit first implementation target

The first meaningful milestone should be exactly this:

```bash
voyage init \
  --output ./runs/test-voyage \
  --style "pastel neon line-art, peaceful, slow cinematic motion"

voyage run --run ./runs/test-voyage --segments 3

voyage validate --run ./runs/test-voyage

voyage finalize \
  --run ./runs/test-voyage \
  --output ./test-voyage.mp4
```

The result should be:

- roughly one minute of continuous video;
- generated autonomously from style-only input;
- continuously evolving rather than hard-cutting between unrelated subjects;
- accompanied by evolving music;
- represented by immutable committed segments;
- fully recoverable after a worker restart;
- finalized into one MP4.

Once that works reliably, increasing the duration to hours is an engineering scaling exercise rather than a fundamentally different application.

---

# 139. End of specification

Coding agents should treat this document as the architectural contract.

Where this document and current upstream model code disagree, **the current upstream implementation is authoritative for backend internals**, while this document remains authoritative for the Voyage application architecture, state management, separation of concerns, and operational guarantees.

Any such discrepancy discovered during implementation must be documented in the relevant source file and in `docs/UPSTREAM_LONG_LIVE_PATCHES.md`.

---

# 140. Implementation progress log (non-spec, handoff record)

## 2026-09-21 — Phase 0 skeleton complete (task groups A-D, G-partial, I, J, K-unit)

What was built in `Voyage/` (all in-container; host carries no voyage deps):

- Package `voyage/`: `errors` (§48 taxonomy), `paths` (run layout), `atomic`
  (§31 temp+fsync+rename), `seeds` (§62 BLAKE2b streams), `models` (pydantic:
  RunState, EvolutionDecision, PromptPlan, ArtifactRef, WorkerRequest/Response),
  `config` (B: TOML + validation + sha256), `persistence` (C: manifest §32,
  state §33), `concepts` (G: append-only JSONL + token-set similarity fallback),
  `prompts` (§§18/75-76, style prefix injected per stage), `director`
  (deterministic fallback, §51), `rpc` (D: JSONL stdin/stdout, subprocess
  workers, stderr logs), `fake_backends` + `backends` (real ffmpeg testsrc/sine
  renders; `LongLiveBackend`/`AceStepBackend` raise NotImplementedError with
  their phase pointer), `workers/{video,audio,director}` + `workers/loop`,
  `supervisor` (lifecycle + transactional commit §30: validate → metadata →
  checksums → DONE → state), `media` (I: probe/validate/finalize §§54-57, one
  final encode, atomic publish), `doctor` (§64), `cli` (J: all 12 commands).
- `Dockerfile` (`python:3.12-slim` + ffmpeg, pip install `-e .[dev]`), plus
  `scripts/{build,test,gates,run}.sh` (thin docker wrappers, all work
  bind-mounted; `PYTHONDONTWRITEBYTECODE=1` so container runs never leave
  root-owned `__pycache__` in the tree).
- `tests/`: 11 unit + 3 integration (worker RPC health/decide, full segment
  commit with real media). Gates green: ruff check + format, mypy strict
  (23 files), 14 pytest.
- Verified E2E via CLI: `init → run --segments 2 → status → validate →
  finalize → inspect` produced 2 DONE segments (96 frames, 4.00 s) and a valid
  final MP4. Notable: segment 2's repeated deterministic concept was correctly
  novelty-rejected (`[-] #1`) while remaining in immutable history.

Deliberate deviations / notes for the next agent:

- `run`s default `min_free_space_gib = 5.0` (spec example says 20) so small
  dev boxes work; raise for production.
- Fake media is 768×432@24fps testsrc + sine, so `finalize` scale/pad is a
  pass-through on fake runs — the transform reporting path still executes.
- `commit_one_segment` requires `start_workers()` first (explicit
  FatalWorkerError otherwise); `run_segments` manages the worker lifetime.
- Host LSP (pyright) diagnostics may appear stale/wrong (host has no pydantic/
  pytest by design) — `scripts/gates.sh` in-container is the authority.
- Root-owned files: anything the container writes under a bind mount
  (`/app`, `/tmp`) is root-owned on the host; delete via
  `docker run --rm -v /tmp:/tmp voyage:latest rm -rf ...`.

Next up (spec order): Phase 1 task group E (LongLive adapter + video worker
`generate_blocks` idempotence), then Phase 2 stream session, Phase 3 Qwen
director worker behind `DirectorBackend`, Phase 4 ACE-Step worker.

## 2026-09-22 — Run loop: infinite mode, restart recovery, crash injection (Phase 1 groundwork)

- `voyage run` without `--segments` now runs indefinitely (the autonomous
  voyage): the loop re-reads `state.json` at every segment boundary, so
  `voyage pause` / `voyage stop` from another process take effect without
  signals. SIGINT is caught and converted to a clean PAUSED rest (never
  kill -9; in-flight partials are harmless because the commit is
  transactional). A pending PAUSE/STOP at startup is honored instead of
  being clobbered by RUNNING. After `stop`, status rests at STOP_REQUESTED
  until `resume` clears it; finite batches still end PAUSED.
- Failure handling per §48 now executes: any worker RPC goes through
  `_call_with_restart` (one restart + retry on RecoverableWorkerError, then
  record `last_error` and abort; FatalWorkerError additionally flips state
  to FAILED). `SubprocessWorker` gained `restart()`, `pid`, `running`;
  `stop()` tolerates already-dead pipes (close() on a SIGKILLed peer raises
  BrokenPipeError — found live, fixed).
- `Supervisor.inject_worker_crash(name)` (§69 seed): SIGKILLs one worker for
  tests/chaos; never used by the run loop itself.
- Metrics: `logs/metrics.jsonl` records `segment_committed` (with elapsed s)
  and `worker_restart` / `segment_commit_failed`; new
  `voyage inspect metrics` tails them.
- Tests (18 green, was 14): supervisor-restart continuation (fresh Supervisor
  commits 000001 after 000000 + CLI validate passes — Phase 1 exit criterion
  on fake backends), SIGKILLed video worker recovers mid-commit with a
  `worker_restart` metric, pause-before-start exits clean, threaded
  pause-mid-run stops at the boundary (1-2 segments).
- Verified live: infinite `run` in one container (25 segments in 5 s on fake
  backends) + `stop` from a second invocation → exit 0 at STOP_REQUESTED;
  `resume` + `run --segments 1` continued at 000029; `validate` passed on
  30 segments. Gates green (ruff + format + mypy strict + 18 pytest).

## 2026-09-22 — Phase 1 checkpoint decision (task group E kickoff)

Decision (user-aligned via Q&A): **BF16 5B + TorchAO FP8 PTQ first** on the
RTX 4060 Ti (sm89, 16 GB, GPU index 0). Research behind it:

- Host 4060 Ti is sm89 → meets upstream minimum (compute capability >= 8.9);
  the RTX 2060 (sm75) is out.
- NVFP4 W4A4 acceleration is Blackwell-only (paper §Limitations); the
  supported non-Blackwell path is BF16 + TorchAO FP8 PTQ (W8A8), first-class
  upstream (`configs/fp8/inference_fp8.yaml`, `utils/fp8.py`,
  `tests/test_fp8_inference.py`, `torchao==0.13.0` pinned in requirements).
- S2 NVFP4 checkpoint additionally needs custom fouroversix/kernel builds with
  CUDA_ARCHS listing only 100/120 — high risk on Ada. Deferred.
- HF repo `Efficient-Large-Model/LongLive-2.0-5B` is NOT gated; single weight
  file `model_bf16.pt` (~10 GB); upstream pin `6b36d20` (2026-09-07).
- Upstream attention (`wan_5b/modules/attention.py`) degrades FA4→FA3→FA2 with
  graceful import guards, so a missing flash-attn build must not hard-fail
  the eager-mode smoke test.
- Disk: 296 GB free — ample for the ~10 GB checkpoint + worker image.

Plan: `Voyage/worker/Dockerfile.video` (CUDA 12.8 + py3.10 + torch 2.8.0/cu128
+ torchao 0.13.0 + LongLive@6b36d20 + voyage package, eager mode,
`torch_compile: false`), `voyage models download longlive2-bf16` with revision
pin + manifest record, real `LongLiveBackend` behind the worker RPC, finite
one-segment E2E on the 4060 Ti.

## 2026-09-22 — Phase 1 group E: finite LongLive segment on 4060 Ti (DONE)

First real-model segment committed (run id `vll`, 1 segment, 8 frames
@ 1280x704 h264 24fps, VALID, frames visually coherent — neon portrait,
stable across the 8-frame span).

What was built:

- `Voyage/worker/Dockerfile.video` (CUDA 12.8 devel + py3.10 + torch
  2.8.0/cu128 + torchao 0.13.0 + LongLive@6b36d20 + flash-attn 2.8.3 +
  voyage, image `voyage-video:latest`). Runs the whole stack (supervisor +
  workers in one container, `--gpus all`); the supervisor/GPU-env split is
  a later-phase refactor.
- `voyage/workers/video_longlive.py`: resident pipeline (strict ckpt load
  via upstream `unwrap_generator_state_dict`, bf16, TorchAO FP8 PTQ eager),
  CPU/bf16 UMT5 twin injected through the pipeline's `text_encoder=` seam
  (upstream's fp32+auto-CUDA wrapper alone exceeds 16 GB), latents +
  `decode_to_pixel_chunk` mp4 write. Backend selected by
  `config.video.backend` (`fake`/`longlive2`); `SubprocessWorker` replays
  `init` after every restart so the resident pipeline rebuilds.
- `voyage/model_registry.py` + `voyage models download/verify`: pinned
  `Efficient-Large-Model/LongLive-2.0-5B@model_bf16.pt`
  (rev 8521079, 9.3 GiB, sha ec9063a4) + `Wan-AI/Wan2.2-TI2V-5B` subset
  (diffusion shards, VAE 2.8 GB, T5 11.4 GB, tokenizer) into `/models`
  (host `~/.cache/voyage-models`), manifest + sha record. Both repos
  ungated. `scripts/run.sh` gained `VOYAGE_IMAGE`/`VOYAGE_GPUS`/`VOYAGE_MODELS`.
- `voyage/workers/loop.py`: quarantines worker stdout to stderr during
  handlers (upstream prints progress to stdout, which broke JSONL framing).

16 GB VRAM findings (measured, RTX 4060 Ti 15.57 GiB):

- Resident after init only ~6 GB (FP8 generator 4.75 + VAE 2.6→1.3 +
  overhead); the FP8 print confirms 300 linears quantized.
- Latent [1,8,48,44,80] decodes to **1280x704** (VAE spatial x16, causal
  temporal 1+7x4=29) — not 640x352. Longlive profile: 1280x704, 8-29
  frames (see decode note).
- KV cache is `local_attn_size x frame_seq_length` bf16 per layer for BOTH
  branches (neg never gated upstream): 32-window default = ~20 GB alone.
  Fix: `local_attn_size` 32→8 in our config + `_install_pos_only_caches`
  (pos-only twins of the two init methods, guidance-1.0-only, all neg
  reads verified use_cfg-guarded except one unguarded crossattn-neg reset
  write covered by tensor-free placeholders).
- VAE full-segment `decode_to_pixel` OOMs (14.3 GB); `streaming_vae=true`
  unusable (needs `VAE.cached_decode`, absent from this VAE build).
  Working point: `decode_to_pixel_chunk(chunk_size=1)` — 8.7 GB peak
  end-to-end. Tradeoff: causal history restarts per chunk (8 latents → 8
  frames, not 29); frames verified coherent, no visible breakage. Fallback
  on record: CPU decode (slow, full 29f causal) or 1024x576 latents.
- Triton JIT (TorchAO kernels) needs `python3.10-dev` in the image
  (diagnosed via `Python.h` missing) + `CUDA_HOME` set.
- `datetime.UTC` is 3.11+ (worker is 3.10): use `timezone.utc`;
  `requires-python` relaxed to >=3.10 with a tomli shim (3.10 branch).
- mypy quirk documented in the worker header: one missing-module error
  per file → ignore on first occurrence per module only.

Next: Phase 2 stream session (`append_blocks`, persistent KV, recovery
replay across segments — currently each segment rebuilds from noise;
no cross-segment continuity yet), then quality path (larger windows,
torch.compile warmup, CPU-decode 29f).

## 2026-09-22 — Phase 2 continuity slice (shared stream session, RoPE off)

Slice implemented (user-aligned: slice-first, RoPE off, 2-3 blocks/segment):

- `LongLiveStreamSession` in `voyage/workers/video_longlive.py`: persistent
  pipeline across blocks AND segments within one worker invocation —
  `next_start_frame`/`blocks_appended` tracked, prompt-embed cache keyed by
  prompt (CPU T5 encode costs minutes; repeated prompts reuse embeds),
  `append_block` via direct `_inference_inner` (upstream `inference()`
  resets cache positions to zero on later calls, so continuation bypasses
  it after block 1 with tracked `current_start_frame`, `cache_start_frame=0`,
  rolling-window eviction → FLAT peak by construction: longer segments cost
  wall time, not VRAM). `reset()` stub reserved for scene-cut recovery.
- `LongLiveSession.generate_blocks(prompts, seeds)`: per-block direct calls,
  per-block `decode_to_pixel_chunk(1)`, concatenated output; RPC accepts
  multi-block prompts/seeds lists (backward-compat single prompt/seed).
  `use_relative_rope:false` recorded in segment metrics (RoPE is a top-level
  config attr defaulting False, applied/restored per upstream call).
- `VideoConfig.blocks_per_segment=1` (+validator, +TOML template);
  supervisor sends prompts/seeds lists for longlive2
  (per-block seeds `video_seed(seed, seg, block)`), records blocks +
  `use_relative_rope:false` in metrics.json.

VRAM sizing (4060 Ti, 15.57 GiB): 3-block probe peaked 13.17 GiB allocated
(init 6.06 GiB) — fits with 2.4 GB headroom. Probe also caught a real bug:
`generate_blocks` indexed shape[3]/shape[4] on the concatenated (B,T,H,W,C)
tensor (width=3/height=1280 — would have failed supervisor validate); fixed
to shape[2]/shape[3].

E2E continuity proof (PASS): 2 segments x 3 blocks, one invocation, both
committed, VALID 48f @ 1280x704. Boundary metrics — mid-block 0.06-0.09,
block boundaries 0.32-0.35 (fresh noise per block, expected this slice),
SEGMENT boundary 23->24 = 0.366 (same magnitude, no reset penalty), far
control 0.344. Visual: same neon-portrait subject across the boundary, no
hard cut (tighter framing, palette drift — block-evolution, not reset).
Tree note: `persistence.py` had regressed to `datetime.UTC` (commit 6b7b6b9
contains it — Phase 1 fix lost pre-commit via concurrent-agent/tree mishap);
re-applied `timezone.utc` + `noqa: UP017` guard so gates stop flagging it.

## 2026-09-22 — Phase 2 remainder (RoPE on, scene-cut, recovery, 29f decode)

- **RoPE on**: `use_relative_rope: true` in built config + dit setup in
  `LongLiveSession` (mirrors `inference()` preamble bypassed by the direct
  `_inference_inner` path). Denoise peak 8.68 GiB — identical to RoPE-off:
  zero VRAM impact (compute-only). Metrics now record
  `use_relative_rope: true`.
- **Scene-cut**: `append_block(..., scene_cut)` → upstream prefix
  `"The scene transitions. "` on raw_prompts only (embeds stay bare);
  supervisor fires it on destination change. Pure helper slim-tested.
- **Recovery (DESIGN §27)**: worker writes `recovery.pt` per segment (last
  block tail latents + embeds + position, ~7 MB); supervisor `_resume_video_worker`
  hook replays the tape after any video restart before the retried op.
  Root cause found empirically: replay must run at current_start=0 with the
  clock REWOUND to tail length (fresh-cache replay at nonzero start gathers
  0 tokens — local wraps at ring size while counters stay absolute).
  Post-resume state is structurally a fresh stream that generated the tail.
- **29f decode**: VAE transient is ~10 GB regardless of chunk size (spatial
  intermediates, not chunk-scalable). Fix: offload generator (FP8 `.cpu()`
  works) + KV caches to CPU → 1.34 GB resident → full causal decode of 24
  latents → true 93f (1+23x4), zero pops. Segments are now 93f @ 1280x704.
- **Kill-test PASS**: SIGKILL mid-run → restart + resume (rewind 8/1 in
  metrics) → retry commits → VALID 2 segments/186f, PAUSED, no error.
- **CpuUmt5Encoder hardened**: inference-mode self-contained (probes calling
  outside `inference_mode` hit inplace-on-inference-tensor errors).
- Pending: torch.compile verdict (probe running), 5-min soak, then Phase 3.

## 2026-09-22 — Phase 2 remainder complete (compile verdict + soak PASS)

- **torch.compile verdict: functional but DEFERRED.** Inductor
  (max-autotune-no-cudagraphs) compiles the FP8 generator and denoises
  correctly; warmup cost 383 s, post-compile VRAM peak 8.68 GiB — identical
  to eager. Warmup exceeds any per-segment saving at our scale (segments
  run minutes each; compile pays once per process but blocks startup ~6 min
  and saves ~0). Revisit for long-running (hour+) workers only. No code
  change (worker stays eager).
- **5-min soak: PASS.** 4 segments x 3 blocks (93f each, 372f = 15.5 min
  timeline, 3x requirement), VALID, PAUSED, no errors, no restarts, timeline
  monotonic. VRAM per-quarter peaks 15578/15578/15598/15498 MiB — flat
  within noise, global max 15598 of 16380 MiB. KNOWN TIGHTNESS: ~350 MB
  headroom at decode peaks; bounded and repeating (rolling window, flat by
  construction), but any +350 MB transient OOMs — the single allowed restart
  + resume hook is the safety net. Later-phase relief: smaller overlapped
  decode windows or CPU VAE.
- Slim gates green (ruff + format + mypy strict 25 files + 25 pytest).
- Phase 2 exit criteria all met: persistent stream, prompt changes per
  block without reset, scene-cut support, recovery tail + replay resume,
  RoPE on, 29f causal decode, flat VRAM, restart/recovery proven by live
  kill test. Next: Phase 3 (Qwen director worker behind DirectorBackend).

## Phase 3 progress (2026-09-22, groups F+G done, E2E PASS)

- **Schemas (models.py, full §19):** StyleSpec (§15 + spec-example defaults),
  TransitionMechanism (8 literals), TransitionPlan, DirectorDestination,
  VideoPlan (3-5 stages), DirectorAudioPlan, DirectorNovelty,
  EvolutionDecision rewritten (destination/transition/video/audio/novelty +
  phase/novelty_accepted/notes, destination_concept property).
- **Config:** DirectorConfig gains enable_thinking=false, max_new_tokens,
  embedding_model_id; new [voyage] table (allow_concept_revisit=false,
  major_transition bounds, world_decision_interval, blocks_per_prompt_stage=3,
  novelty_threshold=0.85, novelty_max_attempts=3, validators).
- **Prompts:** 3-layer compose (§18) + enforce_style (§18.1, code-level) +
  override detection → ProposalRejected (not a worker failure) + staged plans
  (§18.2, balanced staging, last stage absorbs remainder).
- **Novelty (§21):** ConceptStore(directory) with concepts.jsonl +
  concept_vectors.npy + concept_index.json, cosine check on canonicalized
  text (canonicalize sorts tokens — frozenset join was nondeterministic),
  legacy migration, rejections recorded immutably.
- **Director worker:** lazy Qwen3-8B bf16→fp32 CPU + MiniLM; `decide`/`embed`
  ops; §51 chain (generate → validate → retry with validation errors fed
  back → deterministic fallback, fallback=true + notes). Non-thinking
  defaults (temp 0.7/top_p 0.8/top_k 20). Shape instruction pins the 8
  mechanisms, environment-as-array, novelty-as-object (Qwen's first live
  output used free-text mechanism + string environment — prompt fix landed).
- **Runtime:** voyage-director image (slim + torch CPU + transformers +
  sentence-transformers + accelerate + safetensors — accelerate is required
  for device_map=auto, found via live fallback notes) runs supervisor + fake
  workers + director subprocess. longlive2+Qwen combined image deferred.
- **Registry:** Qwen3-8B rev b968826d (~16.4GiB) + MiniLM rev 1110a243 pinned;
  `models download/verify director-qwen8b`.
- **Supervisor (§74):** bounded accept loop (schema → stages → style →
  embed → cosine/token-set → record), exhaustion → local fallback;
  per-block staged prompts into longlive payload; audio from decision;
  prompt_plan.json + transition.json persisted per segment (§75).
- **Live E2E (CPU, qwen backend, fake media):** 1 segment committed, VALID
  (48f); Qwen decision lighting_transformation, 3 stages, novelty 0.000
  embeddings-on; kill-mid-run left an uncommitted partial correctly ignored
  by validate. ~5 min/segment on CPU.
- Gates green in both images (ruff + format + mypy strict + 37 pytest).

## Phase 4 progress (2026-09-22, group H done, live GPU E2E PASS)

- **GPU placement (probe verdicts):** the 2060 cannot host ACE-Step (DiT load
  alone needs 5.30 GiB vs 5.60 total; an LM-on-CPU split changes nothing), so
  audio renders **sequentially on the 4060 Ti**: per commit the supervisor
  evicts the resident LongLive session (`gc.collect()` + `empty_cache()` — a
  bare `del` frees nothing, reference cycles keep the tensors alive), renders
  the ACE take, evicts the audio stack, then rebuilds LongLive from the
  recovery tape. 45 s take ≈ 20 s render (peak ≈ 15.5 GiB).
- **Single GPU image:** no `Dockerfile.audio`; `worker/Dockerfile.video` gained
  the ACE-Step clone (`ace-step/ACE-Step-1.5` @ ca1e85fe) + `soundfile` /
  `loguru` / `numba` / `vector_quantize_pytorch` + `torchaudio==2.8.0` (cu128),
  and later `sentence-transformers` (director embed) with
  `transformers==4.57.6` **pinned after** the audio deps (sentence-transformers
  6.1.0 upgrades transformers to 5.x and breaks LongLive's x_clip imports).
- **Compat (voyage/audio/acestep.py):** `AceStepStack` wraps
  `AceStepHandler.initialize_service` (turbo config) + `LLMHandler.initialize`
  (0.6B planner) with a post-init readiness gate (model/vae/text_tokenizer/
  text_encoder non-None → VoyageError instead of a late "not fully
  initialized"); `render_take` supports text2music and repaint
  (`src_audio` + `repainting_start/end` + 1.0 s wav crossfade, 1.0 s floor);
  `evict()` documented as gc + empty_cache.
- **ACE LM offload:** `LLMHandler.initialize(offload_to_cpu=True)` keeps the
  0.6B planner on CPU when idle and upstream moves it to GPU only during
  planning; without it the LM stays resident (~2.4 GB) and the DiT preflight
  fails at 45 s ("need ~0.8 GB, only 0.7 GB free").
- **Checkpoint layout:** upstream `MAIN_MODEL_COMPONENTS` expects
  `acestep-v15-turbo`, `vae`, `Qwen3-Embedding-0.6B`, `acestep-5Hz-lm-1.7B`
  directly under `<project>/checkpoints/` — the 1.7B weights satisfy the gate
  only; generation uses `lm_model_path='acestep-5Hz-lm-0.6B'` (relative to
  `checkpoint_dir`). Registry `download/verify audio-acestep` targets that
  layout (turbo 4.5 GiB + 0.6B 1.2 GiB of the 11 GiB total).
- **Slow loop (§35/§40):** `audio/planner.py` (`AudioTake`, `PlanDecision`,
  `AudioPlanner` keep/render/repaint with take_seconds 45 + ahead_seconds 20,
  ledger `audio/takes.jsonl`); `media.py` `slice_take` (ffmpeg -ss/-t to
  canonical s16le WAV) + `assemble_segment_audio` (single slice copied through,
  crossfade chain otherwise with the fade clamped to half the shortest slice,
  concat fallback <0.1 s); `AudioConfig` defaults take 45 s / ahead 20 s /
  crossfade 2.0 s / 48 kHz; finalize muxes AAC `-b:a 256k`;
  `audio_buffer_seconds` = coverage ahead of the committed timeline.
- **Repaint anchoring (bug caught by live audio forensics):** repaint output is
  **timeline-aligned with its source** (head before `repainting_start`
  preserved, ~1 s crossfade, rest regenerated), so the new take must inherit
  the source's `covers_from` — anchoring it at `video_time` replayed the
  preserved head and duplicated ~1.2 s of music across segments. Fix +
  regression tests in `tests/test_audio_planner.py`.
- **Slice precision:** `slice_take` formatted seek/duration as `.3f`, so
  `1.208333` became `1.208` (~16 samples) and each segment drifted ~0.33 ms —
  unbounded over an infinite run. Now `.6f` + impulse-position regression test.
- **Take files are `.wav`:** ACE writes FLAC and the worker converts, but the
  supervisor requested `*.flac`; the ffmpeg FLAC muxer rejects pcm_s16le →
  0-byte takes. Supervisor now names `*.wav`.
- **Live GPU E2E (longlive2 + acestep + qwen, 1280×704, 29 f/segment):**
  `/tmp/vphase4` 2 segments VALID 58 f + finalize → 768×432 h264 + AAC 48 kHz;
  `/tmp/vphase4b` 3 segments VALID 87 f with the repaint-anchor fix — slice
  forensics show no duplication (seg vs previous seg ≈ 0.18, seg vs its own
  preserved head ≈ 0.17), boundary continuity diffs sit inside the
  within-segment range, finalize → AAC 3.67 s. The GPU swap ran live on every
  commit without OOM.
- Gates green (ruff + format + mypy strict 29 files + 51 pytest).

## Phase 5 progress (2026-09-23, inspector done, live VLM E2E PASS)

- **Scope (user Q&A):** deterministic metrics + VLM inspector now; style_similarity
  = drift-vs-segment-0 proxy; feedback = director context + prompt amendments;
  async = piggyback at next commit (ordered, no threads — §44's full async
  loop stays future work).
- **`voyage/vision/metrics.py` (§43, pure numpy+ffmpeg, no torch/cv2):**
  `sample_frames` (ffmpeg rawvideo pipe, width 160; count==1 returns the middle
  frame as the VLM view), 8-bin/channel histograms, and the six spec metrics in
  order — motion_energy (changed-pixel fraction), visual_complexity (Sobel edge
  fraction), semantic_change_rate (1 − first/last overlap), palette_distance
  (mean-color walk, tint shift), style_similarity (vs segment-0 anchor
  histogram, None → 1.0), scene_boundary_strength (max pair drop).
- **Worker `inspect` op** on the director worker (§51-style: generate → strict
  retry → skip with `inspected: False`, never raises): Qwen3.5-9B multimodal
  (trust_remote_code, CPU bf16, non-thinking, greedy), single-frame JSON
  `scene_summary`.
- **Feedback (§43):** `format_measured_context` renders a MEASURED block (value
  + StyleSpec band + BELOW/WITHIN/ABOVE per metric) into the director input and
  user message; `feedback_amendments`/`apply_feedback_amendments` map
  out-of-band readings to prompt amendments, applied post-validation
  pre-style-check (markers verified amendment-safe). `StyleSpec` gains
  `style_similarity_min = 0.60` (provisional — see calibration).
- **Supervisor piggyback:** `_inspect_previous_segment` runs before `_accept`
  (flag-off or segment-0 → skip; blanket-except → skip, never blocks the
  voyage); VLM view frame via ffmpeg middle-frame PNG (no PIL in slim/video
  images); the `visual` section (metrics + summary + inspected + amendments)
  merges read-modify-write into the previous segment's `metrics.json`, never
  clobbering. `[experimental] visual_inspector = false` TOML (§132).
- **Registry:** Qwen3.5-9B pin rev `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
  (Apache-2.0, ~19 GiB; the download allow-list MUST include
  `chat_template.jinja` — the Step 0 probe failed without it) +
  `models download/verify inspector-qwen35`; `DirectorConfig.inspector_model_id`
  (plain local path works, e.g. `/models/Qwen3.5-9B`); supervisor passes
  `model_id` in the inspect payload.
- **Transformers conflict:** the video image is pinned to transformers 4.57.6
  (LongLive `x_clip_loss`) and can NEVER load qwen3_5 — the VLM lives in the
  director image (transformers 5.17.0, torch 2.14.0+cpu) with pillow +
  torchvision added to `Dockerfile.director`.
- **Step 0 probe verdict GO:** LOAD 2.3 s (mmap), GENERATE 92.4 s / 128 tokens
  CPU BF16, PEAK_RSS 16.4 GiB (62 GiB host) — ~2 min/segment overhead is
  acceptable next to multi-minute video commits; 1 frame/segment, tight token
  cap. Probe script was throwaway (`/tmp/probe35.py`); model retained in
  `~/.cache/voyage-models/Qwen3.5-9B`.
- **Live E2E:** fake-backend 2-segment run VALID 96 f (skip path: inspected
  False, amendments fire on motion/drift below bands); director-image 2-segment
  run VALID 96 f with `inspected: True` and an accurate testsrc scene summary —
  full loop verified (piggyback → VLM → merge → MEASURED → amendments → VALID).
- **Calibration (open):** synthetic testsrc reads below the StyleSpec bands
  (motion 0.108 vs 0.20–0.35, complexity 0.063 vs 0.30–0.50, drift 0.009 vs
  0.12–0.25). Definitions and bands kept as-is; real-footage calibration is
  deferred to a GPU longlive run with measured justification.
- Gates green both images (ruff + format + mypy strict 31 files + 94 pytest).

## Fast iteration, slice 1: draft mode (2026-09-23)

- `config.py` gains `DraftConfig` (640x352, latent `[1,8,48,22,40]`
  spatial-halved/temporal-untouched, blocks 1, take 15 s) + TOML `[draft]` +
  pure `apply_draft_overrides` (profile + targeted director/blocks/take-seconds,
  re-validating constructors; in-memory only, printed as "effective settings").
- CLI `run` gains `--draft` / `--director` / `--blocks` / `--take-seconds`;
  `./Voyage/output/` added to `.gitignore` (experiment runs preserved there).
- Verified: fake draft 2-seg (VALID 96f, 1.3 s) + GPU draft 2-seg
  (longlive2/acestep/deterministic, 640x352@29f per segment, 15 s chained
  takes via audio-ahead, VALID 58f, finalize → 768x432 h264 + AAC) in
  `./Voyage/output/draft-fake1` + `draft-gpu1` — 5m50s (~2m55s/segment, above
  the ~1-2 min target; per-stage breakdown is the next slice: benchmark).
- Gates green (100 pytest / mypy 31).
