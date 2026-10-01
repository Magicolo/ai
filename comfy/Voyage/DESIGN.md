# Voyage — Autonomous Infinite Audiovisual Voyage

## Design Specification for Coding Agents

**Status:** Implementation specification / architecture baseline  
**Research snapshot:** 2026-09-23  
**Video backend architecture:** pluggable LongLive 2.0, LTX-Video 0.9.8, and CausVid profiles; backend selected per run after local benchmark validation  
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

The project is intentionally **not** a general-purpose video-generation framework. It is a purpose-built autonomous audiovisual director around a pluggable video-generation backend and an independent generative music backend. The first supported video backends are LongLive 2.0, LTX-Video 0.9.8, and CausVid. A run selects exactly one video backend; the supervisor and persistent run format must not depend on renderer-specific state semantics.

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
selected VideoBackend        ACE-Step 1.5
(LongLive / LTX / CausVid)
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

## 5.1 Video generator backend architecture

> As-built (batch-7-2026-09-30): `config.BACKEND_REGISTRY` declared single owner of the streaming set and worker-module maps (test-gated by `tests/test_single_source.py`); supervisor-side literal derivation deferred to the supervisor-split pass.
> As-built (batch-13-2026-10-01, issue 079): `longlive2` (1280×704@24, 29f, `persistent_kv`) is deleted — the backends table keeps `fake` / `ltxv` (default, 768×512@24, 96f, `reconstructable_prefix`) / `causvid` (832×480@16, 72f, `reconstructable_prefix`). Only tape format is the JSON `write_tape_atomic` record (`recovery.pt` filename kept for discovery; old pickle tapes unresumable by design). Stored runs with `backend="longlive2"` fail fast at load (re-init with `--backend ltxv`; tapes do not transfer; existing segments stay valid media). Backend count 4→3.

The video renderer is a replaceable subsystem. The production application must not assume that every video generator is a persistent KV-cache stream. Three concrete backends are now first-class design targets:

| Backend | Family | Continuation mechanism | Renderer state across segments | Native target profile | Primary reason to support |
|---|---|---|---|---|---|
| `longlive2` | Wan2.2-TI2V-5B / LongLive 2.0 | causal AR block append | persistent GPU KV cache, reconstructable from recovery data | 24 fps | true infinite/streaming architecture; baseline correctness target |
| `ltxv` | LTX-Video 0.9.8 2B distilled | video-prefix conditioning / extension | no required persistent diffusion KV cache; restart by replaying the prefix clip | validate 24 or native-supported FPS | low-VRAM, fast iteration, simple crash recovery |
| `causvid` | Wan2.1-T2V-1.3B + CausVid causal DMD | autoregressive chunk rollout with `start_latents` overlap | no required long-lived GPU cache between rollouts; continuation reconstructed from latent prefix | upstream scripts use 16 fps | fast causal continuation and straightforward long-video rollouts |

The default backend must **not** be hard-coded by architecture. `voyage benchmark video` should produce a machine-specific benchmark report and `voyage init` should record the selected backend explicitly in the run manifest.

The supervisor-facing interface is conceptually:

```python
class VideoBackend(Protocol):
    async def initialize(self, profile: VideoProfile) -> BackendCapabilities: ...
    async def generate_segment(self, request: VideoSegmentRequest) -> VideoSegmentResult: ...
    async def checkpoint(self) -> VideoBackendCheckpoint | None: ...
    async def restore(self, checkpoint: VideoBackendCheckpoint) -> None: ...
    async def health(self) -> BackendHealth: ...
    async def shutdown(self) -> None: ...
```

A backend may implement `checkpoint()` as a no-op when all continuation state can be reconstructed from committed media or deterministic latent prefixes. The supervisor must therefore distinguish:

```text
state_mode =
    persistent_kv
    reconstructable_prefix
    independent_clip
```

`LongLive 2.0` uses `persistent_kv`; `LTX-Video` and `CausVid` should initially use `reconstructable_prefix`. This distinction is important: **application-level resumability is mandatory even when model-level statefulness is not**.

### Backend selection rule

Do not select a backend using published FPS alone. Published FPS is useful for identifying promising architectures but is not a local hardware guarantee. A candidate becomes a supported production profile only after:

1. exact model revision and upstream commit are pinned;
2. the generator passes a finite correctness smoke test;
3. the requested resolution/FPS configuration is validated;
4. peak VRAM and host RAM are measured;
5. the steady-state generation ratio is measured after warm-up;
6. crash recovery is tested;
7. visual continuation quality is reviewed for at least 3 consecutive segments;
8. the benchmark record is stored with the run.

The current LongLive implementation remains documented below as the reference persistent-stream backend. The LTX-Video and CausVid sections add alternative implementations without changing the director, audio, novelty, persistence, or finalization architecture.

> **As-built note (2026-09-24):** the live codebase already wires two of these three backends. `voyage/backends.py` carries the sync `VideoBackend` precursor; `voyage/supervisor.py:VIDEO_WORKER_MODULES = {fake, longlive2, ltxv}` with `STREAMING_VIDEO_BACKENDS = (longlive2, ltxv)`; the worker RPC op is `generate_blocks` (not the async `generate_segment` sketched above — that remains the target interface). CausVid has no worker/registry/preset entry (`grep causvid voyage/` is empty) and stays planned-not-built.

> **As-built note (2026-09-24, Stream C):** the single contract is now `voyage/backends.py:VideoBackendAdapter` over the `generate_blocks` wire op — the spec's `generate_segment` + `segment_seconds`/`state_mode` vocabulary lives caller-side and maps onto live payloads/results with no wire or config change. The sketch above stays sync on purpose (live JSONL transport blocks; §46 forbids concurrent GPU ops to one worker, so `async` adds no concurrency).

## 5.2 LongLive 2.0

> As-built (batch-8-2026-09-30, issue 125): all three video backends saturate VAE highlight overshoot before uint8 conversion (longlive `clamp_to_uint8`, ltxv `clip_array_to_uint8`, causvid `np.clip`) — a bare float→uint8 cast wraps modulo 256 and turns blown highlights near-black.

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
## 5.3 LTX-Video 0.9.8 backend

### Upstream sources

- Repository: https://github.com/Lightricks/LTX-Video
- Model collection: https://huggingface.co/Lightricks/LTX-Video
- 2B distilled checkpoint: `Lightricks/LTX-Video/ltxv-2b-0.9.8-distilled.safetensors`
- 2B distilled FP8 checkpoint: `Lightricks/LTX-Video/ltxv-2b-0.9.8-distilled-fp8.safetensors`
- Current ComfyUI extension: https://github.com/Lightricks/ComfyUI-LTXVideo

The 0.9.8 2B distilled model is a particularly important target for the user's 16 GB GPU because the upstream model table describes the 2B distilled variant as intended for light VRAM usage, and the repository also publishes an FP8 variant. The exact model licenses are version-specific and currently presented as `other` on Hugging Face; the implementation must download and archive the exact license text associated with the selected checkpoint rather than infer a generic license from the repository.

### Architectural role in Voyage

LTX-Video is not a persistent infinite KV-cache generator in the same sense as LongLive. Voyage should use it as a **stateless segment-extension generator**:

```text
committed segment N
       │
       ├── extract a prefix/tail conditioning clip
       │
       ▼
LTX-Video 0.9.8 distilled
       │
       ├── prompt for current transition stage
       ├── previous tail as conditioning media
       └── deterministic seed/config
       │
       ▼
extended clip containing conditioning prefix + new frames
       │
       ├── discard duplicate conditioning frames
       └── commit only newly generated frames
       │
       ▼
segment N+1
```

This has a major operational advantage over LongLive: a video worker restart does not require serializing a large diffusion KV cache. The supervisor can reconstruct the next request from the last committed media tail and the stored random seed/configuration.

### Required constraints

The adapter must inspect the exact installed upstream code and model metadata before choosing these values. Do not assume the old LTX-Video 0.9.6 settings remain valid for 0.9.8.

The implementation must validate at runtime that:

- the selected 0.9.8 checkpoint is actually present;
- the checkpoint's allowed inference-step schedule is compatible with the requested profile;
- the requested width and height are accepted by the pipeline and are divisible by the model's required spatial granularity;
- the requested number of frames satisfies the model's temporal-frame constraint (the upstream README documents `8*n+1` style frame counts for the 0.9.x line);
- conditioning-media frame count satisfies the same contract;
- the target frame index supplied to the extension API satisfies its multiple-of-8 requirement;
- the generated frame rate is explicitly recorded rather than assumed to equal the final Voyage frame rate.

### Recommended development profile

Start with the smallest stable 2B checkpoint:

```toml
[video]
backend = "ltxv"
render_width = 768
render_height = 432
fps = 24

[video.ltxv]
repo = "https://github.com/Lightricks/LTX-Video"
repo_revision = "<pin exact commit>"
model_id = "Lightricks/LTX-Video/ltxv-2b-0.9.8-distilled"
precision = "bfloat16"
stochastic_sampling = false
conditioning_tail_frames = 25
conditioning_strength = 1.0
segment_target_frames = 121
```

The numerical values above are **initial experiment defaults**, not upstream guarantees. The adapter must benchmark 81, 97, 121, and any other frame count actually supported by the checkpoint, then select a segment length that gives a useful generation/conditioning tradeoff.

At 24 fps, 121 frames is approximately 5.04 seconds. With 25 conditioning frames, the adapter would commit approximately 96 new frames (4.0 seconds) if the pipeline returns the full conditioned+generated sequence. The precise overlap must be verified empirically from the generated frame count; never assume that the pipeline's returned tensor contains exactly the requested number of novel frames.

### Recovery semantics

LTX recovery should store:

```json
{
  "backend": "ltxv",
  "state_mode": "reconstructable_prefix",
  "source_segment_id": "000123",
  "conditioning_tail_path": "segments/000123/video_tail.mp4",
  "conditioning_tail_sha256": "...",
  "prompt_plan_hash": "...",
  "seed": 123456,
  "model_revision": "...",
  "pipeline_revision": "...",
  "profile_hash": "..."
}
```

No GPU cache tensor is required. The recovery procedure is:

1. verify the last committed segment;
2. verify the conditioning tail checksum;
3. restart the LTX worker if necessary;
4. reproduce the next deterministic generation request;
5. discard the conditioned prefix from the returned clip;
6. commit only the novel frames.

For production, the adapter should persist an explicit conditioning-tail file so recovery does not depend on extracting a video frame range from a large historical segment.

### Performance experimentation

The LTX backend exists primarily to reduce iteration time. Benchmark separately:

- BF16 2B distilled;
- FP8 2B distilled, if the exact 0.9.8 FP8 integration is supported by the installed kernels;
- official/verified Q8 kernel path if the selected revision still supports it;
- TeaCache only as an optional experimental acceleration, never silently in the correctness profile;
- each supported inference-step schedule exposed by the exact checkpoint metadata;
- 768×432 and the closest model-native resolution if 768×432 is internally binned or padded.

Record `time_to_first_output`, `seconds_generated`, `wall_seconds`, `steady_state_ratio`, `peak_vram_bytes`, and `peak_cpu_ram_bytes`. Published LTX timing is not a local guarantee.

### Continuation-quality rule

Because LTX is a conditioned extension rather than a persistent AR cache, the director must avoid changing the semantic prompt too abruptly between every call. The director should hold the same prompt stage for at least one full LTX segment and use the tail-conditioning image/video plus a transition prompt to bridge to the next stage.

The adapter must expose whether a prompt change occurred relative to the previous segment, allowing the benchmark and later visual inspector to correlate prompt changes with boundary artifacts.

### As-built (Stream A, 2026-09-24)

Upstream validation (all state 0.9.x frame/spatial rules, checked 2026-09-24):

- Frame counts satisfy `(F-1) % 8 == 0` (8n+1: 9, 17, 25, …, 121, 257); non-conforming
  requests are padded with -1 then cropped. Sources:
  `https://github.com/Lightricks/LTX-Video` (README "Parameter Guide" + "Extending a
  video" note), `https://raw.githubusercontent.com/Lightricks/LTX-Video/main/README.md`
  (same text), `https://huggingface.co/Lightricks/LTX-Video/blob/main/README.md` (same
  text). Extension inputs must be 8n+1 video segments (9/17/25/…) with the conditioning
  target frame a multiple of 8; `InferenceConfig` defaults to 121 frames.
- Spatial sizes must be divisible by 32 (one-stage) and by 64 for two-stage multiscale
  (`https://ltx.io/blog/run-video-generation-model-locally`: "width and height must be
  divisible by 64 for two-stage pipelines and by 32 for one-stage"). Best below 720x1280
  and 257 frames; Replicate's native LTX entry renders 24 fps at 768x512
  (`https://replicate.com/lightricks/ltx-video`).
- Verdict: **keep native 768x512** (768 % 32 == 0, 512 % 32 == 0, and both % 64 == 0).
  The 768x432 draft text is rejected: 432 % 32 == 16, so the pipeline would pad to
  768x448 then crop — silent binning with no benefit. `validate_spatial_size` refuses
  non-/32 sizes instead of padding; `validate_frame_count` enforces 8n+1;
  `validate_conditioning_start` enforces the multiple-of-8 target.

Segment accounting (implemented in `voyage/workers/video_ltxv.py`): one
`SEGMENT_TARGET_FRAMES = 121` clip per block, conditioned at start frame 0 on the
25-frame tail video when one exists (`CONDITIONING_TAIL_FRAMES = 25`, itself 8n+1);
fresh starts (no tail, missing tail, scene cut) commit all 121 frames, conditioned clips
discard the 25-frame prefix and commit 96 novel frames (`COMMITTED_NOVEL_FRAMES`). All
counts in the result (`requested/generated/conditioning/novel/committed/prefix_discarded`)
are measured from the real tensors per TASK §4.3, never assumed. Multi-block payloads
chain extensions (each block after the first conditions on the previous block's fresh
tail video); the worker also returns `prompt_changed` (vs the previous segment's prompt)
for the continuation-quality rule.

Conditioning tail: **replaced the single-frame tail PNG with `video_tail.mp4`** (last 25
committed frames, written beside `video.mp4`, sha256 recorded). Rationale: the extension
API needs an 8n+1 *video* prefix, and a single image carries no motion — the PNG was
image-conditioning, not the spec's prefix replay. No PNG is written anymore.
> As-built (§5.3-pruning-2026-09-30): `video_tail.mp4` is no longer
> persisted at generate time. The tape still records
> `conditioning_tail_path` (the would-be location, hash omitted until
> observed); at resume `ensure_conditioning_tail` adopts an existing
> tail untouched, otherwise ffmpeg-trims the last 25 frames from the
> sibling `video.mp4` (causvid: `max(25, overlap window)`) into the
> recorded path and re-hashes in-memory when the tape carries a hash.
> Tail sha stays advisory — resume never hard-fails on mismatch; both
> files missing raises `ValueError`.

Recovery tape: **§5.3 JSON written to `segments/<id>/recovery.pt`** (filename kept so the
supervisor's `recovery.pt` discovery is unchanged; content is JSON, atomically
written). Fields: backend/state_mode/source_segment_id/conditioning_tail_path+sha256/
prompt_plan_hash/seed(+seeds)/model_revision/pipeline_revision/profile_hash plus
geometry and an extra `last_prompt` for truthful `prompt_changed` after resume. **Clean
break is intentional: pre-Stream-A torch-pickle tapes (`{"profile": "ltxv", "tail_png"}`)
fail resume/rebuild with an explicit "unresumable by design" error — re-render.**

Deferred: the 81/97/121 benchmark matrix and any TeaCache/Q8/FP8-kernel study (TASK
§30.2 follow-up). Fresh-segment benchmark probes commit 121 frames; extension (96
novel) throughput must be measured separately once the matrix runs.

Addendum (same day, after Stream A E2E): the `ltxv` preset in `config.py` was moved
768x512 → **1024x576** (`ltxv-576p`) by the rhythm-cut stack slice. That geometry also
satisfies the granularity verdict above (1024 % 32 == 0, 576 % 32 == 0, both % 64 == 0,
true 16:9) — the worker is geometry-agnostic (`validate_spatial_size` + padding), so no
Stream A code change was needed. All Stream A live evidence below is at 768x512; a
1024x576 render (2.25x pixels) still needs its own VRAM/throughput probe on an idle GPU.

> As-built (§5.3-ltxv-oom-fallback-2026-09-30, issue 049): OOM-fallback contract — `_generate_block` catches `(torch.OutOfMemoryError, RuntimeError)` filtered by `video_ltxv.is_oom()` (class-name + message substring), runs `gc.collect()` → `empty_cache()` → `synchronize()` and logs allocated/reserved GiB before the one-shot torchao dynamic-fp8 quant + retry; a second OOM re-raises to the supervisor rebuild path.

---

## 5.4 CausVid backend

> As-built (batch-8-2026-09-30, issue 124): video worker sessions place every model shard on their configured `device` (causvid threads it through pipeline build, T5 shuttle, noise, and telemetry); the shared `video_common.cuda_device_index` parse is the single home and `init` fails fast on out-of-range indexes.
> As-built (batch-8-2026-09-30, issue 128): the causvid tail-slice re-encode never builds an empty window — overlap 1 takes the last frame explicitly (agreeing with the window-size accounting); non-positive overlaps and empty slices fail loud at the slice site.

### Upstream sources

- Repository: https://github.com/tianweiy/CausVid
- Model/weights: https://huggingface.co/tianweiy/CausVid
- Paper: https://causvid.github.io/
- Base model: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B

CausVid is a causal autoregressive video diffusion system built by converting a bidirectional Wan2.1-T2V-1.3B model into a causal generator and applying distribution-matching distillation. The upstream README reports streaming generation around 9.4 FPS on a single GPU and provides both a 3-step autoregressive short-video path and an autoregressive long-video rollout path.

The CausVid model release is subject to a **CC BY-NC-SA 4.0** license according to its Hugging Face model documentation. This is compatible with the current noncommercial Voyage use case, but it must remain explicit in `docs/LICENSES.md` and `run_manifest.json`.

### Architectural role in Voyage

CausVid should initially use the upstream long-video strategy rather than inventing a new persistent cache format:

```text
committed video segment N
        │
        ├── encode a short video tail to Wan latent space
        ├── retain the required overlap latent frames
        └── construct `start_latents`
        │
        ▼
CausVid `InferencePipeline.inference()`
        │
        ├── new Gaussian noise for next chunk
        ├── current director prompt
        └── `start_latents` continuation state
        │
        ▼
new video chunk
        │
        ├── remove overlapped decoded frames
        └── commit novel frames
```

The upstream long-video implementation explicitly reconstructs `start_latents` from the tail of the generated video and the previous latent rollout, then calls the causal inference pipeline again. This means a worker restart can be made recoverable without serializing the full model's GPU-side execution state.

### Upstream timing and tensor contract

The public CausVid configuration currently uses:

```yaml
denoising_step_list:
- 1000
- 757
- 522
- 0
num_frame_per_block: 3
image_or_video_shape:
- 1
-  21
- 16
- 60
- 104
```

The upstream long-video script writes at **16 fps** and uses the 21-latent-frame chunk configuration shown above. The exact decoded frame count and effective novel-frame count after overlap must be measured from the current repository rather than inferred from tensor shape alone.

Voyage therefore treats CausVid's native frame rate as backend-specific configuration. Do not silently pretend that native 16 fps is 24 fps. A CausVid run may either:

1. use 16 fps end-to-end; or
2. use a separately validated final interpolation/resampling stage.

Simple frame duplication from 16 to 24 fps is not acceptable for the final production path.

### Recommended development profile

```toml
[video]
backend = "causvid"
render_width = 832
render_height = 480
fps = 16
final_width = 768
final_height = 432

[video.causvid]
repo = "https://github.com/tianweiy/CausVid"
repo_revision = "<pin exact commit>"
base_model_id = "Wan-AI/Wan2.1-T2V-1.3B"
checkpoint_id = "tianweiy/CausVid"
checkpoint_revision = "<pin exact revision>"
num_frame_per_block = 3
latent_chunk_frames = 21
num_overlap_frames = 3
sampling_steps = 3
```

These values mirror the current upstream examples and must be validated against the pinned commit before implementation. The adapter should not hard-code the output resolution from this example into the generic `VideoProfile`.

### Recovery semantics

Persist:

```text
recovery.pt or recovery.safetensors:
    previous_tail_latents
    previous_tail_decoded_tail metadata
    next_rollout index
    RNG seed/state sufficient for replay
    checkpoint/profile identifiers
```

However, prefer the smallest recoverable artifact. If `start_latents` can be reconstructed deterministically from a committed tail video plus a few stored latent blocks, store those components rather than a complete model cache.

At minimum the segment metadata must record:

- `num_overlap_frames`;
- actual decoded overlap frame count;
- actual newly committed frame count;
- latent-tail tensor shape and dtype;
- whether the first latent is an encoded video frame or a generated latent;
- exact CausVid commit/config hash.

### Prompt evolution

CausVid can accept a new text prompt on each inference call. Voyage should not exploit that capability by making every rollout a new scene. Use the same `WorldState`/`TransitionPlan` abstraction as LongLive and LTX, with the rollout prompt selecting the current transition stage.

For gradual semantic drift, one destination should normally span multiple CausVid rollouts. The director may alter camera, material, lighting, motion, and environment incrementally before changing the canonical concept.

### Performance profile

CausVid's primary purpose in Voyage is to test whether a smaller causal Wan-derived model can deliver a dramatically better generation ratio than the user's LongLive measurement while maintaining useful visual continuity.

Benchmark at minimum:

- upstream 16 fps / 832×480 profile;
- reduced 768×432-compatible profile if the model tolerates it;
- 3-step default path;
- all supported fast-step variants found in the exact checkpoint/config;
- overlap values 1, 2, 3, and any larger supported value;
- `torch.compile` only after an eager correctness baseline exists;
- VAE decode isolated from transformer timing.

The benchmark must distinguish **causal-generation throughput** from final encoded-video throughput and must measure novel frames per wall second after overlap removal.

---


## 5.5 LongLive 2.0 video base model: Wan2.2-TI2V-5B

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

## 5.6 LongLive 2.0 checkpoint choices

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

## 5.7 LongLive model licensing

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

> As-built (§7-sfx-conditioning-2026-09-30, issue 045): SFX conditioning is
> one ffmpeg spawn per window (25 fps @ 384 px, streamed into a single
> `torch.stack`; CLIP = even temporal subsample, sync = CPU downscale
> 384→224) with a 2 GiB pre-stack byte budget and pad-to-16-sync-frames
> tails. Evict drops the extra device-to-host copy.
> As-built (§7-stem-cache-2026-09-30, issue 153): fuzzy stem-cache reuse (0.6 s tolerance) + render-to-tmp + atomic-replace + prune-on-plan.
> As-built (§7-sfx-workers-2026-09-30, issue 158): `--sfx-workers 2` fail-fasts via `augment_devices()` unless 2 GPUs are visible.
> As-built (§7-vocoder-2026-09-30, issue 072): vocoder snapshot is data-only — verify rejects any `.py` (canonical note under §85).
> As-built (§7-sfx-ledger-2026-09-30, issue 054): SFX ledger appended serially in plan order after pool joins (no lock by construction); validator dedupes last-wins + start-sorted walk.

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

> As-built (batch-7-2026-09-30, issue 079): `longlive2` deprecated since 2026-09-30 (kept for existing runs; `voyage init --backend longlive2` warns toward `ltxv`); `ltxv` is the default.
> As-built (batch-13-2026-10-01, issue 079): the deprecation above is now a full delete — the 3-backend table (`fake` / `ltxv` default / `causvid`), JSON-only tape format, and stored-run migration are recorded in the §5.1 table note (same batch/issue tag); no table is duplicated here.

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
      │ video backend│   │ ACE-Step 1.5 │
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

> As-built (§12-container-hygiene-2026-09-30, issues 069/075/076): `.dockerignore` 9 patterns + corrected COPY-graph header; apt all 8 packages `=`-pinned (jammy freeze); chmod 777 → 755 + rationale.
> As-built (batch-7-2026-09-30, issues 034/035-doc): `gates.sh` mypy gate now covers `voyage` + 82 test modules (was 4); scoped PLR2004 check on 4 files; `gates.sh` DESIGN-ref check (top-level modules; caught `cli_core.py` same-day); coverage floor 65→73.

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

The supervisor should not attempt to normalize every generator-specific field. Define explicit backend profiles (`LongLiveProfile`, `LTXVideoProfile`, `CausVidProfile`) and translate each profile into the exact upstream configuration contract. The common supervisor-visible fields should remain limited to resolution, timeline, segment duration, continuation semantics, randomness policy, and resource limits.

> As-built (§14-toml-2026-09-30, issue 020): all TOML basic-string quoting
> flows through the one `config._toml_basic_string` helper (full C0 + DEL
> escaped; `tui_state._toml_string` is a back-compat alias), so creator
> free-text can never emit an unparseable charter.

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
backend = "longlive2" # longlive2 | ltxv | causvid
fps = 24
render_width = 1280
render_height = 704
final_width = 768
final_height = 432
segment_seconds = 16

[video.longlive]
repo = "https://github.com/NVlabs/LongLive"
repo_revision = "<pin a commit at implementation time>"
checkpoint = "<local path>"
model_id = "Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2"
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
state_mode = "persistent_kv"

[video.ltxv]
repo = "https://github.com/Lightricks/LTX-Video"
repo_revision = "<pin a commit at implementation time>"
model_id = "Lightricks/LTX-Video/ltxv-2b-0.9.8-distilled"
precision = "bfloat16"
conditioning_tail_frames = 25
conditioning_strength = 1.0
segment_target_frames = 121
stochastic_sampling = false
state_mode = "reconstructable_prefix"

[video.causvid]
repo = "https://github.com/tianweiy/CausVid"
repo_revision = "<pin a commit at implementation time>"
base_model_id = "Wan-AI/Wan2.1-T2V-1.3B"
checkpoint_id = "tianweiy/CausVid"
checkpoint_revision = "<pin exact checkpoint revision>"
num_frame_per_block = 3
latent_chunk_frames = 21
num_overlap_frames = 3
sampling_steps = 3
state_mode = "reconstructable_prefix"

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

> **As-built note (2026-09-24):** the live `VideoConfig` still uses `segment_frames` + `blocks_per_segment` + `quantization (fp8|bf16)` with backend presets for `fake|longlive2|ltxv` (`voyage/config.py:with_video_backend`). `segment_seconds` and `[video.longlive/ltxv/causvid]` blocks above are the target schema — adopt them (or an adapter) before implementing the CausVid backend.

> **As-built note (2026-09-24, Stream C):** duration mapping lives in `voyage/backends.py` (`frames_for_segment_seconds`: ceil so segments never run short, min 1 frame; `segment_seconds_for_frames` for the reverse). `VideoConfig` is unchanged — `segment_frames` + `blocks_per_segment` stay the stored schema; per-backend `[video.*]` blocks remain a follow-up schema migration.
> As-built (§14-finite-config-2026-09-30, issue 100): `rpc_timeout_seconds`
> (and Draft `take_seconds`) reject non-finite/non-positive values at load —
> a config typo can never reach `select()` as a raw `ValueError`.
> As-built (§14-tui-inherit-2026-09-30, issue 023): untouched-at-default TUI
> fields (`quantization`/`min_fps`/`min_resolution`) resolve as `Unset`, so
> stored TOML wins; an untouched form never clobbers a customized run.
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

> As-built (§19-novelty-distinguishes-2026-09-29, issue 076): the prompt
> demands `novelty {why_new, distinguishes_from}`, but `DirectorNovelty`
> persists only `why_new` (extra keys dropped) — distinction rationale is
> asked for but never stored. Keep the prompt or extend the schema; do not
> assume the field survives validation.

---

# 20. Director prompt design

> As-built (batch-8-2026-09-30, issues 136/168): prefetch telemetry has three outcomes — hit, miss, and invalidated (a ready proposal discarded by fresh inspect amendments or by the drift-cadence hold, logged as `director_prefetch_invalidated` with the reason); the reported hit-rate is `hit / (hit + miss)` with invalidated counted separately as the speculative-waste signal (the §140 prefetch paragraph names both discard conditions).

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

> As-built (batch-8-2026-09-30, issue 117): the token-set fallback is script-aware (`[^\W_]+` word runs) — non-Latin concepts tokenize to non-empty sets and score genuine Jaccard values; the embedding path remains authoritative for non-Latin text and the fallback only decides when no embedding backend is available.

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

> As-built (§21-embed-fallback-2026-09-30, issue 103): uncoercible/non-finite embed vectors degrade to the token-set fallback.
> As-built (§21-measured-finite-2026-09-30, issue 144): MEASURED context skips non-finite metrics per-metric (never WITHIN); `feedback_amendments` steers nothing on skipped metrics.

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

## 22.5 Stream-noise continuity, attention preamble, and KV capacity floor

Three deviations from upstream's single-call `inference()` path each broke
chunk-to-chunk continuity on their own (measured ~6x frame-diff jumps at
every block boundary until all three were fixed; see §140 continuity entry).
All apply to the direct-`_inference_inner` session path (§22.1-22.4):

1. **Stream-level noise.** Upstream draws ONE noise tensor per sequence and
   slices it per chunk. The session therefore holds a single persistent
   RNG (`LongLiveStreamSession`, seeded by the first block's seed) and
   draws each block's noise sequentially — the identical trajectory.
   The RNG state is taped per segment (`noise_rng_state` in
   `recovery.pt`, i.e. the "RNG state" of §27) so evict/rebuild cycles
   continue the trajectory instead of restarting it.

2. **Attention preamble mirror.** Upstream `inference()` applies a per-call
   preamble (`dit.local_attn_size`, `_set_all_modules_max_attention_size`,
   `_set_all_modules_sink_size`, `_set_all_modules_global_sink_size`).
   The direct-`_inference_inner` path bypasses it, so attention modules
   kept construction values — notably `sink_size=0`, which disabled the
   leading-frames sink prepend entirely. `LongLiveSession` mirrors the
   full preamble once at build (values are static per config).

3. **KV capacity floor: `sink + block <= local_attn_size`.** The KV cache
   holds `local_attn_size` frames; the sink permanently occupies `sink`
   of them. With `local_attn_size=8, sink=8` the sink consumed the whole
   cache and every chunk attended only to its own window (fresh scene per
   chunk). Production uses `local_attn_size=16, sink=8` (16-frame cache =
   8-frame sink + one    rolling 8-frame block): every chunk sees the anchor
   plus its predecessor. Validated pairs: 16/8 (draft, boundaries smooth),
   12/4 (tighter, slight chunk-2 morph — insufficient). Upstream's 32/8
   needs >24GB VRAM and does not fit 16GB cards.

4. **`cache_start_frame` units: segment-relative FRAMES, not block indices.**
   `_inference_inner` uses its `cache_start_frame` local directly as a
   frame index into the caller-provided noise/output buffers
   (`noise[:, cs:cs+8]`, `output[:, cs:...]`). Voyage's `append_block`
   once passed the block index (0,1,2): block 1 denoised `noise[:,1:9]`
   while chunk reads happen at `[:,8:16]`/`[:,16:24]` — so chunk 1 became
   1 real latent + 7 zeros and chunk 2 all zeros (symptoms: a white flash
   at the real→zero transition, white-locked brightness thereafter,
   and byte-similar brightness envelopes across runs since zeros decode
   identically). Upstream `inference()` advances both its
   `current_start` (absolute) and `cache_start` (buffer-relative) by
   `num_frame_per_block` per chunk; the voyage equivalent is
   `cache_start_frame` = segment-relative start frame (0,8,16 per block)
   while `current_start_frame` stays absolute for KV/RoPE continuity.
   (The generator-internal `cache_start` forward argument is dead —
   it defaults to `current_start` and is never read for any computation;
   do not confuse the two.) `handle_rebuild` additionally tears down the
   old session (`evict()`) before building the new one — without it the
   old session (~11.5GB: fp8 DiT + 16F cache + VAE) and the new build
   (~6GB) coexist and resume replays OOM (measured 14.97GB).

Fitting 16/8 on 16GB VRAM requires the **VAE-offload-for-generate**:
`generate_blocks` moves the 1.31GB VAE to CPU during denoising (it is
idle until the decode) and restores it before `decode_to_pixel_chunk`,
mirroring the existing generator-offload-for-decode. Measured full-res
fp8 stage peaks: build 6.06 / encode 6.07 / buffers+8F-cache 8.67 /
forward peak 13.16GB; local-16 projects to 13.16 + 2.59 (extra 8F cache)
− 1.31 (VAE) = 14.44GB — fits with ~1.5GB headroom.

> As-built (§22.5-chunked-decode-2026-09-30, issue 048): per-block chunked VAE decode with per-chunk write+del — full latents are never `torch.cat`'d.

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

> As-built (batch-8-2026-09-30, issue 122): recovery tapes (`recovery.pt`, JSON content) are rename-atomic and durable — the temp file is fsynced before the rename and the directory fsynced after (`video_common.write_tape_atomic`), so the latest tape survives OS crash / power loss, not just process crashes.
> As-built (batch-8-2026-09-30, issue 123): resume re-verifies the taped conditioning tail — when the tape carries `conditioning_tail_sha256`, discovery recomputes it and skips to the next-newest tape on mismatch (metric-visible); a truncated tail degrades to an older anchor instead of conditioning the next segment on garbage.
> As-built (batch-8-2026-09-30, issue 171): tape paths carry a byte bound (`1 GiB`, ~100× a legitimate tape) — the commit gate rejects oversized reports and discovery skips them to the next-newest tape, both metric-visible; a runaway or hostile tape fails before the first restart, never inside `torch.load`.

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

> As-built (§27.3-orphan-adoption-2026-09-30, issue 013): a retry meeting
> `segments/NNNNNN/DONE` with `state.next_segment_number` unadvanced never
> re-renders — `_adopt_unaccounted_segment` verifies `sha256.json` over the
> existing media and advances counters from the orphan's metadata
> (`segment_adopted` event); unverifiable orphans raise `MediaError`.
> (Superseded §29-pruning-2026-09-30: verification now reads the
> `manifest.json` checksums section, legacy `sha256.json` fallback.)
> Artifact-free DONE dirs log `segment_reclaimed` and render fresh. Both
> paths share the 099 stop/pause compare-and-swap.
> As-built (§27.4-tape-containment-2026-09-30, issue 016): worker-reported
> tapes are resolve-contained + `is_file()`-gated (`MediaError`); discovery
> skips non-conforming tapes with `recovery_tape_skipped`. Residual:
> check-then-use TOCTOU at the consumer.
> As-built (§27-trust-2026-09-30, issues 003/006): the adoption path probes
> orphan media with the existing `validate_video`/`validate_audio` APIs and
> enforces the shared 0.6 s `check_av_alignment` gate before checksums
> advance state, and clamps stored `frames` to the same
> `1..REPORTED_FRAMES_SLACK × segment_frames` ceiling as the live-report
> gate — drifted or absurd orphans raise `MediaError` instead of adopting.
> As-built (§27-tape-discovery-2026-09-30, issue 139): discovery skips empty/un-stat-able tapes with a `recovery_tape_skipped` metric.

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
    ├── recovery.pt
    ├── manifest.json
    └── DONE
```

`manifest.json` (`format: 1`) holds the five metadata sections
(`transition`/`prompt_plan`/`audio_state`/`world_state`/`metrics`) plus
`checksums` (sha256 over the binary artifacts `video.mp4`/`audio.wav`/
`recovery.pt` only — metadata rides on the atomic manifest write, no
self-hash). Pre-prune runs (individual JSONs + `sha256.json`) still
load via legacy fallback in `voyage/segment_manifest.py`. Audio
coverage slices live in a tmpdir and are never persisted; the video
conditioning tail is derived on demand at resume (see §5.3 as-built)
and new runs start without the root `concepts.jsonl` legacy dup.

A segment must never be modified after `DONE` is created.

If an implementation requires a replacement, create a new attempt and update the manifest through a new transaction rather than mutating historical artifacts.

> As-built (§29-path-confinement-2026-09-30, issue 015):
> `paths.resolve_stored_path` confines every stored path to the run dir
> (resolve-then-`relative_to` containment on both branches; `..` escapes
> and outside-the-run absolutes raise `MediaError`); layout re-anchoring
> heals moved runs and wins over a still-existing stale absolute. The
> read-only validate sites (`_check_segment_metrics`,
> `validate_sfx_ledger`) convert the new `MediaError` into error strings
> instead of tracebacks; render paths let it propagate to the existing
> `MediaError` catches.
> As-built (§29-pruning-2026-09-30): per-segment file count 12 → 5
> (`DONE`, `video.mp4`, `audio.wav`, `recovery.pt`, `manifest.json`);
> ~100 fewer files on a 15-segment run. `DONE` stays a separate file
> (kept as the commit gate); slices stay in tmpdir; `video_tail.mp4`
> is resume-derived; root `concepts.jsonl` no longer created;
> `.cache/acestep` upstream writes are chdir-redirected out of the run.
> Readers (`media`, `cli_validate`, `scoreboard`, `sfx_finalize`,
> adopt, inspect-merge) go through `segment_manifest` with legacy
> fallback, so old runs still validate/scoreboard/resume.

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
10. Compute checksums and persist them inside the manifest
> As-built (§30-pruning-2026-09-30): step 10 persists checksums as the
> `checksums` section of the single atomic `manifest.json` write
> (binaries only); there is no separate `sha256.json` anymore.
11. Write DONE.partial
12. fsync
13. rename DONE
14. Atomically update state.json
15. Atomically update run manifest / committed index
```

The precise order can be simplified, but the invariant must remain:

> No state file may claim a segment is committed until the segment's required artifacts are valid and durably present.

> As-built (§30-av-gate-2026-09-30, issue 003): step 3/5 (validate media)
> enforces `|video−audio| ≤ 0.6 s` via probed durations and the shared
> `check_av_alignment` helper (`av_drift_seconds` recorded); misalignment
> raises recoverable `MediaError`, never a silent commit.

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

> As-built (§31-atomic-copy-2026-09-30, issue 043): large-file publish uses
> `atomic.atomic_copy` (sibling `.partial` + chunked ~1 MiB copy +
> flush/fsync/replace/fsync_dir) — staged finals are never `read_bytes()`'d
> into RAM. Same durability, constant memory.
> As-built (§31-ledger-sync-2026-09-30, issue 101): the takes ledger append
> completes the dance (flush + file-fsync + `fsync_dir`); concepts
> jsonl/index, metrics `append_line`, and the SFX ledger stay open
> follow-ups in their owners' scopes.
> As-built (§31-concepts-sfx-sync-2026-09-30, issue 101): concepts append + `fsync_dir` after jsonl; `validate_concepts` missing-index-key detector; SFX ledger append + `fsync_dir`.

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

> As-built (batch-8-2026-09-30, issue 120): the beat grid honors the renderer's tempo ceiling — the supervisor passes the ACE `MAX_BPM` into the grid, so an over-fine `beats_per_segment` degrades to a coarser integer grid and an impossible one fails the commit naming `beats_per_segment`, before any GPU work.
> As-built (batch-8-2026-09-30, issue 121): take quantization uses round-half-to-even — exact-half segment ratios can plan short (a 45 s take on 18 s segments plans 36 s), chaining extra takes; prefer ceil or round-half-up when the rhythm pins are renegotiated (fix deferred, documented here).
> As-built (batch-11-2026-09-30, issue 121): take quantization now uses ceil — never plans short (`45/18 = 2.5 → 3 → 54 s`); `planner._fresh_take` carries the plan-site `duration > ahead_seconds` guard; rhythm/beat-tie pins updated in the same commit.

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

> As-built (§35-slice-bounds-2026-09-30, issue 104): `AudioTake` geometry is
> validated at load (`StateError`); slice walks are bounded (128 slices,
> 50 ms piece floor, `slice_take` rejects slivers) on both the commit and
> finalize paths, so one corrupt ledger line can no longer spawn thousands
> of 0.1 s-floor ffmpeg slices.

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

> As-built (batch-8-2026-09-30, issue 155): ACE take durations validate both halves — finite within `(0, MAX_TAKE_SECONDS]` (120 s, 2× the largest legitimate 60 s take); absurd values fail in validation, never after minutes of DiT render (mirrors the SFX `MAX_WINDOW_SECONDS` bound).

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

> As-built (batch-8-2026-09-30, issue 180): all six MEASURED metrics steer — each out-of-band metric appends exactly one corrective amendment to the next segment's middle-layer text (palette blowout → palette restraint, boundary spike → single-shot continuity), thresholded against the charter's bands.

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

> As-built (§43-streamed-sampling-2026-09-30, issue 044): frame sampling
> streams the rawvideo decode (`Popen` + exact-stride reads, one owned array
> per frame) instead of capturing the whole stream; the select path is
> capped at the pick count and the estimate-drift fallback at
> `FALLBACK_MAX_FRAMES = 2048`. Sampled frames are pixel-identical to the
> old algorithm.

---

# 44. Visual inspector phase

> As-built (batch-8-2026-09-30, issue 126): probe-derived frame estimates are best-effort — non-finite probe data yields no estimate (full-decode fallback), over-requested frame counts clamp to the decoded total (metrics never describe duplicated frames), and degenerate sample geometry fails as `MediaError`.
> As-built (batch-8-2026-09-30, issue 140): the visual inspector never writes inside committed segment dirs — the single-frame VLM view goes to `logs/inspect/<segment>.png` (or a temp dir), so segment dirs stay exactly the checksummed artifacts (residual: the supervisor-side move stays with its owner).

When implemented, the inspector should operate asynchronously after segment commit.

It should:

1. sample a small number of frames;
2. produce a concise textual scene summary;
3. estimate visual metrics;
4. write `metrics.json`;
5. update the director's next-decision context.

It should **not** block video generation unless an explicit closed-loop mode is enabled.

This is important: an unavailable or slow inspector must not stop an otherwise healthy voyage.

> As-built (§44-vlm-trust-2026-09-29, issue 056, superseded 2026-09-29):
> the Qwen3.5-9B inspector loaded with `trust_remote_code=True` under
> transformers 4.57.6 (no native qwen3_5 modeling — the flag was
> load-bearing AND the model was unloadable without Hub code). Since
> the director image moved to transformers 5.17.0 (native
> Qwen3_5ForConditionalGeneration + multimodal auto class), the loader
> uses `trust_remote_code=False` (verified live: processor resolves as
> Qwen3VLProcessor, 9.4B params load) — no remote code executes by
> design. The pin + allow-list (`chat_template.jinja`) remain as
> availability guards.

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

> As-built (§45-resync-2026-09-30, issue 137): timeouts never poison the
> pipe — `call()` reads within one deadline and discards well-formed
> responses from strictly older requests; only future/unparseable ids are
> Fatal id-mismatch. A timeout that consumed *partial* bytes of the late
> line surfaces on the next call as Recoverable `malformed response line`
> (never Fatal); carrying partials across calls is deferred.
> As-built (§45-start-fence-2026-09-30, issue 170): a failed `start()`
> leaves no handle — the child is reaped, `_proc` cleared, the log fd
> closed — so callers retry `start()` cleanly or observe not-running.
> As-built (§45-call-guard-2026-09-30, issue 007): `SubprocessWorker.call`
> fail-fasts malformed requests (`isinstance` checks on `op`/`payload`) as
> `FatalWorkerError` before any pipe use; the op vocabulary stays unchecked
> so unknown ops remain the worker's `UNKNOWN_OP` fatal. Wire mapping
> pinned: `retryable=False -> FatalWorkerError`,
> `retryable=True -> RecoverableWorkerError`.
> As-built (§45-timeout-finite-2026-09-30, issue 100): per-call timeouts are
> validated finite-positive before deadline math (`RecoverableWorkerError`
> otherwise); `positive_seconds` and Draft `take_seconds` reject non-finite
> at config load (the `beat._require_finite` precedent).
> As-built (§45-prefetch-budget-2026-09-30, issue 030): the speculative
> director prefetch carries an explicit 60 s budget (never the 600 s
> default) and `stop_workers` best-effort drains it (~2 s); worst-case exit
> hold is 60 s, sub-5 s needs a daemon-thread follow-up.

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

> **As-built note (2026-09-24):** live worker RPC (`voyage/workers/loop.py`, `voyage/rpc.py`) is synchronous JSONL over stdio with op `generate_blocks` (multi-block payload for streaming backends + resume hook + ACE-Step GPU swap). The async `generate_segment`/`checkpoint`/`restore` sketch in §5.1 is the target, not the current wire contract.

> **As-built note (2026-09-24, Stream C):** `generate_blocks` is the sole wire op. `voyage/backends.py:VideoBackendAdapter` builds single-prompt payloads for `fake` and multi-block prompts/seeds/scene-cuts for streaming backends (`longlive2`/`ltxv`, mirroring the supervisor), and normalizes results (worker-reported frames win with fallback to requested; as-built conditioning is 0 committed duplicates; worker fps wins so a future 16 fps backend is never relabeled; transport errors propagate untouched).
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

> As-built (§49-audio-oom-2026-09-30, issue 052): the ACE-Step compat layer
> (`voyage/audio/acestep.py`) never wraps out-of-memory failures as base
> `VoyageError` — `is_oom()` detects the OOM class/message shape and
> re-raises unwrapped, so the worker loop reports retryable `WORKER_ERROR`
> and the supervisor restart budget engages. Deterministic ACE failures
> stay fatal `VoyageError`. (Raising `RecoverableWorkerError` from the
> compat layer would not work — it is a `VoyageError` subclass and maps to
> the same fatal branch.)

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

> As-built (batch-8-2026-09-30, issue 096): `validate_video` treats an unparseable `avg_frame_rate` as an fps mismatch (`MediaError`), and an unknown frame count (`nb_frames == 0` or unparseable) as a duration-derived estimate (`duration × fps ≥ min_frames − 1`) rather than a pass.
> As-built (batch-8-2026-09-30, issue 098): `validate_run` orphan-scans every atomic-write location — `segments/`, `novelty/`, `audio/`, and top-level run-root `*.partial` staging — so "no orphans" means no crashed-commit residue anywhere, including the state file's own staging (residual: two-line patch at `cli_validate.py` with the split-tree owner).

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

> As-built (§54-probe-taxonomy-2026-09-30, issue 018): malformed ffprobe JSON raises `MediaError` (retryable / `--skip-bad`-able), never a raw `JSONDecodeError`.

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

> As-built (§55-ffmpeg-budget-2026-09-30, issue 019): local ffmpeg/ffprobe spawns carry the 600 s RPC-mirroring budget `FFMPEG_TIMEOUT_SECONDS`; expiry is `MediaError`.
> As-built (§55-concat-ffmpeg-hygiene-2026-09-30, issue 053): all concat-demuxer lists go through `media.write_concat_list` (`'` → `'\''` quoting); all ffmpeg spawns carry `-hide_banner -nostdin` (incl. `audio_acestep._convert`).

---

# 56. Finalization
> As-built (batch-2026-10-01, issue 152): parity mechanism found — `afade` runs in its input's native sample format, so s16-fed fades truncate to the s16 grid while s32-fed (fold blends 2+) keep precision (≤1 s16 LSB, second-and-later overlaps only); chained pairwise stages with an `aformat=s32` barrier per stage == production fold byte-for-byte (N=4 and N=8, max=0). Recorded NOT landed — needs the pin owner's 31-input-scale no-hang proof + ≤2-input pin relaxation first; the pairwise probe-memo fold stands.

> As-built (batch-7-2026-09-30): frame-count math single-homed in `voyage.augment.interpolated_frame_count` (`media` re-exports); `FINALIZE_CRF_*` aliases the augment CRF ladder; `resolve_finalize_settings()` is the single scalar/options= contract (768/432/24 defaults retained for the zero-floor stream-copy fast path); `workers/augment_worker.py` is a quarantined spike (official weights raise `ModelCompatibilityError`).
> As-built (batch-8-2026-09-30, issue 138): `finalize --skip-bad` is input triage with a record — missing artifacts, checksum/metrics/alignment failures, and numbering gaps are each skipped with a `finalize: skipping ...` warning naming the segment; the post-assembly presentation check stays strict, so a corrupt stage still aborts the finalize.
> As-built (batch-8-2026-09-30, issue 152): finalize audio joins are single-graph — each stem/window is read once through one chained adelay+amix filter invocation (O(N) I/O, one spawn), never re-encoded N−1 times through a left fold; soak trends join wall-clock vs timeline length to prove the scaling (residual: SFX-pass owner).
> As-built (batch-12-2026-09-30, issue 152): correction — the batch-8 single-graph note never landed in code: joins stay pairwise by the pinned ≤2-input invariant (`test_final_blend_scale.py`, live 31-segment deadlock). Batch 9 landed probe-memo (O(N²)→O(N) probes) + per-blend timings; batch 12 proved a wide MANUAL-fade graph hangs never but is not bit-identical (≤1 s16 LSB generational ordering), so the fold stays pairwise until parity is proven + the pin owner relaxes it with a 31-input-scale GPU-long-run proof.
> As-built (batch-8-2026-09-30, issue 188): numbering gaps are the fourth skippable leg — strict raises, lenient warns (`finalize: skipping segment numbering gap: ...`) and ships the sorted survivors.
> As-built (batch-8-2026-09-30, issue 190): `finalize_run` knob precedence is uniform — an explicit scalar wins over `options` for every parameter and `None` means use `options`; an explicit `overlap_fraction=0` behaves as a hard splice whichever path built the settings.
> As-built (batch-10-2026-09-30, issue 166): the Real-ESRGAN anime-6B weight loads strict into the upstream-named RRDB builder at its measured depth (6 body blocks) and upscales end to end; the augmentation floor is executable for upscale. FILM stays fail-loud until the full upstream port lands — default CUDA runs still fetch a FILM weight no loader accepts.
> As-built (batch-11-2026-09-30, issue 166): both pinned augment weights now strict-load and run end to end — Real-ESRGAN anime-6B upscales and FILM interpolates (fp16 on CUDA, fp32 on CPU, OOM-halving preserved) — so the augmentation floors are executable; the remaining step is wiring finalize weights→loader, plus a live-GPU fp16 numeric and quality eyeball the next time a CUDA box is available.
> As-built (batch-13-2026-10-01, issue 166): finalize resolves its model pass through `resolve_augment_weights(config.video.models_dir)` — each leg is a loader-ready path or `None` when weights are absent, and absent legs keep the ffmpeg fallback (default-off unless provisioned); the registry-to-loader seam is production code.

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

> As-built (§56-publish-2026-09-30, issue 043): step 13 publishes via
> `atomic_copy` (chunked, fsynced, constant memory).
> As-built (§56-align-2026-09-30, issue 003): step 6 and `voyage validate`
> enforce the same 0.6 s A/V budget through the shared `av_drift_seconds`
> helper — read-only error strings on the validate side.
> As-built (§56-staging-2026-09-30, issue 102): staging uses `TemporaryDirectory(prefix="voyage-final-", dir=run_dir)` — preflighted filesystem, greppable names.
> As-built (§56-single-pass-finalize-2026-09-30, issue 050): single concat-demuxer + vf libx264 pass over originals (`-preset`/`-crf` from FinalizeOptions, defaults veryfast/15); no intermediate parts; native runs stream-copy; every finalize appends `finalize_completed` (`parts_encode_ms` schema-stable 0.0 + `audio_blend_ms` + `final_encode_ms` + effective crf/preset/geometry) to `logs/metrics.jsonl`.

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

> As-built (§§56-57-generation-resolution-2026-09-29, issue 081): finalize
> keeps the generation resolution (no downscale; `cli.py` passes
> `config.video.width/height/fps`) — 768×512 runs finalize natively.
> Steps 10 (§56) and §57 still read as normative 768×432 legacy; treat
> them as legacy unless a downscale pass is deliberately reintroduced.

---

# 58. CLI specification

> As-built (batch-7-2026-09-30, issue 080): `voyage/cli.py` is now a 735-line seam (parsers + `__all__` re-export surface) over 10 verb-group modules (`cli_paths`/`cli_planning`/`cli_core`/`cli_run_ops`/`cli_models`/`cli_status`/`cli_validate`/`cli_finalize`/`cli_generate`/`cli_observe`, each ≤365L); cross-verb calls and test-patched leaves resolve through the `voyage.cli` namespace at call time via function-level imports (seam-dispatch rule); verb modules never bind seam names from home modules.
> As-built (batch-8-2026-09-30, Group A issues 109/179): the `stop --finalize` surface matches `finalize` — `--skip-bad` plus shared augment and console args, so `stop` is no longer the only finalizing verb without console flags.
> As-built (batch-8-2026-09-30, Group A issue 110): `generate` runs a pre-init override gate — it renders the exact TOML `cmd_init` would write, validates it, and dry-runs `apply_draft_overrides` before touching the directory (exit 2 on bad numeric overrides).
> As-built (batch-8-2026-09-30, Group A issue 112): `parse_duration` rejects mixed signs, bare trailing numbers after h/m (with a did-you-mean hint), and interior whitespace — one grammar shared by CLI and TUI.
> As-built (batch-8-2026-09-30, Group A issues 116/148): `_effective_run_id` strips the winning value and `_run_dir_for` defers to it, so check, use, and TUI agree (`" boba "` lands in `output/boba/` on both surfaces); explicit `--output` still wins.
> As-built (batch-8-2026-09-30, Group A issues 143/185/149): `cmd_init` validates output presence, run-id, non-blank style, int seed, and backend registry membership — all before the first `mkdir`, each an exit-2 error; `generate --help` names the `output/<name>` default.
> As-built (batch-10-2026-09-30, issue 024): unbounded `--output`/`--final-video` paths warn on stderr when the resolved target escapes `./output/`; absolute outside-tree paths remain legal. `--run-id` traversal stays a hard error (exit 2).
> As-built (batch-11-2026-09-30, issue 024): outside-tree `--output`/`--final-video`/`--output` (finalize/sfx) targets are legal and warn on stderr; only `--run-id`/`--name` traversal is a hard error (exit 2) — warn-only is the binding decided behavior, not a deferred upgrade.

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

> As-built (§58-handoff-2026-09-30, issue 022): the `generate` → inner
> `run` handoff forwards every in-memory generation override including
> both caption pins (`music_caption`/`video_caption`) — no flag parses
> yet silently does nothing.

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
  Backend: <selected video backend>
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

> As-built (§59-scoreboard-2026-09-30, issue 062, folds 027): scoreboard degrades per-row — errors cell, `baseline_segment_id`, video/audio existence flags, `partial_segment_ids`, int-validated frames.
> As-built (§59-console-2026-09-30, issue 063, folds 028): console stream rule — all output incl. `error()` via the injected stream; elapsed on both success+failure stage lines; validate-then-report tracker.
> As-built (§59-status-detail-2026-09-30, issue 061): Config section (quantization, SFX, floors, inspector, beats/drift, take-ahead, director), recorded-vs-live hardware labels, restart/circuit counts, gauges trend, reserve WARN lines.

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

> As-built (§60-worker-rotation-2026-09-30, issue 056): worker logs rotate
> mid-run copytruncate-style once per committed segment (`rotate_worker_logs`
> tick: previous-day mtime or `> MAX_WORKER_LOG_BYTES` 10 MiB →
> archive to dated sibling, truncate live in place so the open stderr
> handle continues at offset 0). Rename-based `rotate_log` stays
> start-time only; retention via existing dated-sibling pruning.
> As-built (§60-inspect-readers-2026-09-30, issue 029): `inspect metrics`
> reads through the rotation-aware `iter_metric_files` helper (`N events
> across K files`); it is a reader-list member with `status`, `scoreboard`,
> and soak/benchmark — never a direct `metrics.jsonl` open.
> As-built (§60-size-rotation-2026-09-30, issue 057): size-or-time rotation (`MAX_METRICS_BYTES` 10 MiB) + `fsync_dir` after rename; `append_line` flush+fsync+fsync_dir; `_prune_siblings` fsync_dir; gzip follow-up noted as open.
> As-built (§60-metrics-schema-2026-09-30, issue 058): `METRICS_SCHEMA_VERSION=1`, `MAX_METRIC_LINE_BYTES=16KiB`, `format_metric_line` (base-wins stamps, longest-string halving + `truncated:true`), `parse_metric_lines` ((events, torn) + run_id filter).

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

As-built (§64-doctor-coverage-2026-09-25, issue 048): `probe()`
covers python/ffmpeg+ffprobe/ffmpeg-version/nvidia-smi GPU lines/
`torch.cuda.is_available()` behind a `find_spec` guard (None when torch
is absent)/root disk-free/models-dir presence with a per-backend
`verify_*` summary; the `models_ok` flag is False unless the dir exists
and every check passes. Not covered: compute capability, CUDA runtime
version, FlashAttention/Triton, checkpoint compat, fs permissions,
worker interpreters, ACE-Step — see `docs/TROUBLESHOOTING.md`; full
weight checks stay behind `models verify`.
As-built (§64-cuda-preflight-2026-09-30, issue 021): CUDA preflight sets
are derived from `BACKEND_REGISTRY` device columns (video/audio/SFX
vocabularies, union kept for the TUI warning); the SFX branch is checked,
so `fake/fake/mmaudio` on a torch-less box fast-fails naming the backend.
As-built (§64-doctor-depth-2026-09-30, issue 066): `disk_by_mount` + `meets_reserve()`, `models_ok_required`/`all` split, per-GPU VRAM/driver/compute-cap/temp + `cuda_runtime`, `health_alerts()` thresholds.

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

Video-generator profiles also include `ltxv-*` (reconstructable prefix, bf16-first) and planned `causvid-*` names; recovery tapes never resume across backends or numerics (see §118 as-built note).
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

> As-built (§70-segment-manifest-2026-09-30, issue 095): `sha256.json` is the
> full segment manifest (media + `metrics`/`transition`/`prompt_plan`/
> `audio_state`/`world_state`); both verifiers check every recorded entry
> and the adoption path too; pre-fix media-only manifests verify as
> "not covered". The inspector's post-commit `metrics.json` rewrite
> refreshes its checksum entry so validate never false-positives.
> (Superseded §29-pruning-2026-09-30: the manifest is now a
> `manifest.json` section set + binary checksums; `sha256.json` survives
> only as a legacy fallback. Metadata tamper inside `manifest.json`
> passes `_verify_segment` by design — atomic write is the guarantee.)
> As-built (§70-fps-corrupt-2026-09-30, issue 029): `validate` reports
> `state fps is corrupt` for `fps <= 0`; the 24-fallback below it is
> SFX-math-only and runs after reporting.
> As-built (§70-never-raises-2026-09-30, issue 017): `validate_run` never raises — every filesystem anomaly (`IsADirectoryError`/`OSError`/`RecursionError`, dir-as-artifact, torn metrics) maps to an INVALID line.

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
> As-built (batch-2026-10-01, issue 081): `supervisor.py` 2666→2587L via 3 verbatim extractions + 1 derivation (non-quiet tree, foreign hunks preserved) — new `voyage/supervisor_routing.py` (69L: `VIDEO`/`AUDIO_WORKER_MODULES`, `audio/video_worker_module`, `STREAMING_VIDEO_BACKENDS` now derived from `BACKEND_REGISTRY` per 023/083, value-identical `('ltxv', 'causvid')`), `voyage/supervisor_plan_info.py` (82L: stateless `segment_plan_info`, §§73/18.2), `voyage/supervisor_lock.py` (49L: stateless `read_lock_holder`, §73) — each with facade re-export + delegation + agreement tests (6/5/7); split index now reads proposal/prefetch/commit-types/tape/routing/plan-info/lock. Remainder is stateful-only (commit pipeline, lifecycle hub, audio-coverage) and needs a future stateful-group pattern; `sha256_file` shim stays (`cli_validate.py` importer); worker-map merge needs a 023-owner design decision.

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

> As-built (§73-lifecycle-2026-09-30, issue 012): `start_workers()` is
> exception-safe — a partial start unwinds already-started workers in
> reverse order — and `run_segments()` starts inside its `try/finally`,
> so `stop_workers()` always runs.
> As-built (§73-commit-cas-2026-09-30, issue 099): the commit's state
> advance preserves an externally-written `STOP_REQUESTED` /
> `PAUSE_REQUESTED` across its read-modify-write (compare-and-swap just
> before the write); the run loop honors it at the next segment boundary.
> Full mutual exclusion with the CLI/TUI control plane is still open —
> `_set_status` and the TUI Stop button write without the run lock.
> As-built (§69-restart-accounting-2026-09-30, issue 014):
> `restart()`/hook `RecoverableWorkerError`s consume budget attempts
> (`worker_restart_failed` event); exhaustion still opens the circuit
> breaker — at most N restarts holds on all paths. `Fatal` from the hook
> still propagates untouched.
> As-built (§73-run-lock-2026-09-30, issue 004): only
> `EWOULDBLOCK`/`EAGAIN` reads as contention (other flock `OSError`s
> propagate raw); holder-pid reads are liveness-checked
> (`kill(pid,0)`); `state.json.lock` is pid-guarded-unlinked after close.
> Residual: unlink/check TOCTOU (fd-passing future work).
> As-built (§73-report-gate-2026-09-30, issue 006): worker-reported `frames`
> are clamped to `1..REPORTED_FRAMES_SLACK × segment_frames` with an
> `implausible` `MediaError` on violation, and every non-empty string
> `recovery_path` is re-resolved through `_checked_tape_path` — at both the
> live-report gate and the orphan-adoption path.
> As-built (batch-10-2026-09-30, issue 081): director-proposal pure helpers (`previous_transition_captions`, `effective_music_caption`, `effective_video_stages`, `_token_counts`) now live in `voyage/supervisor_proposal.py` with `voyage/supervisor.py` as the re-export facade; future extractions follow the same move-verbatim + re-export + agreement-test pattern, one group per pass.

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

> As-built (§74-accept-order-2026-09-29, issue 077): code enforces style
> before novelty (cheap code-level gate, then embed + similarity), while
> the diagram above lists novelty first. Both gates run; violation
> attribution follows the code order. Reorder code or diagram deliberately —
> do not "fix" one side without the other.

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

> As-built (batch-8-2026-09-30, issue 118): worker/RPC boundaries validate wire types exactly (`checked_request` discipline — presence plus exact type, bools never satisfy int) and raise `TypeError` before any coercion; `None`, numeric strings, truncated floats, and `"false"` strings never execute with invented values.
> As-built (batch-8-2026-09-30, issue 119): contract models validate value domains at parse time, not just ordering — metric bands and style/audio/transition scalars are unit-bounded where the inspector compares 0..1 metrics against them, so one bad director decision fails loudly at validation instead of silently biasing every segment.
> As-built (batch-10-2026-09-30, issue 035): partial `JsonValue` migration — scoreboard `_finite_float`, all 10 registry builders, `record_builder`, `_merge_manifest_record`, `download_model` + wrappers, and `future_to_entry` are now `JsonValue`-valued; `call()` + supervisor `dict[str, object]` sites, `scoreboard_rows`, `snapshot_kwargs`/`file_kwargs`, and `json.loads` `Any` idioms remain.
> As-built (batch-11-2026-09-30, issue 035): partial `JsonValue` remainder — `json.loads` narrowings landed (`model_registry` 3 `loaded` + `record`, `models_ensure` `raw`, `scoreboard` `event`, `supervisor` `raw`/`existing`/`recorded`); `call()` + supervisor `dict[str, object]` sites, `scoreboard_rows` return, hub `snapshot_kwargs`/`file_kwargs`, and `atomic`/`rpc` fd `Any` idioms remain (probed-blocked, invariance).
> As-built (batch-12-2026-09-30, issue 035): `scoreboard_rows` return narrowed — `row: dict[str, JsonValue]` + `cast` on stages/metrics/deltas/errors cells, locals stay `dict[str, float]`, arithmetic untouched; `test_scoreboard.py` re-pinned with matching casts (the batch-12 gate failure was the test file, not the source). `call()` + supervisor `dict[str, object]` chain + hub kwargs remain probed-blocked (invariance, exact mypy sites logged in the issue).

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

> As-built (batch-7-2026-09-30, issues 084/085): per-family split target — `voyage/registry_records.py` owns pins + `_record_*`/`_describe_*` builders (767L); `model_registry.py` keeps dataclasses + `MODEL_SPECS` assembly + download/verify/manifest core (1270L); future per-family `registry_{ltxv,causvid,qwen,audio,sfx,augment}.py` target recorded.
> As-built (batch-14-2026-10-01, issue 082): `registry_ltxv.py` (84L) + `registry_audio.py` (105L) extracted move-verbatim + facade + agreement tests; remaining families: director triple (shared MINILM), causvid+WAN21, SFX triple.
> As-built (batch-15-2026-10-01, issue 082): `registry_causvid.py` (121L) + `registry_sfx.py` (135L) extracted move-verbatim + facade + agreement tests; remaining family: director triple (shared MINILM coupling must be decided explicitly).
> As-built (batch-16-2026-10-01, issue 082): `registry_director.py` closes the family set — film/realesrgan/inspector/ltxv/audio/causvid/sfx/director; `registry_records.py` now pure facade; manifest-race owner-held.

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

> As-built (§84-hashes-2026-09-30, issue 071): registry rows carry
> `expected_hashes`, verified pre-merge at ingest (poisoned bytes never
> attest); `verify_model` runs a manifest-conditional hash leg; unknown
> manifests fail closed by default (`allow_missing_manifest` /
> `VOYAGE_ALLOW_MISSING_MANIFEST=1` opt-in for external volumes).
> `wan_revision` stays nullable until issue 070 pins the 40-hex revision
> (wire + record + pin procedure landed; value awaits a provisioned box).
> As-built (§84-weight-load-gates-2026-09-30, issue 074): weight loads branch
> on suffix (`.safetensors` via `safetensors`, never pickle; `.pth` via
> `torch.load(weights_only=True)`); torch-free size-floor + manifest-sha
> pre-checks run before model build; `UnpicklingError` maps to
> `ModelCompatibilityError`. Pinned RealESRGAN/FILM weights still await the
> upstream port + SRVGG loader (issue 166 follow-up).
> As-built (§84-manifest-repair-2026-09-30, issue 077): per-dir single atomic write under `_MANIFEST_LOCK` with 3 attempts; torn manifests never overwritten; ensure fails loud on unrepaired entries; absent-manifest carve-out documented.

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

> As-built (§85-vocoder-data-only-2026-09-30, issue 072): the MMAudio vocoder
> snapshot is data-only (`config.json` + `bigvgan_generator.pt` — code ships
> in the pinned `/opt/mmaudio` clone); `verify_sfx_models` fails loud on any
> `.py` under the vocoder dir. Per-file vocoder hashes await measurable
> bytes (issue 071 follow-up).
> As-built (§85-mirror-2026-09-30, issue 065, folds 146): every `MODEL_SPECS` key must appear in INSTALL + README + `models list` output — enforced by the mirror test.

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

### LTX-Video 0.9.8 2B distilled

```bash
hf download Lightricks/LTX-Video \
  ltxv-2b-0.9.8-distilled.safetensors \
  --local-dir ./models/LTX-Video 
```

Optional FP8 checkpoint:

```bash
hf download Lightricks/LTX-Video \
  ltxv-2b-0.9.8-distilled-fp8.safetensors \
  --local-dir ./models/LTX-Video 
```

The implementation must also archive the exact license text for the selected checkpoint. Do not assume the repository's overall license applies to the individual model file.

### CausVid

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir ./models/Wan2.1-T2V-1.3B

hf download tianweiy/CausVid \
  checkpoints/model.pt \
  --local-dir ./models/CausVid
```

The exact checkpoint path and repository layout must be confirmed against the pinned CausVid commit; do not silently substitute a different checkpoint.

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
| Video | LongLive 2.0 5B NVFP4 S2 | NVIDIA Open Model License / current model card terms | HF model card |
| Video | LTX-Video 0.9.8 2B distilled | exact checkpoint license; current HF metadata says `other` | HF model card / bundled license |
| Video | CausVid checkpoint | CC BY-NC-SA 4.0 | HF model card |
| Video base | Wan2.2-TI2V-5B | Apache-2.0 | HF model card |
| Video base | Wan2.1-T2V-1.3B | verify current model card | HF model card |
| Video code | LongLive repo | Apache-2.0 | GitHub |
| Video code | CausVid repo | verify current repository license | GitHub |
| Music | ACE-Step 1.5 | MIT | GitHub/HF |
| Director | Qwen3-8B | Apache-2.0 | HF |
| Embedding | all-MiniLM-L6-v2 | verify current card | HF |
| Optional SFX | AudioGen Medium | CC-BY-NC-4.0 | HF |

The table is informational and must point at the authoritative license page.

---

# 87. Reference documentation to include

> As-built (batch-7-2026-09-30): `docs/SFX.md` + `docs/AUGMENT.md` created per the §87 contract and indexed.

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

> As-built (batch-8-2026-09-30, issue 154): benchmark setup blocks record the discriminating knobs per target (SFX: `model_size` × `sfx_workers`; augment: chunk size × upscale factor × CRF × preset), and the soak SFX section aggregates plain window records post-run — no extra renders.
> As-built (batch-8-2026-09-30, issue 163): soak reports carry an SFX section (window render mean, join/mix rollup via `summarize_sfx_windows`, ledger verdict) collected as a post-run pass over stems + ledger; the end-to-end throwaway stays music-only behind the existing `no_sfx` gate.
> As-built (batch-8-2026-09-30, issue 194): every §104 setup block records the presentation floors (`min_fps/min_width/min_height`) plus the resolved output (`out_w/out_h/out_fps`, `needs_reencode/needs_minterpolate`), so re-encode and stream-copy reports are never silently compared.
> As-built (batch-8-2026-09-30, Group A issue 115): `benchmark` (video/audio) and `soak` fast-fail with the actionable CUDA message when the CUDA stack is absent — workers never start; the end-to-end target stays guard-free by construction (hardcoded fake backend).

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

> As-built (§104-gauges-2026-09-30, issues 059/051): resource gauges now carry per-worker `vram_free_first/last/min_gib` + `vram_total_gib` + `vram_workers_reporting`; `timing_stats_ex` (p50/p95/std) is the percentile source, `timing_stats` frozen.
> As-built (§104-bench-devices-2026-09-30, issue 156): SFX/audio bench peaks are session-device-indexed with an honest-null report shape off-GPU.
> As-built (§104-benchmark-env-2026-09-30, issue 060): `report_document()` JSON artifact builder; rich `_benchmark_env` (driver/compute/CUDA-runtime/versions/registry revisions); reports persisted to `logs/benchmark-*`/`soak-*`.

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

# 118. Video backend migration and recovery contract

The backend interface is now a V1 requirement rather than a future optimization. It must permit all supported generators without making the supervisor aware of model-specific tensor shapes.

The backend interface should permit:

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
CausVidBackend
FramePackBackend
MAGIBackend
SkyReelsBackend
```

The supervisor must not depend on a concrete backend implementation.

---

> **As-built note (2026-09-24):** only `longlive2` (persistent KV, `recovery.pt` with tail latents + prompt embeds + `noise_rng_state`) and `ltxv` (reconstructable prefix, `recovery.pt{profile:ltxv, tail_png}` + `<stem>_tail.png`, 768×512, `25+(B-1)*24` frames, bf16-first, `generate --backend ltxv` default) are wired. `CausVidBackend` is still spec-only.
> As-built (§118-unity-2026-09-30, issue 025): backend-set unity across registries is test-guarded (streaming triple-equality, worker-module keys, CUDA projections).
# 119. Generator upgrade and benchmark policy

LongLive 2.0, LTX-Video 0.9.8, and CausVid are all explicit generator profiles. Their selection must be driven by local benchmark evidence rather than a permanent ranking in this document. New model releases may be added behind the same interface.

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

Once the generator-correctness benchmark and recovery tests pass, the first production candidate should be selected from the three validated profiles rather than assumed in advance. The run manifest must preserve the exact selection.

Reference profiles are:

```text
Video profile A — LongLive 2.0:
  backend = LongLive 2.0
  model = LongLive-2.0-5B-NVFP4-S2
  state mode = persistent_kv
  frame block = 8 latent frames
  local attention = 32
  sink size = 8
  relative RoPE = enabled only after correctness validation
  KV quantization = enabled only when supported by the exact hardware/checkpoint
  output FPS = 24

Video profile B — LTX-Video 0.9.8:
  backend = LTX-Video
  model = ltxv-2b-0.9.8-distilled or validated FP8 variant
  state mode = reconstructable_prefix
  continuation = conditioning tail + generated novel frames
  FPS = exact validated native profile (prefer 24 if stable)

Video profile C — CausVid:
  backend = CausVid
  model = Wan2.1-T2V-1.3B + CausVid checkpoint
  state mode = reconstructable_prefix
  frame block = 3 latent frames
  long-video rollout = upstream latent-overlap strategy
  native FPS = 16 unless a validated alternative exists

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
9. Audit the real LongLive implementation and record the measured bottleneck.
10. Establish the backend-neutral `VideoBackend` interface with a fake implementation.
11. Integrate real LongLive in one finite segment and, if retained, its persistent stream/recovery path.
12. Integrate LTX-Video 0.9.8 as a reconstructable-prefix backend.
13. Integrate CausVid as a reconstructable-prefix backend.
14. Add autonomous director.
15. Add ACE-Step audio.
16. Add cross-backend recovery and endurance tests.
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

## Supported alternative video backends

### LTX-Video

- https://github.com/Lightricks/LTX-Video
- https://huggingface.co/Lightricks/LTX-Video
- https://github.com/Lightricks/ComfyUI-LTXVideo

### CausVid

- https://github.com/tianweiy/CausVid
- https://huggingface.co/tianweiy/CausVid
- https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B

## Other alternative video backends

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


# 137A. Video generator benchmark and backend qualification

The video renderer is now a first-class interchangeable subsystem. Before a backend can be used for an autonomous overnight or multi-day run, it must pass a qualification suite that measures both **correctness** and **generation efficiency**.

The qualification command should eventually support:

```bash
voyage benchmark video --backend longlive2 --profile correctness
voyage benchmark video --backend longlive2 --profile throughput
voyage benchmark video --backend ltxv --profile throughput
voyage benchmark video --backend causvid --profile throughput
voyage benchmark video --all
```

Every benchmark record must contain:

- exact repository revision;
- exact checkpoint identifiers and revisions;
- model file SHA-256 where practical;
- Python version;
- PyTorch version;
- CUDA runtime version;
- NVIDIA driver version;
- GPU PCI identifiers, compute capability, total VRAM;
- host RAM;
- OS/kernel;
- resolution and FPS;
- frame count requested;
- novel frame count actually committed;
- warm-up duration;
- steady-state wall time;
- wall time per generated second;
- peak and average VRAM;
- peak host RAM;
- GPU utilization statistics;
- transformer time;
- text-encoding time;
- VAE time;
- CPU↔GPU transfer time where measurable;
- final encoded-media time;
- error/restart count;
- output hashes.

A backend may be considered operationally preferable only after the benchmark demonstrates that its **novel-frame throughput** is materially better than the LongLive baseline or that it offers a required quality/recovery property. No static ranking in this design document overrides empirical benchmark results.

> As-built (§137A-qualify-gates-2026-09-30, issue 064): `qualify.sh` fail-closed gates (nvidia-smi exit 4, absolute-path exit 2, df preflight vs `QUALIFY_MIN_FREE_GIB` exit 5), tee `reports/qual-<run>-<date>.json`.

# 137B. Cross-backend continuation semantics

The supervisor defines one logical segment timeline but allows backend-specific continuation mechanisms.

For each generated segment:

```text
logical segment interval
        │
        ├── conditioning/prefix frames (may overlap previous segment)
        ├── newly generated frames
        └── optional lookahead frames not yet committed
```

Only the novel committed interval advances the voyage timeline. This rule is necessary to avoid duplicate frames when LTX or CausVid uses explicit overlap.

Each backend must report:

```text
requested_output_frames
returned_output_frames
conditioning_frames
novel_frames
committed_frames
native_fps
presentation_fps
```

The supervisor must reject a segment when:

- `committed_frames <= 0`;
- timestamps overlap an already committed interval;
- returned frame count does not match the backend's declared tensor contract;
- audio alignment cannot be established;
- the backend reports a different native FPS without an explicit timeline conversion step.

# 137C. Video backend recovery classes

### `persistent_kv`

Used by LongLive 2.0. The worker may retain an in-memory causal KV cache between blocks, but the supervisor must still persist a compact recovery artifact. After crash, the worker is restarted and the cache is reconstructed by replaying the minimum required clean latent history rather than serializing arbitrary CUDA tensors.

### `reconstructable_prefix`

Used by LTX-Video and CausVid. The worker is intentionally disposable between segments. The supervisor stores the exact media/latent prefix required to reproduce the next continuation request. Recovery therefore means restarting the worker and regenerating the in-flight segment from the last committed prefix.

### `independent_clip`

Future backends that do not condition on prior frames may use this class. The director must then provide an explicit transition image/video keyframe to preserve visual continuity. Such a backend is not acceptable as a default Voyage renderer until transition tests demonstrate adequate continuity.

# 137D. Revised development sequence for interchangeable generators

The generator implementation phases are now:

1. **LongLive audit:** prove whether the current 100:1 observation is caused by implementation, quantization/kernel fallback, CPU/GPU transfer, VAE, configuration, or genuine model throughput on the user's hardware.
2. **Backend interface extraction:** isolate the supervisor from LongLive-specific state and tensor contracts.
3. **LTX-Video finite continuation:** implement and benchmark 2B distilled extension with explicit tail-conditioning and novel-frame accounting.
4. **CausVid finite continuation:** implement the upstream latent-overlap long-video rollout and benchmark at native settings.
5. **Cross-backend recovery tests:** crash each worker during a segment and verify recovery from the previous commit.
6. **Director integration:** make one transition plan work identically through all supported backends.
7. **Long-run bake:** run the selected backend for multiple hours with forced worker restarts and periodic validation.
8. **Production selection:** choose the backend/profile from the measured report and record it immutably in the run manifest.

The old LongLive-only development sequence remains a useful implementation detail, but these phases supersede it as the project-level generator roadmap.

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

- roughly one minute of continuous video using whichever qualified video backend is selected in the run configuration;
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

Any such discrepancy discovered during implementation must be documented in the relevant source file and in the backend-specific upstream notes, for example `docs/UPSTREAM_LONG_LIVE_PATCHES.md`, `docs/UPSTREAM_LTXV_NOTES.md`, or `docs/UPSTREAM_CAUSVID_NOTES.md`.
# 140. Implementation progress log (non-spec, handoff record)
> As-built (batch-2026-10-01, issue 093): `Voyage/TASK.md` removed via `git rm` — brief retirement complete (open §30 items live once in the §140 batch-9 entry, §30.4 procedure in `docs/BENCHMARKING.md`); full text survives in `git log --oneline -- Voyage/TASK.md`; remaining `TASK.md` mentions are historical only.

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

## Fast iteration, slice 2: per-stage timings (2026-09-23)

- `commit_one_segment` records per-stage seconds
  (inspect/director/video/audio/validate/commit) into the `segment_committed`
  metrics event (`tests/test_stage_timings.py` pins the key set + bounds).
- GPU draft breakdown (`/app/output/draft-timing`, 2 segments, 370 s):
  seg0 166.8 s = director 8.8 (first-call warm-up) + video 70.4 (first-block
  CUDA warm-up) + audio 87.4; seg1 137.0 s = video 54.9 (steady) + audio 82.0.
  Audio dominated because draft `take_seconds=15` < `ahead_seconds=20`, so
  coverage was always inside the ahead window → a fresh take + full GPU swap
  (evict video → ACE reload → render → evict audio → rebuild from tape)
  EVERY segment.
- Fix: draft `take_seconds` 15 → 45 (same as full; swaps land ~every
  21 segments, ~7 s amortized) + invariant test `take_seconds > ahead_seconds`.
- Verified (`/app/output/draft-timing2`, 3 segments, 4m58s): seg0 168.2 s
  (one take + swap), seg1 54.6 s (video 54.4, audio 0.06 KEEP), seg2 9.5 s
  (video 9.4, kernel warmth) — one `take_rendered` total; VALID 87f = 29x3;
  finalize → 768x432 h264 + AAC 3.667 s. Draft steady-state ≈ ~1 min/segment
  or better — the iteration target is hit.
- Gates green (101 pytest / mypy 31).

## Fast iteration, slice 3: inspect scoreboard view (2026-09-23)

- `voyage/scoreboard.py`: `scoreboard_rows(run_dir)` reads committed
  segments (metrics.json visual.metrics + transition.json destination/phase +
  audio_state.json take_ids + logs/metrics.jsonl stages) and computes
  per-metric deltas vs the previous inspected row (seg0 deltas zero;
  missing visual → None; partials skipped). Six §43 keys pinned by
  `tests/test_scoreboard.py` (3 commits: the piggyback inspects the
  previous segment, so seg1 visual needs a 3rd commit).
- `voyage inspect scoreboard --run <dir>`: compact per-segment table with
  `value(Δ)` metric cells + stages + destination/phase/takes + segment
  audio/video view paths + final.mp4 line.
- Verified: `/app/output/draft-timing2` (no-visual path) and a fresh
  flag-on fake 3-commit run `/app/output/score-visual` (seg0/seg1 metrics
  with +0.000 deltas on identical testsrc; seg2 no-visual by piggyback
  design; inspect=0.2/0.3 s in stages).
- Gates green (104 pytest / mypy 32).

## Fast iteration, slice 4: bf16 precision + highlight-blowout investigation (2026-09-23)

- Draft runs exposed a quality defect: bright/saturated regions render
  solarized/posterized (Exp A chrome-river draft: neon blobs everywhere;
  Exp C calm dawn lake draft: mostly photorealistic BUT a giant
  yellow/red sun blob + dark edge bands; Exp B full-res confirmed the
  same look — draft shape exonerated).
- Elimination arc (all our-path diffs cleared): pos-only KV patch
  (faithful replication), max_attention_size (7040 = 7040 at full res),
  RoPE/sink/t-scale mirror, KV persistence (block-1 equivalent), VAE
  decode (full causal, chunk_size = ALL latents), text-encoder twin
  (bf16-vs-fp32 rounding only), manual seeded noise both sides,
  guidance 1.0 both, full-res also broken.
- Decisive A/B (same calm prompt + video_seed(7,0,0) + draft shape):
  bf16 probe renders a gorgeous clean lake (frame means ~150) vs fp8
  Exp C blown out (~204). VERDICT: fp8 W8A8 dynamic activation
  quantization (torchao Float8DynamicActivationFloat8WeightConfig,
  per-row) is the amplifier — it mangles extreme magnitudes.
- W8-only alternative is DEAD on sm89/16GB: torchao 0.13
  Float8WeightOnlyConfig dequantizes eagerly per forward (no scaled_mm
  without sm90/compile) and the transients accumulate to ~a full
  second bf16 copy → OOM even after a gc.collect() fix that freed
  4.36 GiB at LOAD (8.91 GiB free → died at the identical point).
  Also no speedup without compile (deferred since Phase 2).
- Adopted: `VideoConfig.quantization: fp8|bf16 = fp8` (+ TOML +
  `--quantization` CLI flag + worker conditional + derived recovery
  profile `longlive2-bf16` vs `longlive2-bf16-fp8`, so tapes never
  resume across numerics) + tests/test_precision.py (7 tests).
- Verified: draft 3-seg bf16 E2E (`/app/output/exp-d-bf16`, VALID 87f,
  GEN 3s same speed as fp8, fits with 2.66 GiB free at GEN) and
  full-res 1-seg bf16 (`/app/output/exp-e-bf16full`, VALID 29f, no OOM).
  One transient silent worker death mid-load on the first full-res
  attempt (no traceback, clean dmesg, GPU contention aftermath with a
  just-finished probe the likely cause); immediate retry on a clean
  GPU committed first try.
- Residual: bf16 still shows a moderate solarized band at peak
  brightness (draft AND full-res) — the band is model/4-step behavior
  at extremes, fp8 only amplifies it. Saturated greens blow out too
  (exp-d seg1 'glowing green aurora' prompt → posterized masses even
  in bf16). Prompt guidance: avoid extreme brightness/saturation;
  calm midtones render beautifully in either precision.
- Determinism characterization (same-process 3-way: bare/bare/stream
  latents bit-identical, A==B==C meanabs 0.000000): the stack is
  deterministic, stream ≡ bare path. Cross-process carries ~±5
  mean-abs PNG noise (cudnn/flash-attn scheduling) — A/B verdicts must
  exceed ~5 or run same-process. A bogus 17.58 reading (violating the
  triangle inequality vs 2.89+3.95) plus biased separate viewings
  caused a long false paradox; contact sheets + the inequality check
  resolved it. Lesson: compare side by side, verify arithmetic.
- Real-footage calibration points (LongLive, testsrc is far below):
  motion 0.076-0.658, complexity 0.06-0.13, drift 0.01-0.10 vs bands
  0.20-0.35/0.30-0.50/0.12-0.25 — bands kept (they encode desired
  ranges, evidence still thin), revisit with more footage.
- Gates green (111 pytest / mypy 32).

Phase 6 slice A (state integrity) done 2026-09-23: closes the
written-never-verified sha256 gap plus the atomicity gaps.
- `validate` recomputes sha256.json per segment (checksum mismatch →
  INVALID), scans orphans recursively (`rglob *.partial`, catches
  in-segment temps + DONE.partial remnants), enforces contiguous
  `^\d{6}$` numbering from 000000, requires positive frame counts and
  media durations, and checks recorded recovery tapes exist. Refactored
  to testable `validate_run()` returning error strings (cli.py).
- `finalize` implements §56 steps 4-6 per segment before any encode:
  checksum recompute, metrics.json frame-range sanity, A/V alignment
  probe (shared `AV_ALIGNMENT_TOLERANCE_SECONDS = 0.6`, same budget the
  commit path enforces) + strict-mode contiguity check; new
  `--skip-bad` flag skips corrupt segments with a warning instead of
  aborting (media.py).
- Durability: new `atomic.fsync_dir()` called after every
  temp+fsync+replace plus after the DONE rename (rename durability —
  file fsync alone does not persist the directory entry); chunked
  `sha256_file` (constant memory, takes are multi-GB); concept vectors
  saved temp+fsync+rename, index via atomic_write_json, jsonl appends
  + legacy migration fsynced (concepts.py); takes ledger appends
  fsynced (audio/planner.py).
- tests/test_state_integrity.py (15 tests, TDD): clean run validates;
  video/audio checksum mismatches, in-segment partials, DONE.partial
  remnants, numbering gaps, non-positive frames, missing recovery
  tapes, impossible durations all fail validate; tampered segments fail
  finalize (checksum), misaligned A/V fails finalize, --skip-bad
  finalizes the rest; concept vector/index + ledger roundtrips.
- Gates green (126 pytest / mypy 32).

Phase 6 slice B (failure policy) done 2026-09-23: closes the
single-restart, dead-timeout, and misleading-status gaps.
- Restart budget + circuit breaker: `[voyage] max_worker_restarts = 3`
  (TOML-configurable, non-negative) bounds restarts per worker per run;
  `_call_with_restart` loops the op until success or budget exhaustion,
  then opens the breaker (FatalWorkerError → run rests FAILED) and logs
  `circuit_breaker_open`; `worker_restart` events now carry
  attempt/budget. Counters reset each `run_segments`; resume retries
  share the same budget (config.py, supervisor.py).
- Honest abort statuses: any VoyageError aborting the run sets FAILED —
  the old only-Fatal mapping left twice-failed recoverable runs
  misleadingly at RUNNING. DiskSpaceError instead rests at
  PAUSED_DISK_FULL; `voyage run` resumes from it (precheck re-pauses if
  still full). The video resume call goes through `_call_with_restart`
  (hook now takes segment_id), so a resume failure gets its own restart
  instead of aborting at once (supervisor.py).
- RPC timeout wired: the dead `timeout` param is now a select-deadline
  on worker stdout (new `SubprocessWorker(timeout=...)`, default 600s,
  `[voyage] rpc_timeout_seconds` plumbing from Supervisor construction
  so model-load `init` honors it too); expiry raises
  RecoverableWorkerError so restart engages and `stop()` kills the hung
  worker (rpc.py).
- Finalize preflight (DESIGN §53): `finalize_run(...,
  min_free_space_gib=0.0)` runs the shared `check_free_space` (moved
  supervisor → media, import direction stays supervisor → media) before
  any encode; CLI passes the run reserve and surfaces DiskSpaceError as
  exit 1 instead of a traceback (media.py, cli.py).
- tests/test_failure_policy.py (9 tests, TDD): breaker opens after the
  budget (attempt counts asserted), zero budget fails fast, repeated
  failure rests FAILED (not RUNNING), disk-full pauses then resumes
  after freeing space, silent-worker call times out, resume failure
  gets a second chance, resume failures share the budget, finalize
  preflight refuses then succeeds, config plumbing end to end.
  test_phase2 restart-hook lambda updated to the segment_id signature.
- Gates green (135 pytest / mypy 32).
- Phase 6 slice C (crash matrix) done 2026-09-23: tests/test_crash_matrix.py
  (7 tests, fake backends, real media) — killed audio worker recovers,
  killed director worker recovers, SIGKILL landing mid-`generate_blocks`
  restarts and retries the op, two straight video kills recover with no
  state leak, truncated-media partial dir without DONE is reused by the
  next commit, DONE-without-state-advance re-commits the same number
  cleanly, dead-director embed degrades to None instead of raising;
  every committed segment asserts validate_run() == [].
- Gates green (142 pytest / mypy 32).
- Phase 6 slice D (observability) done 2026-09-23: voyage/logrotate.py
  (daily rotation `<stem>-YYYY-MM-DD<suffix>` + 30-day retention prune,
  best-effort, live paths unchanged) wired into `_log_metric` and
  `SubprocessWorker.start()`; every metric event now carries `run_id`;
  `voyage status` renders the §59 sections (Uptime from manifest
  created_at, Video backend/render/timeline/segments/blocks/GPU via
  nvidia-smi, World, Audio style/energy/buffer, idle Workers with
  backends, last-commit per-stage seconds, Storage free GiB) degrading
  to 'unknown' instead of failing; tests/test_observability.py (6
  tests: roll, no-op, prune, worker-log roll, run_id on every event,
  status layout). Timezone notes: UP017 noqas are deliberate (worker
  image is py3.10, same precedent as persistence.py).
- Gates green (148 pytest / mypy 33).
- Phase 6 slice E (benchmark + soak) done 2026-09-23: `benchmark` RPC op on
  all workers (fake video/audio time warmup + measured renders; director
  times deterministic decisions; longlive/acestep time real probes with
  torch.cuda VRAM peaks, require init, honest errors otherwise),
  `voyage benchmark video|audio --run` (worker-backed §104 report with
  GPU/driver/torch env, honest unknowns off-GPU) + `benchmark end-to-end`
  (throwaway temp run, per-stage means + gauge deltas, never mutates user
  data) + `voyage soak --run --segments` (stability trend: stage means,
  RSS/disk deltas via summarize_gauges, validate verdict, exit 1 on
  errors); per-segment `resource_gauges` events (disk free, supervisor
  RSS peak, worker VRAM when health reports it; best-effort, never fails
  a commit); tests/test_benchmark.py (9 tests incl. endurance-marked
  3-segment flatness).
- Gates green (157 pytest / mypy 34).
- Phase 6 slice F (§87 docs tree) done 2026-09-23: README rewritten
  (quick install, model prereqs, basic run, docs index) + 11 files under
  Voyage/docs/ — INSTALL (3 images, CUDA, ffmpeg, env vars, downloads),
  MODELS (exact HF repos + revisions + links, mirrored from
  model_registry.py), BACKENDS (RPC adapter contract, fake/GPU/
  experimental backends), ARCHITECTURE (process diagram, ownership,
  commit pipeline, GPU time-sharing, run-dir layout), STATE_AND_RECOVERY
  (8 validate invariants + crash-scenario table), PROMPTING (charter,
  transitions, novelty 0.85, staged plans, code injection),
  AUDIO (slow loop keep/render/repaint, crossfade rules, final mix),
  OPERATIONS (§127 runbook + monitoring), TROUBLESHOOTING (OOM, CUDA,
  mismatch, ffmpeg, audio, disk-full, corruption, worker env),
  BENCHMARKING (§104 protocol, commands, report fields, soak
  acceptance), UPSTREAM_LONG_LIVE_PATCHES (pos-only caches, direct
  _inference_inner session, config deviations, adapter shims).
- Gates green (157 pytest / mypy 34, unchanged — docs only).

## Fast iteration, slice 5: style/coherence experiments + calibration (2026-09-23, worktree fast-iteration)

Style experiments (draft bf16 unless noted, deterministic director): exp-a chrome-river fp8 (motion 0.315→0.658 volatile, complexity 0.10-0.13, drift 0.07-0.10); exp-d calm dawn-lake bf16 (motion 0.076→0.080 stable, complexity 0.06-0.08, drift 0.01-0.02, similarity 1.000→0.662); exp-f ice-caves bf16 (motion 0.126→0.368, complexity 0.09-0.10, drift 0.03-0.04) rendered magenta/green posterized masses from a sane blue-ice prompt — systematic content failure (seed-12 retest in exp-g also garbage → NOT seed luck); exp-b full-res calm showed the same solarized look (draft shape exonerated as a cause); exp-e full-res bf16 viable (VALID 29f, no OOM). Mechanism picture: (M1) weak no-CFG conditioning + 4 distilled steps + rare refractive content → poor convergence and wrong hues (ice fails, dawn-lake works as common training content); (M2) highlight/saturation blowout → solarized blobs even in converged scenes (sun blob, chrome neon, green-aurora posterization — fp8 amplifies substantially per slice 4, bf16 reduces but does not eliminate; prompt care at extremes required, saturated greens included). Motion exists at full res (exp-b f0≠f28); exp-a seg0 frozen was segment-specific.
Decision dynamics (exp-h Qwen+fake 3-seg, CPU-only): destinations ice-caves → 'dynamic visual field' → 'shifting luminescent corridors' (all ESTABLISH/lighting_transformation, coherent luminous-abstract thread); audio energies 0.50→0.65→0.60 with evolving captions; take_0000 render + take_0001/0002 repaints on caption change, all anchored covers_from=0.0 (Phase-4 anchor fix holds under real director variation); the deterministic director instead emits static audio (one 45 s take, energy 0.48 — fit is accidental without director variation). VALID 144f.
Calibration (real footage): complexity 0.06-0.13 and drift 0.01-0.10 read BELOW StyleSpec bands (0.30-0.50 / 0.12-0.25) on all styles; motion spans 0.076-0.658 by style (chrome volatile, calm stable, ice mid-volatile). Bands HELD (they encode desired behavior; 3 styles × 2 segs is thin evidence). Determinism: same-process bit-exact (bare/bare-rerun/stream A==B==C at latent level); cross-process ±5 mean-abs noise floor (a bogus 17.58 reading traced to a mislabeled pair via triangle inequality — side-by-side contact sheets beat separate viewings).
Contention saga: 4 silent worker deaths (4-line log, no traceback, clean dmesg), ALL under GPU overlap with another agent's runs (exp-f vs determined_tu, exp-g vs silly_lamarr, exp-i × 2 vs vcont-seq); rule: verify GPU <2 GB used before EVERY GPU launch, single-GPU-job-at-a-time. Mechanism hunt (no supervisor timeout anywhere; worker stderr IS captured to the log so tracebacks would appear; EOF on stdout → 'closed stdout') proves non-Python death, but the source is unidentified — product fix deferred, never attempted blind.
Collision note: experiments ran in the main tree pre-18:06 UTC; the live tree now carries another agent's worker rewrite (325e6d5 stream fix: 2-arg append_block + _seq_noise, ON TOP of the slice-4 precision work which is intact) plus Phase-6 slices C/D/E and uncommitted work; exp-i's seg1 conv crash is in THEIR rewritten path (traceback shows their 2-arg call), NOT the slice-4 code — do not fix. Slice-5 docs written in worktree fast-iteration (based on bd5c4cd); experiment artifacts live in the main tree under ./Voyage/output/exp-* (preserved, gitignored).
Audio fit: mechanism proven (repaints on Qwen caption change, anchor holds); qualitative judgment (final.mp4) pending a clean GPU run after the seg1 crash is fixed. Gates RED at base (pre-existing slice-E SyntaxError video_longlive.py:925; this docs-only change is unaffected; code last green 111/32).

- Continuity fix done 2026-09-23: every segment started a new scene at
  every block boundary (~6x frame-diff jumps, in-segment and
  cross-segment). Root causes, all deviations from upstream's single-call
  `inference()` path (§22.5): (1) per-block fresh noise RNGs instead of
  one stream-level trajectory — fixed with a persistent session RNG
  seeded by the first block's seed, state taped per segment
  (`noise_rng_state`); (2) the attention preamble (local/sink/global-sink
  module setup) never applied on the direct-`_inference_inner` path, so
  `sink_size` stayed 0 and no history was ever prepended — fixed by
  mirroring the preamble at session build; (3) KV capacity: local 8 +
  sink 8 left zero rolling room (sink consumed the whole 8F cache) —
  fixed with `local_attn_size=16, sink=8` (VideoConfig default, all
  profiles). Full-res 16GB fit via VAE-offload-for-generate in
  `generate_blocks` (VAE 1.31GB parked on CPU during denoising; measured
  projection 14.44GB peak). Validation: draft teacup probes (local 8:
  0.19-0.23 boundary spikes + full viewpoint change at chunk 2; local 16:
  max 0.08, chunk-1 structurally invisible, chunk-2 flicker-level, same
   composition across all frames) + full-res production-path run vcont-prod
   (no-OOM, boundaries, eyeball). Bit-identical split-vs-single probe
   proved the session plumbing equals upstream's path; slim tests pin the
   `local_attn_size=16` default (tests/test_draft.py).
- Gates green (157 pytest / mypy 34).
- Continuity fix, part 2 (cache_start units + rebuild teardown), done
  2026-09-23: post-validation forensics on a 2x3 full-res run showed
  identical mid-chunk-1 white flashes + white-locked brightness + a 0.55
  cross-segment jump in both segments — traced to `cache_start_frame`
  units (§22.5 item 4): block indices (0,1,2) were passed where
  `_inference_inner` expects frame indices, so chunks 1-2 read
  zero-filled latents. Retrospective: the earlier vcont-prod segment 0
  ran with absolute (0,8,16) = already-correct frames and remains the
  true local-16 reference; the flashing run is invalidated (the
  prompt-attractor/white-absorbing-state theory built on it is
  discarded). Also fixed in the same pass: `handle_rebuild` session
  teardown (14.97GB resume OOM). Re-validation run vcont-fixed (2x3
  full-res, fixed code): both segments committed, no crash/OOM; seg0 is
  md5-IDENTICAL to vcont-prod seg0 (the fix is a no-op for the
  already-correct path); seg1 smooth (median 0.0052, max 0.0462 at the
  frame-1 settling transient, chunk boundaries 0.004-0.015, no flash,
  normal brightness); cross-segment 0.1467 with same-scene continuation
  (eyeball: same neon sign, no viewpoint change — a brief ~5-frame
  settling morph, not a cut).
- Gates green (168 pytest / mypy 35 files).
- LTXV alternative backend (Phase 7) slices 1-4 done 2026-09-24: Slice 1
  probe = GO bf16 at native 768x512 (stock `infer()` OOMs moving the fp32
  T5-XXL 18.8GB to GPU and `offload_to_cpu` does not prevent the upfront
  `.to(device)`, so the worker passes `text_encoder=None` + CPU-precomputed
  bf16 embeds — 25s encode, cacheable; attention masks moved to CUDA
  manually; `negative_prompt=None` with negative embeds; VAE bf16 halves
  resident 8.23->5.91GB; load 6s, t2v 25f 5.0s = 0.20s/f @6.68GB peak,
  extension 24f 5.8s = 0.24s/f @7.89GB peak; dawn-lake quality, no
  solarization, extension continuous). Pins: code Lightricks/LTX-Video
  @4b2d053, weights rev 8984fa25 (2B-distilled 0.9.8 6.3GB + spatial
  upscaler 0.5GB, ungated), TE PixArt-XL-2-1024-MS rev b89adade (18G fp32
  2-shard + tokenizer, `cp -rL` into voyage-models after verify flagged
  hub-only). Slice 2: `voyage/workers/video_ltxv.py` (resident
  LTXVSession, full op surface, benchmark saves/restores tail state) +
  registry `download/verify_ltxv_models` + CLI list/verify/download
  ltxv-2b + Dockerfile.video ltx layer (`--no-deps`, transformers stays
  4.57.6) + tests/test_ltxv.py (9 tests). Slice 3: supervisor routing
  (`ltxv` module + `STREAMING_VIDEO_BACKENDS` constant covering init
  payload split, acestep swap, num_blocks, restart hook; metrics.json
  `video_backend`; `use_relative_rope` stays longlive2-only). Slice 4
  live on idle 4060 Ti (/tmp/ltxv-e2e1, seed 11, fake audio,
  deterministic director): seg0 25 native frames VALID, seg1 cross-seg
  0.021 vs 0.018 within, seg2 `--blocks 2` = 49f (25+24, frame-0 dedupe
  OK), VALID 99f total, finalize 768x432 h264 + AAC; benchmark op
  29.4s/block, 6.54GB peak (one fail-then-retry); kill-test
  /tmp/ltxv-kill (`pkill -9 -f video_ltxv` at 96% util mid-seg1-denoise
  -> `worker_restart` attempt 1 -> `video_resumed` -> seg1 25f, VALID
  50f, cross-seg 0.021 vs 0.046 within); draft /tmp/ltxv-draft VALID 25f
  @640x352 (non-native sizes work). Two worker bugs fixed from log
  forensics: (1) randn device flake (upstream overrides device with the
  CPU `_execution_device` under `offload_to_cpu=True` vs the CUDA
  generator, ~40% block failure) -> `offload_to_cpu=False`; (2)
  stale-tail intra-call chaining (`_tail_png` updated only after the
  block loop; seg2 f025->f026 jumped 0.114) -> per-block chain PNGs,
  last renamed to tail, intermediates deleted.
- Gates green (168 pytest / mypy 35 files).
- 2026-09-24 `voyage generate` one-shot command (user request: minimal-param
  fixed-duration video, e.g. `generate --backend ltxv --duration 5s`).
  `config.py`: `_VIDEO_BACKEND_PRESETS` table + pure `with_video_backend`
  (ltxv -> 768x512/`cuda:0`/profile `ltxv-512p`; longlive2 -> default
  geometry/`cuda:0`; fake -> CPU) + `default_config_toml(..., video_backend)`
  kwarg; `init` parser gains `--backend` (default fake). `cli.py`: new
  `generate` subparser (defaults backend ltxv, run-id voyage, seed 0, run
  dir `./output/<run-id>`, final MP4 `<run>/final.mp4` unless
  `--final-video`; reuses existing `--draft/--director/--blocks/--take-seconds/--quantization/--skip-bad`
  names) chaining init -> effective-config segment math -> `cmd_run` ->
  `validate_run` (abort unless `--skip-bad`) -> `cmd_finalize` + summary;
  `parse_duration` (`90`/`90s`/`2m`/`1m30s`/`1h`/`1h2m3.5s`) +
  `segments_for_duration` round-up (`_frames_per_segment`: ltxv 25 +24/block,
  else `segment_frames`); `_warn_if_no_cuda` for GPU presets without a
  visible GPU. `run.sh` needs no change (pure `$@` passthrough). Bug found by
  the new default-dir test: workers spawn with CWD=run_dir, so relative run
  dirs double up in payload paths (take WAV landed under
  `<run>/<run>/...`) -> `cmd_generate` resolves the run dir once; the
  codebase invariant is absolute voyage paths. `tests/test_generate.py`
  (duration/rounding/presets/fake 4s e2e VALID + final.mp4/refusal/default
  dir/CUDA warnings). Docs: README one-shot example, OPERATIONS `generate`
  section. Gates green (198 pytest / mypy 35). Follow-up fix 2026-09-24: the
  first live `generate --backend ltxv` failed with `No module named 'torch'`
  (slim image + no GPUs) -> `run.sh` now auto-selects `voyage-video:latest`
  with `--gpus all` for CUDA backends (explicit env wins), pins `-w /app`
  (the video image WORKDIR `/opt/longlive` swallowed output into the
  ephemeral container), and the CLI fast-fails with a `voyage-video` pointer
  when torch is missing (`_require_cuda_stack`, `find_spec`-based per §83;
  3 tests). Proven live: the exact user command committed 5 ltxv segments
  and wrote `final.mp4` (h264 768x432 + AAC, 5.25s), `validate` VALID 125f.
  Follow-up fix 2026-09-24: that video's audio was a constant sine — the
  presets only set video, audio stayed on the toml-default `fake` backend
  (ffmpeg `sine`). `_AUDIO_BACKEND_PRESETS` now pairs audio with video
  (`ltxv`/`longlive2` -> ACE-Step music on `cuda:0`, `fake` keeps sine),
  applied in both `default_config_toml` and `with_video_backend` (3 tests).
  Proven live: fresh `--run-id voyage-audio` run, 5 ltxv segs, `final.mp4`
  AAC 5.23s with dynamic music (per-window RMS -11 to -59 dB), VALID 125f.

## 2026-09-24 — new-DESIGN.md merged into DESIGN.md (backend-neutral spec)

- Merged the full new-DESIGN proposal per Q&A agreement (append-everything,
  removals handled case-by-case against the tree): header research snapshot
  2026-09-23 + pluggable-backend line; §1 purpose paragraph; §2 renderer
  diagram (`selected VideoBackend (LongLive / LTX / CausVid)`); §5 rebuilt
  as §§5.1 (backend table + `VideoBackend` protocol + `state_mode` trio +
  8-step selection rule), 5.2 LongLive, 5.3 LTXV, 5.4 CausVid, 5.5 base
  model, 5.6 checkpoints, 5.7 licensing; §11.1 worker box genericized;
  §14 backend profiles + `segment_seconds` + `[video.longlive/ltxv/causvid]`
  blocks; §24 production-values paragraph deleted (single-sourced in
  §22.5); §48 whitespace collapse; §59 status backend line; §85.1 LTXV +
  CausVid downloads; §86 license rows; §§118-120 migration/recovery +
  benchmark-driven selection + 3 reference profiles; §128 16-step order;
  §136 supported/other split; §§137A-D (benchmark/continuation/recovery/
  roadmap); §138 qualified target; §139 backend-specific upstream notes;
  §137 single blank-line change.
- Keepers: §22.5 (stream-noise RNG + preamble mirror + 16/8 floor +
  `cache_start_frame` frame-units + `handle_rebuild` evict — live
  `video_longlive.py` depends on each) and the full §140 log kept verbatim.
- Codebase-reconciliation inserts (marked `As-built note (2026-09-24)` in
  §§5/14/46/118 + `ltxv-*`/`causvid-*` profile names in §65): proposal text
  stays canonical while the live contracts are recorded — sync
  `backends.py:VideoBackend` precursor, `VIDEO_WORKER_MODULES` +
  `STREAMING_VIDEO_BACKENDS`, `generate_blocks` RPC, `VideoConfig`
  (`segment_frames`/`blocks_per_segment`/`quantization`), LTXV as-built
  (768×512, `25+(B-1)*24`, tail-PNG, bf16-first, `generate --backend ltxv`
  default), CausVid spec-only.
- TASK.md gained §30 transition checklist (CausVid remaining, LTXV drift,
  config/interface duality, missing benchmark/audit artifacts).
- Verification: §22.5 + §140 entries confirmed present; gates below.

## 2026-09-24 — rhythm-cut music-video stack (clipping fix + beat grid + 1024x576 + parallel director)

- User report on a 1m ltxv `generate` (run-id boba): audible clip at every
  segment joint; motion jumps per segment (kept as a rhythm feature);
  final felt low-res (768x512 segments downscaled to 768x432 with side
  pillars); no prompt evolution in 60 s. Agreed via Q&A: overlap
  crossfade (proportional 10% capped, final-only), adaptive-k beat grid
  (4 → 8 → 16 … until BPM >= 60), native 1024x576 with finalize keeping
  generation size (new runs), director drift every segment with the LLM
  running parallel to segment generation (Qwen3-8B on CPU, must finish
  before video gen; deterministic hold on miss).
- Audio joints (`media.py`): `finalize_run` used the concat demuxer on
  per-segment A/V muxes — a hard splice by design. New `build_final_audio`
  re-slices each segment window (extended half the overlap per side,
  clamped to the timeline) from the takes ledger and chains
  `acrossfade d=overlap`; total stays exactly the video timeline (no A/V
  drift). Overlap = min(0.10 x shortest segment, 0.5 s cap); <0.05 s or a
  missing ledger degrades to the legacy hard-splice concat. Per-segment
  `audio.wav` previews stay hard-cut. Final encode is now video-concat +
  blended mix with explicit `-map 0:v:0 -map 1:a:0` (plus `-shortest`
  safety). New `AudioConfig` fields `beats_per_segment=4`,
  `final_overlap_fraction=0.10`, `final_overlap_cap_seconds=0.5`.
- Beat grid (`voyage/audio/beat.py`, new): `beats_for_segment` (adaptive
  k doubling to hold >= 60 BPM: 4 s LTXV = 4 beats = 60 BPM; 2 s fake =
  4 beats = 120 BPM; 5.04 s cold-start = 8 beats ~ 95 BPM) +
  `quantize_take_seconds` (takes snap to whole segments so downbeats stay
  on boundaries). `AudioTake.bpm` recorded in the ledger (absent on legacy
  lines); planner `segment_seconds` quantizes fresh takes (None = legacy).
  `acestep.render_take` takes explicit `bpm` (else energy mapping);
  supervisor computes the grid BPM per segment, passes it in the audio
  payload (`take_rendered` metric logs beats/bpm). ACE honors tempo as a
  hint — alignment is approximate, documented as such.
- Resolution: ltxv preset stays 768x512, but `cmd_finalize` now forwards
  the run config's generation size instead of the 768x432 default — the
  pillar/downscale loss is gone (768x512 in, 768x512 out; old runs
  refinalize natively too). Native 1024x576 was tried and reverted the
  same day: the forward needs ~15.6 GB (13.9 resident + 1.7 transient),
  beyond the 16 GB card even via the dynamic-fp8 fallback — which itself
  exposed a latent `torchao.quantization.quant` import that no longer
  exists in pinned torchao 0.13.0 (fixed to `quant_api`; the worker runs
  from the bind mount, no image rebuild). 1024x576 needs a dedicated
  memory-optimization pass (VAE tiling/chunked decode); model
  post-upscale stays a Zoomy concern per `docs/MODELS.md`.
- Director: `generate` defaults `--director` to qwen when unspecified
  (file default was deterministic at the time; explicit flags win). New
  `VoyageConfig.drift_every_n_segments=1` — non-drift segments hold via
  the deterministic path (`drift_hold` metric). Parallel prefetch:
  after each accept, the raw N+1 proposal is submitted to a 1-thread
  executor and runs during video+audio render (CPU vs GPU, no
  contention); the next commit consumes it as the accept loop's first
  candidate when it targets the right segment and no fresh inspect
  amendments exist (`director_prefetch_hit/miss` metrics). Store writes
  stay on the commit path. `SubprocessWorker.call` gained a lock so the
  shared director stream cannot interleave. Qwen/embedder loads are
  offline-first (`HF_HUB_OFFLINE=1` default, explicit 0 re-enables) so
  missing weights fail fast to the deterministic fallback instead of
  hanging on a download.
- CLI: `--beats-per-segment` / `--drift-every-n` on `run` + `generate`.
- Tests: `test_rhythm.py` (beat math, quantization, ledger), updated
  ltxv preset asserts, `test_generation_stack.py` (config/validators/
  toml/overrides, drift cadence, prefetch hit, blend duration-exactness,
  overlap-0 legacy path, no-ledger fallback, generate-with-default-qwen
  offline-fallback e2e), `test_failure_policy` partial-worker lock fix.
- Gates green (279 pytest / mypy strict / ruff + format). GPU
  verification: 768x512 ltxv segments + blend + BPM log + Qwen CPU timing
  on idle GPU (1024x576 reverted — see resolution note above).

## 2026-09-24 — Stream A LTXV realign (121/25/96, mp4 tail, spec JSON tape)

- Worker now renders 121-frame clips with 25-frame video-tail conditioning
  and prefix-discard (96 novel committed), writes `video_tail.mp4` + sha256
  and the §5.3 JSON tape (clean break from torch tapes); PNG tail removed.
  Upstream verdict (README + ltx.io): keep 768x512, reject 768x432 (pads to
  448). Live E2E VALID 338f (121 fresh + 121 cut + 96 extension, ~17s video
  stage for the extension). Deferred: 81/97/121 matrix; stale cli duration
  math + relative-`--run` path doubling noted in §30.2. Collided with the
  1024x576 preset move (still /32-clean; E2E evidence is 768x512) —
  1024x576 VRAM probe still open.

## 2026-09-24 — Stream B §137A longlive2 qualification (harness done, GPU leg PENDING)

- Delivered the CPU-runnable harness (`tests/test_qualification.py`: stage
  list, metric math, <3x boundary gate, `summarize_run`, fake 3-seg +
  kill-recovery dry-runs), `reports/video-backends.md` (longlive2 leg with
  registry pins + PENDING table, empty LTXV leg for Stream A), and
  `scripts/qualify.sh` (idle gate + benchmark + 3-seg + validate + JSON
  summary). GPU leg PENDING: 4060 Ti held by Stream A for the full back-off
  window — never ran under contention per the GPU-contention rule. Measured
  harness dry-run instead (fake CPU, NOT longlive2): VALID 144f, steady
  ratio 1.32, continuity 2.16 PASS. To complete: idle GPU → `qualify.sh` →
  fill table.

## 2026-09-24 — Stream C config/interface duality (adapter, no migration)

- Single contract is `voyage/backends.py:VideoBackendAdapter` — spec
  `generate_segment`/`segment_seconds`/`state_mode` vocabulary caller-side
  over the unchanged sync `generate_blocks` wire op and unchanged
  `VideoConfig` schema (sync by design: blocking transport + §46
  no-concurrency rule; no migration needed). 21 CPU-only tests in
  `tests/test_backends_adapter.py`; mypy strict + 250 pytest green.

## 2026-09-24 — Stream D CausVid prep scaffolding (no worker)

- `model_registry.py` gains additive-only `CAUSVID_*`/`WAN21_*` pins (code
  `adb6a5e`, weights `b545eb27`, Wan2.1-1.3B `37ec5126`) +
  `download_causvid_models`/`verify_causvid_models` mirroring the ltxv
  pattern; new `docs/UPSTREAM_CAUSVID_NOTES.md` (pins, CC BY-NC-SA 4.0
  implications, 832x480@16fps/21-latent/overlap-3 notes, 6 worker-slice
  questions) and `tests/test_causvid_prep.py` (5 CPU-only tests). No worker,
  no `cli.py` wiring, no weight downloads. Full pytest 229 passed at the
  time (gates.sh red only on concurrent files).

## 2026-09-24 — Console progress layer (prompts on screen, console-only)

- New `voyage/console.py` renders per-segment progress for `run`,
  `generate`, and `soak`: destination + phase, full video prompts, music
  caption with the beat grid (`beats @ BPM`), then spinner stages and a
  commit summary with take IDs and per-stage seconds. `--verbose` adds
  seeds, transitions, and take reasons; console-only by contract
  (logs/metrics untouched).
- `rich>=13.7` is a core dependency; worker images install it explicitly
  because they use `--no-deps`. Colors/animation engage only on a real TTY
  with `rich` installed and color allowed; pipes/tests get identical plain
  words. `--verbose` / `--no-color` parse on `run`, `generate`, `status`,
  `validate`, `finalize`, `benchmark`, `soak`, and `inspect`.
- The supervisor takes an optional progress sink (default silent), so
  existing callers/tests are unchanged. Covered by `tests/test_console.py`
  (12 tests); gates green (291 pytest / mypy strict / ruff + format).

## 2026-09-24 — Launcher TUI (bare `voyage` configures `generate`)

- Bare `voyage` (no verb) launches a Textual app (`voyage/tui.py` +
  pure `voyage/tui_state.py`): every `generate` setting shown editable
  with its default (required Story fields first, advanced collapsed),
  live validation with a derived segments/frames plan, then Generate
  runs the unchanged init → run → validate → finalize with per-segment
  video + audio prompts, a progress bar, and a Stop button (Back/Quit
  after completion). Subparsers are no longer required; non-TTY or
  missing-Textual invocations get guidance + exit 2, all verbs stay
  non-interactive.
- Generation reuses `cmd_generate` via a `progress_sink` namespace slot:
  when set, the console stays silent and the TUI's `TuiProgress`
  (a `SegmentProgress`) reports instead — existing tests never set it,
  so their stdout assertions hold. `cmd_run` also tolerates a non-main
  thread (no SIGINT handler there); Stop writes STOP_REQUESTED to the
  run-dir state, the same control plane as `voyage stop`.
- `textual>=8.0` is core (`pyproject.toml` + explicit installs in both
  `--no-deps` worker images); `run.sh` allocates `-it` when attached.
  Covered by `tests/test_tui.py` (state/validation/namespace/plan +
  bare-command wiring + app structure, Textual-dependent parts skipped
  without the extra).

## 2026-09-24 — Stream B GPU leg + qual-driven fixes (longlive2 + acestep)

- Ran the §137A leg on an idle 4060 Ti (`output/qual-longlive2`, 1-block
  fp8 segments): `benchmark video` 29f/block @ ~38.5s, 14.28 GiB peak;
  `run --segments 3` → VALID 87f (29f ≈ 1.21s @ 1280x704 each; steady
  ~38.4s/segment, ratio 2.56; audio take kept after seg0). Results table in
  `reports/video-backends.md` (environment PENDINGs filled: CUDA 12.8.0 /
  torch 2.8.0+cu128 / 62 GiB host).
- Continuity FAILs the provisional 3x gate (within 0.0246 / boundary 0.1134
  → 4.61x), but per-boundary isolation exonerates the resume path: the
  no-rebuild boundary (0.1062) matches the post-rebuild one (0.0957), so the
  jump is native inter-block drift (every boundary IS a block boundary at 1
  block/segment — the Phase-2 pass criterion). The 3x gate is miscalibrated
  for slow-motion content; manual eyeball review still open.
- Fix 1 — post-audio `rebuild` OOM (deterministic, 3/3 fresh processes, not
  contention): `resume_from_tape` replayed the DiT forward with the 1.31 GiB
  VAE resident (14.40 GiB + audio residue > 15.57 budget). Now
  `pipe.vae.to("cpu")` + `empty_cache()` around the replay with `finally`
  restore (`voyage/workers/video_longlive.py`), mirroring
  VAE-offload-for-generate.
- Fix 2 — `init --backend longlive2` could never commit: the preset left
  768x432 while the worker always renders native 1280x704 (latent x16),
  failing the commit-time resolution check. Preset now pins
  `longlive2-704p` / 1280x704 (`voyage/config.py`); preset test renamed to
  `test_longlive2_preset_pins_native_geometry`.
- Fix 3 — `_run_dir_arg` resolves absolute (relative `--run` doubled paths
  inside workers, hit live); `_frames_per_segment` gains the longlive2
  branch ((8B-1)*4+1: 29/93 measured) and the ltxv branch moves to the
  Stream-A 96-novel steady state (was pre-realign 25/24). Tests:
  `test_run_dir_arg_resolves_absolute`,
  `test_frames_per_segment_longlive2_follows_decode_expansion`,
  `test_frames_per_segment_ltxv_uses_novel_minimum`. Gates: 309 pytest /
  mypy strict / ruff + format clean.
- 1024x576 LTXV probe (same window): deterministic OOM 3/3 (13.69 GiB +
  1.72 GiB failed) — the 768x512 preset stands; the §30.2 revert note is
  now measured evidence, not just a claim.

## 2026-09-24 — Launcher-TUI usability pass (focus/dropdown/descriptions/chrome/keys)

- Why: user-reported papercuts on the bare-`voyage` form — nothing took
  focus on mount so typing went nowhere (fields felt uneditable), the
  backend Select was not seen as a dropdown, card titles/fields had no
  visible descriptions (verbose help lived in tooltips only), the layout
  felt airy, and keyboard-only operation was not real.
- What changed (`voyage/tui.py` app shell only; `tui_state.py` behavior
  untouched): Style is a soft-wrapped multiline `TextArea`
  (`voyage/tui.py:329-334`) and `AUTO_FOCUS` targets it
  (`voyage/tui.py:218`, mount-time `set_focus` fallback at
  `voyage/tui.py:491-494`) so typing lands with no click; `:focus`
  accent styling on inputs/selects/buttons (`voyage/tui.py:294-299`);
  every Select shows a visible `▾` affordance in its label plus a
  `▾ pick ...` prompt (`voyage/tui.py:357-394`); card titles carry
  one-line `.card-desc` descriptions and technical fields carry
  `.field-hint` lines (`voyage/tui.py:322-394`); chrome tightened (card
  padding `0 1`, margins `0 1`, shrunken key-hint/run-view margins —
  `voyage/tui.py:220-300`) with single-column cards kept; key map
  `ctrl+g` generate / `ctrl+s` stop / `b` back / `ctrl+q` two-press
  confirm-quit while running (`voyage/tui.py:302-307,568-576`) plus
  in-app key-hint bars (`#key-hints`, `#run-keys`); mouse stays optional
  (buttons mirror every action). Remembered settings
  (`~/.config/voyage/tui-last.toml`: `load_last_settings` prefill +
  `save_last_settings` after validation) and the `#gpu-warning` line
  (`tui_state.gpu_warning`) are pre-existing and unchanged.
- How verified: repro-first Pilot — 15 of 16 new headless `run_test()`
  Pilot tests in `tests/test_tui_app.py` failed pre-fix, then the
  implementation until green. Gates: 343 pytest / mypy strict / ruff +
  format clean.
  Follow-up folded in: the `tests/test_tui.py` e2e drives the Style
  editor as a `TextArea` (sets `.text`, not Input `.value` —
  `tests/test_tui.py:138,147`).

## 2026-09-24 — LongLive audit (stage timers + verdict, §16/§137D phase 1)

- Instrumented `LongLiveSession.generate_blocks` with off-by-default
  CUDA-event stage timers (`profile_stages` flag through
  `handle_generate_blocks`/`handle_benchmark`; zero overhead when off —
  proven by test, not inspection) + `tests/test_longlive_stages.py`
  (6 CPU-only tests).
- Steady-state splits (fp8, 1-block/29f @ 1280x704, blocks 1-2):
  DiT denoise ~16.9s (44%), VAE decode ~16.3s (43%), PCIe offload
  roundtrips ~4.4s (11.5%), media write ~0.7s (2%) — budget closes at
  ~100% against ~38.3s block wall (~32:1 wall:video; the old ~100:1 was
  cold-start-dominated). Verdict: §16 category I (mixed) — DiT genuine
  throughput + VAE (D) + transfers (C); A/B/E/F/H excluded with evidence
  (torchao W8A8 provably engaged: 300 linears quantized, 6 kept BF16).
- bf16 is NOT runnable full-res on 16 GiB: init OK (11.1 GiB resident)
  but the first forward OOMs deterministically (14.93 GiB in use, +42 MiB
  requested) — per the user's bf16-first/fp8-fallback call, fp8 stands.
  First-call excess ~55 s (block 0 denoise 72 s vs 16.9 s steady) is a
  once-per-session CPU-T5/autotune cost, excluded per TASK §4.2.
- Full writeup: `Voyage/reports/longlive-audit.md`. Raw probe logs were
  ephemeral (`/tmp/auditprobe/`, host-only). Gates green on scope
  (ruff + format + mypy strict + targeted pytest; full-suite single
  failure is the concurrent TUI agent's `test_tui.py` e2e, out of scope).

## 2026-09-24 — Launcher-TUI frontpage rework (single list + help panel)

- Why: user verdict on the card-based form — "ugly frontpage",
  flaky-feeling keyboard (root cause: `ctrl+s` is terminal XOFF, so
  Stop keystrokes froze the terminal, not the app), no visible help,
  no distinction where errors are, Generate seemingly froze (root
  cause: `_start_generation` exceptions never switched views, and
  worker-thread stdout fought Textual's alt screen), and settings
  sprawl (run-id/output/final-video split, advanced collapse).
- What changed: the form is now one compact required-first row
  list (`_field_row`, `voyage/tui.py:379`) — Style (`TextArea` h3),
  Name, Duration, Backend/Director/Quantization `Select`s, Blocks,
  Take seconds, Beats, Drift, Seed, 5 checkboxes; run-id + output +
  final-video merged into a single `name` (`tui_state.py`: new
  `name` field, `_flat_folder_name` check, `field_errors()` boundary,
  `to_generate_namespace` derives `output/<name>` +
  `output/<name>/final.mp4`; legacy `run_id` TOML migrates silently).
  A focus-driven help panel (`#form-columns` / `#help-body`,
  `NARROW_WIDTH = 100` stacks it below on narrow terminals) shows
  the overview + focused field's `FIELD_HELP` + its error; invalid
  fields get red borders (`field-invalid`). Key map is now
  `ctrl+g` / `ctrl+x` (stop) / `b` / `ctrl+q`; the old `ctrl+s`
  binding is gone with a table note explaining XOFF. Freeze fixes:
  `_start_generation` is exception-guarded (failure returns to the
  form with the error in `#errors-line`) and worker stdout/stderr
  are redirected into the run log.
- How verified: Pilot repro tests first (`tests/test_tui_app.py`:
  freeze regression via monkeypatched `cmd_generate`, help-panel
  focus updates, narrow CSS, red-border class; click tests need
  `scroll_visible` since Generate sits below the fold at 120x40).
  Gates: 350 pytest / mypy strict (39 files) / ruff + format clean.

## 2026-09-24 — Launcher-TUI compact pass (title + single-line rows + stable focus)

- Why: user follow-up — no project title, form too airy, and some
  fields (e.g. backend) changed the vertical spacing on focus.
- What changed (`voyage/tui.py`): `#app-title` heads the form
  ("Voyage — one-shot music-video generator"); every field row is
  now a single line (the Style editor stays 3) via borderless
  `$surface` wells — focus/invalid recolor the background only
  (`$surface-lighten-1` / `$error-muted`), so geometry is identical
  in every state by construction; empty `#errors-line` /
  `#gpu-warning` are hidden (`_set_line` toggles `display`) instead
  of occupying dead rows; checkbox/button/hint margins tightened.
- How verified: repro-first Pilot (`tests/test_tui_app.py`: title
  present, all non-style rows height 1, focus + invalid toggles on
  every field keep all row/widget heights bit-identical, empty
  notice lines take no space). Full gates after: TUI scope green
  (ruff + format + mypy on touched files, full pytest); the only
  gates red is the concurrent agent's in-flight
  `voyage/workers/video_causvid.py` (mypy, out of scope, untouched).
- Hero-title follow-up (same day): `#app-title` moved out of
  `#form-columns` to a full-width screen-level hero above the form —
  massive 5x35 block-letter `TITLE_ART` (bold, accent, centered) plus
  an italic muted subtitle, with spacing below; persists during runs.
  Textual has no font scaling, so presence comes from size + color.
  Full gates green after (387 pytest).
- Clipping + stuck-view follow-up (same day): the 5-row art was
  clipped by `#app-title { height: 5 }` (padding-top stole a row) —
  height is now `auto`; content-height views overflowed small
  terminals with nowhere to scroll, so `#form-view` / `#run-view`
  are `height: 1fr` and scroll internally (backend row reachable via
  `scroll_visible` at 120x24); the worker's `except Exception`
  missed `BaseException` (a `SystemExit`-class failure left a stuck
  run view), now `except BaseException` with `CancelledError`
  re-raised. Repro-first Pilot (title reserves art+pad rows,
  constrained-scrollable form, real fake-backend click-to-✓,
  SystemExit recovery); TUI scope green (58 tests).

## 2026-09-24 — CausVid worker live (first E2E generation)

- Worker `voyage/workers/video_causvid.py` renders: three 16 GB fitment
  fixes were needed beyond the scaffold — (1) upstream hardcodes
  `WanModel.from_pretrained("wan_models/Wan2.1-T2V-1.3B/")` (+ T5/VAE
  same prefix), fixed with a CWD-contract `_enter_causvid_tree`
  (symlink + chdir, longlive precedent); (2) upstream `pipeline.to()`
  parks the 11 GB T5 on GPU (forward OOMs at 14.9 GiB) — T5 now lives
  on CPU, shuttled to GPU once per segment for a pre-encode of all
  prompts (one 11 GB roundtrip/segment; per-rollout shuttling
  fragmented into OOMs), reused via a stub encoder swapped onto the
  resident pipeline (DiT takes embeds with no device transfer, so CPU
  embeds would crash); (3) `WanVAEWrapper` has no `.encode` — re-encode
  goes through `vae.model.encode(x, scale)` (script parity).
- Wiring: `VIDEO_WORKER_MODULES` + `STREAMING_VIDEO_BACKENDS` +
  adapter `_STREAMING_BACKENDS` + `causvid-480p` preset (832x480, fps
  16, latent `[1,21,16,60,104]`) + `models download/verify causvid` +
  `--backend causvid` (init/generate) + `_frames_per_segment` 72/rollout
  + adapter honors reported novel/conditioning + run.sh CUDA sniffing.
  Preset now also carries fps/latent_shape into `voyage.toml`
  (`default_config_toml` hardcoded 24/[1,8,48,44,80] — the supervisor
  sent fps 24 and the worker refused; all four presets pin both
  explicitly, behavior unchanged for fake/longlive2/ltxv).
- Live proof: `generate --backend causvid --duration 5s` → VALID 144f
  (2x72 novel, 81-decoded/rollout) @ 832x480/16fps + AAC music,
  `final.mp4` 9.0s; chained boundary diff 1.02x (seamless);
  minterpolate 16→24 validated on real frames (214f, no luma shift —
  finalize-stage wiring still open). Inspection copy (gitignored):
  `Voyage/output/causvid-e2e_final.mp4`.
- Gates: ruff + format + mypy strict + 387 pytest green. Open:
  overlap sweep (1/2/3+), resume-vs-uninterrupted A/B, 24 fps finalize
  stage (`presentation_fps`), manual eyeball review.

## 2026-09-25 — Launcher-TUI liveness pass (blank dropdowns + proof-of-alive)

- Why: user report — dropdown fields clipped vertically / blank
  focusable boxes under the fields / Generate "still freezes" with no
  monitoring view. SVG-text screenshots of the headless app proved the
  first two are one bug: the `Select` value line renders blank (labels
  fine, `Input` values fine).
- Root cause (bisected property by property on Textual 8.2.8): an
  explicit `height` on `Select` — even the same 3 rows the tall border
  computes to — collapses the value line to blank. `border` /
  `background` / `padding` are innocent. Fix: Selects keep bordered
  chrome with `height: auto` (geometry fixed at 3 rows by the border;
  focus/invalid recolor the border only, spacing constant), rows use
  `tall=True`; the `height must stay auto` NOTE in
  (`voyage/tui.py`) guards the regression.
- Freeze investigation: probes proved the machinery sound — run view
  switches synchronously, `run_worker(thread=True)` never blocks the
  loop (Tab moves focus mid-run), CUDA fast-fail returns to the form
  in ~1s. The remaining failure modes were silence, not deadlock:
  (a) nothing painted before worker boot, (b) no motion during
  minute-long silent stretches (Qwen weight load), (c) exit-code
  failures hid the reason (stderr went to the hidden run log). Fixes:
  synchronous headline + `▶ starting` line before boot (namespace is
  now built up-front, so settings errors also surface on the form),
  1s elapsed heartbeat on the run head (ticks iff the loop is alive;
  stopped with the run), and the worker's last captured line appended
  to nonzero-exit errors.
- Proof (all in `tests/test_tui_app.py`): SVG-text asserts for all
  three dropdown values, Select row height >= 3, slow-run Tab-moves-
  focus mid-run, segment line lands in history before completion,
  fast-fail restores the form with the CUDA reason, elapsed tick
  appears within ~2.5s and the timer stops at finish.

## 2026-09-25 — run.sh bare-TUI CUDA default (ltxv with no variables)

- User report: explicit `VOYAGE_IMAGE=voyage-video VOYAGE_GPUS=1` works,
  bare `run.sh` must work too. Root cause: bare launch (TUI, backend
  picked interactively) carried no CLI backend signal, so run.sh fell
  through to the slim CPU image with no `--gpus` — ltxv then fast-failed
  (or hung at worker init) inside a torch-less container.
- Fix (`scripts/run.sh`): when invoked bare (`$# == 0`) probe
  `nvidia-smi -L`; GPU present → `voyage-video:latest` + `--gpus all`
  (explicit `VOYAGE_IMAGE`/`VOYAGE_GPUS` still win); no GPU → stay slim
  (fake smoke runs; CUDA picks fast-fail to the form with the relaunch
  hint). `VOYAGE_DRY_RUN=1` seam prints `image=`/`gpus=` and exits
  (test-only, never runs docker).
- Proof: `tests/test_run_sh.py` (7 tests, isolated PATH of symlinked
  coreutils + fake nvidia-smi ok/fail/absent, temp HOME): bare+gpu →
  video+gpus, bare w/o gpu or failing smi → slim+none, explicit image /
  `VOYAGE_GPUS=0` win, `generate` → CUDA without host GPU, explicit
  `--backend fake` stays slim on a GPU box. Docs: OPERATIONS TUI +
  run.sh paragraphs rewritten, `cli._cuda_stack_error` / `_require`
  docstring mention bare launches.

## 2026-09-29 — finalize audio: acrossfade out, manual fades everywhere

- User report (run-id poulah, 31 segments): `finalize` never finished —
  the process sat 3+ days with no output. Root causes, both in ffmpeg
  7.1.5 `acrossfade`, both reproduced live: (1) the final blend built
  ONE invocation chaining 31 acrossfades — that graph deadlocks the
  filter scheduler (futex wait, zero bytes out), while a 2-input graph
  finishes in milliseconds; (2) even a 2-input acrossfade collapses
  whenever the FIRST input is much longer than the second (86s+1s at
  d=0.4 came out as ~0.2s; window 21's 3.49s+0.91s slices came out as
  ~1–3s instead of ~4.0s). Short-first and equal-length pairs are fine,
  which is why 4-segment runs (karl) always worked.
- Fix (`voyage/media.py`, new `_blend_pair` + `_audio_duration_seconds`):
  every blend is now afade-out on the first tail + afade-in on the
  second head + adelay + amix (normalize=0), 2 inputs per ffmpeg call,
  s32le intermediates, fade clamped to half the shortest input.
  `build_final_audio` reduces pairwise through it (was: N-input chain);
  `assemble_segment_audio` left-folds through it (was: inline N-chain).
  This is the recipe `docs/AUDIO.md` already prescribed ("manual fades
  + delay, never acrossfade") — the code had regressed away from it.
- Proof: `tests/test_final_blend_scale.py` (+2 TDD tests, watched fail:
  20s+1s pair and 3.49s+0.91s slices both collapsed on acrossfade, both
  exact on manual fades) + an instrumented 31-window blend run logging
  every step exp==actual. Poulah finalized: 125.29s h264 768x512 + AAC,
  A/V drift 0.08s. Known residuals (open, inaudible): take files render
  ~0.175s short of ledger on 45.375s takes (ACE variance, crossfades
   absorb it); take-joint windows shrink by the assemble fade instead of
   tiling exactly (~0.67s over 31 segments, subsumed in wider blends).

## 2026-09-29 — Containers run as the host user (issue 053 follow-up)

- User report: every render left root-owned files on host bind mounts
  (`output/`, `/tmp`, `~/.cache/voyage-models`), needing a manual
  `chown goulade:goulade` after each run. Root cause: no Dockerfile set
  `USER` and no script passed `--user`, so everything ran as root.
- Fix: all three images (`Dockerfile`, `worker/Dockerfile.video`,
  `worker/Dockerfile.director`) create a real `voyager` user carrying the
  host's UID/GID (`ARG UID/GID`, defaults 1000; `getent`-tolerant so
  rebuilds and pre-existing ids never fail), with `HOME=/home/voyager`
  (user-owned, so torch/triton/HF caches work rootless) and `USER voyager`
  as the default. The block sits after the slow pip layers so rebuilds
  stay cached; the slim build-time `voyage doctor` now runs AS voyager,
  proving the stack resolves without root. `scripts/build*.sh` forward
  `--build-arg UID=$(id -u) GID=$(id -g)` (rebuild after a host-uid
  change); `run/gates/test/qualify.sh` pass `--user=$(id -u):$(id -g)`
  explicitly, so even a stale image (built under other ids) still writes
  host-owned files. GPU workers are supervisor subprocesses inside the
  same container, so one flag covers them too.
- `/models` writability (the deferred-follow-up concern in
  `docs/TROUBLESHOOTING.md`): resolved by a one-time
  `sudo chown -R goulade:goulade` of `~/.cache/voyage-models`,
  `Voyage/output/`, and one stray `__pycache__` pyc (zero root-owned
  files remain in either scope; host `/tmp` held no voyage residue).
- Proof: slim rebuild + `scripts/gates.sh` (ruff + format + mypy strict +
  844 pytest; the single `test_pause_mid_run_stops_at_boundary` failure
  is a timing flake — passes on retry, untouched by this change) plus a
  fake-backend `init → run --segments 1 → validate → finalize` into a
  throwaway dir: `find -user root` empty, every artifact (segments,
  logs, `state.json`, `final.mp4`) `goulade:goulade`. Director + video
  images rebuilt from cache with the same user block: director probes
  `uid=1000(voyager)`, `HOME=/home/voyager`, host-owned bind-mount writes
  (ruff + format pass; its mypy gate hits a pre-existing numpy-stub
  syntax error in site-packages — unpinned-numpy drift per issue 011
  remainder, untouched here). Video image probes identical CPU-side
  (`voyage doctor` runs as voyager). GPU-as-voyager execution probe still
  open — GPU 0 held ~7 GiB by another process at the time (contention
  rule: re-run idle).
- `tests/test_run_sh.py`: `id` added to the isolated-PATH core tools
  (run.sh resolves `--user` on every path incl. the dry-run seam) + a new
  test pinning the `user=--user=UID:GID` report.

## 2026-09-29 — Three caption families with drift evolution (SFX slice 1)

- User doctrine: the director generates all three caption families from
  the style charter + evolving general prompt — video captions
  (motion/scenes/visuals/objects/characters/shots/angles), music
  captions (instruments/harmony/melody), SFX captions (concrete audio
  descriptions of objects/environments/creatures) — and every family
  must evolve as the general prompt slowly drifts each segment.
- Schema (`voyage/models.py`): `DirectorAudioPlan` gains `sfx_caption`
  (default `""`, so pre-slice transition.json files validate unchanged);
  `DirectorVideoPlan` keeps `stages` as the video caption family, now
  documented as carrying concrete visual detail. The slow-loop music
  planner keys repaints on `music_caption` only — SFX drift never
  triggers a music take render.
- Drift mechanism (`voyage/director.py` + `voyage/supervisor.py`): new
  pure `format_previous_captions` renders the prior segment's video
  stages + music/SFX captions as a PREVIOUS CAPTIONS block; the system
  prompt + response-shape text now require all three families to
  continue from it with a slow drift, never jump or restart. The
  supervisor feeds it from two sources: `_decide_payload` loads the
  previous committed `transition.json` best-effort (missing/torn/legacy
  → `""`, never breaks a commit) via `previous_transition_captions`,
  and the director-prefetch path formats the just-accepted decision
  directly (its speculative state has no segment number). The Qwen
  worker forwards `previous_captions` (plus the previously-dropped
  `measured_context`) into the user message. `transition.json` already
  persists the full decision, so caption history is automatic.
- Deterministic fallback: captions derive from concept + charter
  (`{charter}: {concept} in continuous gentle motion…` /
  `slow ambient electronic composition for {concept}…` /
  `quiet concrete sounds of {concept}…`) — drifted concepts yield
  drifted captions, held concepts yield bit-stable captions.
- Console: `_segment_plan_info` surfaces `audio_sfx_caption` alongside
  the music caption so the drift is visible per segment.
- Proof: `tests/test_three_captions.py` (9 TDD tests, watched fail on
  missing symbols/params, then green) + `tests/test_phase3.py` payload
  key update; full `scripts/gates.sh` green (ruff + format + mypy
  strict + 867 pytest, 1 deselected). Next: SFX worker + VRAM ladder
  (slice 2), finalize windowing/sharding/mix (slice 3).

## 2026-09-29 — SFX worker + 2060 VRAM ladder (SFX slice 2)

- Worker + fake behind a shared `generate_sfx` contract
  (`voyage/workers/sfx_mmaudio.py` real, `voyage/workers/sfx.py` fake
  seeded pink noise, `FakeSfxBackend` in `fake_backends.py`):
  `{window_id, caption, video_path, start_seconds, duration_seconds,
  seed, output_path, sample_rate, channels}` → WAV + `{artifacts,
  sfx}`. Validation before side effects, benchmark count guard,
  uniform evict — same shape as the ACE-Step pair. The real worker
  extracts CLIP (8 fps @ 384 px) + sync (25 fps @ 224 px) frames via
  two ffmpeg passes and renders native 44.1 kHz FLAC → requested WAV.
- Compat (`voyage/audio/mmaudio_sfx.py`, lazy imports, §12 clean):
  upstream hkchengrex/MMAudio @ `974010a0` with native .pth weights
  (no comfy-loader machinery), fp16 resident, euler 25 steps, cfg 4.5.
  Two pinned-constructor hub hardcodes are redirected at the registry
  files (restored in `finally`): the nvidia 44 kHz vocoder
  (`from_pretrained` accepts the local snapshot dir) and the DFN5B
  CLIP tower (open_clip `create_model` with the pinned .bin under the
  builtin `ViT-H-14-378-quickgelu` arch entry). Caught live: the
  pinned API differs from the newer vendored ComfyUI copy
  (`AutoEncoderModule(vae_ckpt_path=…)`, `FeaturesUtils(tod_vae_ckpt=,
  synchformer_ckpt=, mode=)` — the vendored copy takes state dicts).
- Config + registry + CLI: `[sfx]` section (`SfxConfig`: backend
  fake|mmaudio, device, models_dir, model_size; default fake keeps old
  runs byte-identical), `sfx-mmaudio` MODEL_SPECS row (3 variants +
  VAE/synchformer files, nvidia vocoder + DFN5B CLIP snapshots,
  CC-BY-NC-4.0 recorded like CausVid), `models download/verify
  sfx-mmaudio` (~13 GB), `docs/MODELS.md` mirror. `test_longlive`
  layout-keys test extended with `sfx_dir`.
- Ladder (idle GPUs, 8 s benchmark windows): small_44k fits the 2060
  (4.6 GiB peak, ~6.0 s/window, 1.4 GB headroom); medium_44k OOMs it
  (5.19 GiB PyTorch vs 5.6 usable); large_44k_v2 fits the 4060
  (6.2 GiB peak, ~5.9 s/window). Locked: large on the 4060 primary
  (user's quality-first directive); dual-shard (slice 3) runs small on
  both GPUs for quality consistency across windows. Dockerfile.video
  carries `/opt/mmaudio` (`--no-deps` + inference-only leaves, so the
  transformers 4.57.6 pin still wins) + PYTHONPATH.
- Proof: `tests/test_sfx_contract.py` (9 TDD tests, watched fail, then
  green incl. fake byte-determinism); `models verify sfx-mmaudio` OK
  live; ladder numbers above from live runs. Full gates: pytest 872
  passed + 4 load-flaky TUI Pilot failures (all pass isolated —
  known shared-box flakiness, re-run idle) + 10 ruff errors all in
  `voyage/workers/video_common.py` (concurrent agent's in-flight file,
  untouched per §9). Next: finalize windowing/sharding/mix (slice 3).

## 2026-09-29 — Finalize-time SFX pass (SFX slice 3)

- Post-pass design (zero-touch to `finalize_run`): after the
  music-only final publishes, `finalize_sfx_pass` (`voyage/
  sfx_finalize.py`) renders the SFX bed conditioned on the shipped
  pixels, mixes, and muxes video-copy + mixed audio back over the same
  path (atomic replace — a failed pass never strands a half-written
  final; disabled backends leave music bytes untouched). Windows tile
  [0, timeline) at 8 s / 1 s overlap; junction windows join both
  adjacent segments' `sfx_caption`s (the incoherence the user flagged);
  stub tails < 1.0 s merge into their predecessor; seeds derive
  deterministically so re-finalize re-renders identical bytes and
  ledger-matching stems are reused. Bed joins via manual-fade
  `_blend_pair`s (never acrossfade); amix lays it at -6 dB
  (normalize=0, no auto-gain); every stage verifies duration against
  the 0.6 s A/V tolerance. Stems persist under `audio/sfx/` +
  fsynced `sfx.jsonl` (takes-philosophy); `validate_run` extends
  read-only (missing ledger = clean for old runs).
- Live incident (proof run, drove the fix): `clip_f [1,17] vs
  _clip_seq_len=16` on the first non-8 s window. Root cause: the
  duration-derived `SequenceConfig` asserts EXACT counts (synchformer
  emits S segments × 8 — temporal stride 2 — CLIP passes frames
  through), so the fixed pad-to-17 floor corrupted every non-8 s
  window. Fix mirrors the canonical `load_video`: exact counts
  (int(rate × duration)), truncate-to-reality with a 0.05 s fail-loud
  guard, no padding. Proved by the retry: w0001 [7, 9.04) rendered
  clean (16 clip / 51 sync → lens 16/40/88).
- Proof (idle GPUs, `sfxproof` 2-seg ltxv + deterministic director):
  `final.mp4` 9.042 s h264 + AAC, SFX stem mean -12.3 dB / max
  -2.5 dB (genuine effects, not silence), mix mean -21.9 dB,
  `validate` VALID incl. the new sfx checks. CLI: `--no-sfx`,
  `--sfx-device`, `--sfx-model-size`, `--sfx-workers 1|2` (dual
  forces small/small on cuda:0+cuda:1 for joint consistency; cuda:1
  single with larger models fails fast per the ladder).
- Proof: `tests/test_sfx_finalize.py` (12 TDD tests incl. a fake
  end-to-end over junctions — worker spawn, stems, bed, mix, ledger,
  validate — all in slim gates); my scope ruff + format + mypy clean.
  Full tree: 903 passed + 1 failed in another agent's untracked
  `test_generate_blocks_request.py` (message mismatch inside their
  in-flight `video_common.py` — same file holding the 10 ruff errors
  from slice 2; untouched per §9).

## 2026-09-29 — Director backend qwen by default (poulah freeze fix)

- User report on the poulah run (31 segments): the general prompt never
  evolved — the stored `voyage.toml` carried `backend = "deterministic"`,
  so the director degraded to the hold path on every segment and the
  phase cycle (ESTABLISH/DRIFT/TRANSFORM/…) spun with frozen content.
  User directive: qwen is the default everywhere; deterministic is an
  explicit opt-out only.
- Change (`voyage/config.py`, `voyage/cli.py`,
  `voyage/workers/director.py`): `DirectorConfig.backend` defaults to
  `"qwen"`; `default_config_toml` writes `backend = "qwen"` and takes an
  optional `director_backend` override; `init` gains `--director
  (qwen|deterministic)` defaulting to qwen and `cmd_init` forwards it;
  `generate` already defaulted `--director` to qwen and now forwards it
  into the stored config via `init_args`; `run --director` stays an
  in-memory override (None = respect the stored file); the director
  worker `_CONFIG` and `handle_decide` fallback default to qwen; the
  benchmark probe keeps its explicit deterministic payload (it measures
  that path on purpose). The supervisor drift-hold still uses the local
  `DeterministicDirector` by design (miss deadline → hold, never block).
- Offline behavior unchanged: without cached Qwen weights the worker
  fails fast (offline-first) to the deterministic fallback, so
  file-backed scaffold runs still commit.
- Proof: new `tests/test_director_default.py` (7 tests: config default,
  toml default + explicit deterministic opt-out, worker default, init /
  generate parser defaults, run leaves the file alone — watched fail on
  the missing `director_backend` param, then green); updated
  `test_generation_stack.py` (stored config now asserts qwen) and
  `test_integration.py` (decide probe passes explicit deterministic);
  live `run.sh init` verified default qwen + `--director deterministic`
  opt-out; full `scripts/gates.sh` green.

- Causvid `/opt` permission fix 2026-09-29 (user report: `WORKER_ERROR:
  [Errno 13] Permission denied: '/opt/causvid/wan_models'` on a causvid
  run): the repo clones are root-owned while workers run as the
  host-mapped `voyager` user, and `_enter_causvid_tree` /
  `_enter_longlive_tree` unconditionally unlinked + recreated the
  `wan_models` symlink (a write) on every call. Fix:
  `voyage/workers/video_causvid.py` + `video_longlive.py` skip the
  unlink/recreate when the link already points at the wanted target
  (zero writes, just chdir); `worker/Dockerfile.video` pre-creates the
  default-`/models` links at build (as root), chowns all `/opt` trees to
  `${UID}:${GID}`, and chmods the two symlink-parent dirs 777 (stale-image
  UID-mismatch insurance). A one-off `chown` cannot fix this class of bug
  (ephemeral `--rm` containers + `--user` pin — only the image persists).
- Proof: new `tests/test_enter_repo_trees.py` (4 tests: correct-link +
  read-only parent succeeds for both shims — watched fail with the exact
  reported `PermissionError` — plus stale-link repoint; green); rebuilt
  `voyage-video` (smoke ok) and probed both shims as the host-mapped user
  against `/models` (zero-write path, links resolve to the `/models`
  defaults). Full gates: 901 passed, 1 pre-existing unrelated failure
  (`test_generate_blocks_request.py::test_empty_prompts_rejected` —
  scene_cuts/prompt message mismatch from concurrent in-flight work, not
   this change).

- Backend excerpts 2026-09-29 (1-min silent slow-morph, deterministic
  director, drift 4, blocks=1, fake audio; user 30-min cap per excerpt):
  LTXV 15x96f committed in ~8+6 min, VALID 1490f, finalized
  `output/excerpt-ltxv-slow1_final.mp4` (62s, 768x512 h264 + AAC, 4.7 MiB;
  steady ~15.5 s/segment; seg10 committed 121f fresh after a tail-less
  resume — top-up boundary, validate-clean); CausVid 14x72f committed,
  VALID 1008f, finalized `output/excerpt-causvid-probe_final.mp4` (63s,
  832x480 lifted 16->24fps, 10.3 MiB) with a 4-segments-per-worker CUDA
  OOM pattern (fresh worker ~47 s/seg, degrading to ~106 s, then OOM +
  circuit breaker; 3 resumes + one pkill-9 needed — contention vs leak
  inconclusive per the GPU-contention rule). LongLive2 excerpt DROPPED
  per the user's fall-back: worker dies at `loading generator
  checkpoint ...` with RC=137; kernel OOM record shows
  `anon-rss:39629176kB` (~37.8 GB for the 10 GB `model_bf16.pt`, ~3.8x)
  — filed as `issues/096_longlive_checkpoint_load_spike.md`
  (candidates: `torch.load(mmap=True)`; docs note until fixed).
  Workarounds used (superseded same day by the in-image /opt fix above,
  kept here for stale-image runs): direct `docker run` mirroring run.sh
  + `-e HF_HUB_CACHE=/tmp/voyage-hf-cache -e HF_HOME=/tmp/voyage-hf-home`
  (run.sh `--user` + no HOME breaks HF downloads), `-v
  /tmp/causvid-anchor:/opt/causvid/wan_models`,
  `-e VOYAGE_LONGLIVE_DIR=/tmp/ll-tree` (host symlinks into /models).

## 2026-09-29 — `generate` ensures required models inline (selective + parallel)

- User intent: `voyage generate` must download/verify all models (and
  other external deps) it needs as part of the command, with polished
  console progress — but only the stacks the specific invocation needs
  (ltxv downloads its own models, never the other backends'), and
  parallelized where possible.
- New `voyage/models_ensure.py`: `required_specs(config, sfx_enabled)`
  maps the effective config to registry specs (video backend's own:
  longlive2→`longlive2-bf16`, ltxv→`ltxv-2b`, causvid→`causvid`, never
  the other two; `audio-acestep` when paired; `director-qwen8b` unless
  deterministic; `sfx-mmaudio` only when the finalize pass will run;
  `inspector-qwen35` only when enabled; fake video → empty set, so fake
  runs stay weight-free/offline and the qwen→deterministic fallback is
  preserved). `ensure_models` verify-first, then downloads the missing
  specs on a `ThreadPoolExecutor(min(missing, 4))` with per-model
  `parallel_downloads` progress, re-verifies, fail-fast 1 with the
  verify/download message per model. `--no-download` skips fetching and
  fails on anything missing. Manifest race note: parallel
  `download_model` calls can drop each other's `manifest.json` entries
  (read-modify-write), so a locked `_repair_manifest` re-merges missing
  keys afterwards (best-effort; verify stays authoritative).
- `VoyageConsole.parallel_downloads(labels)` (console.py): rich TTY gets
  one spinner + elapsed timer per model; plain `▸ downloading …` /
  `✓ <label> ready (Xs)` / `✗ <label> failed` lines otherwise (same
  words, TUI-capturable). Tracker is main-thread-only (reported from
  the `as_completed` loop — rich is not thread-safe).
- `cmd_generate` wiring (after effective config, before the banner):
  `check_ffmpeg` gate, `check_free_space` on the run dir + each model
  stack mount (nearest existing ancestor — the mount may not exist
  until downloads create it), then ensure with
  `sfx_enabled = not no_sfx and sfx.backend != "fake"` (mirrors the
  finalize gate). TUI path unchanged: ensure runs under stdout
  redirection into the run log, no `SegmentProgress` change.
- Known pre-existing gap (out of scope, documented): director/inspector
  workers resolve via hub `model_id` + HF cache, not `models_dir` — the
  inline download still warms the cache in the same container, but a
  pre-provisioned `/models` with a cold HF cache still falls back to
  deterministic/skip.
- Proof: `tests/test_generate_ensure.py` (15 TDD tests: mapping incl.
  fake-empty, verify-hit skips download, missing-only download,
  `--no-download`, download-failure nonzero, plain-fallback words,
  ffmpeg/disk/ensure-fail gates, selective-scope spy on a real fake
  e2e); ruff + format + mypy green; full suite 906 passed + 13 failed,
  all 13 in other agents' in-flight areas (fake 48f→96f segment-math
  change, backend-adapter/config/finalize/scoreboard — none touch the
  ensure path; verified by failure signatures + `git diff` scope).

## 2026-09-29 — ltxv wins the backend dilemma: all defaults to ltxv, other models deleted

- User decision: ltxv has by far the best tradeoff between generation
  speed, consistency, quality and memory usage. Other backends stay in
  the codebase (code + tests + docs), but their models are deleted and
  every default option now selects ltxv.
- Code (`voyage/config.py`): `BackendRecord` gained `segment_frames`
  (fake 48 / longlive2 29 / ltxv 96 / causvid 72 — the natural novel
  counts previously hardcoded in `cli._frames_per_segment`), so presets,
  TOML and `VideoConfig` derive it instead of leaking the fake row's 48
  (or ltxv's 96 into fake TOML — `_VIDEO_BACKEND_PRESETS` was missing
  the key, one-line fix). `_FAKE_ROW` renamed `_DEFAULT_ROW` =
  `BACKEND_REGISTRY["ltxv"]`; `VideoConfig` defaults are now the ltxv
  row (backend/profile/geometry/fps/segment_frames/device/latent_shape);
  `default_config_toml` defaults `video_backend` to ltxv, writes
  `segment_frames` into the body, `[video]` comment reordered ltxv-first.
  `voyage/cli.py`: `init --backend` default fake→ltxv, `cmd_init`
  fallback ltxv, `models` default target longlive2-bf16→ltxv-2b,
  benchmark end-to-end passes explicit `backend="fake"` (CPU benchmark
  must stay fake). `voyage/tui_state.py`: `_planning_frames_and_fps`
  passes `segment_frames` from the video preset (was leaking 96 for
  fake). `generate --backend` and the TUI backend list were already
  ltxv-first/default — no change.
- Tests: registry defaults→ltxv, init/models parser defaults, 4
  `test_generate` source-purity asserts (video→ltxv, audio→acestep),
  `conftest.initialize_run_directory` gained `video_backend="fake"`
  (covers all CPU fake-pipeline callers, zero call-site edits), fake-guard
  tests pinned `video_backend="fake"`, adapter-contract payload frames
  48→96 (old value was the fake-row leak), `_video_config` row-aware.
  Note: a concurrent agent reverted one `--backend fake` re-pin
  mid-session; re-applied — re-read before editing shared test files.
- Models deleted from `~/.cache/voyage-models` (~70 GB, 915G disk
  78%→70%): `causvid` 11G + `longlive2` 9.4G + `Wan2.1-T2V-1.3B` 17G +
  `wan_models` 32G (verified no live GPU users — only CPU gates/test
  containers running). Kept: `ltxv-2b`, `PixArt-XL-2-1024-MS` 18G (LTXV
  TE), `acestep`, `Qwen3-8B` (default director backend), `Qwen3.5-9B`,
  `all-MiniLM-L6-v2`, `mmaudio` (SFX agent in-flight). `manifest.json`
  kept untouched — entries are sha bookkeeping for re-download verify,
  not presence claims. README `generate` example comment updated
  (ltxv default 768×512@24; longlive2/causvid need downloads first).
- Gates at switch time: own scope ruff + format + mypy (51 files) clean,
  pytest 895 passed; full-tree ruff red only in the concurrent agent's
  untracked `voyage/models_ensure.py`, 2 test failures in their in-flight
  files (`test_empty_prompts_rejected` ordering, rewritten
  `test_init_accepts_absolute_output` + stray `Voyage/rel-run/`).

## 2026-09-29 — ltxv-default follow-up: fake-plan test fix + TUI×SFX collision filed

- `tests/test_tui_state.py::test_plan_counts_match_cli_truth_all_backends_and_blocks`
  failed post-switch: the test built `VideoConfig(backend="fake")` without
  `segment_frames`, inheriting the new ltxv default (96) while TUI planning
  (correctly) uses the fake preset (48). Production never produces that
  config (`resolve_config` always carries preset `segment_frames`), so the
  test now pins `segment_frames` from the preset like production does.
- Full gates at follow-up: ruff + format + mypy green; pytest 918 passed,
  5 failed — 1 is the known foreign `test_init_accepts_absolute_output`,
  1 the known foreign `test_empty_prompts_rejected`, 2 are TUI e2e
  (`test_generate_end_to_end_fake_backend`,
  `test_generate_button_runs_real_fake_backend_to_completion`) broken by a
  collision between two other in-flight passes: SFX wiring added direct
  `args.sfx_backend` reads in `cmd_generate:1250` while the issue-045
  `to_generate_namespace` rewrite emits no `sfx_*` attrs (CLI `generate`
  is unaffected — verified exit 0 with/without `--no-sfx`). Filed as
  `Voyage/issues/097_tui_generate_missing_sfx_namespace_attrs.md`
  (repro + fix candidates, owner = SFX finalize wiring). The 5th
  (`test_worker_failure_restores_form_with_error`) passes in isolation —
  Pilot flake under load. Own-scope files (config/cli/tui_state +
  registry/cli-split/generate/adapter/hardening/state tests): 197 passed,
  only the foreign absolute-output failure.

## 2026-09-29 — issue 097 resolved: TUI generate × SFX namespace collision fixed

- Root cause confirmed live: `to_generate_namespace` emitted `no_sfx` /
  `sfx_device` / `sfx_model_size` / `sfx_workers` but not `sfx_backend` /
  `sfx_caption`, while `cmd_generate`'s finalize block read all six plus
  `skip_bad` via direct `args.*` access — TUI runs died with
  `AttributeError: 'Namespace' object has no attribute 'sfx_backend'`
  after segments committed.
- Fix (option 1 from the issue file, plus completing the partial TUI
  namespace): `to_generate_namespace` now emits `sfx_backend=None` /
  `sfx_caption=None` (parser defaults, single default source kept), and
  `cmd_generate`'s finalize block + `skip_bad` / `final_video` reads use
  `getattr` with parser-matching defaults; `cmd_finalize`'s
  `sfx_device` / `sfx_model_size` / `sfx_workers` reads hardened the same
  way. Regression test
  `tests/test_tui_state.py::test_generate_namespace_carries_finalize_sfx_attrs`.
- Verified in-container: namespace probe shows all six attrs present;
  `test_tui_state` 57 passed, cli-split/generate/registry/adapter 75
  passed, mypy clean (47 files), and the previously failing
  `test_tui.py::test_generate_end_to_end_fake_backend` passes. Full suite:
  923 passed, 3 failed — all in `tests/test_tui_app.py`, all Pilot
  viewport flakes (`OutOfBounds` on click/scroll, or passing in
  isolation), none touching the SFX path. (The `Voyage/issues/097_*`
  investigation file this section was drafted against has since been
  archived — it lives on in git history only; see AGENTS.md §11.)

## 2026-09-29 — SFX/music on-by-default + explicit caption pins (SFX slice 4)

- User directives: (1) "Ensure that SFX/music generation is enabled by
  default for all relevant cli commands (especially the 'generate'
  verb) and it must use the GPU by default as well." (2) "SFX captions,
  music captions and video captions may be provided as explicit
  arguments to the relevant CLI commands but when using the director,
  they must be driven by it. The goal is to have those captions evolve
  as the general/styling prompt evolves."
- Defaults-on: `BackendRecord` gains `sfx_backend`/`sfx_device`
  (CUDA rows → mmaudio/cuda:0, fake → fake/cpu); `default_config_toml`
  and the `resolve_config` backend-switch pair SFX like audio, so
  `generate --backend ltxv` (the default) renders ACE-Step music AND
  MMAudio SFX on cuda:0 with no flags, while fake stays CPU-only
  (gates never touch weights) and old runs without `[sfx]` keep
  byte-identical behavior via `SfxConfig` fake defaults. `ensure_models`
  already gates the sfx stack (no change needed).
- Explicit pins (in-memory only, never written to voyage.toml):
  `AudioConfig.music_caption` / `VideoConfig.video_caption` via
  `resolve_config` (+ wrapper), flags `--music-caption` /
  `--video-caption` on run+generate (shared `_add_generation_overrides`
  helper); `--sfx-caption` already existed on the finalize family.
  Precedence helpers `effective_music_caption` /
  `effective_video_stages` (supervisor, pure, tested): explicit pin,
  else director's evolving caption/stages, else charter fallback.
  The video pin applies AFTER the accept transaction (novelty +
  destination stay director-driven) but is still style-checked —
  charter-violating pins fail the commit loudly. The decision record
  keeps director stages (drift chain stays director-pure);
  prompt_plan.json keeps what rendered. TUI namespaces untouched
  (getattr-defensive reads; TUI caption fields are a follow-up).
- Poulah SFX attempt: first pass died on the last window
  (w0017 [119, 125.29): source yielded 6.20s for 6.29s — EOF edge
  rounding under the `-frames:v` cap). Fix: truncate-and-continue
  when the shortfall is < 0.6s (ledger records resolved reality),
  fail loud beyond it; plus a sub-1.0s stub-tail merge rule in the
  planner. Retry then failed at startup: `output/poulah/` (31
  segments + final.mp4, gitignored scratch) vanished from disk
  mid-session — likely another agent's scratch cleanup; recovery
  options with the user.
- Gates: own scope ruff + format + mypy clean; 51 SFX/config/registry
  tests green. Full tree 919 passed + 9 failed, all outside this
  slice: 5 finalize-geometry failures traceback into the concurrent
  in-flight `media.py` refactor (e.g. 1280x720 != 768x432 from
  validate_video — untouched by this change), 4 generate_ensure
  failures pass in isolation/grouped reruns (their file + models_ensure
  are concurrent-modified). Poulah recovery still open (see below).

## 2026-09-29 — Director snapshots resolve to the single /models copy (volume-deleted safe)

- User intent: the Qwen gap — `generate` ensured `director-qwen8b` into
  the volume, but the director worker loaded by hub id from the
  ephemeral HF cache only, so a deleted volume silently degraded to
  deterministic/skip. Now every snapshot resolves from /models, and a
  missing one is fetched on demand (no duplication, hub id stays the
  fallback).
- `model_registry`: new `SnapshotRef` (spec_name/repo_id/revision/
  relative_dir) + `resolve_snapshot(repo_id)` (first spec wins across
  ALL snapshot rows — any known repo maps, FileSpec-only rows excluded
  since they have no loadable directory) + `snapshot_present` (runs only
  the checks under the snapshot's own subdir, via an optional `checks`
  param on `_collect_missing`, so a partial volume still serves whatever
  is complete).
- `workers/director`: new `_models_dir()` (init payload wins, then
  `VOYAGE_MODELS_DIR`, then `/models`) + `_resolve_model_source`
  (local-dir passthrough, unknown-id passthrough, known repo →
  `<models_dir>/<subdir>`, absent snapshot → `download_model` into the
  volume with `HF_HUB_OFFLINE` lifted for that fetch only and restored
  after, still-incomplete → raise into the caller's fallback chain).
  Wired into `_load_qwen`/`_load_embedder`/`_load_inspector` AFTER the
  `_require_module` guards (slim stays fail-fast, never downloads without
  the stack); id-change cache keys unchanged. `handle_init` records
  `models_dir` (new `INIT_STR_KEYS` tuple, str-checked like the ids).
- `supervisor`: director worker now gets
  `init_payload={"models_dir": config.video.models_dir}` (the same mount
  `generate` ensures the director/inspector specs under).
- Proof: `tests/test_director_models_dir.py` (19 TDD tests: mapping incl.
  any-known-repo, sparse-file presence, passthrough, download-once +
  env-restore, init recording, from_pretrained-source plumbing via stub
  modules, supervisor payload); ruff + format + mypy green; full suite
  1062 passed + 6 failed, all foreign (5 Track-B finalize-geometry, e.g.
  1280x720 != 768x432, + 1 TUI tick flake — none touch this path).
- Reviewed + re-gated (ruff/format/mypy green, 1065 passed + 3 foreign
  finalize-geometry failures triaged); committed with explicit approval.

## 2026-09-30 — Finalize-time augmentation floors: min-fps 32 + min-resolution 1280x720 (Real-ESRGAN + FILM)

- User intent: every shipped video is >=32fps and >=HD by default, via a
  model-free floor today with the model-augmentation path staged. Knobs:
  `--min-fps` / `--min-resolution` (`0` disables) on `finalize`/`generate`/
  `run`, `[augment]` TOML section, TUI fields, `AugmentConfig`
  (`min_fps=32`, `min_width=1280`, `min_height=720`).
- `media.py`: `plan_augmentation` (pure: `max(requested, min, 24)` fps +
  per-axis `max`, 0/None disables, minterpolate only on lift) gates the
  stream-copy fast path off; vf is minterpolate-when-lifting +
  scale/pad/setsar/fps. 24fps sources re-encode 2x + fps-decimate to 32.
- `model_registry`: `film` (Comfy-Org `film_net_fp16`, 66M) +
  `realesrgan-anime` (x4plus-anime-6B, 18M) specs; `models_ensure`
  `augment_enabled` (CUDA backends include both, fake stays empty);
  `Dockerfile.video` gains leaf-only safetensors+Pillow.
- New `voyage/augment.py` (chunked orchestration, 2-GPU
  video-aug-cuda:0/SFX-cuda:1 pairing) + `workers/augment_worker.py`
  (lazy-torch vendored RRDBNet + FilmNetMini stand-in; full FILM weight
  port is follow-up — official `film_net` weights will NOT load).
- Proof: 6 new test modules (138 tests); the 5 legacy
  finalize-geometry failures other agents triaged as foreign were this
  change's default shift (768x432@24 -> 1280x720@32) — fixed in place
  (fastpath/stack tests pin legacy path with min_*=0, integration pins
  new defaults); full gates 1067 passed + 1 TUI Pilot flake (passes in
  isolation).
- Batch 5 augment notes (2026-09-30, issues 046/047/074): chunk decode uses input `-ss` fast-seek with exact-fallback + fresh `dest_dir` enforcement (046); resident `_RRDB_CACHE`/`_FILM_CACHE` keyed `(weights, device)` + `evict_augment_models` + `load_ms`/`infer_ms` split (047); `.safetensors` decodes via safetensors, `.pth` via `torch.load(weights_only=True)`, size+manifest pre-checks torch-free (074).

## 2026-09-30 — GPU director: Qwen3-4B-AWQ on cuda:1, unified image, eager attention

- User intent: the director must never run on CPU unless explicitly asked
  (a 1-minute `generate` froze 31 segments on 500 s CPU decides). Decision
  after Q&A + live probes: primary Qwen3-4B-AWQ on the idle RTX 2060
  (`cuda:1`), CPU opt-out via `--director-device cpu`, all workers in one
  image.
- Live verdicts that shaped the design: official Qwen3-8B-AWQ OOMs at
  materialization on 6 GB (5.49 GiB > 5.6 GB capacity — rejected, staging
  deleted); Qwen3-4B-AWQ (rev `74d4bd2b…406a3`) loads in 2.5 s / 2.67 GiB
  and generates valid JSON first-attempt at temp 0.7. Qwen's own quant
  table (8B-AWQ-4bit MMLU 73.8 > 4B-FP16 73.0) justified smaller-AWQ over
  larger-fp8.
- `Dockerfile.video` gains `/opt/venvs/director` (CUDA torch 2.11 cu128 +
  transformers 5.17 + GPTQModel 7.5.0; transformers loads AWQ via the
  gptqmodel backend, autoawq deprecated and must NOT be installed);
  `ENV VOYAGE_DIRECTOR_PYTHON` selects it. `worker/Dockerfile.director` +
  `scripts/build-director.sh` retired (refs updated in README/Dockerfile/
  docs/INSTALL.md/docs/BACKENDS.md). `run.sh` unchanged — ltxv already
  selects `voyage-video:latest` with `--gpus all`.
- `DirectorConfig.device` (default `cuda:1`) + `--director-device` on
  init/run/generate + `[director] device` TOML key + manifest record;
  `rpc.SubprocessWorker` gains an `executable` param (supervisor spawns
  the director with `VOYAGE_DIRECTOR_PYTHON`); device flows through the
  decide payload; absent CUDA falls back to CPU loudly, OOM never falls
  back. `models_ensure` picks `director-qwen4b-awq` (registry pin +
  download/verify) unless device is cpu.
- Two live-caught bugs, both fixed: (1) transformers 5.17 defaults to
  flash-attention, whose kernels need Ampere+ — every decide on Turing
  sm_75 fell back with "FlashAttention only supports Ampere GPUs or
  newer". Fix: `attn_implementation="eager"` on the CUDA load path
  (verified: `fallback=false` in 76.8 s on cuda:1 vs 497.4 s CPU).
  (2) The substitution cache key used the raw id while the stored id
  was substituted — every retry reloaded all 902 tensors. Fix: pure
  `_effective_qwen_id` helper normalizes before check and store.
- Proof: `tests/test_director_device.py` (10 tests incl. the
  substitution-cache regression) + `doctor.probe` director-venv fact +
  2 `test_doctor.py` tests; full gates 1089 passed + 1 TUI Pilot flake
  (passes in isolation — shared-box load, known class).

- Batch 3 Rank-1 resolutions (2026-09-30, issues
  003/006/020/022/043/044/045/056/067/071/073; 070 procedure-only):
  `worker/Dockerfile.video` pip rows frozen to the 2026-09-30 live-image
  freeze (067; rebuild + `--require-hashes` still open); LTXV TE resolves
  via `_resolve_te_source` offline-first (`local_files_only` + pinned
  revision, 073 — needs a `HF_HUB_OFFLINE=1` GPU-box probe);
  `WAN_HF_REVISION` wire + record + pin procedure landed, value still
  `None` (070 — pin from a provisioned box, never the API alone);
  registry ingest/ensure/load hash gates + fail-closed manifests (071,
  CausVid DMD baseline still open); `atomic_copy` publish, streamed
  frame sampling, single-pass SFX conditioning (043/044/045 —
  `sfx_finalize.py:563` read_bytes left as noted follow-up, torch
  execution needs a GPU box); mid-run worker-log rotation (056);
  shared TOML escaper, caption-pin forwarding, validate-side AV budget
  via `av_drift_seconds` (020/022/003).
- Batch 5 (2026-09-30): resolved 059/051/062(+027 fold)/063(+028 fold)/017/018/019/102/103/046/047/048/153/156/158/065(+146 fold)/144/139/178/025/026; real 024 (unbounded --output paths) stays OPEN (concurrent owner).
- Batch 6 (2026-09-30): resolved 049/050/053/054/057/058/060/061/064/066/069/075/076/077/101.

## 2026-09-30 — Stage A telemetry: all runtime blind spots instrumented (additive-only)

- User intent: "investigate all telemetry blind spots to be sure that we
  capture the full performance/runtime story" before the speedup stages;
  music coherence "changes unexpectedly" — Stage B's repaint gate should
  help there too. Blind-spot map came from three parallel read-only
  survey subagents (file:line evidence, verified against live code):
  commit→propose gap contents (gauges + rotate + control reads + lock
  acquire + commit-head precheck + per-segment ConceptStore re-read, no
  `time.sleep` anywhere in `voyage/*.py`); accept-chain rejections emit
  no metric (only `drift_hold`); zero token reporting (`_qwen_generate`
  returns str only); LTXV/causvid `generate_blocks` untimed (only
  longlive has `_CudaStageTimer`); audio swap + slice/assemble as one
  coarse `audio` stage; prefetch submit/consume untimed; finalize SFX +
  augment untimed.
- Music-coherence root causes (planner.py): repaint fires on EXACT
  `music_caption` string inequality (one-char LLM rewording repaints the
  whole unconsumed tail); chained takes use fresh derived seeds +
  text2music with no previous-audio conditioning (only a loose BPM hint);
  no music loudness matching (only SFX −6 dB); fallbacks degrade to hard
  cuts. Joining itself is crossfades (commit 2.0 s, finalize ≤0.5 s).
- What landed (all additive, `stages` key set pinned by
  test_stage_timings.py): token counts from live tensor widths through
  `_qwen_decide` into `ProposedSegment.director_tokens` into
  `segment_committed`; `director_rejection` per §74 continue path +
  `director_prefetch_rejected` + `director_fallback`; 6-bucket
  `gap_breakdown` ledger emitted+reset per propose; `prefetch_age_ms`;
  swap-only `audio_swap_breakdown`; `audio_assemble`; LTXV `stage_ms`
  merged at `_render_video` from the raw worker result (adapter strips
  extras by design — never widen the adapter for telemetry).
- Proof: `tests/test_stage_a_telemetry.py` (11 tests, TDD red-first) +
  `tests/test_ltxv_stage_ms.py` (5 tests, parallel subagent with
  exclusive file scope); `tests/test_commit_split.py` canned proposal
  gained the new `director_tokens` field. Gates: ruff + format + mypy
  strict clean; 1404 passed, 5 skipped, 2 foreign failures (concurrent
  agent's qualify.sh lib-path breakage; known TUI Pilot load flake that
  passes in isolation). Live review notes: `B023` — parameterize
  loop-varying values into closure signatures instead of capturing.
- Next: Stage B (caption-similarity repaint gate + prefetch acceptance),
  Stage C (AWQ kernels + gauge cadence); measurement = extend boba +1m
  (15 more segments on the same run dir).
- Stage B repaint gate done 2026-09-30: boba ledger calibration (10 takes,
  consecutive caption Jaccard sims 0.719/0.444/0.750/0.727/1.000/1.000/
  0.809/0.907/1.000) proved every baseline repaint was a rewording
  repaint — takes 2-9 share near-identical captions yet repainted
  repeatedly. `AudioPlanner.repaint_similarity_threshold = 0.5`
  (`AudioConfig` + TOML, validated [0,1] finite; no CLI flag per YAGNI):
  `plan()` repaints only when token-set similarity < threshold, else the
  new caption rides the next chained/keep take joint (source-anchored
  repaint preserved for genuine shifts — threshold 0.5 suppresses all
  but the 0.444 shift). Proof: `tests/test_repaint_similarity_gate.py`
  (5 tests, TDD red-first; existing repaint tests survive —
  brighter-pulse vs ambient-drift sim = 0.0). Gates: ruff + format +
  mypy strict clean; 1428 passed, 6 skipped, 1 foreign failure
  (concurrent agent's qualify.sh lib-path breakage). Boba +1m extension
  runs WITHOUT the gate (no-gate baseline for before/after music
 comparison); gate takes effect on the segment after. Prefetch
 acceptance + Stage C still open.
- Novelty leniency + gauges skip-busy done 2026-09-30 (user-approved):
  boba extension post-mortem (FAILED seg 21, A/V drift 0.959s; Stage A
  metrics showed all 6 rejections novelty, incl. seg16 burning 474s of
  director time into a deterministic fallback). `novelty_max_rejections
  = 2` (`VoyageConfig` + non-negative validator + TOML, no CLI flag per
  YAGNI): at the cap the last generation is accepted with
  `novelty_accepted=False` + a `novelty_overridden` metric (style
  rejections still hard-fail to the deterministic fallback). Retry
  feedback now names the rejected concept, its score vs the threshold,
  and the last-8 visited worlds with a steer instruction. `_sample_gauges`
  skips the director probe while a prefetch is in flight (kills boba's
  5s/timeout + the one 23.7s block). Encode-variance probe verdict:
  expected behavior, not a bug (resident session = 1 T5 encode ~13s vs
  post-audio-swap fresh session = 2 encodes ~25-31s; block-count,
  prompt-cache and prefix-drop hypotheses ruled out). Proof:
  `tests/test_novelty_leniency.py` (5, TDD red-first) + skip-busy unit
  test in `test_stage_a_telemetry.py` (17/17 with the file). Gates: 43
  scoped green; full suite 1632 passed with 16 TUI/worker_perf
  load-flakes on a loaded box (none in scope; clean on re-run).
  Boba resume (9 segs + finalize) is GPU work, parked until the user
  approves per the GPU-prompt rule.
- Commit-time take-joint compensation done 2026-10-01 (boba seg21 root
  cause): the commit slice walk joined abutting slices with a bare
  crossfade, absorbing `fade` seconds per joint (out = sum - fade) — the
  preview ran 0.959 s short of a 5.04 s video and the 0.6 s A/V gate
  refused the commit twice with identical geometry (structurally
  unretryable). The walk now extends the tail slice by exactly the
  absorption — takes are continuous, so the extra content is real music
  — clamped to the probed take file, and passes the fade explicitly so
  assembly consumes exactly what was added (mirrors the issue-095
  finalize compensation; single-slice stream-copy path untouched).
  Proof: `tests/test_commit_slice_compensation.py` (2, TDD red-first —
  pre-fix drift 0.9587 reproduces boba's 0.959 to the millisecond, with
  a two-take-ids assertion so the straddle cannot pass vacuously).
  Gates: 59 green across the compensation + Stage A + 095/101/104 +
  leniency + planner files; ruff + format + mypy strict clean on all
  touched files. Boba resume retries seg21 with this fix on the GPU.
- Novelty steer-and-accept done 2026-10-01 (item 1: director slowness):
  novelty never rejects — the prompt steers (system prompt now demands a
  different setting/element/mood than every visited world; builder adds a
  NOVELTY STEERING section pointing at FORBIDDEN CONCEPT SUMMARY, gated
  by new `REVISITS_ALLOWED_SENTINEL` shared with the supervisor payload),
  every generation is scored (`novelty_scored` {score, threshold,
  embedded, accepted_novel}) and the first schema/style-valid generation
  renders with truthful `novelty_accepted`. Speed comes from
  single-serve accepts: no novelty retry loop burns extra ~120s+ LLM
  calls (boba seg16 burned 474s → fallback under the old regime).
  `novelty_max_rejections` kept deprecated-unused (validator + TOML line)
  so older run dirs still load; `novelty_overridden` event retired (no
  production consumers). Proof: new `tests/test_novelty_steer_accept.py`
  (7, TDD red-first); `test_novelty_leniency.py` deleted (superseded);
  `test_stage_a_telemetry.py` revisit test rewritten (score, no
  rejection); `test_feedback/three_captions/phase3` untouched (`in`
  assertions). Gates: ruff + format + mypy strict clean; full pytest
  1730 passed + 1 foreign flake passing alone (augment model-pass
  selection, shared-box class), 11 skipped; tree format gate blocked by
  foreign `tests/test_run_sh.py` (untouched per §9).
- Stage C done 2026-10-01 (AWQ kernels + gauge cadence): root cause of
  the 0.0 s JIT failures was PATH, not toolchain — the director venv
  ships a `ninja` binary (nvcc/gcc present) but workers spawn by
  absolute interpreter path with the system PATH, so torch cpp_extension
  never finds it and GPTQModel falls to `AwqGEMMTritonLinear` (~9
  tok/s). `rpc._spawn_env` prepends `dirname(executable)` to the child
  env (None = plain inherit, zero change for default workers); live
  proof through the real `SubprocessWorker` spawn on cuda:1: ExLlamaV2
  JIT compiled in 11 s, `selected -> AwqExllamaV2Linear`, 64 tokens in
  1.7 s = 36.7 tok/s (~4x). JIT cache stays HOME-local (12 s per
  container boot, once per run — no persistence wired, YAGNI). Gauges:
  `_director_probe_blocked` also skips when the prefetch future is
  done-but-None (60 s budget expired, worker still chewing — the boba
  5 s/tail case; only a dict result proves the worker free);
  `VoyageConfig.resource_gauge_interval_segments = 1` (positive
  validator + TOML, TOML-only per YAGNI) thins sampling on long runs;
  audio health gains `vram_free/total_gib` via never-raising
  `_cuda_mem_info_gib` (it answered keyless before, hiding the timeout
  attribution). Proof: `tests/test_stage_c.py` (11, TDD red-first);
  ruff + format + mypy strict clean on all 5 touched files; full pytest
  green except foreign in-flight media-split files (untouched per §9).

## Batch 7 (2026-09-30) — structure/toolchain/docs as-builts (ambiguous-header notes folded here per append-only rule)

- Workers seam (batch-7-2026-09-30): shared validators in `voyage.workers._validators` + torch guards/`BYTES_PER_GIB` in `voyage.workers._resident`; fake video serves `standard_serve_map` + `run_benchmark_harness`.
- Tests/process (batch-7-2026-09-30): `conftest` `short_texts`/`bounded_counts` unification + `VOYAGE_HYPOTHESIS_DATABASE=1` replay opt-in; `_init_run` ratchet test caps at 144 (`test_phase2` folded into `test_recovery`).
- Scripts (batch-7-2026-09-30): `scripts/lib/common.sh` shared cache-env/user-args/image helpers; `run.sh` TOML sniff covers video/audio/sfx with CUDA-preferring pick; `qualify.sh` takes `--backend`/`--segments`.
- Batch 7 (2026-09-30): resolved 079-partial/080/082-partial/083/084/085/031-doc/032/033/034/035-doc/036-tracker/037/038/039/040/041/088-proof/089-residual/090/091/092/093-partial/094-policy; real 024 stays OPEN (concurrent owner); 070 value-pin still needs a provisioned box; 086 needs voyage/ scope.

- Run-file pruning done 2026-09-30 (user Q&A: fewer files first, run
  stays resumable/validatable/re-finalizable; future runs only, boba
  untouched): per-segment 12 files → 5 (`DONE`, `video.mp4`,
  `audio.wav`, `recovery.pt`, `manifest.json`). Three parallel tracks:
  A (workers tail: `generate_blocks` no longer persists
  `video_tail.mp4`; `ensure_conditioning_tail` in
  `workers/video_common.py` derives the last 25 frames from sibling
  `video.mp4` at resume, causvid width `max(25, overlap)`, sha
  advisory; `tests/test_tail_derive.py` 22 tests), B (supervisor
  slices to `TemporaryDirectory`; 6 JSONs → `manifest.json`
  `format: 1` via new `voyage/segment_manifest.py` with legacy
  fallback for all readers; `tests/test_segment_manifest.py` 4
  tests), C (`cmd_init` no longer creates root `concepts.jsonl`;
  boba's 0-byte dup deleted after verifying the live store;
  `audio_acestep` worker chdir-redirects upstream `.cache` writes out
  of the run). `DONE` kept separate (round-3 reversal);
  `audio/take_*.wav` + sfx windows kept (primary sources).
  `tests/conftest.py` scaffold drops the root dup. Gates: ruff +
  format + mypy strict clean; 1483 passed, 5 skipped, 1 known TUI
  Pilot load flake (passes in isolation).

## Batch 8 (2026-09-30) — worker/audio/augment/TUI as-builts (ambiguous-header notes folded here per append-only rule)

- Worker init (batch-8-2026-09-30, issue 127): every worker `init` rejects unknown fields as `TypeError` (INVALID_PAYLOAD, fatal) naming the key and the known set — a mistyped model id fails at startup instead of booting defaults (landed in `sfx_mmaudio` + `audio_acestep`; the `director.py` leg stays with its owner).
- Augment worker (batch-8-2026-09-30, issue 157): OOM-split paths call `empty_cache` unconditionally (no-op without CUDA) — allocator relief never depends on device presence (preset knob + fan-out contract stay with the `augment.py` owner).
- Augment weights (batch-8-2026-09-30, issue 166): until the full upstream FILM port + SRVGG anime-6B loader land, `augment_enabled` stays opt-in-gated so default CUDA runs stop fetching weights the spike loaders structurally reject; the weight↔loader round-trip contract test pins the port's completion.
- Audio fast paths (batch-8-2026-09-30, issue 189): every audio fast path verifies its output — `slice_take` and the single-slice assembly copy reject empty outputs at creation, so a degenerate slice fails at slice time, never three call levels up.
- SFX bounds (batch-8-2026-09-30, issue 191): SFX segment bounds search the video stream (`codec_type == video`, the house idiom) and fail loud (`MediaError`) on zero-duration bounds — every later junction caption shifts otherwise (residual: `sfx_finalize.py` owner).
- Augment decode (batch-8-2026-09-30, issue 192): `ffmpeg_decode_chunk` fails loud on both stale-input shapes — a non-fresh dest dir raises before the spawn, and a post-decode frame-count mismatch raises after it.
- Augment device (batch-8-2026-09-30, issue 193): documented CPU fallbacks stay loud — `_resolve_device` emits one per-process stderr line naming the requested device and the CPU execution, so plan-vs-execution placement mismatches diagnose in one line.
- TUI audio floor (batch-8-2026-09-30, Group A issue 111): the form rejects `take_seconds` at or below the audio-ahead window (floor read from `AudioConfig`, no restated literal); blank stays valid.
- TUI namespace parity (batch-8-2026-09-30, Group A issues 145/182/147): `no_download` and `no_sfx` checkboxes ride `to_generate_namespace`, and `generate --verbose/--no-color` (plus the TUI sink) ride the finalize namespace — the TUI no longer silently drops what the CLI honors.

## 2026-09-30 — Issue 133 §140 staleness corrections (console flags, verb count, fps floor, fake pass-through)

- Console flags (corrects §140 console entry at `:7043-7044`): `--verbose` / `--no-color` parse on six verbs — `run`, `generate`, `stop`, `finalize`, `sfx`, `soak` (`cli.py` `_add_console_args` call sites `:499`, `:560`, `:601`, `:624`, `:643`, `:678`) — not the eight listed (`status`, `validate`, `benchmark`, `inspect` parse zero hits; `tests/test_console.py:248-262` pins the 050 rejection). Any agent adding console output to `status`/`benchmark` starts here, not from the stale eight.
- Verb count (corrects Phase-0 entry at `:6145`): "all 12 commands" is 15 as of 2026-09-30 — `{init,doctor,models,run,generate,status,pause,resume,stop,validate,finalize,sfx,benchmark,soak,inspect}` (`cli.py:279-684` subparsers) — the three additions are `sfx`/`benchmark`/`soak`.
- Fps floor (qualifies the augment entry at `:7832` + CLI help `cli.py:449-473` + `media.py:836-873`): `0` disables the 32 fps / 1280x720 floors, never the 24 fps presentation floor — `out_fps = max(requested, floor, PRESENTATION_MIN_FPS)` (`media.py:873`, `PRESENTATION_MIN_FPS = 24` at `:807`); a CausVid source (fps 16) with `--no-augment` still lifts 16 to 24. Geometry axes are true disables (`max(target, 0) = target`).
- Fake pass-through (annotates Phase-0 note at `:6163`): true when written (fake 768x432@24 matched the old target); post-augment defaults ship 1280x720@32, so a fake finalize re-encodes by default — the augment entry's own "768x432@24 -> 1280x720@32" shift applies. Holds re-verified 2026-09-30 (no change): augment flags on run/generate/finalize via one helper, TUI `min_fps`/`min_resolution` fields, `AugmentConfig` 32/1280/720, fake-early-return `[]` + CUDA augment pair, sfx rows, caption pins, director `models_dir`, overrides + `effective_*`, `run.sh` CUDA default + entrypoint bypass, `commit_one_segment` worker gate.

## 2026-09-30 — Issue 135 §5.3 1024x576 addendum correction (live preset stays 768x512)

- The §5.3 closing addendum (at `:559-563`) states the `ltxv` preset "was moved 768x512 to 1024x576 (`ltxv-576p`)" as a completed fact. Live registry disagrees: `config.py:197-204` is `"ltxv": BackendRecord(profile="ltxv-512p", width=768, height=512)`; `video_ltxv.py:811,858-859` defaults 768/512; the §140 rhythm-cut resolution (at `:6952-6960`) records "ltxv preset stays 768x512 — native 1024x576 tried and reverted the same day (forward ~15.6 GB, beyond the 16 GB card)"; the Stream B probe verdict (at `:7110`) is "deterministic OOM 3/3 — the 768x512 preset stands". History kept in one line: 1024x576 tried 2026-09-24, reverted same day (VRAM), see the §140 rhythm-cut note — normative geometry is 768x512 (`ltxv-512p`). The §140 log itself is untouched (history stays history). Gate: `rg -n "576p|1024x576" DESIGN.md` shows only historical/log contexts after this correction, zero normative prescriptions.

## 2026-09-30 — Issue 093 TASK prune-merge DESIGN write (deletion approval pending, nothing deleted)

- TASK-only slice already landed (batch 7): §30.4 heading corrected from "never produced" to "produced (2026-09-24+)" with the open breadth noted (81/97/121 matrix, TeaCache/Q8/FP8, extension-throughput). This is the DESIGN-side write: open §30 items live exactly once here — §30.1 CausVid open (overlap sweep, resume A/B, 24 fps finalize stage, eyeball) tracks in §§137D/128 + the CausVid worker slice; §30.2 LTXV open (81/97/121 matrix, TeaCache/Q8/FP8, extension-throughput, eyeball) tracks in §5.3 deferred + the §137A qualification harness; §30.3 resolved via the `VideoBackendAdapter` contract (§140 Stream C entry); §30.4 open breadth as above; §30.5 is pointers only. Mapping (§30.x to DESIGN §y) is recorded in this entry per the issue's gate. Deletion approval stays pending — `Voyage/TASK.md` (1746 lines) is NOT deleted by this pass (per-commit approval per AGENTS §9); `rg TASK.md` outside `issues/` shows zero non-historical refs, so no ref repair is owed when deletion lands.

## Batch 9 (2026-09-30) — batch-9 resolutions as-builts (one line per issue group)

- Worker tape+resume (129/134/169): save_recovery_tape_atomic (tmp+fsync+replace+fsync_dir) + resume_fallback triple with anchor-clears-after-record (supervisor metric residual).
- TUI parity+help+toggle tests (113/114/181): segment_plan/done parity with console non-verbose lines + FLAG_HELP_FIELDS/tooltips/FIELD_HELP keys + real-interaction toggle tests for all 5 flags.
- SFX render (161): console + TUI print sfx: line (render half; video_caption pin residual).
- Orphan+doctor (098/195): cli_validate audio/ extend + run-root *.partial pass + 4 doctor rows added, 10/10 verifiers.
- DESIGN corrections (133/135/093): 6-verb list, 15 verbs, fps floor, pass-through gap corrected + §5.3 correction append-only with history kept + TASK DESIGN write landed (delete approval pending).
- SFX-bounds+manifest+failsoft+inspect-logs (152/191/141/142/140): probe-memo + per-blend timings (single-graph join blocked by pinned ≤2-input invariant) + video-stream search with zero-duration/fps guards + presentation/final_geometry at init/finalize + concepts/scoreboard fail-soft degrade + frame view → logs/inspect/<segment>.png.
- Bench-targets+soak+floors+preset (154/163/194/157): sfx+augment CLI targets + bench helpers + soak SFX section and setup axes + _presentation_setup spread at all 3 sites with new targets + CHUNK_PRESETS preset knob with warm-first fan-out.
- Test folds+slow mark (088/089): wire_contract fold landed (unset+blocks-request→test_wire_contract, clusters remain one per pass) + tui_app slow-mark landed (lock typo→lock track, tail marking→tests track, mypy-scope→scripts, cache guard→gates).
- Gates.sh wire_contract list fix: test_wire_contract added to the gates.sh test list.
- Ratchet re-pinned at 144 via helper-call conversion.

## Batch 10 (2026-09-30)

- Resolved: 024 / 086 / 094 / 078-accepted / 087-accepted.
- Partial: 035 / 036 / 081 / 166.
- Blocked: 070 / 121.
- Deferred: 031.
- 086/121: no inline DESIGN change (086: zero-caller deletes, no contract change; 121: §35 tie-doc stands — see §35 batch-8 as-built, ceil switch blocked on rhythm pins).

## Batch 11 (2026-09-30)

- Decided: 024 (warn-only outside-tree paths legal, only run-id/name traversal hard error — binding, see §58 batch-11 as-built).
- Fixed: 121 (ceil switch + plan-site guard, pins updated — see §35 batch-11 as-built).
- Full: 166 (FILM port + ESRGAN leg — both weights strict-load end to end — see §56 batch-11 as-built).
- Partial: 035 (`json.loads` narrowings landed; `call()`/supervisor/`scoreboard_rows`/hub kwargs still blocked — see §82 batch-11 as-built).
- Progress: 036/081 (`cli_inspect_metrics.py` 31L + `supervisor_prefetch.py` 43L, move-verbatim + re-export + agreement-test, following existing conventions; no DESIGN contract change, folded here only).
- Still-blocked: 031 (PERF 12 / N 60 / PT 111, no family green), 070 (Wan2.2 `WAN_HF_REVISION` still None, no verified bytes).

## Batch 12 (2026-09-30)

- Fixed: 127-director (`handle_init` unknown-key `TypeError`, all workers strict — shared-helper home residual).
- Proven-blocked: 152 (wide MANUAL-fade N=8 no-hang but not bit-identical — see §56 batch-12 correction; fold stays pairwise).
- Partial: 035 (`scoreboard_rows` return narrowed — see §82 batch-12 as-built), 036/081/082 (`registry_film.py` + `supervisor_commit_types.py` extractions; 088 one cluster fold; 089 slow-marks + cache guard), 093 (TASK.md → pointer stub; `git rm` pending user approval).
- Blocked: 079-delete (quiet tree, decision made — hard error with migration hint; ~24 non-owned follower files need a joint pass), 070 (wan_models absent), 031 (PERF 11 / N 60 / PT 111 — owned `cli_observe` site landed, rest foreign/dirty).

## Batch 13 (2026-10-01)

- Resolved: 079 (full delete executed; 070 superseded by deletion — Wan2.2 pins gone with the worker; see §5.1/§11 batch-13 as-builts).
- Partial: 036/081/082 (+`registry_realesrgan.py` +`registry_inspector.py` +registry-film pattern extractions; no DESIGN text change per convention, folded here only), 088 (+`sfx_parser_parity` fold), 089 (+`integration:119` ignore removed), 166 (`resolve_augment_weights` registry-to-loader seam — see §§56-57 batch-13 as-built).
- Record-only: 031/035 (no DESIGN text change).
- Blocked-remain: 152-proven (pairwise fold stands until parity proven — see §56 batch-12 correction; no new note, standing note already present).

## Batch 14 (2026-10-01)

- Resolved: 089-fastpath/validate-legs (marker-only change otherwise; none outstanding beyond the fastpath/validate legs).
- Partial: 082 (`registry_ltxv.py` 84L + `registry_audio.py` 105L extracted move-verbatim + facade + agreement tests; remaining families: director triple (shared MINILM), causvid+WAN21, SFX triple — see §84 batch-14 as-built), 081 (`supervisor_tape.py` 51L, `tape_tail_sha_matches` extracted move-verbatim + facade + 6 agreement tests; `supervisor.py` 2650→2624L; no inline DESIGN change per convention, folded here only), 036 (see issue), 088 (`prefetch_summary`→`generation_stack` fold landed; remaining clusters listed in the issue).
- Record-only: 031/035 (none new; record-only re-probes; batch-12/13 proposals stand).
- Confirmed-blocked: 152/166 (152 none new; record-only re-probe, batch-12/13 proposals stand).
- Verified: 079-clean (none new; followup sweep clean; one orphan: deleted worker's pyproject per-file-ignores entry removed), 070-tests-updated, 093-stub (none outstanding; stub verified, rm pending approval).

## Batch 15 (2026-10-01)

- Partial: 082 (`registry_causvid.py` 121L + `registry_sfx.py` 135L extracted move-verbatim + facade + agreement tests; remaining family: director triple (shared MINILM coupling must be decided explicitly) — see §84 batch-15 as-built), 081 (extraction skipped with cause; no DESIGN contract change, folded here only), 088 (`prefetch_shutdown`→`generation_stack` fold mechanical; no DESIGN contract change, folded here only).
- Partial: 089 (8 slow marks + `test_integration.py` in mypy gate; no DESIGN contract change, folded here only).
- Record-only: 031/035/152 (no new DESIGN claims this batch).
- 093 rm-ready (no new DESIGN claims; `git rm` pending approval).
- Blocked: 166 (CPU-only; no new DESIGN claims this batch).

## Batch 16 (2026-10-01)

- Resolved: 082 (all 8 families split — film/realesrgan/inspector/ltxv/audio/causvid/sfx/director; MINILM moved as single source; `registry_records.py` now pure facade — see §84 batch-16 as-built).
- Skip: 081 (dirty tree + nothing verbatim-movable; no DESIGN contract change, folded here only).
- Fold: 088 (`final_blend_scale`→`finalize_fastpath`; no DESIGN contract change, folded here only).
- Record-only: 031/035/152 (record-only re-probes; no new DESIGN claims this batch).
- Records: 089/093/166 (no new DESIGN claims this batch).

## Batch (2026-10-01)

- 081 (see §73 as-built): 3 verbatim extractions + streaming derivation landed (`supervisor_routing.py` / `supervisor_plan_info.py` / `supervisor_lock.py`, 18 agreement tests); remainder stateful-only, `sha256_file` shim stays, worker-map merge and audio-coverage deferred with cause.
- 088 (tail-only; DESIGN proposals: none): 6 folds, 11 sources deleted, assertion net-zero (`av_alignment_consumer`→`state_integrity`, `hashing`+`paths`→`unit`, adapter triple→`adapter_contract`, `take_ahead_guard`→`audio_request_validation`, 4 TUI satellites→`tui`, `integration`→`state_integrity`); as-left 163 files / 1625 tests; augment-quad remainder + video-worker quartet + audio/TUI/finalize remainders each carry an exact verbatim-block cause.
- 093 (see §140 as-built): `Voyage/TASK.md` git-rm landed; nothing live lost.
- 152 (see §56 as-built): parity mechanism found (native-format `afade`; staged-s32 == fold byte-for-byte), recorded NOT landed pending the pin owner's 31-input proof + pin relaxation; pairwise fold stands.

## Batch resolve-all (2026-10-01)

- 152 RESOLVED (see §56 as-built): N=31 CPU ffmpeg proof byte-identical (31x 4 s sine stems at 1.0 s overlap — fold 1.9 s vs staged 0.7 s, same 36096102 B, same_bytes=True, no hang under the 600 s guard; each stem decoded once, one spawn, 30 per-stage `aformat=sample_fmts=s32` barriers, same `_blend_fade_seconds`/`%.3f`/integer-ms recipe). Landed construction: `voyage/media.py` adds `PAIR_BLEND_INPUT_COUNT = 2` + `_join_audio_single_graph` (N==2 delegates to `_blend_pair`; N>=3 builds the staged single graph, one `timing_ms` entry; raises on <2 inputs / duration mismatch / non-positive durations; output s32le) replacing the left-fold loops in `assemble_segment_audio` and `build_final_audio`; `voyage/sfx_finalize.py` replaces its fold loop (intermediate names change `joined.wav` vs per-index `*_blend_*.wav` — still rebuilds on retry, O(N) so the quadratic retry cost is gone). Pin update (`tests/test_finalize_fastpath.py:275`): `test_final_blend_never_spawns_wide_acrossfade_graph` now forbids `acrossfade` in every call and pins exactly one wide manual call (8 inputs on the 8-segment synthetic, `afade`+`adelay`+`amix=inputs=2` + `aformat=sample_fmts=s32`, no `acrossfade`) instead of `wide == []` — the poulah 31-`acrossfade` deadlock rationale no longer applies to manual graphs (audio joins are CPU ffmpeg on every run shape, no torch/GPU in the path, so no GPU long-run is owed for this stage). Probe-memo suite kept green by update (3 timing assertions N-1 → 1; all probe-count assertions hold — still O(N) probes). TDD: `tests/test_issue_152_single_graph.py` (3 tests: N=4 byte-identity vs fold reference, one-spawn pin, barriers/no-acrossfade pin). Soak trending moves from per-blend milliseconds (N-1 entries) to per-join milliseconds (1 entry).
- 081 REMAINDER CLOSED (see §73 as-built): no verbatim-movable group remains under the landed move-verbatim + explicit-`as` facade + delegation pattern (verified via AST `self.`-use + caller + dirt rescan over all 45 methods at 2587L — the take-joint compensation hunk has landed as `d04bdb5`, region quiet). Must-stay verdicts with live evidence: stateful commit-pipeline method groups (`_decide_payload` 1 … `_call_with_restart` 9 with 20+ `self._log_metric` fan-out sites — moving one method rewrites its signature, moving a group needs mixin inheritance, MRO risk); lifecycle remainder (`_held_run_lock`, `_stored_relative`, `_checked_tape_path`, `start/stop_workers`, `run_segments` 25 `self.`, `_log_metric` hub with 39 fan-out sites); audio-coverage group (`_ensure_audio_coverage` 7 … — the only `self`-free sub-block is an in-method slice, extracting it splits the method); `sha256_file` shim (`voyage/cli_validate.py:23` imports it — removal belongs to the cli track); worker-map merge (VIDEO 3 keys vs AUDIO 2 keys, different error contracts — the 079 longlive2 hint — merging breaks `test_longlive2_removed_079`); deterministic-payload compat (in-method block inside stateful `_decide_payload`); `run_id` legacy (single line inside the hub); legacy-migration threading ABSENT (`grep LEGACY_MIGRATION voyage/supervisor.py` empty — the constant lives in `voyage/concepts.py:34`). Landed this issue: 7 split files (`supervisor_proposal.py` / `supervisor_prefetch.py` / `supervisor_commit_types.py` / `supervisor_tape.py` / `supervisor_routing.py` (+ streaming derivation per 023/083, value-identical `('ltxv', 'causvid')`) / `supervisor_plan_info.py` / `supervisor_lock.py`); `supervisor.py` 2760→2587L. Next split needs a stateful-group pattern decision (mixin vs service object), one group per quiet-tree pass — explicitly out of verbatim scope.
- 035 RESOLVED joint leg (see §82 as-built): `rpc.py call()` is now `RpcPayload → RpcResult`; the supervisor `dict[str, object]` chain bridges at the seam via `cast` (no producer/consumer re-annotation — `cast(dict[str, JsonValue], dict(payload))` in, `cast(dict[str, object], ...)` out at `_call_with_restart`; embed payload `cast(dict[str, JsonValue], {"texts": texts})`; `float(value)` → explicit `bool`/`(int,float)` guard, same `None` outcome for hostile vectors); `bench.py`/`cli_observe.py` reporting takes `Mapping[str, object]` (covariant — both `dict[str, object]` and `dict[str, JsonValue]` pass; end-to-end `setup`/`metrics` renamed `e2e_setup`/`e2e_metrics` with explicit `dict[str, object]` fixing `no-redef` vs the benchmark-branch `metrics: RpcResult`). Residuals: `rpc.py:281,284` fd juggling (by construction); supervisor chain stays object-typed by design; `model_registry.py` hub kwargs + `atomic.py` write/read sides (by design); `bench.py:16` `_finite_float` (blocked by the `summarize_sfx_windows list[dict[str, object]]` callers).
- 166 FINALIZE THREADING WIRED (see §§56-57 as-built): opt-in `use_model_pass` (default off) runs `AugmentConfig` → config (`resolve_config` / `apply_draft_overrides` via `is_provided`, `--no-augment` forces False; `[augment] use_model_pass = false` TOML) → CLI (`--use-model-pass` `store_true` default None via shared `_add_augment_args`, all finalizing verbs incl. `stop --finalize` handoff; `cmd_finalize` passes `config.augment.use_model_pass` + `config.video.models_dir`; `cmd_generate` fan-out) → TUI (`use_model_pass` checkbox `flag-use-model-pass` + FIELD_HELP + tooltip + FLAG_HELP_FIELDS + `_read_form` + last-settings persistence) → `FinalizeOptions.use_model_pass` (appended last — positional compatibility kept; strict-bool `__post_init__`) → `ResolvedFinalizeSettings` (`resolve_finalize_settings` scalar-wins) → `finalize_run(use_model_pass=None, models_dir=None)` (consults `resolve_augment_weights(models_dir)` only when on; absent legs / no dir = ffmpeg fallback, never an error). Identity proof (real ffmpeg, fake-backend 1-segment commits, sha256): explicit-False == default AND knob-on with empty models dir == knob-off (plus seam test: off never consults the seam, on resolves exactly once; strict-bool rejections at every layer; pre-change determinism probed first so equality assertions are meaningful). TDD: `tests/test_issue_166_finalize_knob.py` (11 tests) red→green; companion pin updates (`test_augment_config` exact-dump + `--no-augment` pin, `test_cli_tui_split` 25→26 children, `test_tui` seven→eight flag ids + toggle coverage). Chunk-scale model pass behind it (`augment.py`: `enhance_frames` + `make_enhance_chunk_worker` + `run_model_augment_chunks`, idle-CUDA proven — 210 frames flat at 0.372 GB peak, fp16 numerics within 0.00327 mean-abs, eyeball true-blend). Residual (GPU-box + tensor encode): when legs ARE provisioned the knob still encodes via the ffmpeg vf path (resolve consulted, tensor chunk encode not selected); wiring the present-legs selection plus the idle-CUDA proof (LTXV 768x512 segments → 1280x720@32, fp16 numeric + eyeball + VRAM-flat) is the remaining slice.
- 088 TUI fold + 089 mypy 8/8 (tail-only; DESIGN proposals: none): `tests/test_tui_state.py` (26 tests) → `tests/test_tui.py` (33→59 defs, verbatim + banner + autouse-fixture NOTE; pre-delete 89/89, post-delete 89/89; gates.sh entry removed in the same edit) — TUI remainder `test_tui_app` 33 Pilot stays solo per the issue's own demotion recipe. 089 closed all 8 mypy legs (34→0: `test_video_common` 6 + `test_causvid_worker` 14 + `test_commit_hardening` 5 + `test_finalize_encode_rank2` 1 + `test_ledger_rotation_rank2` 4 + `test_novelty_leniency` 1 + `test_stage_a_telemetry` 2 + `test_perf_regressions` 1) via 8 facade-only voyage re-exports (`logrotate`/`concepts` `fsync_dir`, `prompts.StyleSpec`, `vision.metrics.probe`, `workers.audio_acestep` `subprocess`, `workers.video_ltxv`/`video_causvid` TAIL/TAPE_FILENAME + CAUSVID_COMMIT/CHECKPOINT_FILE, `supervisor.validate_video`) + 5 test-side fixes; gates.sh mypy list 68→154→162 modules; lock legs MOOT per batch-2 068 refutation (httpx2/httpcore2 genuine distributions, tomli omission marker-correct). 088 residuals: augment quad (foreign hunk in `test_augment_models.py` — retry post-land), video-worker quartet (needs explicit fold target), audio remainder (no same-area target), finalize/commit (owner `test_finalize_fastpath.py` foreign-dirty at fold time — now quiet).
- Batch tail: 031 DEFERRED (PERF 10 / N 51 / PT 111 — batch-12 `cli_observe.py` handoff holds, no owned-clean site exists); 036 TRACKED (no 036-seam extraction this batch — quota filled by 081/082/088 work; signal table otherwise unchanged).
