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
> As-built (ltx-integration-2026-10-01): two LTX backends join the table — `ltx25` (LTX-2.5 22B distilled Q3_K_M GGUF, 1216×704@24, 96f, `reconstructable_prefix`) and `ltx23` (LTX-2.3 22B distilled Q3_K_M GGUF, same geometry/accounting). Backend count 3→5; `ltxv` stays the default. Both run ComfyUI in-process inside the `voyage-ltx` worker image (pinned ComfyUI @2f35f4a + ComfyUI-GGUF @6ea2651 + gemma4 patch) — the §4 "no ComfyUI runtime dependency" non-goal is held by containment (never a host dependency, never on the `voyage` image path). Quantization/TE/VAE are implicit per backend (Q3-only; OOM is a clean failure, no fallback ladder). Both generate joint audio, so their rows pair `audio_backend=fake` + `sfx_backend=fake` and the supervisor skips the ACE-Step/MMAudio stacks via `JOINT_AUDIO_BACKENDS` (manual `voyage sfx` stays available).

| `causvid` | Wan2.1-T2V-1.3B + CausVid causal DMD | autoregressive chunk rollout with `start_latents` overlap | no required long-lived GPU cache between rollouts; continuation reconstructed from latent prefix | upstream scripts use 16 fps | fast causal continuation and straightforward long-video rollouts |
| `ltx25` | LTX-2.5 22B distilled (Abiray Q3_K_M GGUF, Gemma4-Q2_K TE, conv VAE) | 25-frame frozen-prefix conditioning (`LTXVImgToVideoInplace`, strength 1.0) | no persistent diffusion state; continuation reconstructed from the 25-frame tail + `recovery.pt` | 1216×704@24 quality path (Mode A two-stage) | high-quality joint audio-visual segments on 16 GB |
| `ltx23` | LTX-2.3 22B distilled (unsloth Q3_K_M GGUF, Gemma3-Q2_K DualCLIP TE, distilled VAEs) | same frozen-prefix mechanism, profile `ltx23` | same reconstructable-prefix discipline | same Mode-A geometry | fallback family; only one with a viable 2-GPU TE split |

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
> As-built (final-2026-10-01, issue 031): ACCEPTED-RESIDUAL — select stays 16 families, gap declared in `Voyage/pyproject.toml` comment with PERF->N->PT retry list (078/087 precedent).

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
very slow continuous camera movement, strong fluid motion throughout, ultra high definition, hyper detailed, sharp crisp image, simple refined composition, no abrupt cuts, no scene change within the shot
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
    ├── recovery.pt
    ├── manifest.json
    └── DONE
```

`manifest.json` (`format: 1`) holds the five metadata sections
(`transition`/`prompt_plan`/`audio_state`/`world_state`/`metrics`) plus
`checksums` (sha256 over the binary artifacts `video.mp4`/
`recovery.pt` only — metadata rides on the atomic manifest write, no
self-hash). Pre-prune runs (individual JSONs + `sha256.json`) still
load via legacy fallback in `voyage/segment_manifest.py`. Recorded
`audio.wav` checksum entries from old runs are ignored. Music takes
are rendered at finalize from the takes ledger and never persisted
per-segment; the video conditioning tail is derived on demand at
resume (see §5.3 as-built) and new runs start without the root
`concepts.jsonl` legacy dup.

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
> (kept as the commit gate); `video_tail.mp4` is resume-derived; root
> `concepts.jsonl` no longer created; `.cache/acestep` upstream writes
> are chdir-redirected out of the run. Readers (`media`,
> `cli_validate`, `scoreboard`, `sfx_finalize`, adopt, inspect-merge)
> go through `segment_manifest` with legacy fallback, so old runs
> still validate/scoreboard/resume.
> As-built (all-deferred-2026-10-04): per-segment file count 5 → 4 —
> `audio.wav` is gone (all backends deferred, takes render at
> finalize), so commit-time audio slices no longer exist either.
> As-built (run-scratch-2026-10-06): every temp file a run produces
> lives under `run_dir/tmp/` (`paths.SCRATCH_DIRNAME`), never on host
> /tmp — the supervisor creates it at construction, passes it explicitly
> (`scratch_dir` init field) to the session-owning workers (ltx25/ltx23
> work_roots incl. ComfyUI folder_paths dirs + mux PNG staging; ACE
> upstream-CWD redirect + take staging; MMAudio window staging;
> finalize-time ACE/SFX spawns; sfx-final + assemble staging), and
> points process TMPDIR at it in `start_workers` as a backstop for
> bench harnesses, bare `TemporaryDirectory` calls, and third-party
> libs (workers inherit the env via `rpc._spawn_env`). `validate_run`
> ignores `tmp/` (orphan scan covers segments/novelty/audio/augment +
> run-root `voyage-final-*`/`*.partial` only); stale `voyage-*`
> session dirs from killed runs are pruned at supervisor construction
> (single generate path builds it; evict paths already rmtree live
> sessions). The `run.sh` /tmp mount stays (torchinductor + HF cache).

---

# 30. Transactional segment commit

A segment is committed in this order:

```text
1. Generate video
2. Validate video
3. Write metadata to .partial files
4. fsync metadata
5. Atomically rename metadata
6. Write recovery checkpoint to .partial
7. fsync checkpoint
8. Atomically rename checkpoint
9. Compute checksums and persist them inside the manifest
> As-built (§30-pruning-2026-09-30): step 9 persists checksums as the
> `checksums` section of the single atomic `manifest.json` write
> (binaries only); there is no separate `sha256.json` anymore.
10. Write DONE.partial
11. fsync
12. rename DONE
13. Atomically update state.json
14. Atomically update run manifest / committed index
```

Audio is not committed per-segment: every backend is deferred, so
takes render once at finalize from the stored director decisions
(`audio/takes.jsonl`) and the commit path never touches audio.

The precise order can be simplified, but the invariant must remain:

> No state file may claim a segment is committed until the segment's required artifacts are valid and durably present.

> As-built (§30-av-gate-2026-09-30, issue 003): step 3 (validate
> media) enforced `|video−audio| ≤ 0.6 s` via probed durations and the
> shared `check_av_alignment` helper (`av_drift_seconds` recorded);
> misalignment raised recoverable `MediaError`, never a silent commit.
> As-built (all-deferred-2026-10-04): the gate is retired with the
> per-segment audio it policed — commit validates video only, and the
> A/V alignment check lives at finalize, where the takes ledger meets
> the committed video timeline.

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

Music takes render once at finalize from the stored director
decisions into `audio/takes.jsonl` (one take per plan entry, sliced
and blended onto the committed video timeline); the SFX bed renders
alongside from `audio/sfx/sfx.jsonl`. No per-segment audio exists —
every backend is deferred, so there are no `music.wav`/`ambience.wav`/
`sfx.wav` intermediates and no per-segment `audio.wav` mix target.

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
> As-built (final-2026-10-01, issue 036): CLOSED media slice — `voyage/media_audio.py` owns the 935-line audio-join block verbatim (test patch-sites re-homed), `voyage/media.py` keeps facades.
> As-built (final-2026-10-01, issue 166): CLOSED+PROVEN — present-legs tensor encode selected on knob-on via `resolve_augment_weights` → `augment.py` chunk worker (cuda:1 fp16 numerics + eyeball + flat VRAM proof).
> As-built (batch-2026-10-01, issue 152): parity mechanism found — `afade` runs in its input's native sample format, so s16-fed fades truncate to the s16 grid while s32-fed (fold blends 2+) keep precision (≤1 s16 LSB, second-and-later overlaps only); chained pairwise stages with an `aformat=s32` barrier per stage == production fold byte-for-byte (N=4 and N=8, max=0). Recorded NOT landed — needs the pin owner's 31-input-scale no-hang proof + ≤2-input pin relaxation first; the pairwise probe-memo fold stands.

> As-built (batch-7-2026-09-30): frame-count math single-homed in `voyage.augment.interpolated_frame_count` (`media` re-exports); `FINALIZE_CRF_MINIMUM/MAXIMUM` alias the augment CRF ladder (the DEFAULT diverged 2026-10-06: finalize 30/slow vs chunk 15/veryfast — intermediates are re-encoded at publish, so their quality setting is transient); `resolve_finalize_settings()` is the single scalar/options= contract; `workers/augment_worker.py` is a quarantined spike (official weights raise `ModelCompatibilityError`). (The old "768/432/24 defaults retained for the zero-floor stream-copy fast path" clause died with the 2026-10-06 compression change — there is no stream-copy publish anymore.)
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
> As-built (finalize-compression-2026-10-06): the native branch no longer stream-copies — every publish encodes libx264 (`-preset slow -crf 30` defaults, AAC `-b:a 128k`), so a default finalize shrinks segments ~60× (measured 1024×576@24 line-art: ~20MB/min source → 6.1MB/min final, SSIM 0.967 / PSNR ~37dB vs source, ~10× realtime CPU); the stream-copy fast path is dead (gate still computes `native`, `fast_path` metric reads False).

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
> As-built (§59-console-progress-2026-10-04): hybrid checklist+bars — `stage()` checklist steps plus `bar()` X/Y counters (SFX windows, ACE takes, drained segments; % + elapsed + ETA on TTY, boundaries-only off-TTY), `timing_table()` finalize summary (per-stage + slowest + total), `--quiet` (failures + final paths only); library layers take an optional `progress` sink (`None` = silent, so all existing callers are unchanged); finalize stages (triage/model-pass/music/mix/publish) + SFX load/render/join + ACE takes + worker boot report through it, and the generate plan line shows the segment span (`a..b of P planned`).
> As-built (§59-bg-progress-2026-10-05): background-worker feedback — upscale/interp pollers fire `on_chunk(segment_id, index, total)` per rendered chunk (optional, default None); `_poll_to_completion` renders one determinate `model-pass chunks` bar (total learned after pass 1 from done + skipped + interp-waiting, per-chunk advance with a (leg, segment, index) seen-set, legacy done-only fakes tolerated), replacing the silent poll stage; `BackgroundPrewarm.ledgered_totals()` accumulates (passes, upscale, interp) for a post-commit supervisor report of newly ledgered chunks (the background thread never touches display code); director prefetch announces submit + hit via a new `SegmentProgress.note()` line (quiet-aware). One bar, never two concurrent Live displays; non-TTY stays boundaries-only.

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
> As-built (final-2026-10-01, issue 081): CLOSED — verbatim pattern exhausted 7/7 (proposal/prefetch/commit_types/tape/routing+streaming-derivation/plan_info/lock; `voyage/supervisor.py` 2760->2587L); remainder must-stay (stateful groups need future pattern decision, shim/map/compat/run_id contract-bound, legacy absent).
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
> As-built (final-2026-10-01, issue 035): CLOSED — `call()` is `RpcPayload->RpcResult` with supervisor cast-bridge at `_call_with_restart` + `bench.py`/`cli_observe.py` `Mapping` covariance; remaining `Any` sites must-stay by design (fd juggling, hub-kwarg invariance, `json.loads` idiom).

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
> As-built (final-2026-10-01, issue 036): CLOSED workers slice — `voyage/video_ltxv_validators.py` + `voyage/video_causvid_frames.py` extracted verbatim+facade+TDD; registry split set complete (082).

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

> As-built (all-deferred-2026-10-04): the `audio.wav` box in the
> diagram above no longer exists at commit time — every backend is
> deferred, so the commit persists video only and the ACE-Step music
> takes render once at finalize from the takes ledger.

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
   crossfade 2.0 s / 48 kHz; finalize muxes AAC `-b:a 128k` (2026-10-06
   compression change — was 256k; video budget then has ~9MB/min of the
   10MB/min target);
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
- Llama-director stack done 2026-10-01 (options 2/3/4a, two parallel
  tracks, uncommitted): Track A — `workers/director.py` gains PLD
  (`prompt_lookup_num_tokens=10` with TypeError-fallback since GPU
  acceptance is unproven), interim strict JSON-coercion 4a (`outlines`
  absent in the test container, so true constrained decoding waits on
  an image change; replacement point documented), and the llama HTTP
  client (`_post_llama_chat` seam: `response_format` json_schema
  `evolution_decision` + thinking-off, `usage` token counts, all
  failures degrade via the existing §51 chain); `tests/
  test_director_assisted.py` (16, TDD red-first). Track B — sidecar
  `voyage/llama_server.py` (start/stop/readiness, verified-real flags
  only, no shell), `director-qwen35-gguf` registry pin (`bartowski/
  Qwen_Qwen3.5-4B-GGUF` rev `4168f45`, file
  `Qwen_Qwen3.5-4B-Q4_K_M.gguf` — the `Qwen_` prefix is load-bearing),
  pinned nightly b11146 CUDA `llama-server` in `Dockerfile.video`
  (stdlib fetch + sha gate, `--version` smoke), `llama` backend +
  `llama_endpoint` plumbing (`models_ensure` pick; wire backend stays
  `qwen` so the client routes correctly; readiness failure aborts
  loudly, never falls back). Integration: 59 + 63 scoped green, ruff +
  format + mypy strict clean, httpx 0.28.1 confirmed in the director
  venv; full suite 1880 passed (1 pre-existing citation gate).
  Open: image rebuild + live GPU A/B (needs GPU signal), then commit
  (needs per-commit approval).
- Llama-by-default done 2026-10-02 (A/B-proven: sidecar + Qwen3.5
  Q4_K_M GGUF on cuda:1 gives 10.7s/11.0s per directive at 67-70
  tok/s vs 23.4s AWQ ExLlamaV2 — ~2.2x wall-clock with ~40% more
  tokens; clean first-try JSON both attempts, no thinking traces,
  no --jinja trap): `DirectorConfig.backend`, `default_config_toml`,
  `init`/`generate` parsers, `cli_run_ops` and TUI form all default
  to `llama` (`qwen` AWQ stays an explicit opt-in, `deterministic`
  the LLM-off opt-out; wire backend still `qwen`); TUI `DIRECTORS`
  gains `llama` so stored configs validate; suite-wide
  `conftest` no-spawn fixture (exempting `test_llama_sidecar.py`,
  which owns spawn behavior) keeps offline fake-stack tests green
  via the §51 fallback. Image saga: prebuilt b11146 links
  GLIBC_2.38 vs image 2.35 (no 2026 binary can run) + build hosts
  lack libcuda (no build-time `--version` gate possible) — the
  Dockerfile block is now a pinned-source build (b11146 tarball,
  cmake pip wheel, GGML_CUDA=ON, arches 75;86, `test -x` +
  symlink only; live GPU run is the smoke test).

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

## Final closeout (2026-10-01)

- 031 ACCEPTED-RESIDUAL (see §12 as-built): select stays 16 families, gap declared in `Voyage/pyproject.toml` comment with PERF->N->PT retry list (078/087 precedent).
- 035 CLOSED (see §82 as-built): `voyage/rpc.py call()` is `RpcPayload->RpcResult` with supervisor cast-bridge at `_call_with_restart` + `bench.py`/`cli_observe.py` `Mapping` covariance.
- 036 CLOSED (see §84 + §56 as-builts): workers `voyage/video_ltxv_validators.py` + `voyage/video_causvid_frames.py` verbatim+facade+TDD, registry split complete (082); media `voyage/media_audio.py` owns the 935-line join verbatim, `voyage/media.py` keeps facades.
- 081 CLOSED (see §73 as-built): verbatim 7/7 exhausted, `voyage/supervisor.py` 2760->2587L; remainder must-stay by contract.
- 088 CLOSED (see Batch resolve-all as-built): folds landed, assertion net-zero; augment-quad/video-quartet remainders carry verbatim-block cause.
- 089 CLOSED (legs green on final tree): all 8 mypy legs 34->0 via facade-only `voyage` re-exports + test-side fixes; `gates.sh` mypy list 68->154->162.
- 152 RESOLVED-VERIFIED (see §56 as-built): N=31 CPU ffmpeg proof byte-identical via `voyage/media.py` `_join_audio_single_graph` + `tests/test_issue_152_single_graph.py` (3 tests).
- 166 CLOSED+PROVEN (see §56 as-built): present-legs tensor encode selected on knob-on via `resolve_augment_weights` → `augment.py` chunk worker (cuda:1 fp16 numerics + eyeball + flat VRAM proof).

## LTX backend integration (2026-10-01)

- User request (verbatim): "integrate both C1 and C2 to the voyage project as backend 'ltx25' and 'ltx23'; parameterization for quantization and the text encoder and VAE must be implicit; also make the quality path the default; implement the most native mechanism for video continuation for each; take advantage of their audio generation capacity which may mean that for those backends, the other audio generative models can be omitted; ltx25 and ltx23 should produce long lasting fully audio-visual high quality videos; do integrate with the upscaling/interpolation post processing if needed". Follow-up directives (verbatim): "if the upscaling and interpolation processes generate a video with higher specs than 1280x720@32, preserve the higher specs; 1280x720@32 is a minimum quality requirement, not a ceiling" (floors-as-minimum — RECORDED requirement, finalize `plan_augmentation` change still open) and "let's not duplicate the models so move the models from '../../video/ltx-experiments and clean up the unused models".
- Locked decisions: ComfyUI in-process execution; default Mode A 1216×704@24, 96 novel frames (121f windows, 25-frame carry); Q3-only per family (no fallback rung — OOM is a clean failure); full audio omission with manual `voyage sfx` kept; models consolidated into `~/.cache/voyage-models/ltx25/` + `ltx23/` (~98 GB unused rungs deleted, experiment tree kept working via symlink-back).
- Landed: registry pins (`registry_ltx25.py`/`registry_ltx23.py` + `MODEL_SPECS` + download/verify + `models_ensure` `JOINT_AUDIO_BACKENDS` skipping acestep/mmaudio); `voyage-ltx:latest` worker image (CUDA 13 + py3.11 + torch 2.14/cu130 + ComfyUI @2f35f4a + ComfyUI-GGUF @6ea2651 + gemma4 patch, smoke gate passes); workers `video_ltx25.py`/`video_ltx23.py` + validators + 28 tests (standard `standard_serve_map` contract, §5.3 tapes, ffmpeg-direct `_save_mp4` since imageio is absent by design, executor `cache_args` lru+ram+ram_inactive per pinned `main.py`); supervisor joint-audio bypass (`RenderedVideo.joint_audio_path`, `_cover_audio` joint branch writing segment `audio.wav` with empty takes, swap untouched since ltx rows pair `fake`); config rows (`ltx25-704p`/`ltx23-704p`, latent `(1,128,16,19,11)`) + `_LTX_NOVEL_BLOCK_FRAMES=96` + routing modules + CLI `--backend` + TUI `BACKENDS` (image-aware gpu_warning: `voyage-ltx` vs `voyage-video`) + `qualify.sh` + docs (`BACKENDS.md` table+sections, `ARCHITECTURE.md`, `UPSTREAM_LTX25/LTX23_NOTES.md`, `MODELS.md`, this §5.1 table).
- Proofs (all on RTX 4060 Ti 16 GB, evidence in `Voyage/LTX2.md` + `ltx2-experiments/results/`): Spike A Mode-A-121f PASS (peak 14933 MiB, locks 96-novel); Spike B 3-segment handover PASS (seam 0.93x/1.02x vs 3x gate, Mechanism 1 frozen-prefix sufficient); live ltx25 worker test FULL PASS (INIT 3.8s, fresh 121f + continued 96f @1216×704 h264 + joint 48kHz stereo audio.wav, seam 10.2/255, clean evict, peak ~14.8 GiB).
- ltx23 live worker test FULL PASS (INIT 4.5s, FRESH 121f denoise 134.1s + CONT 96-novel, joint 48kHz stereo, prefix fidelity 0.1089, seam 0.0148 ratio 0.92x vs 3x gate, eyeball identical glass pavilion, clean EVICT, zero OOM; verdict in `Voyage/LTX2.md` Phase-0). The ltx23-specific Mode-A-121f VRAM question is thereby answered: both 121f windows committed with zero OOM (polls 11933–14871 MiB) — no separate probe owed.
- Phase 5 qual legs (2026-10-01) — both FULL PASS via `run.sh generate --backend <ltx25|ltx23> --duration 8s --director deterministic --no-sfx` (2 segments each: fresh 121f + continued 96f = 217f @24fps; runs in gitignored `Voyage/output/qual-ltx25/` + `qual-ltx23/`): leg-1 ltx25 (seg0 video 157.5s + audio 0.0s joint, seg1 video 129.1s + audio 0.0s, validate VALID 2 segs/217f, finalize → final.mp4 h264 1280×720@32 287f 8.97s + aac48k stereo, eyeball prompt-faithful + 3 distinct frames = motion); leg-2 ltx23 (seg1 video 128.8s + audio 0.0s joint, VALID 2 segs/217f, finalize 8.978s → final.mp4 same 1280×720@32 287f 8.97s + aac48k). Both prove the supervisor drive + joint-audio bypass (audio stage ~0s, `no take (joint)`) + commit validation + floors-lift finalize end to end.
- Scoped gates on the ltx scope green in-container: ruff check + format (26 files) + mypy strict (88 files) + 243 pytest (workers, registry, adapter, ensure, tui, run_sh, commit_types).
- Still open: floors-as-minimum finalize change (`plan_augmentation` must treat 1280×720@32 as a floor, never a ceiling — recorded user requirement). Full `gates.sh` remains tree-owned (foreign in-flight files untouched per §9).
- Tree note: concurrent agents are splitting `supervisor*.py` and editing shared tests in this tree — ltx scope is additive-only (new files + registry/CLI/TUI rows + bypass branch); never touch their hunks.

## Finalize GPU defaults (2026-10-01)

- User directives (verbatim): "Do run the MMAudio model on the 4060 at finalize time. This must be the default behavior." / "In parallel, do run the Real-ESRGAN and FILM models on the 2060 on the video. This must be the default behavior." / "If you need to run tests on the GPUs, you may need to wait for them to be available. Proceed." Topology: cuda:0 = RTX 4060 Ti 16 GB, cuda:1 = RTX 2060 6 GB.
- SFX default-on for joint backends: `ltx25`/`ltx23` registry rows now pair `sfx_backend=mmaudio` / `sfx_device=cuda:0` (music stays `fake`/cpu — the native audio.wav is the soundtrack); `models_ensure.required_specs` dropped the `not joint_audio` carve-out from the SFX gate only (ACE-Step stays skipped). MMAudio dubs effects under the native soundtrack at finalize, after the video worker stops — no GPU contention by construction.
- Model pass default-on + pinned: `AugmentConfig.use_model_pass` / `FinalizeOptions` / `default_config_toml` / TUI form all default True (`--no-augment` stays the off switch; stored TOMLs rule, so in-flight runs keep their setting); new `augment.model_pass_devices()` returns `("cuda:1",)` when two GPUs show, else the legacy `augment_devices()` selection; `finalize_run` threads it into `run_finalize_model_pass` (1-GPU behavior unchanged).
- Parallel finalize: new `voyage/finalize_parallel.py` — `should_run_parallel` (SFX-on + knob-on + provisioned legs + one SFX worker + two visible GPUs; dual-worker SFX already spans both GPUs) and `run_parallel_finalize` (Thread A = whole `finalize_run` to a tmp music-only final incl. its standard `finalize_completed` event; Thread B = stream-copy reference concat + `render_sfx_bed` on cuda:0; then demux/mix/remux video-copy + publish). Shared `media.committed_usable_segments` extraction keeps triage single-homed. Bed failure still publishes music-only and raises `SfxBedError` (CLI keeps the legacy message + exit 1); video failure raises with nothing published. Output is byte-identical to sequential (same calls, same argv).
- TDD: `tests/test_finalize_gpu_defaults.py` (6 tests incl. slow parallel==sequential-bytes with stubbed GPU stages on fake commits); updated pins (`test_issue_166_finalize_knob` defaults on, `test_augment_config` dump, `test_generate_ensure` joint spec set, `test_tui` toggle test derives initial state from form defaults). Affected suites green (100 + 116).
- Still open: idle-GPU live verification (2060 6 GB fit for 1216×704 model-pass chunks is the risk — OOM-halving covers batch, per-frame peaks unmeasured); floors-as-minimum finalize change (recorded above) untouched.

## Finalize GPU defaults follow-up: 2060 fit (2026-10-01)

- Live incident: a single 1216x704 frame through RRDBNet-x4 needs ~5.5 GiB — batch-halving bottoms out at one frame and still OOMs the 6 GB 2060 (`conv_first`), while 768x512 peaks at 3.29 GiB and fits. Fixed by spatial tiling in `workers/augment_worker.py`: `tile_grid` (stdlib-pure output-block partition, gated) + `needs_upscale_tiling` (500,000 px budget — every native backend size at/below keeps the direct path byte-for-byte) + overlap-64 context per tile (measured: overlap 32 leaves diffs up to 0.066 past the RRDB receptive field, 64 drops them to <=0.0008 smooth / 1e-5 noise; tiled-vs-direct pin 0.02). Tiled frames postprocess inline (never accumulate x4-native device tensors — the interp-first order hands the path 4x the frames).
- Second live incident (same probe): FILM pairs at 2432x1408 need ~5.9 GiB — same one-pair floor, same OOM. Fixed by device-aware leg order in `augment.enhance_frames`: `interp_first_for_small_device` (CUDA device under 8 GiB free → interpolate at 1x first, then upscale; else the validated upscale-first recipe byte-for-byte; non-CUDA/unparsable/probe-failure → False). Both-legs-only; single-leg paths untouched.
- TDD: `tests/test_upscale_tiling.py` (grid partition + budget + torch equivalence, skips torch-free) + `tests/test_interp_first_order.py` (probe fail-safe + stubbed-leg order switch incl. 5-frame blend count).
- Idle-GPU proofs (both cards idle at probe time): full 96-frame 1216x704 model pass on cuda:1 → `model_intermediate.mp4` 2432x1408@96, 759 s (~7.9 s/frame), peak 2.10 GiB, pixels sane (0-255, testsrc pattern); MMAudio large 8 s window on cuda:0 → 6.9 s, peak 6.23 GiB (matches the ladder), FLAC 8.01 s RMS -23.6 dB (real bed, not silence). Full suite 1899 passed + 1 foreign (citation gate, concurrent scope); ruff + format + mypy strict clean on all touched files.
- Observed (foreign, not fixed — §9): `tests/test_worker_perf_rank2.py` leaks its stub `torch` via immediate `monkeypatch.delitem` (teardown re-inserts it), so torch-needing tests fail instead of skipping when that file runs first; default suite order is unaffected.

## ltx25/ltx23 continuation fix (2026-10-02)

- Symptom (user report, verbatim): "Currently, the continuation between segments when using the ltx25 backend simply does not work. There is a hard cut between each segment and they are not correlated at all. Use 'voyage/output/ltx25-compare' as an example". Root cause: NOT the worker mechanism — the supervisor sent `scene_cut=True` on every destination change (`supervisor.py` `scene_cut=(destination_concept != current_concept)`), and with qwen `drift_every_n=1` every segment drifts, so every segment rendered FRESH (121f each; manifests confirm) and the 25-frame frozen-prefix path never engaged. The worker continuation itself (Spike-B Mechanism 1, `LTXVImgToVideoInplace` strength 1.0) was proven sound once fed.
- Fix (Track A): `supervisor.py` always sends `scene_cut=False` for streaming backends — the worker goes fresh only when no tail exists (cold start / evict / restart without resume). Pinned by `tests/test_ltx_always_continue.py` (2 tests) + updated `test_adapter_contract.py` streaming expectation to `[False, False, False]`.
- Proof (Tracks B0/B1, RTX 4060 Ti): B0 deterministic 3-seg baseline 121+96+96=313f, seam ratio 0.59x vs 3x gate; B1 worker-level replay sweep over the pinned compare-chain prompts (fracture→void→lattice) at strength 1.0/0.8/0.6 → ratios 1.45x/1.37x/1.38x, all PASS, deltas within noise → strength stays 1.0 (Spike-B validated). Strength is now a wired parameter (`build_mode_a_graph(..., strength=1.0)` + `VOYAGE_LTX_STRENGTH` env override, fail-loud, in BOTH `video_ltx25.py` and `video_ltx23.py`; pinned by `tests/test_ltx_strength.py`, 4 tests) for future bake-offs — production default unchanged.
- Track B2 (probing only, NOT wired): Lightricks/ComfyUI-LTXVideo pinned at `ac4d998` (2026-08-11, contemporary with ComfyUI @2f35f4a, no new pip deps) cloned into `voyage-ltx` via `worker/Dockerfile.ltx`; `LTXVExtendSampler` INPUT_TYPES AST-verified (strength default 0.5, overlap default 16). Wiring deferred by user decision — Inplace passes with margin; known wiring risks recorded in the build log (guider must satisfy `STGGuiderAdvanced` surface — our `CFGGuider` exposes `original_conds`+`set_conds`, likely OK; overlap multiple of 8; sampler drops first latent + blends overlap, budget frames).
- Track C: ltx23 2-seg smoke (`output/ltx23-smoke/`, deterministic) 121+96=217f, ratio 0.49x PASS — same shared mechanism, no per-backend port needed. Track D (qwen-drifted re-run) skipped by user decision.
- Incidental findings (do not regress): the transient Gemma TE OOM (`process_tokens` 3.75 GiB @ 12.47 in use, also killed `ltx25-compare` seg000003) is flaky `expandable_segments` fragmentation — B0 ran zero-OOM; ComfyUI's `PromptExecutor` swallows node exceptions (worker OOM-retry never fires, surfaces as 0-frames `ValueError`) so production relies on supervisor-level re-issue; a retry MUST re-issue in place with NO evict — evict clears the resident tail and silently turns a continued block fresh (matches the 121f retried manifests in `ltx25-compare`).
- Tree note: this scope is additive-only (supervisor 1-line + strength params + 2 new test files + Dockerfile pack); concurrent agents own the other hunks in shared files — re-read before touching.

## generate defaults for the 2-GPU box (2026-10-02)

- User directives (verbatim): backend ltx25; "Do run the MMAudio model on the 4060 at finalize time. This must be the default behavior." / "In parallel, do run the Real-ESRGAN and FILM models on the 2060 on the video. This must be the default behavior." / "Qwen3.5 on 2060 as director" / "ltx25 joint video+audio mixed with SFX at finalize" / "slow prompt evolution per segment" / "ltx25 prompt carries style + video + audio details" / "random initial seed". Follow-ups (verbatim): "Keep 'drift_every_n_segments' to 1, but steer the director to produce prompts that are slightly different that the input one." / "drift a little bit every segment; motion should be derived from the general prompt (motion can be fast if prompt demands it)" / "non-reproducibility is ok by default" / "yes!" (llama Qwen3.5-4B sidecar on cuda:1 satisfies the Qwen3.5-director ask) / "if no duration is provided, perhaps generate until a key is pressed (say the 's' key); pressing the key would complete the current segment and would then move to the finalization stages; ensure that there is plenty of obvious feedback in the console experience" / "Proceed."
- Defaults sweep: `VideoConfig.backend` + `_DEFAULT_ROW` + `default_config_toml` + `init`/`generate --backend` + `cmd_init` fallback + `models download` target + TUI form + `run.sh` generate sniff + `qualify.sh` all move ltxv → ltx25 (1216×704@24 joint A/V on cuda:0; SFX MMAudio on cuda:0 at finalize; augment model pass pinned cuda:1; director llama/Qwen3.5-4B sidecar on cuda:1 — all already the ltx25-row/augment/director defaults, now also the entry-point defaults). `ltxv` stays fully supported via explicit `--backend ltxv`.
- Gentle drift (drift_every_n stays 1): `DIRECTOR_SYSTEM_PROMPT` + NOVELTY STEERING + user message now ask for a destination only slightly different from the input world (small variation in setting/element/mood, never a sharp break); caption doctrine (slow drift, never restart) unchanged. Tests updated to pin "slightly different".
- Prompt-derived motion: `prompts.motion_constraints(style, middle="")` — explicit fast cues (fast/rapid/dynamic/energetic/sweeping/...) in the general-prompt middle layer render "fast dynamic"; glide/drift/flow cues render "moderate"; otherwise the charter band decides (very slow/slow/moderate). Matching is word-boundary (`_mentions_motion_cues`, hyphens count as spaces) so "pan" stays quiet inside "company"/"span" and "flow" inside "flower"; inflected forms ride explicit cue entries. `enforce_style` threads the middle text through. Pinned by `test_phase3.py::test_motion_follows_general_prompt_not_just_charter`.
- Random seed: `init`/`generate --seed` default None → `seeds.random_master_seed()` (`secrets.randbits(31)`, signed-31-bit like every derived seed) with `random seed for this run: N (re-run with --seed N)` printed; TUI blank seed randomizes the same way, explicit pins. Pinned by `test_generate.py::test_omitted_seed_randomizes_init` + `test_tui.py::test_blank_seed_randomizes_but_explicit_seed_pins`. Deterministic derivation (`video_seed/audio_seed/director_seed`) unchanged.
- Press-s-to-finish: `generate --duration` is now optional; omitted means open-ended (`segments=None` through the existing `run` indefinite path) with a daemon stdin thread (`_start_stop_key_listener`) — a line starting with 's' flips state.json to STOP_REQUESTED so the current segment finishes before validate+finalize. Console feedback at arm (rule + plan line + "PRESS 's' + Enter" + "keep this terminal focused"), at trigger, and on stdin loss (Ctrl-C / `voyage stop` fallback). Pinned by parser-default tests + `test_stop_key_listener_requests_stop_at_boundary`. TUI keeps its duration field (scope guard).
- Verification: ruff + format + mypy strict clean on all touched source files; 1906 passed + citation-gate pre-existing failure (issues/ holds only the index since 2026-10-01 — cites across the tree predate this change) + 4 ruff errors in concurrent agents' in-flight files (conftest/test_benchmark/test_ltx_always_continue/test_ltx_strength — untouched per §9). No GPU run in this change (defaults-only + unit coverage; live ltx25 proofs stand from the 2026-10-01/02 entries above).
- Tree note: several touched files carry concurrent agents' staged hunks (MM in git status) — this change commits own scope only, their hunks left untouched per §9.

## Augment floors track ltx25 HQ native + trigger-gated model pass (2026-10-02)

- User directive (verbatim): "let's change the minimum quality outputs (that would trigger augmentations) to be exactly the high quality output of ltx25 (meaning 1280x704@24). This means that the high quality videos of ltx25 would not be augmented by default." True native verified read-only first: Mode-A stage-2 is 2x of the 608x352 stage-1, latent (1,128,16,19,11) → 19*64=1216, 11*64=704, qual legs confirm 1216x704@24 — so floors are 24fps / 1216x704 (a literal 1280-wide floor would still lift true-native 1216→1280 and defeat the goal; user chose 1216x704@24 when asked).
- Floor-number sweep (32/1280x720 → 24/1216x704): `voyage/config.py` (`AugmentConfig` defaults, half-disable docstring, `parse_min_resolution`, `default_config_toml [augment]`), `voyage/media.py` (`AUGMENT_DEFAULT_*`, `plan_augmentation`/`FinalizeOptions`/`finalize_run` docstrings+defaults), `voyage/tui_state.py` (help text, `_DEFAULT_*`, form defaults), `voyage/tui.py` (placeholder), `voyage/cli.py` (help texts); docs `AUGMENT.md`/`README.md`/`OPERATIONS.md`/`BENCHMARKING.md` rewritten to the new floors (ltxv 768x512@24 → 1216x704@24 no-minterpolate; CausVid → 1216x704@24 ~2.2x pixels + 16→24 lift).
- Load-bearing correction found during implementation (read-only plan missed it): `model_pass_active` (`augment.py:688`) is knob-on AND legs-present — it never consulted the plan — and the tensor branch (`media.py` pre-edit `:747`) took precedence over the `native` stream-copy fast path, so with provisioned weights every native ltx25 finalize still ran the full 2x-SRVGG + 4x-FILM pass (~1-1.5 h) and the floor change alone changed nothing by default. Fix: the tensor block is now trigger-gated on `plan.needs_reencode` (`media.py:749-759` + docstring `:653-663`) — geometry/fps lift runs models (FILM replaces minterpolate), at-or-above-floors ships stream-copy. `needs_reencode` covers both axes (`media.py:131-153`), so one flag suffices.
- Tests: new `tests/test_native_skips_model_pass.py` (2 tests — matching source + present legs + forbidden stub ships without calling; lifted request + recording stub engages) ; `tests/test_issue_166_model_pass_select.py::test_finalize_present_legs_selects_tensor_path` updated to the new contract (request lifts 768x432@24 → 1216x704 so the plan flags work; a matching request now takes the fast path — cross-ref added). Passthrough pinned at plan level by `test_augment_plan.py::test_ltx25_native_passes_through_default_floors`.
- Verification: ruff + format clean on all touched files; `mypy voyage` clean (90 files); 261 passed / 9 skipped across the full augment/finalize-adjacent scope (new file, issue-166, gpu-defaults, all augment/e2/cli/media/state/fastpath/generation-stack modules). Tree-level `ruff check .` stays red on foreign in-flight files (latent_io `worker/custom_nodes/`, untracked `test_ltx_*`, `test_benchmark.py:86`, `conftest.py:153` — untouched per §9). Observed-foreign: `test_parallel_finalize_matches_sequential_bytes` stays green but its tensor stub now never fires under floors-0 + matching geometry (both sides fast-path) — owning track may want a lift added to re-arm the orchestration coverage.
- Still open from the same thread: estimate comparison native 1216x704@24 vs 16fps + 2x FILM (delivered in chat).

## Independent augment sidecar workers + SRVGG upscale leg (2026-10-02)

- User directive (verbatim): "Perhaps the upscaling worker can run independently to the video generation worker. Basically, it polls for segments that need to be upscaled and upscales them without needing to synchronize with the video generation process. Same could be done with the interpolation worker. Independent workers running on each their own polling loop. When finalizing, you'd need to run each augmentation worker to completion to have them complete work that may not be fully completed. These workers must be able to resume their work if ever they fail at any moment." Follow-up constraints (verbatim): quick iterations (few segments/chunks); max parallelism incl. SFX+music where possible, sequential where OOM risk is too great; 2x-only upscale, upscale-time-per-segment <= video-gen-time-per-segment at max quality within budget; retry picks up where it left off even on failure; native LTX upscaler investigated; upscaler runs on the 2060 without interfering with generation.
- Architecture (additive, finalize `run_finalize_model_pass` tmpdir flow kept for partial-legs): `voyage/augment_sidecar.py` (stdlib-only durable ledger `run/augment/<plan-hash-16hex>/chunks.jsonl`, SFX-ledger-shaped: `ChunkKey` segment/chunk/stage/source-frames/expected/weights-hash/geometry/recipe/source-checksum, append flush+fsync+fsync_dir, serial plan-order append, last-wins stages, `missing_chunk_indexes` = ledger-truth resume, `prune_partial_outputs`, `plan_dir_for_segment` with multiplier=1 fixed), `voyage/augment_upscale_poller.py` (cuda:1 default, DONE+manifest+checksum gated discovery, per-segment plan dir, stub-seam decode/upscale, atomic .partial+replace+fsync publish, ledgered outputs skipped / unledgered dirs dropped+redone), `voyage/augment_interp_poller.py` (cuda:1 default, re-derives the upscale plan dir, chunks lacking `upscaled` records wait, ledgered-but-broken fails loud, FILM `interpolate_pair/triplet` at (1..m-1)/m moments, single-frame passthrough), `voyage/augment_drain.py` (`drain_interpolated_plan` → per-chunk mp4 + `concat_chunk_mp4s` stream-copy; missing interp record or PNG-count mismatch fails loud), `voyage/augment_finalize.py` (`run_durable_model_pass`: poll loop cap 10 with zero-progress MediaError, per-usable-segment plan+drain, concat in usable order). `media.finalize_run` branches: both legs present → durable path on `model_pass_devices()[0]`; partial legs → legacy tmpdir flow. `source_fps_key=int(round(source_fps))` shared so float/int fps never forks plan hashes. Upscale outputs are multiplier-independent; interp ChunkKeys record out_fps*multiplier (96 for 24fps 4x). Upscale + interp both default cuda:1 = staggered, never co-resident (FILM at upscaled size ~5.9 GiB alone).
- Upscaler swap (Phase 4c): websearch shortlist (realesr-animevideov3 XS, HFA2kCompact, NomosUni, AnimeSharp, LTX latent sidecar) probed — realesr-animevideov3 (SRVGGNetCompact XS, 16 conv/64 feat, native 4x) ADOPTED: 2,504,012 B, sha256 `b8a83768...08000a18b75d` matching HF mirror nateraw/real-esrgan@`44ad8adf` LFS oid, 768x512 0.055s/frame on 2060 (~11.5x vs ESRGAN-anime-6B 0.63s baseline; 1216x704 A/B 0.105s vs 1.4s), peak 0.34 GiB, 96f segment ≈10s upscale vs minutes of video gen; A/B eyeball on real ltx25 frames on par (crisp, no seam, faithful colors). `realesrgan-anime` pins replaced in place (same spec id; rev `44ad8adf6069185b86df22349b12f255821c86ab` verified live via HF API, floor 2_100_000, v0.2.5.0 URL, BSD-3-Clause); worker gains `_build_srvgg_net` + `_is_srvgg_compact_state` key-sniff + `_load_rrdb_net`→`_load_esrgan_net` rename (unwrap-then-sniff; the file wraps params under `params` — caught by provisioned test, production path was already correct). SRVGG raw rings wider (-0.825..1.883) but production clamps [0,1]. Weights-hash change forks plan-hash → old sidecar dirs orphan (expected, not migrated). Old 18 MB weights kept provisioned; prune note recorded. OpenModelDB alternatives deferred (budget already crushed).
- Tests (all TDD red-first): `test_augment_sidecar` (8), `test_augment_upscale_poller` (5), `test_augment_interp_poller` (6), `test_augment_drain` (5), `test_augment_finalize_wire` + 3 pre-existing tests repointed to the durable entry with matching fakes. Provisioned legs verified in voyage-video (ephemeral pip-installed pytest, current tree + `~/.cache/voyage-models` mounted): 20 passed incl. new 53-key strict SRVGG test. Integrated cuda:1 timing: 0.227s/frame wall (4x768x512 incl. load), 0.3 GiB peak. Gates: ruff + format + mypy strict clean on own scope; 190 passed/10 skipped across 14 augment/finalize files (incl. slow parallel byte-identical). Docs updated (`MODELS.md`/`INSTALL.md`/`AUGMENT.md`/`TROUBLESHOOTING.md`). Full `gates.sh` + soak still open at entry time.
- Soak closeout (same day, both GPUs idle): ephemeral end-to-end probe in voyage-video (tree + `~/.cache/voyage-models` mounted, driver at `/tmp/augsoak/driver.py` never committed) — 8 synthetic 768x512 frames through the REAL default seams (no stubs): upscale poller (SRVGG, cuda:1) → interp poller (FILM, cuda:1) → drain (real ffmpeg encode+concat) → 26-frame 1536x1024@96fps `model_intermediate.mp4`, 16.4 s wall, 3.79 GiB peak on the 2060 — budget met with headroom (96f segment projects to ~1-2 min vs minutes of video gen). The probe caught two real bugs, both fixed TDD red-first: (1) ffmpeg image2 numbers from 1 by default while `write_tensors_as_png_frames` writes 0-based — the poller sharing one dir for decode+upscale left a native-size orphan that broke the interp count check; fixed with `-start_number 0` in both `ffmpeg_decode_chunk` argv paths (precedent: the cli_observe probe already used it) + real-ffmpeg regression test `test_ffmpeg_decode_chunk_names_start_at_zero`; (2) drain temp `chunk_00.mp4.partial` destroys the file extension so ffmpeg cannot infer the muxer; fixed to SFX-convention `chunk_00.partial.mp4` via `with_name(stem + \".partial\" + suffix)` + `prune_stale_partials` now covers `*.partial.*` (deduped seen-set) + drain test asserts the temp naming. Fixture lesson: `testsrc duration=0.34@24` yields 9 frames, not 8 — the driver pins `-frames:v 8`. Gates after the fixes: ruff + format + mypy strict clean on own scope, 70 passed across the sidecar/poller/drain scope. Full `gates.sh`: ruff + format clean tree-wide; mypy has 1 error in foreign `tests/conftest.py:175` (concurrent agent's file, untouched per §9); pytest 1968 passed / 14 skipped / 1 failed — the failure is the pre-existing citation gate (`test_every_issue_cite_resolves_to_a_file` vs Dockerfiles citing pruned `Voyage/issues/` files; none of the new files appear in its list). Soak verdict: durable sidecar path PROVEN on the 2060 with real weights; long-run soak (multi-segment generation + concurrent video on cuda:0) deferred to an idle-GPU window with user approval.

## Full-resolution ltxv 5s video with upscale-only model pass (2026-10-02)

- User directive (verbatim): "Generate a full resolution ltxv video of 5s using the upscaler and the qwen director. Enable all features including music and sfx generation (no need for interpolation right now). Measure every part of the generation and report on elapsed times; add measurements if some are missing. Output the generated video to 'voyage/output'. Use the same prompt as was used for 'voyage/output/boba'."
- New `interp_multiplier` knob (TDD red-first, `tests/test_interp_multiplier.py` 6 tests): `AugmentConfig.interp_multiplier=4` (>=1 validator) + `resolve_config`/`apply_draft_overrides` params + `default_config_toml` literal + `--interp-multiplier` CLI flag (shared generate/finalize `_add_augment_args`) + `cli_core._augment_overrides`/`cli_finalize` video_kwargs passthrough + `finalize_parallel.run_parallel_finalize` param/forward + `media.FinalizeOptions` field + `__post_init__` check + `ResolvedFinalizeSettings` + `resolve_finalize_settings` scalar/options branches + `finalize_run` scalar + thread into both model-pass call sites (durable gets `multiplier=` + `timings=`; legacy `run_finalize_model_pass` gets `multiplier=`). `1` = upscale only, no interpolated mids. TUI form intentionally left without the knob (stored-config default 4 rules).
- New timing instrumentation: `run_durable_model_pass` takes an optional `timings` dict (schema: `upscale_poll_s`/`interp_poll_s`/`drain_s`/`concat_s` + `upscale_chunks_done`/`interp_chunks_done`/`chunks_drained`), accumulated in `_poll_to_completion`; `finalize_completed` gains additive `interp_multiplier` + `model_pass_timings_s`/`model_pass_chunks` keys; `media` pre-initializes `model_pass_timings={}` beside `tensor_intermediate`.
- Missing-measurement fix (TDD red-first): no SFX wall-time existed anywhere, so `render_sfx_bed` now emits `sfx_pass_completed` (`windows`/`backend`/`device`/`model_size`/`sfx_pass_s`) via `_emit_sfx_pass_completed` on both return paths (single-stem early return + normal join). First placed in `finalize_sfx_pass`, but a re-finalize proved ltxv5s takes the parallel path (which funnels through `render_sfx_bed` directly) — moved there, removed the first emission. `tests/test_sfx_finalize.py` 18 passed.
- Run `Voyage/output/ltxv5s` (init flags: `--backend ltxv --director qwen --seed 0`, boba style verbatim, acestep ambient electronic default): 2 segments committed = 217f ≈ 9.04 s @24fps (5 s request rounds up to 2×96f segments; seg0 fresh 121f, seg1 continued 96f). `final.mp4` 1216x704@24 h264 + AAC, 9.042 s, 5.0 MiB. Measured times — seg0 total 137.8 s (director qwen 47.0 / video ltxv 37.3 / audio ACE 53.5 incl. 45 s take_0000 render / validate+commit ~0); seg1 total 45.8 s (director 0.1 prefetch hit / video 45.6 / audio 0.1 keep); finalize model pass upscale_poll 82.9 s + interp_poll 70.0 s (m=1 PNG passthrough, no FILM inference) + drain 2.0 s + concat 0.1 s over 7 chunks (217f, chunk 32); audio_blend ~0.2 s, final_encode ~0.6 s; SFX MMAudio-large 2 windows on cuda:0 = 31.5 s. Re-finalize ledger-resume verified (poll 0 s, drain+concat only). Upscale verdict: 82.9 s for 217f ≈ 0.38 s/frame — inside the per-segment video-gen budget (37-46 s/segment) with margin, and that includes PNG decode/re-encode overhead, not just SRVGG inference (0.055 s/frame isolated).
- Path lesson (do not regress): `run.sh:20` cds to `Voyage/`, so `/app` is ALWAYS the Voyage dir — in-container run paths are `/app/output/<name>`, and the host-side TOML sniff fails for `/app/...` paths, so `VOYAGE_IMAGE=voyage-video:latest VOYAGE_GPUS=1` must be exported explicitly for run/finalize.
- Gates: ruff (incl. PLR2004 on `sfx_finalize.py`) + format + mypy strict clean on all touched files; affected scopes green. Full `gates.sh` re-run still open (expect only the 2 known foreign failures).

## Slow-mo finalize Phase 0: presentation_fps knob + slow-mo factor (2026-10-02)

- User directive (verbatim m0496): ACE music moves to finalize (ltxv/causvid only); 2x interp 24→48 presented at 32fps slow-mo with upscale→interp→music→SFX order; seam interpolation never dropped; 2060 sharing probe live. Q&A (m0499): generate ~6.7s content for a 10s final (1.5x stretch; NEVER discard frames); ACE FULL move; music steering in the changed default; 2060 investigation includes a LIVE probe.
- New `presentation_fps` knob (TDD red-first, `tests/test_presentation_fps.py` 10 tests): `AugmentConfig.presentation_fps=None` (0 maps to None, ge=1 rejects negatives) + `resolve_config`/`apply_draft_overrides` params + `default_config_toml` literal + `--presentation-fps` CLI flag (shared `_add_augment_args`, generate/finalize via `**_augment_overrides` spread) + `cli_core` overrides + `cli_finalize` video_kwargs + `finalize_parallel` param/forward + `media.FinalizeOptions` field + `__post_init__` check + `ResolvedFinalizeSettings` + `resolve_finalize_settings` scalar/options branches + `finalize_run` scalar + `finalize_completed` `presentation_fps` + `slowmo_factor` fields.
- Pure `slowmo_factor(source_fps, multiplier, out_fps) = source*multiplier/out` (`media.py`): 24x2/32 = 1.5. `plan_augmentation` gains `interp_multiplier=1` (lift measured against source*multiplier; default 1 = byte-identical old behavior); `finalize_run` passes `plan_multiplier = effective_interp_multiplier` only when model legs are present, else 1 (absent legs fall back to the minterpolate lift, never to slow motion).
- Gates: ruff + format + mypy strict clean on touched scope; 100+ passed across the knob scope. `supervisor.py` untouched (concurrent agent in flight).

## Slow-mo finalize Phase 1: ACE full move to finalize (2026-10-02)

- Design (full move, option (a) dry-ledger REJECTED for the fuller cut): `ltxv`/`causvid` commit writes a timeline-exact silent stub and appends NOTHING to `takes.jsonl`; finalize replays stored director decisions in segment order and renders takes through one shared ACE `SubprocessWorker` on cuda:0, then mixes on the stretched timeline. `ltx25`/`ltx23` joint path byte-identical (never enters the deferred branch); `models_ensure.JOINT_AUDIO_BACKENDS` skip untouched.
- New `voyage/audio_finalize.py` (stdlib-only at scope): `is_deferred_backend` (ltxv/causvid only); `write_deferred_stub_audio` (ffmpeg anullsrc timeline-exact silent wav, mono/stereo guard, fail-loud); `ensure_deferred_takes` (`render_take_fn` seam; replays decisions via `load_transition` + `AudioPlanner` + `audio_seed` + `beats_for_segment` + STRETCHED durations; ACE payload incl. repaint triple; `probed_take_seconds` clamp; incremental `append_take`; `MediaError` fail-loud with nothing appended) with ledger-load resume (re-finalize no-ops instead of double-append) + shared helpers `_load_existing_takes`/`_segment_stretched_seconds`/`_segment_music_inputs`; `deferred_render_pending` (pure dry walk, True on first non-keep — the parallel gate and the no-spawn check); `ensure_deferred_for_finalize` (pending-check, else `spawn_ace_render_fn` + ensure + best-effort shutdown, returns bool); `spawn_ace_render_fn` factory returns `(render_fn, shutdown_fn)` via `SubprocessWorker` + `audio_worker_module("acestep")`, init `{models_dir, device}`, `generate_audio` payload/return (same spawn pattern as `render_sfx_bed`).
- `supervisor._cover_audio` deferred branch (after the joint branch): stub + `take_action='deferred'`, never touches `_ensure_audio_coverage`. `music_style` default changed to the user experimental text (`config.py:327` + `:843` TOML + `models.py:284` AudioPlan; director fallback captions untouched — different field, concept-derived).
- `media_audio.build_final_audio` gains keyword-only `deferred:bool=False` (single-segment shortcut bypassed when deferred — else it copies the stub; all 5 fallbacks fail loud via `_fallback_or_raise` when deferred) + `stretch:float=1.0` (passes `fps/stretch` into `_segment_timeline`, widened to float); single-window fix: one rendered window converts straight to dest (the N-way `_join_audio_single_graph` requires ≥2 inputs — deferred single-segment runs landed there and raised).
- `media.finalize_run` reorder to model-pass → `ensure_deferred_for_finalize` (only when effective deferred; take sizing via `getattr(audio_config,...)` defaults 45/20/4/''/None; `stretch = stretch_full if tensor_intermediate else 1.0`) → `build_final_audio(fps unchanged + deferred/stretch)` → encode. Threading: `FinalizeOptions.deferred_audio:bool=False` (+post_init TypeError) + `ResolvedFinalizeSettings.deferred_audio` + `resolve_finalize_settings(deferred_audio param, both branches)` + `finalize_run(deferred_audio/seed/audio_config:AudioConfig|None` appended after `options`, TYPE_CHECKING import) + `finalize_completed` `deferred_audio` key. `finalize_parallel.run_parallel_finalize` gains `deferred_audio/audio_config` (+TYPE_CHECKING): pre-fork ensure via probe+plan+`slowmo_factor` precompute (superset stretch, ledger no-ops Thread A's internal ensure) before the ThreadPoolExecutor fork, Thread A forwards all three flags. `cli_finalize` video_kwargs gains `deferred_audio=is_deferred_backend(config.video.backend)` + `audio_config=config.audio` + `seed=config.seed`, and forces sequential when deferred (ACE cuda:0 vs SFX cuda:0 would collide on the 4060; the duplicate `seed=` in the parallel call removed — it now rides video_kwargs).
- Tests (`tests/test_deferred_audio.py`, TDD red-first, now 12): backend predicate, stub timeline-exact, `_cover_audio` deferred branch on duck-typed self (take-path forbidden), finalize replay renders+ledgers, fail-loud on render error with empty ledger, stretched 1.5x coverage, stub fail-loud, blend rendered, stretch 6.0s mix, spawn factory callable, `finalize_run(deferred_audio=True)` end-to-end with tuple-shape mock, ledger-resume no-op (`ensure_deferred_for_finalize` returns False on a complete ledger with `models_dir=None` — the pending check fires before any spawn).
- Gates: ruff + format + mypy strict (`mypy voyage` 96 files) clean on touched scope; 97 passed/1 skipped (deferred+resolve+fastpath+issue-166+native+interp+presentation+augment-config) + 69 passed (gpu-defaults slow parallel byte-identical + sfx + e1 + generation-stack). Full `gates.sh` still open. Phases 2 (seam joint-dir), 3 (10s experiment), 4 (2060 live probe) follow.

## Slow-mo finalize Phase 2: seam interpolation joint-dir (2026-10-02)

- Shape: without seams the durable path hard-cuts every segment joint (the pair A-last/B-first never saw FILM). New `voyage/augment_seam.py` (stdlib-only at scope): `seam_plan_dir` (joint dir via composite source key `"<key-a>|<key-b>"` through the shared `plan_dir_for_segment` — order-sensitive, either side re-rendered or any recipe change forks the dir instead of reusing a stale seam), `seam_endpoints` (A's last PNG of its max-index interp chunk + B's first PNG of `interpolated_00`; None while either side is incomplete = wait, never fail-loud here), `render_seam_once` (renders the `multiplier - 1` mids into single-chunk `interpolated_00/` + exact-key `interpolated` ledger record with `expected_frames=m-1`, `source_frames=2`, lifted `out_fps`; ledger hit + PNGs verified = resume no-op; unledgered output dropped and redone; `multiplier<2` is a caller ValueError). Default render is the resident FILM leg (lazy torch import, same moments as the interp poller); tests inject `SeamInterpFn` (`(before, after, dest, m) -> m-1 mids`). Endpoints are never dropped or duplicated — they stay in their segments, the seam contributes mids only (`multiplier=1` = zero mids, wiring skips seams entirely).
- Wiring in `run_durable_model_pass` (new `seam_interp_fn` param, None = default FILM): after `_poll_to_completion`, the drain loop carries the previous segment's (source_key, plan_dir) and interleaves [A, seam, B] — seam render (missing endpoints post-poll fail loud) + same `drain_fn` at `source_fps*multiplier` (stream-copy concat stays safe: identical codec/fps) before each non-first segment. New `timings` keys `seam_s`/`seams_done` (flow through `finalize_completed` generically); seam drains count under the shared `drain_s`/`chunks_drained`. A/V impact: +(S-1)*(m-1) frames ≈ 21-31 ms/joint, far under the 0.6 s gate — no audio change.
- Tests (`tests/test_augment_seam.py`, TDD red-first, 7): pair-sensitive plan dir, boundary-frame endpoints, incomplete-side wait, mids-only render + ledger shape + resume no-op, degenerate-multiplier rejection, durable-pass interleave order [A, seam, B] with stub seams (incl. re-run resume) + `multiplier=1` hard-concat skip (forbidden seam raises if called). `test_augment_finalize_wire` m=4 drain test repointed to `multiplier=1` (it carries no interp fixtures; m>1 interleave is pinned by the seam tests).
- Gates: ruff (`--fix` dropped 3 unused imports) + format + mypy strict clean on own scope; 35 passed across the sidecar/poller/drain/seam scope; 148 passed/1 skipped on the wider affected scope (incl. slow parallel byte-identical, no regressions). Full `gates.sh` + Phases 3 (10s slow-mo experiment) + 4 (2060 live probe) still open.

## LTX seamless continuation: stage-2 prefix freeze + sigma-0.45 + morph 2+2 (2026-10-02)

- Problem (user-verified): ltx25 multi-segment video has hard cuts at every segment joint (frozen-prompt Phase-0: within-SSIM 0.967-0.980 vs boundary 0.786-0.817; eyeball CONFIRMED).
- Root cause: `LTXImgToVideoInplace` freezes the 25-frame prefix through stage 1 via `noise_mask` (strength 1.0 → mask 0.0), but `LTXVLatentUpsampler.execute` ends with `return_dict.pop("noise_mask", None)` (pinned ComfyUI@`2f35f4a`, verified in-image at `/opt/comfyui/comfy_extras/nodes_lt_upsampler.py:62`), so the stage-2 refine (3-step euler from sigma 0.85, no mask) repaints the frozen prefix. `LTXVConcatAVLatent`/`LTXVSeparateAVLatent` propagate the mask as a NestedTensor pair (nodes_lt.py:800-815/846-851) — the upsampler pop is the ONLY drop. `SetLatentNoiseMask` builds 4D (B,1,H,W) masks only — unusable for video (needs 5D (B,1,T,1,1)).
- Fix (1) new torch-only local pack `Comfy/custom_nodes/ltx_mask_utils/` (`LTXPrefixFreeze`: LATENT + INT prefix frames → LATENT with 5D mask, first K frames 0.0 rest 1.0, range-validated; precedent: rotation_utils/frame_utils/video_utils), wired as node 57 into `video_ltx25.py` + `video_ltx23.py` prefix branches (samples ← upsampler out, K=(carry-1)//8+1 from prefix length, 19.video_latent rewired to [57,0]). Fix (2) `STAGE2_SIGMAS` 0.85-set → `0.45, 0.3, 0.15, 0.0` in both workers (profile hash covers it — old tapes refuse cleanly). Fix (3, finalize phase, open): morph-cut 2+2 — replace 2+2 frames around each segment joint with 4 FILM moments (count-preserving).
- Measured (fast 768x448/49f/9-carry harness, then production 1216x704/121f/25-carry proof): mask alone pin 8.7/9.2→2.8/2.7, seams 8.6/9.1→3.1/3.3; +sigma-0.45 pin 2.25/1.98, seams 2.32/2.30; wide-carry 17/25 identical (context width irrelevant); feathered mask (ramp) WORSE (5.4) — binary freeze stands; VAE roundtrip floor 1.1 (residual not VAE-bound). Production proof (121/96/96, K=4, T=16): pin 2.42/1.87, seams 2.42/1.94 — transfer verified. Morph 2+2 (user-approved winner, incl. fullres): mean steps at within-motion, motion preserved; plain 7-frame FILM splice froze motion (rejected); correct dissolve still showed seams.
- Tests: `tests/test_ltx_prefix_freeze.py` (graph-shape slim-safe + torch tensor tests via importorskip; PACK_DIR candidate search — host Comfy tree / VOYAGE_LTX_MASK_PACK env / /opt/comfyui, skip when absent); existing sigma/count pins updated (56→57 nodes). 62 passed/3 skipped on the ltx scope.
- OPEN: (a) pack delivery to `voyage-ltx` image — LANDED: stage-then-COPY bake (`build-ltx.sh` stages `Comfy/custom_nodes/ltx_mask_utils` to `worker/ltx_mask_utils/`, Dockerfile COPYs to `/opt/comfyui/custom_nodes/`, trap cleanup, smoke assert; `worker/ltx_mask_utils/` gitignored as build debris); image rebuilt, smoke passed, baked-load GPU-verified numbers-identical with no mount. (b) ltx23 GPU verify (wired, tests green, no live run yet). (c) finalize-time morph 2+2 — LANDED: new `voyage/augment_morph.py` (pair/recipe-sensitive joint keys, anchors A[-3]/B[+2], trims A[:-2]/B[2:], extensionless MPEG-TS pieces — MP4 byte-join decodes first-file-only so TS is required, CFR select+setpts trims, count-preserving, audio untouched) + `morph_joints`/`morph_interp_fn` on `run_durable_model_pass` (mids-seam bypass, morph assembly over drained intermediates, morph_s/morphs_done timings) + media.py native-branch morph (voyage.toml backend gate, film-gated, graceful concat fallback). `tests/test_augment_morph.py` 9/9 green (TDD red-first; stub interp returns valid PNG bytes via stdlib zlib builder). Production assembly GPU-verified with real FILM (129f count-preserved, seam max 2.16/2.53 mean 1.02/1.31 — matches eyeball-approved morph 2+2). Harness/monkeypatch scripts are ephemeral under `/tmp/opencode/phase1/` (never committed). No commits without per-commit approval.

## Slow-mo finalize Phase 3: true retime (no decimation) + timeline-accurate audio/SFX (2026-10-02)

- Problem (found by reading, not by GPU): the tensor branch presented the source*m-fps intermediate through a bare `fps={out_fps}` filter, which DECIMATES frames to hold the duration — slow-mo shipped the same wall-clock with fewer frames, and the stretched ACE takes (1.5x) mismatched it. Three more latent mismatches: (1) `audio_stretch = stretch if tensor else 1.0` armed on every default m=4/out=24 run (4x audio vs decimated video); (2) SFX `segment_sfx_bounds` walks the source timeline while `finalize_sfx_pass` probes the shipped (stretched) duration as its timeline; (3) parallel Thread B conditions SFX on a source-timeline reference concat.
- Pure helpers (`media.py`, `tests/test_slowmo_retime.py` TDD red-first): `slowmo_video_active(tensor, presentation_fps, stretch)` — tensor path AND explicit presentation fps (opt-in; default runs keep the legacy fps-filter timeline byte-identical) AND stretch != 1 (tolerance `SLOWMO_STRETCH_TOLERANCE = 1e-9`); `tensor_presentation_vf(..., slowmo)` — `setpts=<stretch>*PTS` prefix iff slow-mo (every FILM frame survives, timeline stretches), legacy string byte-identical otherwise; `tensor_path_armed(model_selected, weights_present, source_fps, needs_reencode, devices_available)` — one shared gate used by `finalize_run` AND the parallel pre-fork (a superset-stretch pre-fork renders takes for a timeline Thread A never builds, and its ledger no-op then blends the wrong takes silently). `media.finalize_run` uses `tensor_presentation_vf(slowmo=slowmo)` + `audio_stretch = stretch if slowmo else 1.0` (computed before the deferred branch); the gate refactor precomputes `tensor_devices` (probes only when knob+legs select) with explicit-None narrowing for mypy.
- Pre-fork rewrite (`finalize_parallel.py`): presentation-aware requested fps (mirrors Thread A's presentation-first rule), static `tensor_path_armed` prediction (`use_model_pass=None` mirrors the stored-default True; devices probed live), slow-mo-gated stretch — residual divergence only if devices flap mid-finalize (catastrophic anyway).
- SFX (`sfx_finalize.py`): `_scale_bounds_to_timeline` (identity within `BOUNDS_RESCALE_IDENTITY_TOLERANCE = 1e-6` returns bounds untouched — byte-identical legacy; stretched timelines scale uniformly so captions track retimed video), wired into `finalize_sfx_pass` after `segment_sfx_bounds`.
- CLI (`cli_finalize.py`): explicit `presentation_fps` forces sequential (Thread B's reference concat is source-timeline) — one line, always correct, slightly conservative; deferred sequential gate unchanged.
- Gates: ruff + format + PLR2004 + mypy strict clean on media/sfx_finalize/finalize_parallel/cli_finalize; 133 passed (slowmo + sfx + gpu-defaults + deferred + presentation + interp-multiplier + augment-config + e1). GPU experiment (fresh ltxv 2-seg run, finalize m=2/presentation 32, timing report) + 2060 live probe still open.

## Slow-mo Phase 3 experiment: ltxv-slowmo 5-seg run + deferred-continuity fix (2026-10-02)
- Run `output/ltxv-slowmo` (boba style, seed 0, ltxv 768x512@24, acestep experimental default, mmaudio large cuda:0, qwen Qwen3-8B cuda:1): seg0 83.6s (dir 44.7/vid 38.8/aud 0.1 — no ACE at commit), seg1 25.4s (prefetch hit), seg3 97.3s, seg4 71.5s. Commit audio 0.0-0.2s throughout (stub only).
- REGRESSION FOUND + FIXED (TDD): Phase 1's skipped audio swap removed the rebuild that derived `video_tail.mp4`, so every same-run segment went fresh (121f; seg1 proved it). New `audio_finalize.derive_conditioning_tail(segment, tail_frames)` + `deferred_tail_frames(backend)` (ltxv 25; causvid max(25, 4*(overlap-1)+1) with default overlap 3 = 25 — pins each worker's resume derive; short tails would adopt wrong) wired into the deferred branch in `supervisor._cover_audio` (own hunk only; concurrent scene_cut hunk untouched). New tests: 3 derive tests + count pins + branch test asserts the tail (16 passed incl. full file). Scoped gates clean.
- PROOF: seg1 tail manually re-derived (ffmpeg trim, 25f) -> seg2 fresh (121f, new process — see below) with seg2 tail auto-derived by the fix -> single invocation seg3 121f fresh + seg4 96f CONTINUED, no new "starting fresh" in video-worker.log.
- PRE-EXISTING (fixed 2026-10-04, see "Swansy cross-invocation continuity" below): a fresh `run` invocation never resumed the video worker from the latest tape (`_resume_video_worker` fired on retry/proactive-refresh only), so each invocation's first segment went fresh. Intra-run continuation is what the earlier fix restored; startup resume now covers the rest.
- Finalize `--interp-multiplier 2 --presentation-fps 32`: `final-slowmo.mp4` 1145f 1216x704@32 h264, 35.78s video + 35.71s AAC (580f/24.17s source -> 1.5x stretch; 4 seam mids included; chunk-joint hitches account the rest). Deferred takes rendered at finalize (take_0000/0001.wav + ledger). Timings: upscale_poll 208s, interp_poll 663s (FILM dominates), drain 7.4s, seam 4.2s (4 seams), audio_blend 62.8s (incl. ACE renders), final_encode 1.9s, SFX 50.5s (5 windows). VALID before finalize.

## Slow-mo Phase 4: 2060 live co-resident probe — llama sidecar + augment (2026-10-02, report only, no code)
- Probe (ephemeral `/tmp/probe4/driver.py`, never committed; `voyage-video` with ONLY the 2060 visible as cuda:0; host `nvidia-smi -i 1` 1s sampling): sidecar via `llama_server.start('/models')` (contract argv `-ngl 99 --ctx-size 4096 --cache-prompt`), one 226-token chat completion, then SRVGG upscale of a real ltxv-slowmo 768x512 frame + FILM pairs through the production `augment_worker` entry points.
- Measured (2060 = 6144 MiB): sidecar resident settles at 3248 MiB (~3.17 GiB, load 4.5s); 226-token completion +14 MiB (3262); SRVGG native-2x 768x512->1536x1024 peaks 3564 MiB total (own torch peak 196 MiB) = 58% — FITS comfortably co-resident. FILM full-res pair at 1536x1024 OOMs co-resident (only 283 MiB free at the fail; sidecar 3.2 GiB + SRVGG cache resident) — CONFIRMED IMPOSSIBLE, no chunk knob saves it (batch-halving bottoms out at one pair). Interp-first recipe (FILM at 768x512 + SRVGG tiled 2x) peaks 5460 MiB = 89% (own torch peak 1022 MiB) — FITS but tight, ~680 MiB margin.
- Verdict: sharing cuda:1 is viable ONLY with the enforced interp-first + tiled recipe (+ reduced chunk_frames) and a cuda:1 lease that does not exist today (no lock/mutex/flock for GPUs in tree — only the per-run state.json.lock and per-worker RPC _call_lock). Default staggered finalize-time model pass stays the safe path. Residuals: full-4096-ctx KV growth unmeasured (226 tokens added ~14 MiB; a full cache adds low-hundreds MiB — eats the recipe margin, not the SRVGG margin); AWQ 4B path (2.8 GiB peak) not re-probed, still the comfortable-share fallback per the read-only arithmetic.

## Slow-mo backend-agnostic: joint-audio atempo stretch + seam ordering pin (2026-10-02)
- Question: does the stack work for ltx25/ltx23, and is the stitch-then-interp order safe? Read-only sweep (concurrent worker files never touched): NO FILM/stitch/morph logic exists in any video worker (rg zero hits incl. the concurrent prefix-freeze diff; zero in DESIGN/LTX2/docs). The only stitch artifacts are the untracked prototype `Voyage/prodverify/carry09/` (blend-vs-discard experiment, someone's — not shipped); morph-cut 2+2 finalize wiring still OPEN. Ordering is structural, not conventional: the sidecar consumes only committed `segments/NNNNNN/video.mp4` + manifest, so any generation-time stitched frames enter 2x FILM as ordinary frames; seam joint key `H(A)|H(B)` is pair-sensitive (test-pinned) so either side's rewrite forks the joint dir.
- REAL gap found + fixed (TDD, user decision atempo-stretch over fail-loud): joint backends (ltx25/ltx23, empty takes ledger) hit the single-segment shortcut (verbatim copy) and the multi-segment plain-concat fallback in `build_final_audio`, both ignoring `stretch` — a 1.5x final would ship 1x audio under 1.5x video and `-shortest` would trim video frames. Deferred backends were already correct (takes render stretched; ltxv-slowmo proven 35.71s vs 35.78s). New `media_audio._stretched_fallback_audio` (single-graph concat→atempo chain; single input skips concat) + `_atempo_stages` (tempo split into the `[0.5, 2.0]` legal band, `_ATEMPO_MIN/MAX_TEMPO` consts; MediaError on non-positive) wired via a `retime` flag on the shortcut and `_fallback_or_raise`; identity within `STRETCH_IDENTITY_TOLERANCE` keeps the legacy path byte-identical. Pitch drops with the slowdown (standard slow-mo tradeoff, sync preserved); SFX needs no change (windows derive from the shipped stretched duration).
- Seam ordering pin (characterization, `tests/test_augment_seam.py::test_rerendered_side_never_reuses_stale_seam`): after B re-renders under a new checksum, the pass renders the seam exactly once under the NEW joint dir and the concat contains `[A, fresh seam, B']` — the stale dir (records intact on disk) is never referenced. Drain re-derives every seam dir from current adjacent source keys, so stale dirs are orphans by construction.
- Gates: ruff + format + PLR2004 + mypy strict clean on media_audio + both test files; 38 passed (slowmo + seam + deferred). Pitch-drop tradeoff and causvid-tail-never-live-proven noted as residual risks.

## Audio continuity: ltx25/23 music back to the ACE-Step planner (2026-10-02)
- Ear verdict on joint-audio experiments (frozen 128-BPM prompt, fast 49/40/40): every clip renders a FRESH bed from an empty audio latent (no cross-clip audio conditioning — only the video prefix is frozen), so key/tempo/texture restart per segment; tempos locked (130.8x3) but phase jumps at joints (beat-phase surprise 0.40/0.26 vs controls 0.25/0.17). Post-hoc beat-aligned overlap crossfade softened seams (0.26->0.18 at J1) but cannot fix different musical ideas per clip — user: "2 disjoint takes forced together with a fade; we want truly continuous audio".
- Decision (user §9 Q&A): switch ltx25/23 music back to the ACE-Step planner path (long caption-driven takes, repaint-only-on-caption-change + similarity gate + steer-and-accept — built for continuous mood; proven ltxv pairing pattern with sequential evict/rebuild residency). Reverses the JOINT_AUDIO_BACKENDS decision. Worker joint-audio files stay on disk (cheap, ignored).
- Implementation (6-part scope, all landed): models_ensure.py (JOINT_AUDIO_BACKENDS set + joint gate deleted — ACE required whenever audio.backend==acestep); config.py ltx25/23 rows audio fake/cpu -> acestep/cuda:0 (+ 4 stale comments); supervisor.py (JOINT_AUDIO import + _render_video joint block + _cover_audio param/branch + caller arg deleted); supervisor_commit_types.py RenderedVideo.joint_audio_path deleted; tests updated (commit_types pin; generate_ensure/generate/finalize_gpu_defaults x2/sfx_contract pairing pins renamed to ACE expectations). Zero joint_audio references in voyage/+tests. Pre-existing foreign files untouched.
- Open: GPU verify (ACE take render + commit on ltx25 with the upbeat prompt) + listen; morph-cut 2+2 video joints already land in finalize for ltx25/23 (separate entry).

## Slow-mo joint-audio wiring fix: ltx25 finalize passes stretch (2026-10-02)
- Symptom (live, `output/fett` ltx25 1216x704@24, 2 segs / 217f, finalize `--interp-multiplier 2 --presentation-fps 32`): video stretched true 1.5x slow-mo (428f @32fps = 13.375s) but music stayed 1x (8.98s), so the SFX mix gate failed loud (`sfx mix 10.15s drifts from music 8.98s`, exit 1, music-only kept).
- Root cause (`voyage/media.py`): the `else` (joint/non-deferred) branch hardcoded `audio_stretch = 1.0`, so `build_final_audio` took the plain-concat fallback and ignored `stretch` — even though `_stretched_fallback_audio` (atempo retime for exactly this case) already existed per the backend-agnostic entry above. Deferred backends were unaffected (stretch wired in their branch).
- Fix (one line + comment, own hunk): `audio_stretch = stretch if slowmo else 1.0` in the joint branch — takes-ledger path widens windows over `fps/stretch`, empty-ledger path retimes via atempo (pitch drops, sync preserved). Identity (no slow-mo) keeps the legacy 1.0 byte-identical.
- Proof (same box, GPUs idle): re-finalize (model pass + SFX stems fully cached) exit 0 → `output/fett/final.mp4` 428f 1216x704@32 h264, 13.375s video + 13.31s AAC (diff 0.06s < 0.6s gate), audio RMS -13.5 dB (real music+SFX mix, not silence).
- Incidental (no code change): the first finalize attempts hit two environment facts worth recording — (1) FILM interp at upscaled 2432x1408 OOMs the 6 GB 2060 even solo (~5.9 GiB need; `interp_first_for_small_device` covers only the legacy chunk path, not the durable sidecar path), so this finalize ran with only the 4060 visible (`CUDA_VISIBLE_DEVICES=0` → model pass + SFX sequential on cuda:0); (2) `voyage-ltx` carries no mmaudio stack, so SFX finalize must run in `voyage-video` (run.sh picks ltx by video backend — override `VOYAGE_IMAGE` or invoke docker directly for the SFX leg).
- Gates: ruff + format + mypy strict clean on `voyage/media.py`; 45 passed (slowmo + media_audio + finalize_fastpath + sfx suites). Full suite last green except 2 known-foreign (citation gate vs pruned issues dir; TUI Pilot load flake passing in isolation) — both outside this scope, untouched per §9.

## Native-resolution finalize: upscale only on genuine lift (2026-10-02)
- Policy (user directive): ltx25's native output (1216x704) is considered sufficient — by default the model pass must NOT upscale it. Upscale runs only on a genuine resolution lift (either output axis above source); interpolation then runs at source resolution, which also fits small GPUs (FILM at 2432x1408 OOMs the 6 GB 2060 even solo; at 1216x704 it peaks comfortably inside).
- Implementation (TDD red-first, 6 tests): pure `media.upscale_factor_for(source_w, source_h, out_w, out_h)` (1 when the output box fits the source box, else the validated 2; non-positive geometry fails loud) wired into the durable-path call in `finalize_run`; `_default_upscale_pngs` in `augment_upscale_poller.py` handles factor 1 as a CPU file copy (same names/bytes, no torch/PIL/model — provable in the slim test image, which carries neither; same-file (decode-staged) paths skip the copy; factors outside (1, 2, 4) fail at the seam). Plan hash already keys on the factor, so factor-1 plans get fresh ledger dirs with no collision. Identity behavior (genuine lifts) is byte-identical.
- Proof (fett, idle GPUs, both cards visible as designed): refinalize `--interp-multiplier 2 --presentation-fps 32` → `output/fett/final-native.mp4`, exit 0, wall 144s. Per-step timings from the finalize metrics event: upscale_poll 1.77s (decode + CPU copy + chunk encode, zero GPU model work), interp_poll 135.49s (7 FILM chunks at native 1216x704 on cuda:1, no OOM), drain 1.92s, concat 0.10s, seam 0.59s (1 joint), audio_blend 49ms (music rebuild via atempo stretch), final_encode 0.80s, sfx_pass 0.26s (stems cached from the upscaled run). Output: 427f 1216x704@32 h264, 13.34s video + 13.37s AAC (diff 0.03s; exactly 1 frame under the 428-frame intermediate — `-shortest` trimming video to the audio track, correct), RMS -13.5 dB real mix. Contrast with the upscaled recipe on the same run: ~497s interp on the 2060 with OOM risk (lucky pass) for a 2432x1408 intermediate that the presentation vf downscales straight back to 1216x704.
- Gates: ruff + format + mypy strict clean on `voyage/media.py`, `voyage/augment_upscale_poller.py`, `tests/test_upscale_factor.py`; 6 new + 36 neighboring (upscale poller, finalize wire, slowmo, sidecar, interp poller) green.

## Audio prompt direction: always-on dark/experimental music captions (2026-10-04)
- User directive: every ACE music caption must convert the general prompt into music-specific language (instrumentation, harmony, melody) with a dark/experimental touch, always, on top of the provided style.
- Grounding: DIRECTOR_SYSTEM_PROMPT (voyage/director.py) already mandated music captions "in musical terms (instruments, harmony, melody, texture)" from charter + general prompt — only the dark touch was missing. Deterministic fallback built "slow ambient electronic ..." from the same template.
- Mechanism (user §9 Q&A: director-source + exempt pins): dark/experimental direction added to the system prompt's caption doctrine (covers llama+qwen) + deterministic template rewritten to dark experimental electronic (covers deterministic/fallbacks, bit-stable holds preserved); explicit CLI --music-caption pins pass through untouched. Stable anchor wording also damps planner repaints via the similarity gate → longer takes → more continuity.
- Tests: tests/test_director_dark_direction.py (system-prompt markers, deterministic dark markers + derivation + bit-stability); no existing test pinned the old wording (all substring/self-referential/drift-based). Ruff + format clean; 35 passed with neighboring director suites.

## SFX venv in voyage-ltx (2026-10-04)
- Finalize with default SFX on ltx25 failed: `sfx_finalize.py` spawns the MMAudio worker without an `executable=` override, so it used the voyage-ltx interpreter, which never had the MMAudio stack (`No module named 'mmaudio'`). Pre-existing gap (SFX-on-ltx finalize never ran; joint-audio era paired SFX too but nobody exercised it).
- Fix (user §9 Q&A: third venv, mirrors ACE precedent): `/opt/venvs/sfx` in `worker/Dockerfile.ltx` (verified-fetch MMAudio@974010a + no-deps editable install + torch 2.14 trio cu130 + open_clip/librosa/torchdiffeq/accelerate/colorlog/soundfile/einops/safetensors/pydantic, no transformers; missed omegaconf+easydict+ftfy on first pass — caught by the new smoke assert, fixed); chown + `VOYAGE_SFX_PYTHON` env; `executable=` plumbing in `sfx_finalize.py`; build-ltx.sh smoke assert (mmaudio submodules + FlowMatching + OmegaConf). voyage-ltx now carries THREE venvs (director + ACE + SFX).

## CLI-is-config: voyage.toml deleted, effective config lives in run_manifest.json (2026-10-04)
- Problem (user run): `generate --name jango --interp-multiplier 2 --presentation-fps 32` left `output/jango/voyage.toml` with `interp_multiplier=4` and no `presentation_fps` — flags applied only in-memory (`apply_draft_overrides` in `cmd_generate`), the stored file was the stale `default_config_toml` preset. Same run FAILED seg0 on ACE-Step OOM vs the resident ltx25 worker (joint-audio era contention).
- Decision (user §9 Q&A): full TOML removal — CLI-only config with expanded/documented/defaulted flags; the effective config (preset + every flag override) is saved into `run_manifest.json` (`effective_config` model dump + argv), `state.json` stays progress-only; clean break (no legacy importer, fail loud); `init` verb deleted (generate+run only); flag judgement to the implementer.
- Implementation: `config.py` gains `preset_config(run_id, style, seed, video_backend, director_backend, director_device)` and loses `default_config_toml`/`load_config`/`_toml_basic_string`; `persistence.py` gains `create_run_dir()` + `read_effective_config()` (fail loud, no fallback) and `build_manifest` writes argv + full effective dump; `cmd_generate` builds preset→overrides→`create_run_dir` directly (run-id/style/seed/backend guards moved with it); `_load_run` reads the manifest; `init` parser/verb deleted; ~70 test modules migrated (`default_config_toml`→`preset_config`, `load_config`→`read_effective_config`, TOML asserts→model asserts, `cmd_init`→`create_run_dir`/generate); `run.sh`/`qualify.sh` sniff `run_manifest.json` (`effective_config.{video,audio,sfx}.backend`) instead of TOML; docs (README/OPERATIONS/ARCHITECTURE/SFX/PROMPTING) and the cheat sheet use `generate --name`; `ARCHITECTURE.md` GPU time-sharing section corrected to the deferred truth (old "ltx25/ltx23 joint audio skips the swap" contradicted the deferred code — every streaming backend is deferred, only fake commits audio inline).
- Gates: 2045 passed / 17 skipped, 1 failed = `test_issue_citation_gate` only (pre-existing/foreign — cites live in concurrent agents' worker files/Dockerfiles against the pruned issues dir, none from this track).
- Open: jango reset+retry on idle GPUs — LANDED 2026-10-04: `rm -rf output/jango`, retried the user's exact flags + `--duration 8s --seed 31984046` (2 × 96f segments). Both segments committed (seg0 121f fresh incl. 25f prefix, seg1 96f continued — deferred stub+tail, commit audio 0.0s, no ACE OOM); director evolved prompts across a drift (Void-Flux Sector → The Inverted Flux, both novel); manifest carries `effective_config` (`interp_multiplier=2`, `presentation_fps=32`, ltx25 1216x704, acestep) + full argv. Finalize exposed a REAL bug: `spawn_ace_render_fn` built `SubprocessWorker` with no `executable=` override, so on voyage-ltx it used the image interpreter → `ModuleNotFoundError: No module named 'acestep'` in ace-finalize.log (the ACE venv existed but was never selected — unlike the SFX fix which wired `VOYAGE_SFX_PYTHON`). One-line fix mirroring the SFX precedent: `executable=os.environ.get("VOYAGE_ACESTEP_PYTHON")` (verified present in the image + venv imports `acestep.handler`); scoped gates green (ruff+format+mypy+17 deferred-audio tests). Refinalize exit 0 → `output/jango/final.mp4` h264 1216x704@32, 425f ≈ 13.28s (217f@24 = 9.04s source → 1.47x ≈ 1.5x slow-mo), AAC RMS -20.3 dB real mix; `audio/take_0000.wav` (ACE music) + `audio/sfx/w0000+w0001.wav` (separate SFX bed) both rendered at finalize. Follow-up (not this track): manifest summary fields `committed_segments: 0` + `timeline.final_width/height 768x432` go stale after creation (progress lives in state.json, geometry in final_geometry) — harmless but confusing; schema cleanup is a separate decision.

## CLI `--name` shortcut + auto-refinalize on extend (2026-10-04)
- User request: `voyage run --name jango` must resolve to the default output path, and extending a run must re-finalize.
- `resolve_run_ref()` (voyage/cli_paths.py, re-exported via voyage.cli): `--name jango` → `output/jango` (cwd-relative, same root as `generate`'s default and the TUI); both flags or neither → exit 2; non-flat names rejected (existing traversal guard). All run verbs carry it (run/status/pause/resume/stop/validate/finalize/sfx/soak/inspect + benchmark); `run.sh` sniff loop mirrors `--name`/`--name=` → `output/<name>` symmetrically with `--run` (issues 037/069 contract).
- `run` auto-refinalizes `final.mp4` via `maybe_refinalize()` (cli_run_ops) iff new segments committed (empty commit list or `--no-finalize` → skip); sfx/augment/skip-bad/console flags forward into the finalize call (same namespace shape as `generate`). `generate`'s inner `cmd_run` passes `no_finalize=True` (it finalizes explicitly after validate — no double assembly). `resume` flips status only (commits nothing → gate never fires, wired through the helper so the rule stays in one place).
- Tests: tests/test_run_name_refinalize.py (resolver + all-verb parser pins + refinalize/skip/no-finalize/resume behavior with faked Supervisor/finalize seam); ruff + format clean; 180 passed with neighboring CLI suites.

## llama-server GPU pinning: sidecar masked to the director device (2026-10-04)
- User report: two `llama-server` lines in nvidia-smi during generation, spread over both GPUs — must run strictly on the 2060. Diagnosis: a single process (PID 2140187) holding 2256 MiB on GPU 0 + 1302 MiB on GPU 1 — nvidia-smi lists one line per (GPU, PID), so the "two processes" were one straddled server. Root cause: `llama_server.start()` passed no device constraint (no `CUDA_VISIBLE_DEVICES`, no `--tensor-split`) and `Popen` inherited both GPUs (`--gpus all`), so llama.cpp's default split the ~3 GiB Q4_K_M across both cards — stealing ~2.3 GiB + compute from the 4060 Ti mid-render (14555/16380 MiB at probe time). The default director device was already cuda:1 everywhere (config/generate/CLI help); only the spawn path ignored it.
- Fix: `llama_server.visible_devices_for(device)` (pure: `cuda:N`→`N`, bare `cuda`→`0`, anything else→None) + `start(..., visible_devices=None)` passing `env={**os.environ, CUDA_VISIBLE_DEVICES: ...}` to Popen only when set (None = today's exact inheritance, zero behavior change); `_start_llama_sidecar` forwards `visible_devices_for(config.director.device)`. Masking beats `--tensor-split`: the process never opens a context on the other card. `backend=llama` + `device=cpu` stays unmasked (a non-cuda device carries no index; documented in the helper — the sidecar is GPU-only by design).
- Tests: 6 new pins in tests/test_llama_sidecar.py (TDD red-first: 4 failed pre-fix), incl. a seam fix — the Popen stub never recorded kwargs, so the env assertion could never observe anything. 39 passed in-file.

## CLI two-verb redesign, step 2: `configure` verb + manifest rename (2026-10-04)
- `configure <NAME>` (voyage/cli_configure.py) inits `output/<NAME>/manifest.json` (full effective config + planned segment count + final_video/skip_bad/no_sfx policy) or updates it touching only provided flags. Creation requires style + one of --segments/--duration (--duration converts via segments_for_duration); seed randomizes when omitted. Full option set mirrors generate (backend/draft/director/generation-overrides/sfx/augment/final-video/skip-bad/no-download/force) with positional NAME.
- `run_manifest.json` → `manifest.json` (paths.MANIFEST_FILENAME), clean break: old runs fail loud (no migration). Segment-level `manifest.json` (per-segment dir) and models-cache `manifest.json` are different paths, untouched. run.sh/qualify.sh sniff + ARCHITECTURE/OPERATIONS/tests literals updated.
- Update rules: backend change on a committed run refused (exit 2); shrinking the count trims overflow segment dirs + recomputes state counters from surviving DONE manifests + drops takes covering at/after the new end (with wav files); style/caption changes store only (director drifts future segments). Configure is offline-safe (no CUDA-stack gate; slim image configures CUDA runs).
- Tests: tests/test_configure.py (7: rename pin, create, validation, provided-only update, shrink+recount, backend guard, legacy ignore); EXPECTED_VERBS gains configure. 130 passed with fallout suites; ruff + format clean.

## `configure --from <NAME>` run inheritance (2026-10-04)
- `configure <NAME> --from <OTHER>` (create only): base is the source manifest's effective config via `read_effective_config` + `read_manifest` (missing/corrupt/legacy-nested fails loud, exit 1; `--from` on an existing manifest refused, exit 2). Inherits `style` (so `--style` is optional with `--from`) + all tuning sections + planned `segments` as defaults; `name` always fresh, `seed` fresh-random unless `--seed`, `final_video`/`skip_bad`/`no_sfx` flag-driven. Explicit flags override via the existing `resolve_config` path — with `--from`, absent `--backend`/`--director`/`--director-device` keep the source instead of resetting to `ltx25`/`llama`/`cuda:1` (the `backend=` kwarg now rides `resolve_config`, idempotent on the preset path). `--segments`/`--duration` override the inherited count; both-passed still errors (no silent inherit). Non-stored pins (`music/video_caption`, `sfx_caption`/`sfx_workers`) are not inheritable by construction.
- Tests: 4 new in tests/test_configure.py (TDD red-first: copy, override-wins, missing-source, existing-target-refused); gates ruff + format + mypy strict clean on touched files, full suite 1797 passed / 18 skipped.

## Background model-pass pre-warm during generation (2026-10-04)
- User request (verbatim): "Proceed. Be sure that a resumed generation or finalization can trivially resume unfinished interpolation/upscaling work. Generation would always run that work in the background while finalization would wait for it to complete."
- Architecture (additive, no finalize change — `run_durable_model_pass` already polls to completion): new stdlib-only `voyage/augment_background.py` — `resolve_background_plan()` derives the finalize-equivalent plan (target box/fps from `config.video`, floors from `config.augment`, `plan_augmentation` + `upscale_factor_for` + integer `source_fps_key`, `weights_key_for` over both legs, device from `model_pass_devices()`), `prewarm_once()` runs one upscale sweep then one interp sweep (staggered, same order/keys as finalize so every background chunk is a finalize ledger hit), `BackgroundPrewarm` single daemon thread (condition-variable wakeups, `idle_fn` gate, exceptions swallowed). Returns None when moot: knob off, models_dir unset, either leg absent (durable path needs both — partial legs keep the legacy flow with nothing resumable), no device visible, no committed segments, unprobable source. `probe_segment_source()` (ffprobe via `media.probe` + `_probe_video_fps`/`_probe_video_geometry`) is the injectable seam so the suite stays ffmpeg-free.
- Supervisor wiring (all best-effort, never fails a commit): `self._background` created in `start_workers` (after the prefetch executor) with `idle_fn = not _prefetch_in_flight() and not _director_probe_blocked()` — the director always wins `cuda:1`; stopped first in `stop_workers`; `start()` queues an initial sweep (covers generation resume — ledgered chunks skip, partials re-render); `_notify_background_committed()` after `_commit_segment` (covers the fresh segment; orphan-adoptions are swept by the next notify). Finalize needs no change: it polls the same ledger to completion, so pre-warm only shrinks it to drain + concat + encode.
- Tests (TDD red-first): `tests/test_augment_background.py` (12: knob-off, plan==finalize derivation, poll kwargs incl. shared weights_key + integer fps key + multiplier, no-segments skip, single-leg skip, no-device skip, probe-failure skip, real-ledger resume incl. interp skip on re-poll, thread notify/idle-skip/error-swallow/stop-without-start). Verified green: 12 new + 31 neighboring (upscale/interp/sidecar/drain/finalize-wire/supervisor-lifecycle/commit-types/hardening/recovery); ruff + format + mypy strict clean on `voyage/augment_background.py`, `voyage/supervisor.py`, `tests/test_augment_background.py`, `tests/conftest.py`.
- Foreign breakage (not this track, per §9 untouched): concurrent agent's in-flight CLI/persistence refactor leaves `voyage/cli_configure.py` unimportable (`effective_config_digest` gone from `persistence.py`) and drops `ExperimentalConfig` from `config.py` — `voyage doctor`, the image build, conftest collection, and ~33-41 CLI-surface tests fail at HEAD. Own scope re-verified green in isolation; full `gates.sh` blocked on their tree (incl. the pre-existing DESIGN-ref ratchet miss on their `cli_configure.py`).

## Augment sidecar resume hardening (2026-10-04)
- User request (verbatim): "Ensure that the augment_worker can always pick up where it left in the event of an interruption/resume/crash/corruption. Then review, refactor and commit."
- Two real gaps found by live-code audit of the durable path (own scope: `voyage/augment_sidecar.py` + upscale/interp pollers + tests only; vendored `workers/augment_worker.py` nets and `augment_drain.py` fail-loud behavior left alone): (1) `load_chunk_ledger` raised `JSONDecodeError` on a torn mid-append tail line, killing the whole poll — now skips undecodable lines (the interrupted chunk has no record and re-renders). (2) Ledger-done-but-output-missing deadlock: both pollers skipped on ledger alone, so deleted/corrupted outputs skipped forever while drain failed `rerun interp poller`; interp additionally crashed fail-loud (`MediaError`) when the upscaled input vanished. Fix: new sidecar `chunk_output_complete(output_dir, expected)` output-truth helper (present dir + exactly N non-empty `frame_*.png`); upscale re-adds ledgered-but-incomplete chunks to missing; interp treats ledgered-but-incomplete upscaled inputs as waiting (upscale heals them) and re-adds ledgered-but-incomplete interpolated outputs to missing. Crash between `os.replace` publish and ledger append already re-rendered correctly (unledgered output dropped); partials pruned each poll; concurrent background+finalize append interleaves tolerated (union + last-wins).
- Tests (TDD red-first): `tests/test_augment_resume_hardening.py` (5: torn tail skipped; upscale re-render on deleted/short output with ledger intact; interp re-render on deleted output; interp waits-then-heals when upscaled output deleted) — all 5 failed pre-fix for the right reasons. Gates on own scope green: ruff check + format check + mypy strict clean on all 4 files; 62 pytest passed across sidecar/upscale/interp/hardening/drain/finalize-wire/seam/morph/background suites.

## CLI two-verb redesign, step 3: `generate` reconcile rewrite (2026-10-04)
- `generate <NAME>` (voyage/cli_generate.py rewritten, positional NAME only + console flags) reconciles `output/<NAME>/` against the manifest: missing manifest → exit 2 + `configure` hint; `state.committed_segments` is the source of truth — numeric segment dirs at/after the committed count plus `*.partial`/`*.tmp*` are deleted (uncommitted work assumed torn); `remaining = manifest.segments − committed`; fully done (validate clean + `final.mp4` covers `state.timeline_frames`) → "nothing to do", exit 0; else Supervisor runs exactly `remaining`, then validate + `_finalize_run_dir` under manifest policy (`final_video`/`skip_bad`/`no_sfx`), honoring the same CUDA gate, sfx, augment, and console seams `run` uses. One frame-accurate completion rule everywhere: `final.mp4` frame count == `state.timeline_frames`.
- CLI error-message convention: numeric/option validation failures print `error: invalid numeric override: ...` (matches `run`; `configure` aligned to it). `generate` parser takes no run options (all moved to `configure`); `AugmentConfig`/`_add_augment_args` now shared by configure/run/stop/finalize only.
- TUI Generate button = configure-then-generate in the worker thread (form owns the plan, manifest persists it, generate runs it); CUDA-gate and model-ensure failures surface on the form via the existing error path (no stuck view).
- Test migration (41-file fallout from the flat-manifest + two-verb changes): old create-flow `generate` tests deleted or rewritten to configure-then-generate; `inspect`-stage expectations dropped (chain removed in the manifest/state simplification); poulah-effective augment defaults pinned (interp 2, presentation 32); TUI Pilot tests mock `cmd_configure`/`ensure_models` at the `voyage.cli` seams. Gates: ruff + format + mypy strict clean on all touched files; full suite 2064 passed, 18 skipped, 3 foreign failures (issue-140 supervisor frame-view tests orphaned by the simplification; citation gate on the pruned issues dir — both untouched per §9).

## Pre-warm VRAM gate + healable SFX-shortfall filter (2026-10-04)
- User report: `configure poualh --segments 2` + `generate poualh` committed seg1 (121f ltx25) then failed with (1) 25x CUDACachingAllocator expandable-segments warnings on device 1 (2060, ~13MB free), (2) INVALID `sfx coverage 7.44s short of timeline 10.08s` -> abort before finalize, (3) `terminate called without an active exception`. Acceptance: resume generation with zero errors/warnings/exceptions.
- Fix 1 (sfx): SFX renders only at finalize, so EVERY resume-generate with sfx enabled + a pre-existing ledger trips the pure-shortfall line — which finalize heals via `render_sfx_bed` (cache-hits old windows, renders new ones; failures raise). New `sfx_finalize.is_healable_sfx_shortfall` (regex anchored to the exact `%.2f` format line) + `cli_generate._pre_finalize_errors` filtering that line at both pre-finalize gates, only when the SFX pass will run (no_sfx/fake-backend keeps it hard). Gaps/missing/unreadable/escapes stay fatal. `voyage validate` still reports the shortfall (read-only truth; it heals at finalize).
- Fix 2 (pre-warm): `BackgroundPrewarm` ran torch sweeps in-process on cuda:1 while the llama sidecar permanently holds ~5GB of the 6GB 2060 (`idle_fn` gates compute, not memory). New `augment_background.device_free_gib` (nvidia-smi per-index query, never-raising, unknown -> allow) + `PREWARM_MIN_FREE_GIB = 3.0`: `prewarm_once` skips the pass when headroom is short — finalize covers the same ledger keys later.
- Fix 3 (terminate): `stop()` joined the daemon 5s then abandoned it mid-sweep -> CUDA teardown race at exit. The VRAM gate removes the trigger (sweeps never start contended); residual narrowing via `should_stop` between the upscale/interp sweeps (default driver path only — custom 2-arg `prewarm_fn` signatures unchanged).
- Tests (TDD red-first): `tests/test_generate_sfx_shortfall_gate.py` (4: predicate matches a real ledger shortfall, rejects hard errors, gate filters only the shortfall, no_sfx/fake keeps it hard) + 3 new pre-warm pins in `tests/test_augment_background.py` (skip-when-full, run-with-headroom, unknown-allows) + `device_free_gib` stub pinned in the 2 pre-existing sweep tests (else box-dependent). Green: 19 new/pinned + neighbors; ruff + format + mypy strict clean on all 5 touched files. Full suite: 2058 passed; 9 failures all foreign/flakes (5 TUI + recovery pass in isolation under load; 2 issue-140 + citation gate fail on untouched files, pre-existing on HEAD).
- Verify (2026-10-04, idle GPUs, fix committed as 189ebfb): `configure poualh --segments 2` exit 0 + `generate poualh` exit 0 -> `final.mp4` 48.5 MiB, 14.906s, h264 1216x704@32 + AAC; console log 2 lines, zero warnings/errors/exceptions (no expandable_segments, no INVALID, no terminate); metrics `finalize_completed` + `sfx_pass_completed` windows 2; ledger gained w0001 [7.0,14.9s) — the exact shortfall window healed at finalize as designed. Acceptance met. Observation (not acceptance-blocking): metrics show TWO finalize pairs ~13 min apart (first did the real work — 4 interp chunks + 35s SFX render; second fully cached — 0 interp + 0.29s SFX), matching the two `_pre_finalize_errors` call sites — possible redundant second finalize in one generate run, worth a look on the next pass. State rests PAUSED (unchanged by successful generate).

## Presented-frames coverage gate (2026-10-04, redundant-finalize fix)
- The two `_pre_finalize_errors` call sites are exclusive branches (one generate finalizes at most once) — the poualh double-finalize was revisits, not one run. Root cause: `_final_covers_timeline` compared ffprobe presented frames (476) to `state.timeline_frames` (242 source frames), never equal under interp m=2 + 1.5x slow-mo, so the 'nothing to do' gate could never fire: seg-1 era #5 real then #7/#9 fully cached; seg-2 era #16 real then #18 fully cached 805s later (each cached run still re-drained/re-encoded ~1s and rewrote final.mp4).
- Fix: `persistence.record_final_coverage` stamps `{segments, presented_frames}` into the manifest (extra keys ignored by `read_effective_config`; `build_manifest` never emits it so every `configure` wipes = conservative invalidation free); `media.presented_frames` is the one shared ffprobe shape; `cli_generate._final_is_fresh` compares presented-against-presented; `cmd_finalize` records after full success only (SFX failure returns early, keeping music-only finals re-finalizable); `finalize_run`/`run_parallel_finalize` gained optional `invoker` ('generate' via `_finalize_run_dir`, direct finalize leaves None) recorded on `finalize_completed` for forensics.
- Tests (TDD red-first): new `tests/test_final_coverage.py` (10: record round-trip + config tolerance, build_manifest carries none, 6 gate cases, invoker recorded/default-None); `test_generate_reconcile.py` noop seeds coverage (48 native frames, same contract) + 3 patch targets fixed to `voyage.cli_finalize.cmd_finalize` (foreign breakage from the two-verb split — `voyage.cli` no longer exposes it). Green: 25 across coverage/reconcile/fastpath; ruff + format + mypy strict clean on all 7 touched files. Full suite: 1774 passed, 44 failed — all foreign two-verb-migration fallout (deleted verbs -> SystemExit 2, deleted `tui`/`cli_*` modules -> ImportError; spot-verified `invalid choice: 'validate'`; no exact-dict assertions exist on the touched event/manifest shapes). UNCOMMITTED (per-commit approval required).

## CLI two-verb redesign, step 4: delete 12 verbs + TUI (2026-10-04)
- `voyage` is now exactly two verbs: `configure` (plans `output/<NAME>/manifest.json`) + `generate` (reconciles + renders + validates + finalizes, with approved `--segments`/`--duration` plan-extension shorthand). Deleted: `run`, `status`, `pause`, `resume`, `stop`, `validate`, `finalize`, `sfx`, `benchmark`, `soak`, `inspect`, `models`, `doctor` parsers + the bare-launch TUI (prints help, exit 2). Mid-run control is Ctrl-C (rests PAUSED; re-`generate` resumes); monitoring is file-based (`state.json`, `logs/metrics.jsonl`, `final.mp4`).
- Modules deleted: `tui.py`, `tui_state.py`, `cli_run_ops.py`, `cli_status.py`, `cli_observe.py`, `cli_scoreboard.py`, `cli_inspect_metrics.py`, `cli_models.py`. Libraries stay: `scoreboard.py`, `bench.py`, `doctor.py` (probe), `models_ensure.py`, `console.py`, `cli_validate.py` (`validate_run`), `cli_finalize.py` (`cmd_finalize` minus `cmd_sfx`, imported directly by `generate`). Pruned dead path helpers (`_run_dir_arg`, `resolve_run_dir`, `_effective_run_id`); moved rotation-tolerant metric readers to `logrotate.py` (`read_all_metric_events`, `last_commit_stages`). Seams re-homed: `check_ffmpeg`→`doctor`, `check_free_space`→`media`, `_torch_available`→`cli_planning`.
- `textual` removed (pyproject + lock + worker Dockerfiles); build gates + CMD are now `--help` probes. `qualify.sh` migrated to configure+generate (run dir must be `$PWD/output/<name>`). README + docs tree rewritten for two verbs.
- Tests: verb/TUI-only files deleted (`test_tui*.py`, `test_run_name_refinalize.py`, `test_cli_scoreboard.py`, `test_cli_inspect_metrics.py`, `test_inspect_metrics_fps_029.py`, `test_cli_split.py`, `test_cli_benchmark_sfx_augment.py`); mixed files migrated (parser pins → configure, seams → home modules, status/benchmark/soak/sfx/models validations → library calls or deleted). Gates: ruff + format + mypy strict clean; full suite green.

## `generate --segments/--duration`: additive plan extension (2026-10-04)
- User request: `generate <NAME>` accepts `--segments N` / `--duration D` (mutually exclusive, configure precedent — both together is exit 2) meaning *add* to the existing video: new plan = manifest segments + added, written back in place before reconciling. `--duration` reuses the configure path (`parse_duration` human format at argparse + `segments_for_duration` round-up over steady-state `_frames_per_segment`, so output never runs short); non-positive `--segments` is exit 2. Everything else in the manifest (config, finalize policy, `final_coverage`) is preserved — the coverage stamp goes stale on its own once new segments commit, so the next finalize re-runs correctly.
- Impl: `cli._add_generate_parser` flags + `cli_generate._extend_plan` (returns added count, None on error with the message already printed; `getattr` defaults so old namespaces without the flags read as neither); `cmd_generate` validates the stored plan first, then extends, then `planned += added`. Docstring step 2b added.
- Tests (TDD red-first, 6 failed pre-fix): new `tests/test_generate_extend.py` (8: segments add, duration round-up 2.1s→2 on the 2.0s/segment fake backend, exact-boundary 4.0s→2, both-flags exit 2 with manifest untouched, 0/-1 exit 2, coverage stamp preserved, no-flags plan untouched). Green: 8/8 + 34 neighbors (reconcile/coverage/configure); ruff + format + mypy strict clean on all 3 touched files. UNCOMMITTED (per-commit approval required).

## Manifest stale-field removal (2026-10-04)
- User question ("why does the manifest carry Qwen3-8B when the director uses Qwen3.5?") → audit → removal. Three distinct Qwen models were in play: `DirectorConfig.model_id` (Qwen/Qwen3-8B, legacy bf16-CPU default, ignored on every live path — default backend `llama` serves the Qwen3.5 GGUF via the loopback sidecar before `model_id` is read, and the qwen+cuda path substitutes the AWQ id anyway), `inspector_model_id` (Qwen/Qwen3.5-9B, Phase-5 VLM inspector only — the supervisor never calls the inspect op, never the decider), and the actual default decider (bartowski/Qwen_Qwen3.5-4B-GGUF Q4_K_M via the sidecar).
- Removed from `DirectorConfig`: `model_id` (no CLI setter — dead override), `embedding_model_id` (supervisor never sends it in init/embed payloads; worker keeps its hardcoded default), `inspector_model_id` (supervisor never calls inspect; worker keeps its hardcoded default). Supervisor decide payload drops the `model_id` key. Worker payload keys stay (standalone/back-compat, already default-tolerant).
- `build_manifest` strips the in-memory-only caption pins (`video.video_caption`, `audio.music_caption`): documented never-persisted but leaked via `model_dump`, which would freeze every future segment to one caption on resume. CLI flags + `resolve_config` pins + supervisor reads unchanged — only the manifest write drops them.
- Live director knobs kept: `backend`, `device`, `llama_endpoint`, `temperature`, `max_new_tokens`, `enable_thinking`.
- Tests (TDD red-first, 3 failed pre-fix): new `tests/test_manifest_no_stale_fields.py` (4: no stale director keys, no caption pins, live knobs kept, round-trip drops pins); `test_director_device.py` stale `model_id` assertion removed. Gates: ruff + format + mypy strict clean; full suite 1810 passed / 17 skipped (two parallel flakes on the first run passed in isolation and on re-run). UNCOMMITTED (per-commit approval required).

## Swansy cross-invocation continuity: startup resume in run_segments (2026-10-04)
- User report: extending `output/swansy` (ltx25, 1216x704@24, 96 novel frames/segment) segment-by-segment hard-cut at every cross-invocation joint (000000->000001, 000001->000002 each committed 121f = a fresh LTX25 window), while same-batch joints continued (000002->000003 committed 96f = 121f minus the 25f frozen prefix). All extensions must be continuous regardless of batching.
- Root cause (metrics.jsonl + video-worker.log forensics): `run_segments` resumed the video worker between same-batch commits (mid-batch restart+tape resume for `VIDEO_BACKENDS_NEEDING_FRESH_SESSION={'ltx25'}`) but never at startup, so every new process started with an empty session tail (`has_tail=False` -> fresh 121f render). Together = tape resume; alone = fresh.
- Fix: `run_segments` resumes streaming backends from `_latest_recovery_tape()` once after workers start (fresh process needs resume only, no restart), targeting the pending `next_segment_number`; no tape (first segment) is a no-op inside `_resume_video_worker`; fake (non-streaming) never resumes. Startup-resume failure rests the run at FAILED like in-loop failures (fail loud — degrading to fresh would silently reintroduce the cut). New `video_startup_resume_failed` metric event.
- Tests (failing-first, 1 failed pre-fix): new `tests/test_video_continuation_across_invocations.py` (3: streaming resumes with the tape path for the pending segment, no-tape sends no resume, fake never resumes even with a tape). Gates: ruff + format + mypy strict clean; full suite 1813 passed / 18 skipped serial, plus one parallel-only `test_issue_152_single_graph` flake that passes in isolation and with neighbors.

## `configure --low-definition / --high-definition` definition tiers (2026-10-04)
- User request: `voyage configure` gains `--low-definition` / `--high-definition`, configuring the lowest/highest native reasonable resolution for the currently selected backend (default backend when none configured); fresh creates without either flag implicitly use high.
- Map (`DefinitionTier` + `DEFINITION_TIERS` in `voyage/config.py`, applied by `resolve_config(definition=...)` AFTER the backend preset so `--backend X --low-definition` always yields X's low): fake 512x288/768x432, ltxv 512x320/768x512, causvid 832x480 both tiers (fixed geometry — flags are accepted no-ops), ltx25/ltx23 768x448 (`ltx25-448p`/`ltx23-448p`, latent (1,128,16,12,7)) / 1216x704 (`ltx25-704p`/`ltx23-704p`). High tier is byte-identical to the old `BACKEND_REGISTRY` rows. Low 768x448 is /64-clean (S24 ladder: 512x320/640x384/768x448 all PASS).
- Medium tier (`voyage configure --medium-definition`, Track C of the low/medium/high change): 1024x576 everywhere except causvid 832x480 (fixed geometry — all three flags are accepted no-ops there). Profiles `fake-576p` / `ltxv-576p` / `ltx25-576p` / `ltx23-576p` (causvid keeps `causvid-480p`); ltx25/ltx23 medium latent (1,128,16,16,9) = 512x288 stage 1 (commit and stage 1 both /64-clean: 1024=16x64, 576=9x64). VRAM rationale: medium is 0.69x high pixels (589,824 vs 856,064; latent tokens 144 vs 209 for the same ratio) while staying native 16:9, and the 2x finalize upscale lands at 2048x1152 — where FILM fits the 6 GB card — instead of 2432x1408, which OOMs it (~5.9 GiB need). Any two tier flags together (or all three) is exit 2; creates still default to high; updates without flags keep stored geometry.
- `configure` semantics: both flags together is exit 2; create defaults to high unless `--from` without `--backend` (inherits the source geometry untouched); updates without flags keep stored geometry; a tier-driven geometry change on a committed run is refused (exit 2, same rule as backend changes — mixed-geometry sequences never commit). LTX workers accept exactly the two configured commit sizes (`COMMIT_SIZE_OPTIONS`, stage-1 halves threaded through the Mode-A graph + tail decode); anything else fails loud.
- Tests: `tests/test_definition_tiers.py` (23). Gates: ruff + format + mypy strict clean; 127/127 related green, full suite 1835 passed + 1 xdist timing flake (`test_recovery.py` pause-at-boundary, passes alone). UNCOMMITTED (per-commit approval required).

## All backends deferred: per-segment audio.wav removed (2026-10-04)
- User directive (verbatim): "All backends must be deferred. This makes audio.wav obsolete. Clean up associated files/tests/code." Q&A pinned the terms: legacy runs clean-break (old `audio.wav` entries load but are ignored, never migrated), `fake` is deferred like every other backend, commit is video-only (`_ensure_audio_coverage` deleted, not stubbed), and per-segment audio checks/columns go everywhere (no drift gates, no scoreboard audio columns).
- Commit path (`voyage/supervisor.py`, −381 lines): deleted `_with_audio_gpu` (ACE take render with video-evict/audio-evict/rebuild GPU swap), `_best_effort_audio_teardown` (only caller was the swap), and `_ensure_audio_coverage` (planner 3-attempt loop, take ledger, tmpdir slice walk, joint-fade-compensated assembly into `segment/audio.wav`). New `_cover_audio` is always-deferred: records the `audio` stage timing, derives the conditioning tail via `deferred_tail_frames`/`derive_conditioning_tail` for streaming backends (`STREAMING_VIDEO_BACKENDS`; `fake` skips — no tail needed), returns `CoveredAudio(AudioPlan(segment_id), 0.0, "deferred", ...)` with empty `take_ids`. `_adopt_unaccounted_segment` and `_commit_segment` validate/checksum `video.mp4` (+ `recovery.pt`) only; both A/V drift gates removed; `segment_committed` keeps `"av_drift_seconds": 0.0` so log readers never see a missing field. Commit-time audio worker spawn/stop/gauges left intact (still used at finalize).
- Finalize path: deleted `DEFERRED_AUDIO_BACKENDS` / `is_deferred_backend()` / `write_deferred_stub_audio()` (`voyage/audio_finalize.py`) — the deferred-vs-joint distinction no longer exists. `deferred_tail_frames` gates on `STREAMING_VIDEO_BACKENDS` instead (25 for ltxv/ltx25/ltx23, `max(25, reencode_window)` for causvid, `MediaError` for non-streaming incl. fake). `ensure_deferred_for_finalize` routes on the *audio* backend: `fake` spawns the fake sine worker (no models), `acestep` spawns ACE, else `MediaError`. `media_audio.build_final_audio` is ledger-only (no `deferred` flag, no single-segment shortcut, no concat/atempo fallbacks — every gap raises `MediaError`); `_verify_segment`/`_check_segment_committed` video-only. `media.py`/`cli_finalize.py` dropped the `deferred_audio` plumbing (`finalize_run` always ensures takes then mixes); deleted `voyage/finalize_parallel.py` (the parallel gate required non-deferred, so all-deferred kills it — finalize is sequential only).
- Manifest/validate/scoreboard: `REQUIRED_CHECKSUM_ARTIFACTS = ("video.mp4",)`; `cli_validate` DONE/missing/checksum/metrics video-only (recorded `audio.wav` entries skipped silently so old runs still validate); scoreboard dropped `audio_path`/`audio_exists`, kept `take_ids` (empty on new commits — finalize takes live under `run/audio/takes.jsonl`). Segment file count 5 → 4 (`DONE`, `video.mp4`, `recovery.pt`, `manifest.json`); the §30 A/V gate retired with the audio it policed (alignment is a finalize concern now, where the ledger meets the committed timeline).
- Workers: `video_ltx25.py`/`video_ltx23.py` dropped the joint-audio commit path (`SEGMENT_AUDIO_FILENAME`, `_read_block_audio`, the `novel_audio_wavs` FLAC pipeline, the `audio_path` result key); `_execute_graph` requests `["28"]` only — the 29-node graph keeps its audio nodes unexecuted because the joint AV latent is the validated Spike-A denoise path. `finalize_run` resolves `audio_config` from the stored run config when the caller passes none (else every test-less finalize defaulted to acestep and failed offline).
- Tests: reworked ~25 files (deleted `_with_audio_gpu`/`_ensure_audio_coverage`/parallel/joint-fallback/atempo pins, re-pinned manifests/scoreboard/reconcile/observability to video-only, `test_recovery.py` pause test made deterministic — the old version asserted on a pre-existing read-then-write race in `_write_state_preserving_control_plane` and only passed by luck). Gates: ruff check + format + mypy strict clean; 1843 passed, 17 skipped, 1 foreign failure (concurrent agent's in-flight `sfx_finalize.py` work, untouched per §9). Normative sections updated in place (§§29/30/38/137, `docs/ARCHITECTURE.md`, `docs/UPSTREAM_LTX25_NOTES.md`, `docs/UPSTREAM_LTX23_NOTES.md`); history entries left verbatim. UNCOMMITTED (per-commit approval required).

## Ambient slow music + UHD camera-motion direction (2026-10-04)
- User directives (verbatim): "Modify the music system prompt to be more ambient, slow, morphing, weird, experimental, dark, pads, music." + "Modify the music system prompt to be more ambient, slow, morphing, weird, experimental, dark, pads, held chords, large, vast, powerful, music. Add/modify the video system prompt to be ultra high definition, hyper detailed, sharp, crisp, simple, refined, lots of camera motion (rephrase/expand my requirements to be best suited for ltx25)."
- No single music/video system-prompt file exists — direction lives in three places, all retuned: (1) `voyage/models.py` `DEFAULT_MUSIC_STYLE` (was atonal/Messiaen/fusion-rock-jazz, now "ambient dark experimental music, slow morphing pads and held chords, vast powerful drones, weird slow-evolving textures, deep sub-bass") — flows via `config.audio.music_style` into the director CURRENT AUDIO STATE and finalize music inputs; (2) `voyage/director.py` `DIRECTOR_SYSTEM_PROMPT` caption doctrine — item (1) now requires every video stage ultra high definition / hyper detailed / sharp-crisp / simple refined composition with one concrete continuous camera move per stage (push-in, lateral drift, orbit, crane rise, pan), phrased for LTX-25 (physical motion, no cuts, no scene change); item (2) keeps the dark experimental touch (minor/modal harmony, sub-bass pressure, dissonant accents) over a new ambient slow core (morphing pads, held chords, vast drones, weird textures); the `build_director_user_message` schema hint mirrors both (camera-move + UHD video, ambient/pads/held-chords music); (3) deterministic fallback `music_caption` rewritten to the ambient vocabulary (keeps concept + charter derivation).
- Code-level video tail (`voyage/prompts.py` `motion_constraints`, immutable — LLM cannot drop it): was "{pace} continuous camera movement, gentle organic motion, no abrupt cuts, no scene change within the shot", now "{pace} continuous camera movement, strong fluid motion throughout, ultra high definition, hyper detailed, sharp crisp image, simple refined composition, no abrupt cuts, no scene change within the shot". Charter pace bands + middle-layer cue override untouched (the user's "lots of camera motion" arrives via the stronger tail + the doctrine's per-stage camera move, which flows through the designed cue-override channel).
- Tests: extended `tests/test_director_dark_direction.py` (renamed module docstring, new ambient-pads/UHD-camera-move/`DEFAULT_MUSIC_STYLE` pins) + `tests/test_phase3.py` (UHD + strong-fluid-motion pins on the enforce_style test). Gotcha: `DIRECTOR_SYSTEM_PROMPT` is one triple-quoted literal, so source line-breaks are literal newlines — pinned phrases must sit on a single source line. Gates: ruff check clean; format + mypy strict clean on all touched files; full suite 1855 passed / 18 skipped. §18 MOTION example updated to the new tail.

## Finalize overlap: model pass + deferred music fork-join on 2-GPU (2026-10-05)
- User observation: audio never overlaps augmentation at finalize — the two GPU stages run back to back while the cards sit half-idle. Q&A locked the terms: ACE-music-first, finalize-only, 2-GPU parallel with 1-GPU serial fallback, fail-soft (a music failure ships nothing silently — retry resumes from ledgers).
- `finalize_run` (`voyage/media.py`) was strictly sequential: triage → `run_durable_model_pass` (cuda:1) → `ensure_deferred_for_finalize` ACE on cuda:0 → `build_final_audio` mix → publish, then the SFX dub after. New order on 2-GPU boxes: the model pass (cuda:1) and the deferred ACE takes (cuda:0) overlap under one outer stage; mix/publish/SFX stay ordered after the join.
- Impl: module-level `DEFERRED_MUSIC_DEVICE = "cuda:0"` + pure gate `model_music_parallel_armed(tensor_path, deferred_pending, model_devices)` (tensor path taken + dry walk shows takes pending + every model device differs from cuda:0 — `model_pass_devices` collapses to cuda:0 on 1-GPU, so single-GPU finalizes stay sequential) + `run_model_pass_and_music_parallel(model_work, music_work)` fork-join (always joins both threads; model error takes precedence, music error re-raises after the join; both ledgers make every outcome resumable). `finalize_run` extracts `_do_model_pass` / `_do_music_takes` closures so the sequential path stays byte-identical; audio-config resolution moved above the fork and `audio_stretch` is precomputed from the gate — exact because both model-pass entries return non-None, so the gate predicts `tensor_intermediate is not None`. Branch progress is None inside (one-bar rule), single outer stage outside. No concurrent metrics writes exist between the branches (only the media tail appends `finalize_completed`).
- Tests (new `tests/test_finalize_model_music_parallel.py`, 7): gate conjunction + 1-GPU collapse, thread rendezvous proving real overlap (sequential execution would deadlock the 10 s waits), model/music error propagation with the sibling branch still joined, model-error precedence, stretch precompute/post-branch parity. `test_finalize_gpu_defaults.py` docstrings updated (sequential-only claim was stale). Gates: ruff + format + mypy strict clean; full suite 1862 passed / 17 skipped. UNCOMMITTED (per-commit approval required).

## Explicit finalize quality: --upscale/--interpolate replace floors (2026-10-05)
- User directive (verbatim): "For the configure verb, rename '--interp-multiplier' to '--interpolate', and add a '--upscale' with a multiplier ('--upscale 2' doubles the resolution, '--interpolate 2' doubles the number of frames). Also remove the minimum quality requirements, quality will be specified explicitely." Q&A locked: defaults upscale=1/interpolate=1/presentation unset (None, native ship); REMOVE `--no-augment` AND `use_model_pass` (multipliers are the only knobs; 1/1 means no work); REMOVE `PRESENTATION_MIN_FPS` too (CausVid 16fps ships 16fps unless pinned); old manifests fail loud with a re-configure hint.
- `config.py`: `AugmentConfig` is now `upscale=1/interpolate=1/presentation_fps=None` with a (1,2,4) worker-vocab validator on both multipliers (fail fast at configure — the SRVGG worker + pollers only serve 1/2/4); `resolve_config` params swapped; `parse_min_resolution` deleted.
- CLI: `_add_augment_args` carries only `--upscale/--interpolate/--presentation-fps`; `_augment_overrides` maps the three; `cli_configure`/`cli_generate` model-ensure gates are `upscale>1 or interpolate>1`; `persistence` fails loud on the seven legacy keys.
- `media.py`: `plan_augmentation(source_w, source_h, source_fps, *, upscale, interpolate, presentation_fps, model_interpolate)` — out = source×upscale, out_fps = presentation or round(source×interpolate); unprobable dims → MediaError always, unprobable fps → MediaError unless pinned; `upscale_factor_for`/`PRESENTATION_MIN_FPS`/`AUGMENT_DEFAULT_*`/`_record_final_geometry` deleted; `FinalizeOptions`/`ResolvedFinalizeSettings`/`resolve_finalize_settings`/`finalize_run` all carry upscale/interpolate (finalize drops width/height/fps — out derives from the probed source; tensor gate + native fast path kept).
- `augment.py`: `use_model_pass` thread removed everywhere (`enhance_frames`/`make_enhance_chunk_worker`/`run_model_augment_chunks` take legs as given; `model_pass_active(weights)` single-arg = any-leg-present); `augment_background` moot = 1/1, factor = config upscale.
- Tests migrated 26 files (3 parallel tracks, all green in scope); `docs/AUGMENT.md` rewritten to the multiplier contract; README/OPERATIONS/BENCHMARKING one-liners updated. History entries left verbatim. UNCOMMITTED (per-commit approval required).

## Post-commit cheap motion sense: steer-next-segment, never gate (2026-10-05)
- Problem: frozen segments committed silently — the six-metric §43 inspector was never wired on the live path and the commit gate is frame-count only, so a static segment commits identically to a fluid one.
- New `voyage/motion_sense.py`: `sense_motion(video_path)` returns `MotionReading(energy, seconds)` via the existing `sample_frames` path (3 frames @160px) + pixel-delta `motion_energy`; never raises (missing/corrupt clip → `energy=None` = unknown, steers nothing); typical cost 100-300ms of ffmpeg thumbnail decode, no GPU, no model load.
- Supervisor wiring (`voyage/supervisor.py`): `_last_motion` state; `_commit_segment` senses AFTER the state advance so sensing can never strand a commit, emits a `motion_sensed` metric event and passes `motion_energy` / `motion_seconds` into `segment_done`; `_propose_segment` derives `measured_context` + `amendments` from the previous reading via the existing `format_measured_context` / `feedback_amendments` (previously dead on the live path — callers passed `""` / `None`) and forwards them into `_accept_director_decision`; the drift-hold path also applies amendments (the deterministic "gentle motion, held wide shot" stage is the most freeze-prone prompt on the live path, so a low-motion reading hardens it instead of being dropped).
- Console (`voyage/console.py` `segment_done`): shows `motion energy X.XXX (Y.YYs sense)` plus a LOW-MOTION flag when below the 0.20 StyleSpec floor ("next prompt steers harder"); unknown reads show a "sense skipped" line; missing keys tolerated for older callers.
- Tests: new `tests/test_motion_sense.py` (6 — unknown-never-raises, unknown-steers-nothing, low-steers, healthy-silent, console low-motion + latency line, console missing-keys tolerance). Gates: ruff + format + mypy strict clean; full suite 1883 passed / 17 skipped.

## Model-pass progress in frames: per-leg bars, split timing, output count (2026-10-05)
- User directive (verbatim): "in the console during 'generate' show more granular details, especially for the augmentation phase; I want to see how much time did upscaling take and interpolation take separately; show a progress bar with processed-frames / total-frames (include frames that have been processed concurrently to the video generation in processed-frames)". Q&A locked: interp counts source frames (not ×m outputs; output totals appear in finish/finalize detail only), persistent bar + per-commit delta lines both, per-chunk detail --verbose only.
- Units: source frames per leg everywhere — pollers already own the exact window counts, so `UpscalePollResult` / `InterpPollResult` gain `frames_done` / `frames_skipped` (interp also `frames_waiting`), all defaulted (positional test constructions keep working). New keyword-only `on_chunk_frames(segment_id, source_frames)` fires alongside `on_chunk` for live bar advance (`**kwargs` stubs pass it through untouched).
- Background (`voyage/augment_background.py`): `prewarm_once` times each sweep → `PrewarmResult` gains `upscale/ip_frames_done` + `upscale/ip_seconds`; the driver tuple store grows to (passes, up chunks, ip chunks, up frames, ip frames, up s, ip s) with a new `ledgered_frames()` reader (`ledgered_totals()` kept chunk-shaped for old readers) plus `last_result` for verbose sweep lines. The bg thread still never touches display — all reads are main-thread post-commit.
- Generate (`voyage/supervisor.py`): post-commit report is now `pre-warm ledgered +Nf upscale in Ns, +Mf interp in Ms (total X/Yf of Zf committed)` (Z via one `read_state`, omitted when unreadable); a persistent `model-pass frames` bar opens lazily when augmentation is demanded and advances by both legs' new frames against twice the committed frames (every frame owes one upscale + one interp pass), closing in the `run_segments` finally; `--verbose` adds one sweep line per pass (segments, per-leg frames + seconds, identity-deduped). `SegmentProgress` gains `bar()` + `verbose` (`RichSegmentProgress` delegates to the console; legacy chunk-only drivers stay silent via the `ledgered_frames` fallback).
- Finalize (`voyage/augment_finalize.py` + `voyage/media.py`): the merged `model-pass chunks` bar is replaced by sequential per-leg frame bars (`upscale frames`, then `interp frames` — never two concurrent Live displays; totals from each sweep's result, skipped caught up at close, live advance via `on_chunk_frames`); per-leg rates ride the finish line via the new opt-in `BarTracker.set_extra` (unset bars keep the historical format byte-identical); `--verbose` logs per-segment chunk ranges (`upscale 000003: chunks 0-2`); the timing table splits via `_model_pass_stage_rows` (`model pass · upscale` / `model pass · interp`, aggregate kept for legacy paths); publish announces `final video Nf WxH@Ffps (S segments)` (count from `validate_video`, free) and `finalize_completed` gains `output_frames`. `timings` accumulation is `.get`-tolerant so direct `_poll_to_completion` callers with `{}` never KeyError.
- Tests: rewritten per-leg bar pins (`test_augment_finalize_wire`), frame-count pins on both pollers, 7 new in `tests/test_model_pass_progress.py` (ranges, extra on/off, prewarm timing/frames, driver accumulation, leg-row split + fallback), rewritten prewarm report tests (`test_console`, incl. legacy-driver silence). Gates: ruff + format + mypy strict clean; full suite 1893 passed / 17 skipped. UNCOMMITTED (per-commit approval required).
- **Live concurrent progress (generation video∥prewarm + finalize model∥music, DESIGN §59)**: the video render no longer freezes the console — `Supervisor._generate_segment_with_live_prewarm` runs `adapter.generate_segment` on a `voyage-video-render` worker thread while the main thread pumps pre-warm events into the persistent `model-pass frames` bar (0.2 s tick; join in `finally`, errors re-raised after join; direct-call fallback when progress/driver absent or the driver thread died, so silent/library runs never pay for the fork). Event flow is thread-safe by construction: `prewarm_once` + `BackgroundPrewarm` accept `on_upscale_chunk/on_upscale_frames/on_interp_chunk/on_interp_frames` (forwarded to the pollers' `on_chunk`/`on_chunk_frames`; rendered chunks only — skipped chunks never fire, so pass-end ledger deltas stay truth), the supervisor's driver is built with queue-appending callbacks, `_drain_prewarm_queue` advances the bar live (frame events) while ignoring chunk lifecycle events, and `_report_background_prewarm` subtracts pumped counts before advancing (bar moves once, note keeps the full ledger delta) then discards stragglers. Finalize's model∥music fork shares one `ParallelFinalizeDisplay` (`GenerationDisplay` alias for generation-side call sites): a span view owns the outer stage as plain lines (no second spinner Live fighting the shared Progress Live), model-pass and music-takes views carry the per-leg bars; the sfx∥publish fork already used the same coordinator. `getattr` guards on the pumped counters keep `__new__`-built doubles (no `__init__`, never pump) working. Tests: new `tests/test_live_progress_display.py` (13 — callback passthrough incl. moot-plan skip + custom-fn opacity, 3-view concurrency, alias pin, fork-join, pump advance/reconcile/fallback/error-resurface/slow-render). Incidental: `scripts/gates.sh` now captures `SCRIPT_DIR` before `cd` (the trailing cache-guard `source` broke under the documented `./Voyage/scripts/gates.sh` invocation from the repo root). UNCOMMITTED (per-commit approval required).

## Two-stream finalize: SFX bed ∥ publish via proxy conditioning (2026-10-05)
- User directive (verbatim): "During finalization, music and augmentation seem to happen in parallel to each other, but sfx seems to happen sequentially. SFX must happend after music, interpolation must happen after upscaling, but these 2 streams can happen fully in parallel from one another. Ensure this is the case and that as soon as music is done, that SFX starts. Show the progress of each of those in the console."
- Shape (Q&A-locked): stream A = model pass (upscale→interp); stream B = music takes → SFX bed → dub. The bed starts after the model∥music join (cuda:0 is free — ACE worker stops at take-render end, verified `audio_finalize.py:414/460`; SFX tears down best-effort) and overlaps the publish encode on main; the dub runs on main before validate so `validate_video` + `output_frames` cover the dub. A bed failure ships music-only and `cli_finalize` falls back to the legacy shipped-pixels pass (the keyed ledger keeps proxy stems from false-hitting there).
- Proxy conditioning (user counter-proposal, Q&A-locked): the bed conditions on a stream-copy concat of the committed segments (`build_proxy_reference`, source timeline 1:1, no setpts) instead of shipped pixels; slow-mo k applies once at dub via a fresh atempo chain (`atempo_chain_for_stretch`, identity-skipped within 1e-6, factors decomposed into [0.5, 2.0] filters). Approximations documented on the builder: duplicated-frame slow-mo vs FILM-smooth, hard cuts vs morph joints, no upscale (near-irrelevant at the worker's 384px resample). The worker contract is unchanged (seeks are `(video_path, start, duration)` in the reference's own timeline).
- Keyed ledger: records carry `conditioning_source` (shipped/proxy; legacy lines read as shipped) + `conditioning_timeline`; the hit-test and `validate_sfx_ledger` group by identity (dedupe key + per-group timeline walk) so a shipped re-finalize never reuses proxy stems and vice versa. The shortfall-line format is unchanged (generate gate regex intact).
- Shared display (`console.py`): `ParallelFinalizeDisplay` (one shared rich Progress + lock) + `ParallelStreamDisplay` views (real VoyageConsoles, zero signature churn) + `SharedBarTracker`; branch stages degrade to lock-guarded lines. SELF-DEADLOCK found live and fixed: the view overrode `info` and re-acquired the coordinator lock via `self.styled` redispatch — a deterministic self-deadlock on the same thread at the first `view.info` call (the 60-minute sfxab stall; pipe back-pressure was investigated as the cause and disproven by a single-thread faulthandler dump + leaf print). Rule, pinned by comment: only leaf emits (`line`/`styled`/`error`) may lock. Regression test in `tests/test_sfx_parallel.py`.
- Harness lesson (not a code bug): never pipe a GPU run's stdout into `tail` — tail doesn't drain until EOF, the 64KB pipe fills, and the run blocks in `print()` looking exactly like a code hang. Redirect GPU-run logs to files.
- Orchestration (`media.py`): `SfxParallelRequest` + pure `sfx_parallel_armed` gate (request present + non-fake backend); `finalize_run` gains `sfx_request`/`sfx_report` (both default None, signature compatible); `cli_finalize` builds the request and skips the legacy pass when the report says dubbed.
- GPU A/B (both cards idle): `sfxab` run (ltxv 512x320, 2 segs, deterministic director, ACE music, upscale 2) — legacy shipped bed vs parallel proxy bed on identical segments+music. Artifacts `/tmp/sfxab-legacy.mp4` vs `/tmp/sfxab-parallel.mp4` (both 9.04s 1024x640@24 AAC 48k stereo; 48B size delta). Metrics: RMS 2976 vs 2692, peak 12143 vs 11343, mean abs diff 2218 — a genuinely different render (~10% quieter), as designed. User listen verdict: ship it.
- Tests: new `tests/test_sfx_parallel.py` (13 — atempo chains, gate, keyed hit-test incl. legacy-no-key default, append/validate round-trip, proxy→shipped re-render + third-run hit, proxy concat + rejects-empty over real ffmpeg, dub k=1/k=2 over real ffmpeg, deadlock regression). Gates: ruff + format (317 files) + PLR2004 ratchet + mypy strict clean; 1973 passed / 17 skipped.

## Verbose-gated segment prompts (2026-10-05)
- User directive (verbatim): "during generate, do not show the prompts of each segment, gate those behind --verbose".
- `console.py segment_plan` gates the video-prompts loop, music line, and sfx-caption lines behind `self._verbose` (compact default = headers/timings only); test pins updated to hides-by-default + verbose-present (`test_console`, `test_sfx_caption_render_161`).
- Companion upscale-only generation pre-warm work (driver `include_interp=False`, wait-for-idle, leg-aware VRAM floors, held-back notes) is implemented and scoped-green but HELD uncommitted — it depends line-level on the frames agent's uncommitted base and needs the §9 joint-commit exception.

## Finalize A/V stream: 2060 upscale-only, 4060 music -> SFX -> interp (2026-10-05)
- User directive (verbatim): "In voyage, interpolation currently may happen in parallel to upscaling and/or audio. Since it is memory hungry and takes a decent amount of time, I think it'll more efficient to run Music->SFX->Interpolation on the 4060 GPU as a stream and use the 2060 GPU only for upscaling. As frames are upscaled by the 2060 GPU, the interpolator must pick them up and interpolate them on the 4060 GPU. Also, do not interpolate frames during video generation (including seam fixing). This must all happen at finalize time."
- Shape: both legs provisioned splits the tensor pass into Phase A (upscale-only on cuda:1 via `run_upscale_phase`, overlapping the deferred ACE music takes on cuda:0 under the existing fork-join) and Phase C (`run_interp_phase` on cuda:0: interp sweep over the Phase A ledger, then drain + seam/morph FILM joints on the same card, then publish). Order on 2-GPU: upscale ∥ music takes -> mix -> SFX bed (synchronous, cuda:0) -> interp+drain -> publish -> dub -> validate. The old two-stream bed∥publish overlap survives only on the legacy path (partial legs / no tensor work). 1-GPU boxes stay fully sequential (fork gate disarms: upscale leg collapses to cuda:0).
- Seams: `augment_finalize` already had per-leg `upscale_device`/`interp_device` + `run_upscale_phase`/`run_interp_phase` + `_drain_to_intermediate` (FILM joints on the passed device); this change adds `upscale_devices`/`interp_devices` next to `tensor_devices` in `finalize_run`, a `phased` predicate (tensor path + both legs + devices), `_do_upscale_phase`/`_do_interp_phase` closures, the sync-bed branch (`bed_done_sync`, dub gate widened), `upscale_pass_devices` (cuda:1 on 2-GPU) + `interp_pass_devices` (cuda:0 whenever visible) in `augment.py`, and pins the native-path morph joints to the interp device (`resolve_morph_device` fallback moved `model_pass_devices` -> `interp_pass_devices`). No generation-side change was needed: the background driver already passes `include_interp=False` (upscale-only pre-warm on the 2060) and workers contain no FILM/interp code — all interpolation/seam/morph lives in the finalize durable path.
- Tests: `tests/test_finalize_av_stream.py` (4 — device split pins, upscale+music-before-interp order, SFX-bed-before-interp with stubbed bed/dub); re-pinned `test_interp_multiplier` (multiplier + both devices reach Phase C), `test_issue_166_model_pass_select` + `test_native_skips_model_pass` lift case (both phases called) and the native skip case (both phases forbidden). Gates: ruff check + mypy strict (94 files) clean, pytest 1986 passed / 17 skipped; `ruff format --check` tree gate stays red on a concurrent agent's untracked `tests/test_director_parallel_prefetch.py` (not this scope, untouched per §9). UNCOMMITTED (per-commit approval required).

## Director parallel to ltx25 generation: blocking prefetch take (2026-10-05)
- User directive (verbatim): "In voyage, ensure that the director always runs in parallel to ltx25 generation. Currently, I see the director work sequentially between segments. This is 30-40s lost at each segment when the director can already generate the next segment's prompts as soon as the current segment's prompts are done. So it can definitely work at the same time as video generation. The director also resides on a separate GPU, so there's no compute/memory interference."
- Diagnosis (all in `voyage/supervisor.py`): the N+1 prefetch submitted after each accept should overlap the video render, but three defects serialized it — (1) `_take_prefetch` instantly missed a still-running future, then the sync decide queued behind the orphan on the serial `SubprocessWorker._call_lock`, doubling latency; (2) `_accept_director_decision` discarded the prefetch whenever motion-steering amendments existed (`prefetched_raw is not None and not amendments`) although amendments apply post-hoc to both paths; (3) `_prefetch_decide_for_next` submitted wasted prefetches for drift-hold targets, consumed as `invalidated` while contending with prompt enhancement on the single-slot llama sidecar.
- Fix: `_take_prefetch` waits out a still-running same-target future's remaining RPC budget (`PREFETCH_TIMEOUT_SECONDS - age`, never when `invalidated`) via `future.result(timeout)` then re-checks `done()` — the wedged-worker case still falls through to a miss so the sync path's restart budget owns recovery; `prefetch_age_ms` stays honest via a local `_age_ms()` closure (no helper imports). Prefetched proposals stay usable with amendments (gate removed, comment records why). Submit skips targets where `(number+1) % drift_every != 0` (comment records the sidecar-contention motive). The §20 three-outcome telemetry is unchanged; held segments now log `miss` (no future submitted) instead of `invalidated`.
- Tests: new `tests/test_director_parallel_prefetch.py` (4 — blocking-take wait-then-hit with a 0.5 s background completion, invalidated-never-waits, prefetch usable with amendments with the sync path stubbed to raise, drift-hold skip-submit); re-pinned hold e2e in `test_prefetch_invalidated_136_168.py` (held seg1 logs miss, no hit, no invalidated). Gates: ruff + format + mypy strict clean on all touched files (the `format --check` red flag on the new file in a concurrent entry predates its reflow — verified clean); full suite 1985 passed / 17 skipped + 1 known parallel-only flake (`test_issue_152_single_graph`, 3/3 green in isolation, unrelated). UNCOMMITTED (per-commit approval required).
- Known follow-up (not restructured): `prompt_enhancer.enhance_many` (main thread, sequential per-block sidecar calls at render start) shares the single-slot llama-server (no `--parallel` flag) with the background prefetch — overlap is real but the two contend for the one slot.

## Deferred-take exactness: chain-once, start-segment prompts, replay convergence, 6s take crossfades (2026-10-05)
- User questions (verbatim): "If 26 takes are needed (say with some buffer for crossfades), why is the finalization process generating *way* more takes (currently at 71 takes and counting)?" then "Proceed with the fix. Ensure that exactly the correct number of takes are generated. Ensure that the audio prompt from the corresponding starting segment for the take is used as the prompt for ACE. Clean up invalid takes. Ensure that running finalization again will regenerate the music correctly. Ensure there are generous crossfades (say 6s) between music takes."
- Diagnosis (kaolin ledger forensics, read-only): 73 takes but 19 distinct `covers_from` (~4 per 48s joint), all byte-identical captions. Two defects: (1) `AudioPlanner.plan` chained on `current.covers_until()` (the take covering NOW) while chained takes anchor in the future — every later segment inside the 20s ahead window chained another duplicate at the same `covers_from`; (2) `audio_finalize._segment_music_inputs` read `transition.decision.audio`, but production manifests store the decision flattened at `transition` (supervisor persists `decision.model_dump()` directly) — every take fell back to the run `music_style` instead of its segment's evolving caption.
- Fixes (`voyage/audio/planner.py`, `voyage/audio_finalize.py`): chain on `self.coverage_until()` (max over all takes) so exactly one take chains per joint; caption reader accepts wrapper and flattened shapes; new `_replay_segments` table (start/stretched/number/caption/energy) shared by `deferred_render_pending` and `ensure_deferred_takes` so gate and render can never disagree; `_coverage_start_position`/`_resolve_take_caption` give fresh/chain takes the caption + segment_index of the segment where coverage begins (repaints keep the current caption — the new caption is their purpose); `take_for_time(video_time, max_segment_index)` bounds serving to takes rendered for segments ≤ N in both `plan` and the `build_final_audio` window walk — future repaints no longer retroactively serve (and re-trigger repaints at) earlier cursors, so replays converge instead of appending forever. `coverage_until` stays global (future takes count as coverage).
- Proof on real kaolin data (throwaway stub renderer, sine takes deleted after): 68 takes / 26 distinct covers (zero identical-caption same-cover groups), 45 distinct captions, `deferred_render_pending` False after. Count = 26 chain joints + 42 genuine-shift repaints (consecutive caption sims mostly < 0.5 — this run's director shifts music nearly every segment); 6 chains superseded by an immediate next-segment repaint serve nothing (structural: quantized grid lands mid-segment + shift; accepted ~9%, no gap-free alternative).
- 6s take crossfades: takes chain on segment-aligned boundaries, so the segment-window overlap IS the take crossfade. `MAX_FINAL_OVERLAP_FRACTION` 0.5 → 1.0 (windows extend ±half overlap per side; takes are 30-60s so joints stay defined; other runs keep 0.10/0.5 defaults) + kaolin manifest `final_overlap_fraction` 1.0 / `final_overlap_cap_seconds` 6.0.
- Kaolin cleanup: deleted 95 invalid take wavs + `takes.jsonl` (835M; pinned captions + duplicates); `audio/` empty, ready for a clean finalize. No live voyage/ACE/ffmpeg process and no containers at fix time.
- Tests: chain-once (`test_audio_planner`), start-segment captions + flattened shape + repaint replay-convergence (`test_deferred_audio`), serving-bound replay (`test_audio_planner`), overlap-ceiling pin update (`test_generation_stack`). Gates: ruff + format + mypy strict clean on all touched files; 119-file audio/finalize scope green; full suite red only from concurrent in-flight generate-scope work (`cli_generate.py`/`models_ensure.py` signature mismatch) plus ordering flakes — untouched per §9. UNCOMMITTED (per-commit approval required).

## Generate-only post-processing skip flags (2026-10-05)
- User directive (verbatim): non-persistent `--no-music`/`--no-sfx`/`--no-upscale`/`--no-interpolate` flags skipping post-processing, plus `--no-audio` (= music + sfx) and `--no-augment` (= upscale + interpolate) shorthands; `--no-upscale`/`--no-interpolate` force multiplier 1 (not model-pass-only).
- Shape (Q&A-locked): generate-only, never stored — `--no-music` ships silent AAC sized to the timeline, SFX independent of music (bed dubs over silence), freshness forces re-finalize when the behavior key differs (a music-only diff must never read fresh).
- Impl: `cli._add_generate_parser` flags; `cli_core.resolve_generate_skips` (shorthands OR-ed, getattr-safe) + `generate_skip_key` (canonical `music/sfx/up/interp` combining generate skips with manifest policy + stored multipliers); `cli_generate._finalize_run_dir` translates forces to explicit `upscale=1`/`interpolate=1` (None = inherit stored) and ORs `no_sfx`; `cli_finalize` threads `no_music` into `finalize_run` and stamps the key via `persistence.record_final_coverage(skip_key=...)` (omitted = legacy 2-key dict); `media.finalize_run(no_music=...)` skips deferred ACE takes and renders anullsrc silence sized to the stretched timeline while the SFX dub still applies; `models_ensure` gains `music_enabled` (drops the ACE-Step spec); `_final_is_fresh(expected_skip_key=...)` (None = legacy path) + `_expected_skip_key` helper; `_pre_finalize_errors` honors the generate SFX skip for the healable-shortfall filter.
- Tests: new `tests/test_generate_skip_flags.py` (9 — resolver defaults/shorthands, parser accept + configure reject, key combining, key-mismatch stale, legacy-no-key stale-once, namespace threading, ACE gate); re-pinned `test_generate_noop_when_complete` (production-shaped coverage stamp) + `test_generate_ensure_receives_selective_scope` (music_enabled spy). Gates: ruff + format + mypy strict + full pytest green (2000 passed, 17 skipped).

## Music-take continuation + SFX duplicate-audit + SFX 1s crossfade pin (2026-10-05)
- User tasks (verbatim): "music takes must EXTEND the previous take (continuation, not restart)"; "verify no similar duplicate-take bug exists for SFX"; "ensure a 1s crossfade between SFX takes."
- Continuation design: chained take N+1 overlaps take N by O = 6.0s (`_CHAIN_OVERLAP_SECONDS`, tuned to one stretched segment — the overlap is covered by exactly the takes it joins, so no extra renders) and renders as an ACE repaint, not fresh text2music: `_build_continuation_src` (ffmpeg: previous take's tail O via `atrim=start=prev-O` + `(D-O)`s `anullsrc` silence, concat to a full-D canonical wav; fail-loud on missing/short-prev/ffmpeg-failure/empty output; src file is never ledgered and is unlinked best-effort after a successful render, missing prev falls back to fresh) + repaint payload `reference_audio=src, repaint_start=O, repaint_end=D`. ACE repaint preserves the head `0..repaint_start` near bit-exact (`repaint_wav_crossfade_sec=1.0` internal), so the joint is genuinely continuous music, and the 6s build_final_audio take crossfades still apply on top.
- Planner: new `chain_overlap_seconds` field (default 0.0 = byte-identical legacy path; `__post_init__` rejects `overlap < 0` and `overlap >= take_seconds`) + `tail_take()` helper (max `covers_until`, tie-break `segment_index`); chain branch backdates to `max(0, coverage_end - O)` with `current=tail` when O > 0. Threaded through `deferred_render_pending` / `ensure_deferred_takes` / `ensure_deferred_for_finalize` (all default 6.0); `media.py` untouched (new kwargs ride `AudioConfig` getattr defaults). `take_for_time` newest-wins already serves the continuing take inside overlaps.
- SFX audit verdict: NO duplicate bug possible — `plan_sfx_windows` (`sfx_finalize.py`) is pure timeline tiling (`step = window - overlap`), never chains on coverage, ledger dedupes last-wins per `(conditioning_source, window_id)`, and `validate_sfx_ledger` heals only tail shortfall. The 1s crossfade is already real (`SFX_WINDOW_OVERLAP=1.0`, `_join_audio_single_graph` chains pairwise manual `afade-out/afade-in + adelay + amix` in one ffmpeg spawn, never `acrossfade`) — pinned by tests, zero code change.
- Tests: 3 planner continuation tests (`test_audio_planner` — overlap backdate + single-chain-per-joint at 42.0/84.0, guard rejects, `tail_take` ordering), 1 wiring test (`test_deferred_audio` — chained take emits exactly one repaint payload with `repaint_start=6.0`, `repaint_end=duration`, `*_src.wav` cleaned post-render); re-pinned the shifted chained-caption test (50.0/verse-5/index-5 → 44.0/verse-4/index-4 under the 6.0s ensure default); 1 SFX tiling test (`test_sfx_finalize` — adjacent windows share exactly `SFX_WINDOW_OVERLAP`). Gates: ruff + format + mypy strict clean on all touched files, 64 scoped + 693 neighbor tests green (9 skipped). UNCOMMITTED (per-commit approval required).

## SFX bed resume fix (2026-10-05)
- User report (verbatim): "the sfx generation process fails to resume where it left off." Live proof on kaolin: `audio/sfx/` held 27+ stems but zero `sfx.jsonl` — ledger appends happened only after the whole worker pool joined, so any kill/cancel/single-window failure recorded nothing and orphaned every completed stem; the rerun restarted at w0000 (two mtime batches) instead of resuming.
- Fix (`voyage/sfx_finalize.py`): per-window durable append — each window's ledger line is appended under a `threading.Lock` inside `_render_one` right after `os.replace` + probe (join loop now only collects stems); safe because the ledger is keyed by `window_id` and `validate_sfx_ledger` dedupes last-wins, so append order is irrelevant. Orphan-stem adoption on load: stem files matching plan identity (window in plan, no ledger line, probed duration within `SFX_ORPHAN_ADOPT_TOLERANCE = 0.05`s) are ledgered under the current conditioning source with a loud stderr log; anything else re-renders normally. Adoption never breaks finalize (best-effort probe `noqa: BLE001`, T201 via new per-file-ignore).
- Contract change: issue-054's "serial append in plan order" pin is deliberately superseded (durability > plan order — buffering for order reintroduces the kill-loses-bookkeeping hole); `test_054_two_workers_append_in_window_order` now asserts order-insensitive landing (both lines whole and parse). Consumers were already order-insensitive.
- Tests: 2 new in `test_sfx_finalize.py` (failed batch keeps completed lines; orphan adopted with zero re-render calls + second run fully cached) + re-pinned 054 test; full gates green (2007 passed, 17 skipped). Uncommitted (per-commit approval per §9).

## Interp progress feedback + full-stage resume hardening (2026-10-05)
- User request (verbatim): "Interp feedback lacking — display frames processed/total, elapsed + ETA; ensure interp fully resumable after crash/cancel; audit every generation/finalization step for resumability and tighten."
- Interp progress (`voyage/augment_interp_poller.py`, `voyage/augment_finalize.py`): `_default_interp_pngs` accepts `on_pair(pair_index, pair_count)` fired per finished FILM pair; `interp_poll_once(..., on_pair_frames=None)` converts to fractional source-frame advances (`count / max(count-1, 1)` per pair, summing to the chunk — units stay source frames like the upscale bar); injected `interp_fn` is tried with the `on_pair` kwarg first, `TypeError` falls back to the legacy 3-arg call; chunk-end fires `on_chunk_frames` when given, else the `on_pair_frames` fallback for zero-pair passthrough chunks. `_expected_frame_totals` (best-effort, `(0, 0)` on any failure) derives committed sources × `chunk_windows` × `interpolated_frame_count`; both upscale and interp bars `set_total` upfront so elapsed + ETA show from the second chunk. Finish line shows both units (`{rate} frames/s ({expected_output} out frames)`). Corrupt upscaled input (PNG count mismatch) now counts as WAITING so the upscale healer fixes it instead of raising `MediaError`.
- Retile safety (`voyage/augment_sidecar.py`, both pollers): `ChunkKey` gained `chunk_frames: int = 32` (defaulted tail field, old constructors still work); new `stage_indexes_matching` matches exact `(start_frame, source_frames, chunk_frames)` per window, grandfathering pre-field records on window alone; both pollers build `missing` from the guarded union only (the first version matched exactly but still rendered the pre-guard list — caught by the retile test, fixed in both pollers).
- ACE takes (`voyage/audio_finalize.py`, `voyage/audio/planner.py`): keep-verdict requires output-truth (`_take_output_complete`: ledger + file exists + non-empty), missing files re-render in place (same id/path/seed/caption); orphan take adoption (file matches planned duration within 0.05 s → ledger append + loud stderr log); `*_src*.wav` continuation sources pruned at ensure entry; `load_takes` tolerates a torn trailing line (last line only, sidecar twin).
- Seam/morph (`voyage/augment_seam.py`, `voyage/augment_morph.py`): ledger-hit-but-wrong-PNGs now drops and re-renders (last-wins) instead of fail-loud; morph trims are content-keyed via per-trim JSON sidecars (legacy record-less trims re-render once, then stay keyed). `prune_orphan_plan_dirs` (`voyage/augment_drain.py`, grace 7 d, newest-mtime-under-dir) is implemented but UNWIRED — open decision where finalize calls it.
- SFX ledger (`voyage/sfx_finalize.py`): `load_sfx_ledger` skips torn lines (sidecar twin).
- Reconcile (`voyage/cli_generate.py`): `_discard_uncommitted_segments` skips DONE-bearing dirs (DONE is empty-by-design — presence, not size, is the adoptable signal) so the locked commit adopts-or-refuses; `test_generate_reconcile.py:176` updated to assert survival.
- Tmpdirs (`voyage/media.py`): `FINALIZE_TMPDIR_PREFIX` + `prune_stale_finalize_tmpdirs` at finalize entry (best-effort, exact-prefix dirs only).
- Prewarm observability (`voyage/augment_background.py`, module side only): `PREWARM_NOTHING_LOUD_NOTE` + `prewarm_pass_did_nothing` / `report_prewarm_pass` / `did_nothing_all_run` / `report_at_stop`; supervisor wiring left as follow-up (supervisor.py under concurrent migration, off-limits).
- Validate (`voyage/cli_validate.py`, read-only): orphan scan covers `run/augment/` + `voyage-final-*` dirs + new `*.partial.*` pattern, plus a ledger-vs-output notice per plan dir.
- Tests: new `tests/test_interp_progress_resume.py` (5 — per-pair sums, 3-arg fallback, corrupt-input wait+heal, retile re-render + key pin, totals helper), `tests/test_deferred_resume_hardening.py` (8), `tests/test_reconcile_resume_gaps.py` (10), `test_seam_morph_resume_{heal,trim,gc}.py` (11); `test_augment_finalize_wire.py` stub moved to the `on_pair_frames` contract. Gates: 73/73 scoped green; ruff check + format + mypy strict clean on all touched files. Full `gates.sh` on the idle box + final review still open at time of writing. UNCOMMITTED (per-commit approval required).

## SFX ledger timeline-union fix (2026-10-05)
- User report (verbatim): `./scripts/run.sh generate kaolin` aborted at the pre-finalize gate with "INVALID: sfx coverage gap: window w0117 starts at 819.00s, expected ~0.00s".
- Root cause (verified against live kaolin data): 256 committed segments (~1025s timeline), no final.mp4 yet; `audio/sfx/sfx.jsonl` held 148 lines in two conditioning_timeline groups — (proxy, 821.04) head w0000..w0116 + (proxy, 1025.04) tail w0117..w0146 (w0117 duplicated, last-wins keeps the new 8s line). This mixed state is exactly what `_stem_cache_hit` legitimately produces: it ignores conditioning_timeline by design (append-only proxy implies head pixels unchanged), so after a timeline extension the head stems cache-hit and only the tail re-rendered. But `validate_sfx_ledger` grouped by (source, timeline) and demanded each group tile from zero — the tail group can never do that.
- Fix (`voyage/sfx_finalize.py::validate_sfx_ledger`): group by conditioning source only (legacy lines still read as shipped), walk the per-source union from zero (gaps stay fatal), shortfall target is the passed source timeline (stays healable). `conditioning_timeline` remains as provenance. Production always passes the source timeline (cli_validate uses timeline_frames/fps), so group timelines never exceed it.
- Tests: rewrote `test_proxy_group_validates_against_its_own_timeline` as `test_proxy_short_bed_reports_healable_shortfall` (`tests/test_sfx_parallel.py` — expects the single shortfall line + asserts `is_healable_sfx_shortfall`, so the generate gate still proceeds); new `tests/test_sfx_timeline_union.py` (5 — mixed head+tail clean, head-only healable shortfall, cross-timeline gap fatal, tail-without-head fatal, sources never mix). Scoped suite 42 green; real kaolin ledger validates to [] in-container; both GPUs idle at probe time. User re-ran `./scripts/run.sh generate kaolin` end-to-end (SFX gate passed, 81 ACE takes + interp phase observed in progress; user monitors the run). UNCOMMITTED (per-commit approval required).
- Corrections to the previous entry ("Interp progress feedback + full-stage resume hardening"): (1) the injected-`interp_fn` capability probe is now a 3-tier `inspect.signature` sniff (`_supports_on_pair`: True → 4-arg direct, False → legacy 3-arg, None/unintrospectable → legacy try-4/except-TypeError-then-3) instead of the bare try/except, so a TypeError raised *inside* a 4-arg fn no longer triggers a wasteful second full render; (2) `_default_interp_pngs` is self-contained (HEAD `interpolate_pair` per-pair loop + `on_pair` firing — no `interpolate_mids` dependency, which lives in the concurrent frames agent's uncommitted worker code); (3) `test_augment_finalize_wire.py` is NOT part of this commit — its per-leg-bars hunks interleave with the frames agent's uncommitted poller fields and stay untouched for their joint commit; (4) `augment_background.py` + prewarm parts of `test_reconcile_resume_gaps.py` are likewise held (they depend on the frames agent's uncommitted `PrewarmResult` fields); only the `cli_generate.py` DONE-skip production hunk + its `test_generate_reconcile.py` survival assertion ship here.

## Generate self-heals orphan transients before the pre-finalize gate (2026-10-05)
- User report (verbatim): "There is still an error with: comfy/Voyage/scripts/run.sh generate kaolin INVALID: - orphan transient files: ['voyage-final-pycwey4r'] aborting before finalize (re-run `voyage configure` with --skip-bad to salvage) Incomplete/corrupted files must be restored/fixed/cleaned automatically when restarting. Fix the error, but do not complete generation (I'll do that myself). The generate verb should be self-healing without needing special flags, unless signficant destructive actions would occur."
- Root cause: `validate_run` is read-only and flags `voyage-final-*` staging dirs plus `*.partial` / `*.partial.*` / `*.tmp.npy` / `*.tmp*` files, while `media.prune_stale_finalize_tmpdirs` only ran at `finalize_run` entry — after generate's pre-finalize `validate_run` gate — so a crashed finalize's staging (kaolin: 2.3G of window wavs, intermediates the takes ledger re-renders) aborted generate before the prune could run. `cmd_generate` never reached finalize; `--skip-bad` was the only way through.
- Fix (`voyage/cli_generate.py`): new `_heal_safe_transients(run_dir) -> int` mirrors the validate orphan scan exactly (reuses `_ORPHAN_PATTERNS`; recursive files-only deletion under segments/novelty/audio/augment + run-root `*.partial`, all symlink-skipping and best-effort, plus `prune_stale_finalize_tmpdirs` for the staging dirs) — crash-torn staging the next pass re-creates, never DONE/video.mp4/manifests/state/takes. `_pre_finalize_errors` heals before validating (both gates self-heal, impossible to forget a future gate); `cmd_generate` additionally heals+prints (`healed: removed N transient file(s)/dir(s)`, matching the `reconcile:` line) after the reconcile discard and after `run_segments`. Validate stays read-only; checksums, missing artifacts, numbering gaps, and hard SFX errors still require `--skip-bad` (significant/destructive stays flagged).
- Tests: new `tests/test_generate_self_heal_transients.py` (5 — exact-set removal incl. `voyage-final-*` dir, symlink/target preservation, symlinked-tmpdir skip, heal-before-validate order via spies, heal-and-report print contract). Gates: `gates.sh` green (2063 passed, 20 skipped). No live generate run (user re-runs kaolin themselves). UNCOMMITTED (per-commit approval required).

## Durable finalize audio caches: no-change resumes skip the re-mix (2026-10-05)
- User report (verbatim): "Everytime I resume generation for output/kaolin (which goes directly to finalization), it re-mixes the final audio. This seems like redundant work. How can the work can be spared if nothing changed in the video and audio?" Log showed `mix final audio (152.3s)` on a 256/256-committed run with no `final.mp4`.
- Root cause: whole-finalize freshness (`cli_generate._final_is_fresh`) only skips when `final.mp4` exists, and kaolin has none — so every resume ran full `finalize_run`, whose `build_final_audio` re-sliced 256 windows from `audio/takes.jsonl` and re-joined them into the ephemeral `voyage-final-*` tmpdir (pruned on next entry, so the previous mix never survived). The SFX bed had the same shape (ledger-hit stems, but re-render + re-join every time).
- Fix: new `voyage/final_mix_cache.py` — two single-slot durable caches, `audio/final_music_cache.wav+json` (fingerprint: segment manifest checksums+frames, `takes.jsonl` sha, take-file stat identity, mix knobs incl. stretch) and `audio/sfx/final_bed_cache.wav+json` (segment identity, `sfx.jsonl` sha, stem stats, SFX knobs, music digest, cached `source_seconds`). `media.finalize_run` copies the cached file into the tmpdir on hit, else renders and overwrites the same slot via `atomic_copy` — a later publish overwrites, so the cache never grows (two wavs + two sidecars max). Cache I/O never raises (corrupt = miss, failed store keeps the old slot); `no_music` silent path stays uncached (cheap). Publish skip is unchanged (`_final_is_fresh` once `final.mp4` lands).
- Tests: new `tests/test_final_mix_cache.py` (9 — miss-before-store, hit, ledger/knob/segment invalidation, single-slot overwrite incl. byte check, corrupt-sidecar miss, bed hit + seconds round-trip, bed miss on music change). Gates: `gates.sh` green (2072 passed, 20 skipped; ruff + format + mypy strict clean). UNCOMMITTED (per-commit approval required).

## Progress wording unification + determinate bars (2026-10-05)
- User request (verbatim): "In voyage, simplify/unify/clean the wording of the progress feedback and generally unify the experience. Use UX online references for console applications. Make suggestions to make relevant information available/digest/visual."
- Interp-bar root cause (the reported `⠧ interp frames 0:01:54` with no X/Y or ETA): `BarTracker._begin` picked a spinner-only rich `Progress` (Spinner+Text+Elapsed) whenever the bar was born with `total=None`, and `set_total` only updated the task total without rebuilding columns — so a bar that learned its total mid-pass stayed spinner+elapsed forever. `_poll_to_completion` created both leg bars total-less with gated/late `set_total`, and `_expected_frame_totals` fail-softs to `(0, 0)`, so the upfront `set_total` was routinely skipped entirely.
- Fix (`voyage/console.py`, `voyage/augment_finalize.py`): `BarTracker` always builds the one determinate layout (Text+Bar+MofN+Elapsed+Remaining) — rich renders a `total=None` task without X/Y or ETA and the same columns pick up the bar + ETA live once `set_total` learns it, no rebuild needed. Both leg bars pass `expected_source or None` as the creation total (upfront X/Y + ETA whenever the committed segments read; spinner-count-then-upgrade otherwise). Stage finish unified to `✓ {label} (Ns)` on both TTY and non-TTY (was `in Ns` off-TTY); bar finish `✓ {label} (done/total, Ns[, extra])` unchanged (test-pinned).
- Stream contract (stdout progress / stderr errors): generate/finalize progress lines that used raw `print` (healed, extended plan, reconcile, nothing-to-do, generating, generated/finalized duplicates) now go through the console (`ok`/`info`, quiet-aware, same words so existing stdout pins hold); the `generated`/`finalized` double-print duplicates are deleted (single `console.ok` each). Usage/validation failures (`error:`, `INVALID:`, abort hints) stay on raw stdout/stderr untouched — failures bypass quiet legitimately. `_heal_and_report`/`_extend_plan` take an optional console (bare-`print` fallback keeps direct-caller contracts); triage skips (`media.committed_usable_segments`, new optional `progress` param) and orphan adoptions (SFX/MMAudio + deferred ACE takes) warn through the sink with print/stderr fallbacks when `progress is None`.
- Bar/stage labels unchanged (test-pinned: `upscale frames`, `interp frames`, `sfx windows`, `ace takes`, `model-pass frames`, `triage segments`, `morph joints`, `drain segments`).
- Gates: `gates.sh` green (ruff + format + mypy strict clean; 2073 passed, 20 skipped).

## Generate-verb finalize header + cache-hit notes (2026-10-05)
- User report (verbatim): "I just ran generate kaolin again and can't see what changed..." → answer: output looks unchanged. Kaolin sits at 256/256 segments committed with no `final.mp4`, so `generate` takes the finalize-only branch (`remaining <= 0`) — zero segments render, none of the generation-path wording executes, cache hits are silent, and piped/non-TTY output degrades to boundary lines where the determinate-bar fix is invisible. Follow-up directive: "Improve all outputs to the console for the generate verb."
- Fix (`voyage/cli_generate.py`, `voyage/media.py`, generate verb only): new `_announce_finalize` helper prints a finalize-branch header (`voyage finalize · {backend} {WxH} @{fps}fps` rule + `finalizing N segment(s) (F frames, ~Ss) -> {final} · {features}` info, where features = `music|silent + sfx|no sfx + upscale xN|native size + interp xN|no interp` from `resolve_generate_skips` + manifest `no_sfx` + stored augment multipliers), called at both `_finalize_run_dir` sites (resume-direct and post-generation, the latter after a `read_state` refresh) gated on `sink is None` exactly like the generation header. Music/SFX cache hits now announce (`music: cache hit — reusing last mix (N segments)` via `progress`, `sfx: cache hit — reusing last bed (N segments)` via `bed_view`; both quiet-suppressed, `None`-safe). Generation header de-duplicated (`resuming at segment N` repeated the range start — dropped). Scope note: the header lives in `cli_generate`, not `cmd_finalize` (shared with standalone `finalize`).
- Test pins honored: `nothing to do`, `extended plan`, and the exact healed stdout line stay byte-identical; no test pins `run finished`/`generating`/`generated (`/`finalized ->` (verified by grep), so those were free to polish. Usage/validation failures (`error:`, `INVALID:`, abort hints) stay on raw stdout/stderr per the prior stream contract.
- Gates: ruff + format + PLR2004 clean on both touched files; mypy `voyage` shows only the pre-existing foreign `workers/augment_worker.py:1919` error; own-scope suites green (257 passed: console/generate×6/skip-flags/finalize×2/media×4/sfx×2/deferred/planner/manifest); full suite 2056 passed with 17 failures all in the concurrent agent's uncommitted RIFE/registry scope (augment-finalize-wire/background/config/interp-order/chunk-worker/multiplier/surface suites asserting the old no-`interp_backend` defaults — e.g. live config now carries `interp_backend: 'rife'`; untouched per §9), and the tree-wide `ruff check` gate is red on 10 errors in their files (`augment_finalize/interp_poller/morph/seam`, `registry_records`).

## RIFE-by-default finalize interpolation + 2-stream finalize (2026-10-06)
- User directives (verbatim): (1) "Ensure that you use the best quality settings for RIFE"; (2) "Ensure that best quality RIFE is the default interpolation model for all interpolation needs (including fixing seams)"; (3) "Since RIFE requires much less VRAM, we should be able to make it run in parallel to the ltx25/video generation with the upscale process. It would run on the 2060 GPU like the upscale process. Ensure it can share with the llama director without OOM."; (4) "For finalization, ensure 2 parallel streams: 4060 GPU: Music, then SFX / 2060 GPU: Upscale, then interpolate"; (5) "There must be progress feedback both during the video generation phase and finalization phase to observe live progress on each stream in detail."; (6) "Changing the interpolation model must clean up interpolated frames from the previous one; otherwise, video folder gets too large."
- Quality probe (idle 4060 Ti, kaolin top-motion pairs, SRVGG upscale + FILM ref at 2048x1152): heavy 4.25 wins — sharpest (0.09257 vs FILM 0.09759), closest-to-FILM (0.02881), 0.068 s/pair vs FILM 0.85 s (~12x), identical 0.65 GiB peak all variants, clean line-art eyeball (no ghosting/warping/shimmer); fp16 stays (fp16 vs fp32 mean 0.00041). Pin: `rife_v4.25_heavy.safetensors` (86.7 MB, sha 8d0f6be4, same Comfy-Org revision as FILM, MIT), provisioned into `~/.cache/voyage-models/frame_interpolation/` + registry-verified.
- Implementation: vendored IFNet port in `workers/augment_worker.py` (`interpolate_rife_mids` — per-moment forwards, RIFE has no flow-once factorization; pad-64 + crop; resident `_RIFE_CACHE`); `AugmentConfig.interp_backend` Literal film/rife default rife + `--interp-backend` + skip-key backend component; threaded through enhance/legacy-chain/poller/seam/morph/key/background/ensure/doctor/media; `interp_pass_devices` now cuda:1 (RIFE fits the 2060 beside the ~3.2 GiB llama sidecar); background prewarm both-legs by default (1.0 GiB floor, director-idle gated); finalize fork-join restructured to Thread-A 2060 upscale→interp + Thread-B 4060 takes→bed on one shared N-stream display (mix/publish after join; 1-GPU/legacy paths byte-identical); backend-switch prune (`prune_stale_interp_plans`: whole plan-dir deletion for known-other-backend interp records, conservative keep otherwise); continuity: `weights_key_for` keeps the legacy `sha|sha` shape (the interp-leg sha already forks ledger identity on switch; kaolin keeps its 953 legacy records with zero re-render) + `read_effective_config` migrates augment-without-backend to film (RIFE never existed when those runs rendered).
- Tests: new `tests/test_interp_backend.py` (~27: registry mirror, worker validation, provisioned load/CPU, key-shape, defaults, ensure, migration, prune, dispatch) + full sweep updates (stubs/fakes/retargets/pins; racy 2-stream premises rewritten deterministically); `docs/AUGMENT.md` rewritten (RIFE default section); `docs/MODELS.md`/`INSTALL.md`/`README.md` rife entries.
- Gates: `gates.sh` green (ruff + format + mypy strict clean on 247 files; 2101 passed, 21 skipped, 0 failed; coverage 77%). UNCOMMITTED (per-commit approval required).
- Flag: kaolin holds 27 prefixed `rife|40aa...` sidecar records from another agent's in-flight RIFE experiment (different rife file, unknown-sha goes to conservative-keep — not this track's to touch).

## Chunk-interleaved upscale→interp pipeline + seam-early (2026-10-06)
- Follow-up directives (verbatim): fix the upscale producing output no interp consumes (resume wait); drop the parallel-audio scope (no audio changes); focus on upscale+interp concurrently; render seams first; same during generation; progress on upscaled/interp incl. seams during generation. Root cause: the old driver ran a full upscale sweep then a full interp sweep — interp started only after the last upscale chunk landed (~1h stall on long finalizes).
- Implementation: `_poll_to_completion` (`voyage/augment_finalize.py`) enumerates committed segments once per pass and runs each segment's upscale immediately followed by its interp (single-threaded — shared 2060 VRAM + resident caches forbid threads); pollers gained additive `segment_ids: list[str] | None = None` scoping (probed via `_accepts_keyword`, legacy fakes without the kwarg keep working); `maybe_render_seam_joint` (`voyage/augment_seam.py`: endpoints-gated, fail-soft False, same ledger identity as the drain path) attempts each boundary right after its interp when `multiplier > 1` and `seam_early=True` (off for morph-cut timelines); per-segment-leg bars (`upscale frames` then `interp frames`, source-frame units, sequential — never two concurrent Live displays); settle/cap semantics unchanged. `prewarm_once` runs the same per-segment loop (VRAM probed once per leg, `should_stop` checked per segment); Thread-A collapsed to one `run_durable_model_pass` call (split `run_upscale_phase`/`run_interp_phase` + `_do_upscale_phase`/`_do_interp_phase` + `_do_upscale_then_interp` + E6 tail interp block deleted); supervisor bar advances both legs (fixed `pumped_ip` never subtracted) + `PrewarmResult.seams_done` powers a `pre-warm seams: N rendered early` note.
- Tests: new `tests/test_model_pass_interleave.py` (5: poller scoping, up→ip-per-segment order, seam-early once with key/multiplier/backend pins, seam-early-off, per-segment-leg bars) + reworked 6 phase-fake tests to a single durable-pass fake (`test_finalize_av_stream` ×2, `test_native_skips_model_pass` ×2, `test_interp_multiplier`, `test_issue_166_model_pass_select`) + console bar pin 96→160 (both legs advance the shared bar now).
- Gates: ruff + format clean on all touched files; affected suites green. Full `gates.sh` green (exit 0: 2125 passed, 23 skipped) — committed as part of the bidirectional parallel model-pass change below.

## Bidirectional parallel model pass, finalize-only RIFE-only (2026-10-06)
- User decisions (verbatim): "1. Finalize only, yes. 2. RIFE only, yes. 3. Let us do work-stealing. 4. Yes, llama will not be up. 5. Fix [evict]." New idea driving the design: multiple concurrent interpolator/upscaler instances per GPU sharing loaded weights on different segments, spawned on both GPUs bursting to finish.
- Implementation: new `voyage/augment_parallel.py` — shared in-process task deque over every (segment, chunk); worker A pops from the front on cuda:1 (2060), worker B from the back on cuda:0 (4060), meet-in-the-middle; the same worker renders a chunk's upscale immediately followed by its interp (no cross-worker waiting; different devices = different CUDA default streams, no per-thread streams needed). Tasks dispatch as per-chunk poller calls (`segment_ids=[seg]`, `chunk_ids=[idx]` — new additive scoping on both pollers mirroring `segment_ids`, unowned missing counts as skipped); pruning runs once upfront per segment and workers pass `prune_partials=False` (new flag; a per-pass whole-dir sweep would delete the other worker's live `.partial`); ledger appends are `flock`-guarded (`fcntl LOCK_EX/UN`, matching the supervisor pattern); settle is verified on the global ledger (fail-loud `MediaError`); workers render no seams (drain fallback owns them); per-worker `stream_view` labels (`model pass A/B`); lazy torch preload in worker threads. Worker hardening in `workers/augment_worker.py` (torch thread-safety: reads safe, writes not): locked get-or-load + prepare-once (`_get_prepared_model`, `_AUGMENT_MODEL_LOCK`, `_PREPARED_MODEL_KEYS`), warp-grid per-geometry memo with `clear()` removed + double-checked locking (`_WARP_GRID_LOCK`), `cuda_stream` params on `upscale_frames`/`interpolate_rife_mids` (private `torch.cuda.Stream` per call on CUDA, nullcontext elsewhere, `record_stream` after H2D), `evict_augment_models()` under lock clearing prepared flags. `media.py` wiring: `parallel_model_pass_armed` (RIFE-only + both cuda:1/cuda:0 visible) gates an audio-first branch taking precedence over the model∥music fork — audio (music takes, then SFX bed when armed) runs first, then the bidirectional pass, then fall-through to the existing mix block with `fork_ran=True`/`bed_done_sync=True` (downstream bed/publish-thread logic skipped unchanged); post-publish `evict_augment_models()` drops resident nets. Fixed live while wiring: `_worker` passed `progress` unconditionally but neither poller accepts it (probed via `_takes_keyword`, now conditional); `augment_devices`/`augment_parallel` imported unconditionally at function level (previously branch-conditional NameError risk).
- Tests: `tests/test_parallel_model_pass.py` extended (gate matrix, `chunk_ids` scoping incl. out-of-range, `prune_partials` flag, tasks+settle fail-loud) + 3 phase-fake suites disarmed (`parallel_model_pass_armed→False`, pins hold) + new audio-first branch test (music-before-parallel order, multiplier/backend pins, output); `docs/AUGMENT.md` parallelism contract extended.
- Gates: full `gates.sh` green (exit 0: ruff + format + mypy strict clean, 2129 passed / 26 skipped). Uncommitted (per-commit approval required).
- Self-healing finalize inputs, same day (user report: kaolin finalize aborted INVALID — `upscaled_01` ledgered but output missing after a manual heal deleted mixed-geometry dirs, plus `make generate self-healing`): root cause was adoption propagating a mixed-geometry donor (frames 0-6 at 1024x576 hardlink-copied in ~6ms vs 7+ rendered at 2048x1152; the render path is whole-chunk atomic into `.partial` so only adoption's non-atomic final-dir write could do it; all 9 output-truth gates were count-only). Fix: new `augment.chunk_frames_match_size` (PIL header-only reads, fail-open without PIL so slim/fakes are unaffected, fail-closed on OSError/unparseable) hardening every gate (upscale missing loop, adopt post-copy, interp inputs + own outputs, seam cache-hit, parallel combined condition, sidecar donor pre-copy) + new `sidecar.heal_augment_ledgers` (strips any-stage records failing output count-truth mirrored on the validator, plus geometry via the record's own out_width/out_height, atomic rewrite, never raises) wired into `_heal_safe_transients` (both generate gates) and the media.py finalize-start prune site (bare `voyage finalize` verb; `validate_run` stays read-only). Healed kaolin live (validator→heal→validator in-container: 2 findings → 2 strips → 0 findings; next `generate kaolin` passes validation and the pollers re-render those chunks). Tests: 3 heal tests (count-strip/reheal = the kaolin case, healthy-keep, geometry-strip PIL-guarded) with validator-agreement pins.
- Parallel-driver progress bar, same day (user report: the kaolin log went silent after the sfx windows — `run_parallel_model_pass` rendered hundreds of chunks with zero console output): the driver now opens one `model pass chunks` bar (total = queued tasks) and advances it per finished task under the existing `timing_lock` (BarTracker._done is not thread-safe); the advance sits outside the `timings is not None` gate (user-facing progress must not depend on timings collection — the new test caught it nested inside). Test: stub-progress trio asserting total == done == queued tasks.

## Run-scoped generation scratch: /tmp must not be used (2026-10-06)
- User directive (verbatim): "/tmp was heavily in use during generation. /tmp must not be used. Use the output folder of the generation for all files, including temporary ones." Trigger: `run.sh generate boba` failed at segment 12 with `circuit breaker open for video/generate_blocks: 3 restarts exhausted (WORKER_ERROR: [Errno 122] Disk quota exceeded)` — LTX25 session work_root + mux PNG staging on the host /tmp tmpfs (31G, usrquota).
- Implementation: `paths.SCRATCH_DIRNAME = "tmp"` + `scratch_dir()`/`ensure_scratch_dir()` (mkdir + best-effort prune of stale `voyage-ltx25-*`/`voyage-ltx23-*`/`voyage-acestep-cwd-*` session dirs; only `cli_generate` constructs Supervisor so prune runs on the generate path only); `video_common.session_scratch_parent()` (explicit payload field honored, `CWD/tmp` fallback — workers spawn with CWD=run_dir); supervisor `__init__` passes `scratch_dir` in streaming-video + acestep init payloads and `start_workers` calls new `point_temp_at_run_scratch()` (TMPDIR env for spawned workers + `tempfile.tempdir` reset for self — covers bench harnesses, fake workers, and third-party libs); ltx25/ltx23 `handle_init`/`handle_rebuild` route `mkdtemp` under it (recorded in `_INIT_PARAMS` for rebuild) and `_save_mp4` gains keyword-only `staging_parent` (session work_root at both call sites); acestep/sfx `handle_init` accept + record `scratch_dir` (added to `_INIT_STR_KEYS`), take/window/bench staging + the upstream-CWD redirect parent to it (None = TMPDIR default); finalize-time ACE/SFX spawns pass it; `sfx-final` staging under run scratch; `assemble_segment_audio` gains `staging_parent` (caller passes the run-scoped finalize tmpdir). `run.sh` /tmp mount kept (torchinductor + HF model cache); `validate_run` untouched (`tmp/` invisible to the orphan scan by construction).
- Drive-by fix (same lines): ltx23 `generate_blocks` referenced undefined `profile.tail_frames` at the tail mux (NameError on every ltx23 commit — dormant backend, no test coverage) → `CONDITIONING_TAIL_FRAMES` (same value used 13 lines above).
- Tests: new `tests/test_run_scratch.py` (11: layout/ensure/prune, payload honored/fallback/reject, supervisor video+audio pins, TMPDIR backstop incl. bare-TemporaryDirectory landing, acestep/sfx init record + reject + fallbacks); `test_ltxv` payload pin extended; `test_audio_acestep_cwd` hardened with `_scratch_dir` save/restore.
- Gates: full `gates.sh` green (exit 0: ruff + format + mypy strict clean, 2144 passed / 26 skipped). Uncommitted (per-commit approval required).

## ltx25 maximum-length segments: 257f windows / 232 novel (2026-10-06)
- User directive (verbatim): "In voyage, ensure that the length of the segments that ltx25 produces are of the maximum length that is natively supported and that fit in VRAM. Do some GPU probing as needed."
- Native ceiling: 257 frames (upstream "best below 720x1280 and 257 frames"; validators already quote 257 as the 8n+1 example). VRAM fit was the open question — Mode-A-121f Spike A peaked at 14933 MiB, so a longer two-stage window was not obviously safe.
- GPU probes (idle 4060 Ti 16 GB, ephemeral drivers under `/tmp`, never committed; voyage-ltx image + host tree bind mount; monkeypatched `resolve_experiment_profile` only, production 1216x704@24 geometry): fresh 121/169/193/225/257 windows commit 121/169/193/225/257 frames at peaks 13.93/13.75/13.65/13.37/13.37 GiB (walls 150/174/196/224/265 s) — VAE tiled decode bounds memory, so length is flat-to-decreasing, not growing. 257f mp4 verified frame-exact (ffprobe 257 frames h264 1216x704@24). Production-shape probe (fresh process + `resume` from the 257f tape + NEW prompt + `scene_cut=False`, mirroring the supervisor's per-segment restart+resume): 257 generated, 25 conditioning, 232 novel committed at 13.37 GiB peak.
- Incidental (not a length regression): same-process continuation with a fresh prompt OOMs in the Gemma TE encode (`process_tokens` 3.75 GiB alloc vs ~12.4 GiB resident DiT) — the known flaky `expandable_segments` fragmentation class already recorded for 121f production (supervisor re-issues in place, never evict); production restarts the worker process per segment so it never takes that path.
- Implementation (ltx25 only; ltx23 keeps 121f/96-novel on its own row): `video_ltx25.SEGMENT_TARGET_FRAMES` 121→257 (`COMMITTED_NOVEL_FRAMES` derives 232); registry row `segment_frames` 96→232 + stage-1 latent T 16→33 (`(257-1)//8+1`) on the row and all three definition tiers (spatial axes unchanged); `cli_planning` splits `_LTX_NOVEL_BLOCK_FRAMES` (ltx23 keeps 96) with new `_LTX25_NOVEL_BLOCK_FRAMES = 232`. Old 121-era tapes resume fine (the 25f tail is length-independent; the profile hash forks but resume never gates on window length). Duration math, beat grid (adaptive-k lands 16 beats/segment), and audio takes adapt automatically.
- Tests: 257/232 pins (`test_video_ltx25` production-window test + node-8 length, `test_generate` 232-novel planning test, `test_definition_tiers` ltx25 T=33 latents split from ltx23 T=16).
- Gates: PENDING (run `Voyage/scripts/gates.sh` before commit). Uncommitted (per-commit approval required).
