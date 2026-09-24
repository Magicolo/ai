# Voyage — Video Backend Audit and Generator Substitution Task

**Task type:** architecture + performance investigation + implementation plan

**Status:** executable by coding agents

**Date:** 2026-09-23

**Primary objective:** determine whether the currently observed LongLive 2.0 performance of approximately **100 seconds of wall time per 1 second of generated video** is caused by a bad/inefficient local implementation, and then extend `DESIGN.md` so that Voyage can use either **LTX-Video 0.9.8** or **CausVid** as a video generator in place of LongLive.

**Target machine:** Linux Ubuntu, 64 GB system RAM, one 16 GB NVIDIA GPU and one 8 GB NVIDIA GPU. The exact GPU models/compute capabilities must be probed at runtime and must never be assumed from this document.

**Source-of-truth design document:** `/mnt/data/DESIGN.md`

**Important:** This task is not asking the implementation agent to immediately declare LongLive “slow.” The first requirement is to establish whether the ~100:1 ratio is genuine for the current hardware/configuration or is the result of an incorrect installation, unsupported kernel path, CPU offload, synchronization, VAE bottleneck, cache pathology, or another implementation error.

---

# 1. Background and motivation

The current Voyage design was originally centered on LongLive 2.0 / Wan2.2-TI2V-5B because it is an autoregressive long-video architecture. After actual experimentation, the observed generation rate is approximately:

```text
1 second generated video
≈ 100 seconds wall-clock time

=> 0.01 generated video seconds / wall-clock second
=> 0.24 video FPS at a 24 FPS timeline
```

At that rate:

```text
1 hour final video
≈ 100 wall-clock hours
```

This is operationally unacceptable for development and highly undesirable for production because a multi-day uninterrupted process has a high probability of encountering an OOM, driver problem, process crash, numerical failure, stalled kernel, storage problem, or semantic degeneration that could produce many hours of unusable output before the problem is noticed.

The immediate engineering goal is therefore to investigate the entire rendering path and, independently, make the video backend pluggable so that Voyage can test faster alternatives without redesigning the director, audio, persistence, novelty memory, or finalization system.

The alternatives to integrate are:

1. **LTX-Video 0.9.8 2B distilled**, preferably using the exact current 0.9.8 model/checkpoint and an appropriate low-VRAM path.
2. **CausVid**, based on Wan2.1-T2V-1.3B, using its causal DMD long-video continuation strategy.

---

# 2. External source baseline that must be verified

Do not copy benchmark numbers into code or claim that they predict the user's hardware. They are investigation baselines only.

## 2.1 LongLive 2.0

Current upstream repository:

- https://github.com/NVlabs/LongLive
- https://nvlabs.github.io/LongLive/LongLive2/docs/
- https://arxiv.org/abs/2605.18739

Current upstream README lists approximately:

| Model | Upstream published FPS | Parameters |
|---|---:|---:|
| LongLive-1.3B | 20.7 FPS | 1.3B |
| LongLive-2.0-5B | 24.8 FPS | 5B |
| LongLive-2.0-5B NVFP4 4-step | 29.7 FPS | 5B |
| LongLive-2.0-5B NVFP4 2-step | 45.7 FPS | 5B |

The README also states that the NVFP4 inference path was optimized with fused Triton RoPE/adaLN kernels, reduced KV-cache synchronization overhead, in-place quantized KV-cache updates, faster FP4 dequantization, and pinned VAE transfers.

These figures are reference evidence that the architecture itself is capable of high throughput on the upstream test hardware. The exact local ratio must be measured with CUDA events and wall-clock accounting.

## 2.2 LTX-Video

Current upstream repositories:

- https://github.com/Lightricks/LTX-Video
- https://huggingface.co/Lightricks/LTX-Video
- https://github.com/Lightricks/ComfyUI-LTXVideo

Current model collection includes:

- `ltxv-2b-0.9.8-distilled`
- `ltxv-2b-0.9.8-distilled-fp8`
- corresponding 13B variants

The upstream repository describes the 2B distilled model as a light-VRAM/fast-generation model. The 0.9.x line supports video extension / conditioning and requires frame counts compatible with its temporal architecture. Do not assume the exact 0.9.6 rules or sampling schedule are unchanged in 0.9.8; inspect the pinned source and checkpoint metadata.

The Hugging Face metadata for the 0.9.8 models currently reports license `other`; treat the exact bundled/versioned license as authoritative.

## 2.3 CausVid

Current upstream repositories:

- https://github.com/tianweiy/CausVid
- https://huggingface.co/tianweiy/CausVid
- https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B

The CausVid README reports approximately 9.4 FPS on a single GPU and provides both a short autoregressive path and a dedicated long-video autoregressive rollout implementation. The Hugging Face documentation currently identifies the CausVid model release as CC BY-NC-SA 4.0.

The upstream long-video implementation reconstructs continuation state using an overlap of decoded video and latent tensors rather than requiring an indefinitely retained Python/GPU process.

---

# 3. Required deliverables

This task is complete only when all of the following exist:

```text
TASK.md                         this task specification
DESIGN.md                      updated architecture
reports/longlive-audit.md     detailed LongLive performance/correctness report
reports/video-backends.md     comparative backend benchmark report
docs/UPSTREAM_LONG_LIVE_PATCHES.md  LongLive-specific deviations/patch notes

docs/UPSTREAM_LTXV_NOTES.md         LTX-Video-specific integration notes
docs/UPSTREAM_CAUSVID_NOTES.md      CausVid-specific integration notes
```

The implementation may create additional benchmark artifacts such as:

```text
reports/benchmarks/<backend>/<timestamp>/
    metadata.json
    timings.json
    profiler.json
    nvidia-smi.txt
    torch-config.txt
    environment.txt
    output.mp4
    frame-samples/
```

`DESIGN.md` must become backend-neutral at the supervisor/application layer while retaining precise backend-specific implementation notes.

---

# 4. Non-negotiable investigation rules

## 4.1 Do not use wall-clock timing alone

All GPU-sensitive timing must use CUDA events or another GPU-aware timing mechanism in addition to wall-clock time.

Bad:

```python
start = time.perf_counter()
run_model()
elapsed = time.perf_counter() - start
```

This measures submission/host behavior and may be misleading if CUDA work is asynchronous.

Required pattern:

```python
start_event.record()
run_model()
end_event.record()
end_event.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

Also record wall-clock time because CPU scheduling, data movement, encoding, and blocking operations matter operationally.

## 4.2 Separate first-call, warm-up, and steady-state timing

At minimum report:

```text
cold start
model load
first inference
warm-up inference(s)
steady-state block/segment timings
VAE timings
final encode/write timings
```

If `torch.compile` is used, compile/warm-up time must be reported separately and excluded from steady-state throughput.

## 4.3 Do not optimize around an unverified output count

Compute the ratio from **actually committed novel frames**, not from requested tensor shape.

For every segment record:

```text
requested frames
returned frames
conditioning frames
overlap frames
novel frames
committed frames
native FPS
presentation FPS
```

Then compute:

```text
video_seconds_committed = committed_frames / native_or_timeline_fps
ratio = wall_seconds / video_seconds_committed
novel_fps = committed_frames / wall_seconds
```

For LTX and CausVid, overlap/conditioning frames can make a naive FPS calculation look significantly better than the true incremental generation rate.

---

# 5. Phase A — Hardware and software inventory

Create `reports/longlive-audit.md` with an immutable hardware snapshot.

## 5.1 NVIDIA inventory

Run and capture:

```bash
nvidia-smi
nvidia-smi -q
nvidia-smi --query-gpu=index,name,uuid,compute_cap,memory.total,memory.used,memory.free,driver_version,pstate,power.limit,power.draw,temperature.gpu,clocks.gr,clocks.mem,utilization.gpu,utilization.memory --format=csv
```

Also capture:

```bash
lspci -nn | grep -i -E 'vga|3d|display|nvidia'
```

Determine:

- exact GPU model for 16 GB device;
- exact GPU model for 8 GB device;
- CUDA compute capability;
- driver version;
- PCIe link width and speed if relevant;
- whether MIG is enabled;
- whether persistence mode is enabled;
- current power limit;
- clock throttling indicators;
- whether ECC is active on a workstation/datacenter GPU;
- whether the GPU is shared by another process.

Do not assume that `cuda:0` is the 16 GB GPU. Record the actual mapping.

## 5.2 PyTorch/CUDA inventory

Capture:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("torch cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory, getattr(p, "major", None), getattr(p, "minor", None))
    print(torch.cuda.get_device_capability(i))
print(torch.__config__.show())
PY
```

Capture:

- PyTorch version;
- CUDA runtime compiled into PyTorch;
- cuDNN;
- compute capability;
- enabled attention backends;
- BF16/FP8 support;
- FlashAttention/SageAttention/Triton availability where used.

## 5.3 Process isolation

During benchmarks, verify that the GPU is not being used by:

- ComfyUI;
- another Python process;
- desktop display server if relevant;
- browser GPU acceleration that materially consumes VRAM;
- another model worker;
- orphaned CUDA processes.

Record `nvidia-smi` both before and during inference.

---

# 6. Phase B — Verify the LongLive installation is exactly what the code thinks it is

This phase must happen before profiling the model.

## 6.1 Pin the repository revision

From the LongLive checkout capture:

```bash
git remote -v
git status --short
git rev-parse HEAD
git show -s --format='%H%n%ci%n%s' HEAD
```

Also capture all local modifications:

```bash
git diff --no-ext-diff > reports/longlive-audit/local-diff.patch
git diff --cached --no-ext-diff > reports/longlive-audit/staged-diff.patch
```

The report must state explicitly whether the code is pristine upstream, locally patched, or a mixture.

## 6.2 Verify model checkpoint identity

For every model/checkpoint file record:

- repository ID;
- exact revision/commit/tag;
- file path;
- local file size;
- SHA-256;
- whether the file is BF16, FP8, NVFP4, prequantized, or a base checkpoint;
- whether an attached LoRA is loaded;
- whether EMA weights are used.

Never benchmark a model merely because the filename says `NVFP4-S2`.

## 6.3 Verify the configuration actually used at runtime

Dump the resolved merged configuration, not just the source YAML.

Confirm all of the following:

```text
model_name
num_frame_per_block
local_attn_size
sampling_steps
sink_size
multi_shot_sink
multi_shot_rope_offset
use_relative_rope
model_quant
kv_quant
model_quant_use_transformer_engine
streaming_vae
async_vae
vae_device
guidance_scale
resolution / latent shape
num_output_frames
```

If the running code reads configuration through a normalization layer, dump the **post-normalization** configuration.

## 6.4 Verify no accidental debug path is active

Inspect environment variables:

```bash
env | sort | grep -E 'LLV2|CUDA|TORCH|PYTORCH|NVIDIA|TRITON|FLASH|ATTN|OMP|MKL'
```

LongLive currently exposes instrumentation variables such as:

```text
LLV2_TIME
LLV2_PROFILE
LLV2_PROFILE_OUTPUT_DIR
LLV2_DUMP_LATENT_DIR
LLV2_DEBUG_KV
```

Debug instrumentation that forces GPU-to-CPU synchronization can destroy throughput. In particular, scalar CUDA tensors must not be printed from the hot loop unless deliberately benchmarking the debug path.

The upstream source already contains a note about a removed per-step KV debug print because it forced GPU→CPU synchronization repeatedly. Verify that the local checkout has not reintroduced an equivalent path.

---

# 7. Phase C — Establish a minimal correctness benchmark

The purpose is to separate “the model is slow” from “the model is doing something unintended.”

## 7.1 Benchmark a tiny finite render

Run the smallest legal upstream configuration that still exercises the chosen checkpoint.

Produce:

```text
1 prompt
1 sample
1 block
upstream reference dimensions
exact upstream sampling steps
VAE enabled
video written to disk
```

Record:

- total wall time;
- GPU kernel time;
- model time;
- VAE time;
- output dimensions;
- output FPS;
- output frame count;
- peak VRAM.

## 7.2 Benchmark latent-only mode

Run the same generation while suppressing final MP4 encoding and, where supported, VAE decoding.

This isolates:

```text
transformer/KV-cache cost
vs.
VAE cost
vs.
video encoding cost
```

If latent-only generation is fast but full video generation is near 100:1, the bottleneck is not the diffusion model.

## 7.3 Benchmark VAE-only decode

Take a fixed latent tensor generated once and decode it repeatedly.

Measure:

```text
VAE device
VAE dtype
per-chunk decode time
transfer time
synchronization time
```

Repeat with:

- synchronous decode;
- streaming VAE;
- asynchronous VAE if enabled;
- VAE on the same GPU;
- VAE on the secondary GPU only if the pipeline explicitly supports this topology.

Do not move the VAE to the 8 GB GPU merely because that GPU is less busy; PCIe transfer can dominate.

## 7.4 Benchmark “save nothing” mode

Do a run that generates and immediately discards outputs after correctness checks.

This identifies disk encoding as a hidden bottleneck.

Do not measure generation throughput through an H.264 encoder unless the encoder cost is explicitly separated.

---

# 8. Phase D — Verify the quantized kernel path

This is a critical investigation area because the LongLive README's headline throughput depends heavily on optimized NVFP4 execution.

## 8.1 Verify the selected quantization backend

For an NVFP4 model, determine whether the runtime is using:

```text
Transformer Engine
or
FourOverSix
```

based on the actual checkpoint and configuration.

Print/record:

```text
model_quant = true/false
kv_quant = true/false
model_quant_use_transformer_engine = true/false
checkpoint prequantized = true/false
```

Fail the audit if the model file says NVFP4 but the code silently loads it as BF16 or invokes an incompatible fallback.

## 8.2 Check for Python/reference fallback kernels

Use:

```bash
python -c 'import triton; print(triton.__version__)'
python -c 'import torch; print(torch.cuda.get_device_name())'
```

and inspect import/module availability for the exact LongLive quantization stack.

Do not infer kernel use merely because the import succeeded. Profile actual CUDA kernels.

## 8.3 Profile representative blocks

Use `torch.profiler` and/or Nsight Systems for:

- first generated block;
- middle block;
- late block;
- scene/prompt transition block.

Look for:

- tiny CUDA kernels dominating time;
- large numbers of kernel launches;
- repeated quantize/dequantize operations;
- host-device synchronization;
- `cudaMemcpy` between host and device;
- `cudaDeviceSynchronize` or equivalent barriers;
- attention operations falling back to generic implementations;
- VAE kernels serialized on the main stream;
- KV-cache copies that scale with total history rather than local window size.

## 8.4 Compare FP4 and BF16 paths

Run exactly the same finite output through:

```text
NVFP4 S2
NVFP4 S4
BF16 4-step
```

where the exact checkpoint/config supports each path.

Interpret results:

```text
NVFP4 much slower than BF16
    => likely broken/unsupported quantization execution path

BF16 also extremely slow
    => investigate model size, resolution, CPU offload, attention, VAE, or hardware

NVFP4 fast for one block but slow for many blocks
    => investigate cache growth, cache synchronization, relative RoPE, or repeated recompilation
```

---

# 9. Phase E — Detect CPU offload, PCIe transfers, and hidden synchronization

This is one of the highest-priority checks on a 16 GB GPU.

## 9.1 Detect model/device placement

At runtime print the device and dtype for:

```text
generator model
text encoder
VAE
KV cache tensors
cross-attention cache tensors
prompt embeddings
noise
latents
streaming VAE buffers
```

Every hot-path tensor must have an intentional device placement.

## 9.2 Search source for explicit transfers

Audit the running LongLive code for:

```python
.cpu()
.cuda()
.to("cpu")
.to(device)
.to(vae_device)
torch.cuda.synchronize()
torch.cuda.empty_cache()
```

Classify each occurrence:

```text
startup-only
per-segment
per-block
per-denoising-step
per-token
```

Any CPU/GPU transfer inside the denoising loop must be treated as suspicious until justified.

## 9.3 Detect allocator churn

A call to `torch.cuda.empty_cache()` in the hot loop can cause expensive allocator synchronization and memory churn.

Measure with it enabled and disabled where safe.

Do not blindly remove it from the upstream code. First establish whether it is protecting against fragmentation/OOM and whether a fixed memory profile removes the need for it.

## 9.4 Measure PCIe traffic

Use Nsight Systems or another approved profiler to look for:

```text
Host To Device
Device To Host
```

transfers during a single generated block.

The expected pattern is model computation on GPU with only deliberate VAE/output transfers. Unexpected repeated transfers are a likely root cause.

---

# 10. Phase F — Measure cache complexity and block scaling

LongLive is supposed to use local attention and a bounded cache, but the implementation can still become slow if history management is accidentally linear in total video length.

Generate progressively longer sequences:

```text
1 block
2 blocks
4 blocks
8 blocks
16 blocks
32 blocks
64 blocks if feasible
```

For each block index record:

```text
block_index
block_wall_ms
transformer_cuda_ms
KV update ms
VAE ms
output/transfer ms
peak VRAM
```

Plot:

```text
block index → milliseconds per block
block index → peak VRAM
```

Interpretation:

```text
flat timing curve
    => bounded steady-state cache behavior

linear increase
    => likely growing attention/cache work

sawtooth increases
    => periodic cache compaction/flush/recompile

large spikes at prompt changes
    => cross-attention or recompilation overhead

large spikes at scene cuts
    => sink/pinned-cache operations
```

The performance report must explicitly state whether generation time per newly generated block is approximately constant.

---

# 11. Phase G — Test `torch.compile` and shape specialization correctly

LongLive's current FP8 documentation uses `torch.compile` and warns that different KV-cache shapes can trigger additional compilation or eager fallback.

This is a critical possibility for an implementation that appears catastrophically slow.

## 11.1 Run eager mode first

Establish a clean baseline with:

```text
compile disabled
```

## 11.2 Run compile mode second

Then enable the exact upstream-supported compile setting.

Capture:

- compile/warm-up duration;
- number of recompilations if visible;
- steady-state throughput after all relevant shapes are compiled;
- whether longer rollouts introduce new shapes.

## 11.3 Detect accidental compile thrashing

If every block or every cache shape causes a compilation, throughput can collapse.

The report must identify whether:

```text
cache shape is static
or
cache shape changes every block
```

and whether PyTorch recompiles as a result.

If necessary, pre-warm the exact shape sequence used by the production profile before measuring throughput.

---

# 12. Phase H — Validate VAE configuration and decode overlap

The current LongLive source supports:

```text
streaming_vae
async_vae
vae_device
```

The audit must benchmark all relevant combinations:

| Mode | Transformer | VAE | Purpose |
|---|---|---|---|
| A | GPU | same GPU | baseline |
| B | GPU | separate GPU | test PCIe/overlap tradeoff |
| C | GPU | streaming VAE | memory behavior |
| D | GPU | async streaming VAE | overlap |
| E | GPU | no decode | diffusion-only ceiling |

The result must state the fraction of total generation time spent in VAE work.

If `async_vae` is enabled, verify that it really overlaps with subsequent transformer work rather than merely moving decode to another CUDA stream while the main stream immediately waits for it.

Use CUDA events and Nsight Systems timeline visualization to prove overlap.

---

# 13. Phase I — Verify prompt/text encoder costs are not hidden in the benchmark

LongLive's per-block prompt conditioning should not cause full text encoding every block unless the implementation intentionally does so.

Measure:

```text
prompt tokenization
text encoder
prompt embedding reshaping
cross-attention cache build
```

Compare:

```text
same prompt for every block
vs.
prompt changed every block
```

This tells us the upper bound on director-driven prompt changes.

For Voyage, prompt stages should normally span multiple blocks specifically because regenerating expensive prompt features can otherwise destroy throughput.

---

# 14. Phase J — Validate the actual LongLive streaming implementation

The current Voyage implementation previously required extraction/refactoring of LongLive's per-block loop into a persistent session. That code is especially vulnerable to performance mistakes.

Audit the custom adapter for:

## 14.1 Accidental pipeline reset

Check that the code does not recreate `CausalDiffusionInferencePipeline`, generator objects, KV caches, or VAE objects for every block/segment.

## 14.2 Accidental `pipeline.inference()` full-output path

The upstream `inference()` implementation builds output buffers and manages call-level state. If Voyage calls it repeatedly expecting a persistent stream, the cache may be reinitialized or the whole output may be reallocated.

Confirm that the production path uses a genuinely persistent session or an upstream-supported long-rollout interface.

## 14.3 Hidden full-history copies

Inspect every operation that concatenates video/latents/cache tensors.

Suspicious patterns include:

```python
torch.cat(history + [new])
output = torch.cat(outputs)
cache = torch.cat([cache, new_cache])
```

inside the per-block loop.

The correct architecture should use bounded ring/rolling buffers where possible and append only to durable segment files at the application layer.

## 14.4 Excessive Python-side bookkeeping

The hot loop must not:

- serialize JSON per denoising step;
- write latents per step;
- calculate CPU-side statistics every token/block;
- print CUDA scalar values;
- run embedding searches every denoising step;
- invoke the director during diffusion;
- invoke ffmpeg during diffusion.

All such operations belong outside the render critical path.

---

# 15. Phase K — Test for hardware saturation and power/clock anomalies

During steady-state rendering capture:

```text
GPU utilization
GPU memory utilization
SM clock
memory clock
power draw
power limit
temperature
```

Interpretation:

```text
low GPU utilization + high CPU utilization
    => CPU bottleneck or synchronization/offload

high GPU utilization + low clocks/power
    => throttling or clock configuration

high GPU utilization + high power + low throughput
    => actual compute-heavy model path; optimize model/resolution/steps

GPU utilization oscillates 0%↔100%
    => synchronization, data movement, or serial VAE likely
```

Use the 16 GB GPU for the video benchmark unless the exact profile explicitly requires another device.

Do not co-locate ACE-Step on the same GPU during video benchmarks.

---

# 16. Phase L — Produce a LongLive root-cause decision

At the end of the audit, the report must choose exactly one primary diagnosis category, with evidence:

```text
A. local implementation bug
B. unsupported/fallback kernel path
C. CPU/offload/transfer bottleneck
D. VAE/encoding bottleneck
E. compile/recompilation problem
F. cache/history scaling problem
G. hardware limitation
H. configuration/resolution/step mismatch
I. mixed causes
J. insufficient evidence
```

If category I is selected, quantify the main contributors.

The report must include a time budget that approximately closes:

```text
wall time
≈ transformer
 + VAE
 + transfers
 + CPU orchestration
 + compile
 + I/O
 + unexplained overhead
```

Target accounting completeness: **at least 95% of wall-clock time** for the representative benchmark must be assigned to measured categories or explicitly marked as profiler overhead.

Do not conclude “LongLive is simply slow” unless the measured transformer execution itself accounts for the majority of wall time and no implementation error explains the discrepancy.

---

# 17. Phase M — Refactor DESIGN.md to a backend-neutral video architecture

Update `/mnt/data/DESIGN.md` so the following become explicit:

1. LongLive is no longer the only renderer assumption.
2. `VideoBackend` is a first-class interface.
3. Each backend declares capabilities and continuation state mode.
4. Run configuration selects exactly one backend.
5. Run manifests store the exact backend and model revision.
6. Segment persistence is independent of backend internals.
7. Finalization is independent of backend internals.
8. Audio and director components remain unchanged.
9. Backend-specific configuration stays in backend profile namespaces.
10. Public CLI options should expose only a small common set plus backend selection; specialized settings live in config profiles.

The following common interface must exist conceptually, even if the actual Python protocol differs:

```python
class VideoBackend(Protocol):
    async def initialize(self, profile: VideoProfile) -> BackendCapabilities: ...
    async def generate_segment(self, request: VideoSegmentRequest) -> VideoSegmentResult: ...
    async def checkpoint(self) -> VideoBackendCheckpoint | None: ...
    async def restore(self, checkpoint: VideoBackendCheckpoint) -> None: ...
    async def health(self) -> BackendHealth: ...
    async def shutdown(self) -> None: ...
```

The interface must not expose raw CUDA tensors to the supervisor.

---

# 18. Phase N — LTX-Video backend implementation

Implement `LTXVBackend` in a dedicated worker environment.

## 18.1 Model target

Initial model:

```text
Lightricks/LTX-Video
ltxv-2b-0.9.8-distilled
```

Optional performance model:

```text
ltxv-2b-0.9.8-distilled-fp8
```

Do not upgrade automatically to a 13B model. The first reason for selecting LTX is low-VRAM speed and iteration throughput.

## 18.2 Worker responsibilities

The worker owns:

- model load;
- tokenizer/text encoder;
- LTX VAE;
- latent generation;
- conditioning-media encode;
- prompt embedding;
- optional supported acceleration kernels;
- output tensor/temporary clip generation;
- worker-local cleanup.

The worker does **not** own:

- novelty memory;
- world state;
- audio;
- final MP4 assembly;
- persistent run transaction state.

## 18.3 Request schema

Conceptual request:

```json
{
  "request_id": "...",
  "segment_index": 42,
  "prompt": "...",
  "negative_prompt": "",
  "seed": 123,
  "width": 768,
  "height": 432,
  "target_frames": 121,
  "fps": 24,
  "conditioning_path": "segments/000041/video_tail.mp4",
  "conditioning_start_frame": 0,
  "conditioning_strength": 1.0,
  "stochastic_sampling": false
}
```

The exact request must be translated into the pinned LTX API rather than relying on the example names forever.

## 18.4 Prefix accounting

The worker response must return:

```json
{
  "requested_frames": 121,
  "returned_frames": 121,
  "conditioning_frames": 25,
  "novel_frames": 96,
  "native_fps": 24,
  "output_path": "...",
  "metrics": { ... }
}
```

If the actual pipeline returns a different frame count, compute the real counts rather than trusting the request.

## 18.5 Crash behavior

If the worker dies after generating an uncommitted LTX clip:

- discard the temporary clip;
- keep the previous committed segment;
- restart worker;
- regenerate using the stored seed/profile/prompt/conditioning tail;
- commit only after media validation succeeds.

This is substantially simpler than serializing LongLive's KV cache and is one of the reasons the backend must be supported.

---

# 19. Phase O — CausVid backend implementation

Implement `CausVidBackend` in its own environment.

## 19.1 Model stack

Use:

```text
Wan2.1-T2V-1.3B
+
CausVid autoregressive checkpoint
```

Start from the upstream `configs/wan_causal_dmd.yaml` and `minimal_inference/longvideo_autoregressive_inference.py` for behavior.

## 19.2 Preserve the upstream continuation algorithm first

Do not redesign the long-video recurrence on the first implementation.

The upstream script performs approximately:

```text
for each rollout:
    sample noise
    generate with start_latents
    encode selected decoded tail
    concatenate encoded tail + previous latent overlap
    use result as start_latents for next rollout
    drop overlapped decoded frames
```

Reproduce this behavior first, benchmark it, then optimize.

## 19.3 Request schema

Conceptual request:

```json
{
  "request_id": "...",
  "segment_index": 42,
  "prompt": "...",
  "seed": 123,
  "target_chunk_latent_frames": 21,
  "num_overlap_frames": 3,
  "profile": "causvid-default",
  "native_fps": 16
}
```

The actual implementation may use file references plus a small latent artifact path to avoid copying large tensors through JSONL.

## 19.4 Latent continuation artifact

The worker should preferably persist continuation state as a bounded artifact:

```text
segments/000042/
    video.mp4
    causvid_start_latents.safetensors
    causvid_state.json
```

`causvid_state.json` must record:

- latent shape;
- dtype;
- overlap count;
- source frame range;
- source segment ID;
- RNG seed;
- backend revision;
- checkpoint revision;
- checksum of latent artifact.

## 19.5 FPS policy

The upstream examples use 16 FPS. The generic Voyage `fps = 24` assumption must not leak into CausVid.

The backend must report native FPS, and the supervisor/finalizer must either:

- make the whole CausVid run 16 FPS; or
- invoke a separately specified and validated frame-rate conversion stage.

Do not simply relabel 16 FPS media as 24 FPS.

---

# 20. Phase P — Cross-backend tests

Create parameterized tests over:

```text
LongLive2Backend
LTXVBackend
CausVidBackend
```

## 20.1 Interface tests

Every backend must satisfy:

- initialize;
- health;
- generate one finite segment;
- report actual frame counts;
- report native FPS;
- produce an artifact that the supervisor can validate;
- shutdown cleanly.

## 20.2 Recovery tests

For each backend:

```text
start segment N
kill worker at ~25% progress
restart worker
resume from segment N-1 commit
validate
```

For LongLive, recovery may replay cached latent context.

For LTX, recovery reuses the previous segment tail.

For CausVid, recovery reuses the latent overlap artifact or reconstructs it from the committed tail.

## 20.3 Prompt-transition tests

Generate:

```text
prompt A for segment 0
prompt A/B transition for segments 1-2
prompt B for segment 3
```

The result must not hard-cut unless the transition plan explicitly requests a scene cut.

## 20.4 Timeline tests

Verify:

```text
no duplicated committed frames
no missing timeline ranges
monotonic timestamps
correct FPS metadata
correct segment duration
```

## 20.5 Deterministic replay tests

For each backend, where deterministic mode is supported:

```text
same seed
same model revision
same environment
same input prefix
```

should produce sufficiently similar output for recovery validation.

Do not require bit-exact equality from all GPU kernels unless the backend explicitly guarantees it.

---

# 21. Comparative benchmark matrix

After implementations exist, run a standardized benchmark.

## 21.1 Required dimensions

At minimum:

```text
backend:
    longlive2
    ltxv
    causvid

resolution:
    each backend's validated native/low-res profile
    approximately 768×432 when supported

segment duration:
    ~3 s
    ~5 s
    ~10 s

quality mode:
    default production candidate
    fastest validated mode
```

## 21.2 Metrics

Report:

```text
cold_start_seconds
model_load_seconds
first_segment_seconds
steady_state_segment_seconds
novel_video_seconds
novel_fps
wall_to_video_ratio
peak_vram_gib
peak_ram_gib
transformer_seconds
vae_seconds
transfer_seconds
encode_seconds
cpu_utilization
mean_gpu_utilization
p95_block_ms
p99_block_ms
worker_restart_recovery_seconds
```

## 21.3 Primary decision metric

For Voyage development, the primary metric is:

```text
wall_seconds / committed_novel_video_second
```

Lower is better for iteration speed.

The secondary metrics are:

```text
recovery simplicity
continuation quality
VRAM headroom
stability over 10+ minutes
```

Do not select a backend solely on model quality or benchmark FPS.

---

# 22. Performance target for this task

No single target is required before measurements exist. However, the investigation should explicitly answer whether each backend falls into one of these practical ranges:

```text
< 2:1       excellent for Voyage; near-real-time production
2:1–5:1     very strong; multi-hour generation becomes practical
5:1–15:1    useful; one-hour video takes 5–15 hours
15:1–30:1   usable mainly for development / selected production runs
30:1–100:1  poor for a one-hour autonomous run
> 100:1     unacceptable for the current goals unless quality is extraordinary
```

These are **engineering categories, not quality ratings**. The document should record the measured number, not award a model an overall score.

---

# 23. Required updates to DESIGN.md

The implementation agent must verify that `/mnt/data/DESIGN.md` contains all of the following:

## 23.1 Backend-neutral architecture

The architecture diagram must show:

```text
Supervisor
   │
   ▼
VideoBackend interface
   ├── LongLive2Backend
   ├── LTXVBackend
   └── CausVidBackend
```

## 23.2 LongLive details

Keep:

- causal KV-cache architecture;
- relative RoPE;
- multi-shot sink;
- NVFP4/FP8 paths;
- persistent stream/recovery semantics;
- exact upstream references.

Add the audit conclusion once Phase A–L is complete.

## 23.3 LTX details

Add:

- exact 0.9.8 checkpoint identifiers;
- low-VRAM rationale;
- conditioning-prefix continuation model;
- frame-count constraints;
- target frame multiple-of-8 constraint;
- stateless/reconstructable recovery;
- exact licensing caveat;
- performance benchmark modes;
- worker API and result accounting.

## 23.4 CausVid details

Add:

- Wan2.1-T2V-1.3B base;
- causal DMD configuration;
- 3-step default path;
- 21-latent-frame chunk example;
- overlap reconstruction;
- native 16 FPS behavior;
- latent continuation artifact;
- CC BY-NC-SA 4.0 license note;
- long-video upstream example;
- worker API and result accounting.

## 23.5 Configuration

`[video] backend` must accept:

```text
longlive2
ltxv
causvid
```

Backend-specific values must live under:

```text
[video.longlive]
[video.ltxv]
[video.causvid]
```

Do not flatten all model-specific fields into the generic configuration.

---

# 24. Common implementation mistakes to avoid

## 24.1 Do not implement LTX as an “infinite latent cache”

LTX's initial Voyage integration should be a restartable conditioned extension system. Trying to force it into the LongLive cache abstraction will add complexity without a clear benefit.

## 24.2 Do not force CausVid through the LongLive cache interface

CausVid already has a practical continuation mechanism using overlap latent state. Use it directly.

## 24.3 Do not send tensors over JSONL

Use filesystem artifacts or a local shared-memory mechanism later if necessary. V1 should prefer files plus checksums over complicated IPC tensor transport.

## 24.4 Do not call the director from the diffusion hot loop

The director operates at segment/prompt-stage cadence, never per denoising step.

## 24.5 Do not use ffmpeg in the renderer worker's hot path

The renderer should emit validated intermediate artifacts. Encoding/muxing belongs in separate application-level stages.

## 24.6 Do not silently upscale during benchmark runs

Benchmark native generation first. Final output scaling belongs to finalization.

## 24.7 Do not compare backends at different definitions of “one second”

A CausVid segment with 16 FPS and an LTX segment with 24 FPS must be normalized by actual committed video seconds.

## 24.8 Do not hide warm-up

A benchmark that reports only warm steady-state throughput is valid, but it must say so. Also report cold-start cost because Voyage must survive worker restarts.

---

# 25. Acceptance criteria

The coding agent should not mark the task complete until all criteria below are satisfied.

## 25.1 LongLive audit acceptance

- [ ] exact hardware recorded;
- [ ] exact LongLive commit recorded;
- [ ] exact checkpoint hash recorded;
- [ ] resolved runtime config recorded;
- [ ] quantization path verified;
- [ ] actual CUDA kernels profiled;
- [ ] CPU/GPU transfer analysis performed;
- [ ] VAE-only benchmark performed;
- [ ] latent-only benchmark performed;
- [ ] save-disabled benchmark performed;
- [ ] block-scaling experiment performed;
- [ ] eager vs compile comparison performed if compile is available;
- [ ] cache growth/shape behavior measured;
- [ ] timing accounting closes to ≥95% of wall time or explicitly documents profiler uncertainty;
- [ ] root-cause category chosen with evidence;
- [ ] conclusion does not blame the model without eliminating implementation causes.

## 25.2 LTX acceptance

- [ ] exact 0.9.8 model/revision recorded;
- [ ] legal/license metadata archived;
- [ ] 2B distilled finite generation works;
- [ ] continuation from a previous tail works;
- [ ] novel-frame count is correct;
- [ ] worker restart recovers from previous segment;
- [ ] benchmark has a real wall/video ratio;
- [ ] low-VRAM profile fits the 16 GB target or reports why it does not;
- [ ] no dependence on ComfyUI runtime nodes exists in production worker.

## 25.3 CausVid acceptance

- [ ] exact CausVid commit/checkpoint recorded;
- [ ] Wan2.1-1.3B base model recorded;
- [ ] upstream 3-step causal inference works;
- [ ] long-video overlap continuation works;
- [ ] latent continuation artifact is persisted;
- [ ] worker restart can continue from a committed segment;
- [ ] native 16 FPS behavior is represented explicitly;
- [ ] benchmark reports novel FPS and wall/video ratio;
- [ ] license metadata is archived.

## 25.4 DESIGN.md acceptance

- [ ] LongLive remains supported;
- [ ] LTX-Video is a first-class generator option;
- [ ] CausVid is a first-class generator option;
- [ ] supervisor architecture is generator-neutral;
- [ ] backend capabilities/state modes are documented;
- [ ] backend-specific configs are separate;
- [ ] recovery semantics differ explicitly between persistent-KV and reconstructable-prefix backends;
- [ ] model-download commands are documented;
- [ ] license table includes all three video backends;
- [ ] canonical references point to current upstream sources.

---

# 26. Final report format

At the end of the implementation work, `reports/video-backends.md` should contain this exact high-level structure:

```markdown
# Voyage Video Backend Benchmark Report

## Hardware

## Software / revisions

## LongLive audit conclusion

## LongLive measurements

## LTX-Video measurements

## CausVid measurements

## Recovery measurements

## Visual-continuation observations

## Resource utilization

## Failure modes encountered

## Interpretation

## Selected development backend

## Selected production candidate

## Known unresolved risks

## Reproduction commands
```

Do not put a subjective “winner” section into the report. State the measured properties and engineering trade-offs. The user will choose the backend.

---

# 27. Suggested implementation order

Execute in this order:

```text
1. Hardware inventory
2. LongLive version/checkpoint/config verification
3. LongLive finite correctness benchmark
4. LongLive profiler / root-cause audit
5. VideoBackend interface refactor
6. LTX-Video worker + finite generation
7. LTX continuation + recovery
8. CausVid worker + finite generation
9. CausVid long continuation + recovery
10. Cross-backend benchmark suite
11. DESIGN.md reconciliation
12. Documentation / license capture
13. Multi-hour bake test of selected backend(s)
```

Do not skip directly from LongLive's poor measured performance to selecting LTX or CausVid without completing the audit. The audit is important because any optimization discovered in the LongLive path may also improve the alternative backends, and an implementation bug could otherwise be mistaken for a model limitation.

---

# 28. Source-code inspection checklist for coding agents

Before making architectural claims, inspect the actual pinned source for:

## LongLive

```text
pipeline/causal_diffusion_inference.py
utils/inference_utils.py
utils/prompt_conditioning.py
wan_5b/modules/causal_model.py
utils/wan_5b_wrapper.py
configs/inference.yaml
configs/fp8/inference_fp8.yaml
configs/nvfp4/inference_nvfp4.yaml
```

Specifically trace:

```text
model construction
checkpoint loading
quantization wrapping
prompt encoding
KV cache allocation
KV cache update
local attention rolling
relative RoPE
scene-cut / shot-sink logic
VAE decode
asynchronous VAE queueing
output conversion
```

## LTX-Video

Inspect:

```text
ltx_video/inference.py
ltx_video/pipelines/pipeline_ltx_video.py
configs/ltxv-2b-0.9.8-distilled.yaml
configs/ltxv-2b-0.9.8-distilled-fp8.yaml
```

Trace:

```text
conditioning item construction
media VAE encoding
prompt encoding
frame count normalization
resolution binning
sampling timestep schedule
VAE decode
extension output frame accounting
```

## CausVid

Inspect:

```text
minimal_inference/autoregressive_inference.py
minimal_inference/longvideo_autoregressive_inference.py
configs/wan_causal_dmd.yaml
causvid/models/wan/causal_inference.py
```

Trace:

```text
noise construction
start_latents semantics
overlap selection
VAE tail encoding
latent concatenation
causal KV cache
chunk output slicing
FPS metadata
```

The coding agent must not rely solely on README descriptions when implementing these contracts.

---

# 29. Definition of done

This task is complete when Voyage has:

```text
one supervisor
three interchangeable video backends
one audio backend
one director
one persistent run format
one finalizer

and the following property:

A video backend can crash without invalidating already committed segments,
and another backend can be selected for a new run without changing the
supervisor or director architecture.
```

Most importantly, the project must have an evidence-backed answer to:

> Why did LongLive take approximately 100 seconds to produce 1 second of video on this machine, and how much of that was caused by the implementation rather than the model?

That answer is a required engineering artifact, not an assumption.

---

# 30. Transition checklist: new-DESIGN.md → DESIGN.md merge (2026-09-24)

Status of the proposal-to-spec merge and what still needs building. The merge
itself is done (DESIGN.md §§5/11/14/24/48/59/85/86/118-120/128/136-139 carry
the new-DESIGN text; §22.5 and §140 kept verbatim; §§137A-D inserted; as-built
admonitions in §§5/14/46/118 + §65 profile names added). Items below are the
code/config work the merged spec now requires.

## 30.1 CausVid — fully remaining (spec-only)

- No worker (`voyage/workers/video_causvid.py`), registry entry
  (`model_registry.py`), config preset (`config.py`), or
  `VIDEO_WORKER_MODULES` entry — `grep causvid voyage/` is empty.
- Implement per TASK §§19/23.4/25.3: upstream pin (tianweiy/CausVid commit +
  Wan2.1-T2V-1.3B revision), `generate_blocks`/`generate_segment` op,
  `reconstructable_prefix` continuation (latent overlap, 21-latent chunks,
  3-frame overlap), recovery tape, `models download/verify causvid-*`.
- Deliverable: `docs/UPSTREAM_CAUSVID_NOTES.md` (does not exist).

## 30.2 LTXV proposal-vs-built drift — reconciled (Stream A, 2026-09-24)

- Proposal (§5.3): 768×432, 121-frame segments, 25-frame conditioning tail
  (mp4 prefix replay).
- Built (`voyage/workers/video_ltxv.py`, Phase 7): 768×512, `25+(B-1)*24`
  frames, tail-PNG chaining (`<stem>_tail.png`), `recovery.pt{profile:ltxv,
  tail_png}`, bf16-first with torchao-fp8 OOM fallback, `generate --backend
  ltxv` default.
- Resolution (Stream A): code realigned toward §5.3 accounting (121-frame
  target / 25-frame video tail / 96 novel committed with prefix-discard,
  `video_tail.mp4` + sha256, §5.3 JSON tape in `recovery.pt` with a clean
  break from old torch tapes); geometry amended toward as-built native
  **768×512** (evidence: 432 % 32 != 0 → pads to 768x448; see §5.3 as-built
  for URLs). Drift removed.
- Follow-up (explicitly out of Stream A scope): the 81/97/121 benchmark
  matrix + TeaCache/Q8/FP8-kernel study; extension-throughput (96 novel)
  still unmeasured. Also stale: `voyage/cli.py::_frames_per_segment` still
  plans `generate --duration` with the pre-Stream-A 25/24-frame math
  (`_LTXV_NATIVE_BLOCK_FRAMES`) — `cli.py` was out of Stream A scope, so
  duration planning over-estimates segment counts until that helper is
  updated to 121 fresh / 96 extension (worker-reported frames remain the
  timeline truth). Also `run --run <relative-path>` doubles segment paths
  (`output/<run>/output/<run>/...`) because workers spawn with CWD=run_dir
  — pass the absolute in-container path (`/app/output/<run>`) until the
  supervisor normalizes run_dir (same cli/supervisor family).
- Deliverable: `docs/UPSTREAM_LTXV_NOTES.md` (does not exist; only
  `UPSTREAM_LONG_LIVE_PATCHES.md` does).

## 30.3 Config/interface duality — adapter or spec update

- Target (merged §§5.1/14): async `VideoBackend.generate_segment` +
  `segment_seconds` + `[video.longlive/ltxv/causvid]` blocks with
  `state_mode`.
- Live (`config.py`, `workers/loop.py`, `rpc.py`): sync JSONL RPC with op
  `generate_blocks` + `VideoConfig(segment_frames, blocks_per_segment,
  quantization)` + `_VIDEO_BACKEND_PRESETS`.
- Either implement the async interface as a wrapper over the current RPC or
  amend §§5.1/14/45-46 to standardize on `generate_blocks`. Do not leave
  both as if they were the same contract.

## 30.4 Benchmark + audit artifacts — never produced

- `reports/longlive-audit.md` (100:1 root-cause, §16 categories A–J
  unanswered), `reports/video-backends.md` (§21 benchmark matrix, never run
  across backends), per-run benchmark records in the manifest (§5.1
  selection rule step 8).
- Run the §137A qualification (smoke → resolution/FPS → VRAM/RAM → steady
  state → crash recovery → 3-segment visual review) for each backend before
  calling any profile production-ready.

## 30.5 Cross-references

- LTXV was built out-of-TASK-order (Phase 7, DESIGN §140); this checklist
  records the remaining delta rather than re-specifying the built worker.
- CausVid stays the next backend slice; §§137D/128 give the implementation
  order (interface → LTXV reconcile → CausVid → cross-backend recovery →
  endurance).
