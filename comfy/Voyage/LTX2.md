# LTX-2 / LTX-2.5 Feasibility Experiment Plan

> **Purpose:** determine, empirically and reproducibly, whether LTX-2.x can become a useful renderer for the `voyage` infinite audiovisual project on the local machine.
>
> **Hardware target:** NVIDIA RTX 4060 Ti 16 GB + NVIDIA RTX 2060 (VRAM must be measured; this plan assumes the common 6 GB model until `voyage doctor`/`nvidia-smi` proves otherwise) + 64 GB system RAM + Linux Ubuntu.
>
> **Primary objective:** test the highest-quality LTX configurations that could plausibly run on the available hardware, then progressively reduce model precision and increase offloading until a stable/performance-acceptable configuration is found.
>
> **Secondary objective:** determine whether a two-GPU architecture is actually useful: RTX 4060 Ti for the LTX video/audio transformer, RTX 2060 for the LTX Gemma text encoder, with CPU/system RAM used aggressively as overflow/offload storage.
>
> **Deliverable after experimentation:** a measured comparison report covering model quality, prompt adherence, temporal consistency, audio quality, VRAM/RAM use, generation speed, startup cost, reliability, complexity, and suitability for an indefinitely running `voyage` worker.

---

## 1. Scope and non-goals

This document is an **experiment protocol**, not the final `voyage` architecture.

The experiments should answer these questions in order:

1. Can LTX-2.5 run at all on the 4060 Ti 16 GB when the model is heavily quantized and/or CPU-offloaded?
2. Can the RTX 2060 materially improve the configuration by hosting the Gemma text encoder?
3. Which quantization is the highest-quality configuration that remains stable?
4. At approximately 768×448, can LTX produce clips fast enough to be useful for iterative development?
5. Can the renderer generate longer clips/chunks without VRAM growth or reliability problems?
6. Does LTX-2.5 or LTX-2.3 provide the better quality/performance/complexity tradeoff on this exact machine?
7. Is the resulting renderer good enough to justify integrating LTX into `voyage` as either the primary backend or a secondary high-quality transition renderer?

The experiments are **not** intended to establish generic LTX benchmark numbers. All meaningful conclusions must be tied to the exact local hardware, driver, CUDA/PyTorch stack, model revisions, and workflow revisions used.

---

## 2. Important current findings that determine the order

### 2.1 LTX-2.5 should be tested before LTX-2.3

The current LTX repository's main model family is LTX-2.5. The repository describes the distilled pipeline as the fastest starting point and provides FP8 and CPU/disk-offload controls for constrained memory. The production-quality DFR route uses more VRAM and is therefore not the first target here.

Official repository:

- https://github.com/Lightricks/LTX-2
- https://huggingface.co/Lightricks/LTX-2.5

### 2.2 A 16 GB GPU has already been shown to run LTX-2.5 Q5 GGUF

A current, model-specific audit for `elix3r/LTX-2.5-22b-distilled-GGUF` reports:

- Q5_K_M transformer: 16,878,535,040 bytes.
- ComfyUI 0.33.0.
- ComfyUI-GGUF commit `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`.
- PyTorch 2.10.0 + CUDA 12.8.
- RTX 4070 Ti SUPER 16 GB.
- Single-stage 608×352×49 inference completed with video and audio.
- Two-stage 1216×704×49 inference completed in 267.91 seconds from a cold server state.
- Two-stage peak VRAM was approximately 14,554 MiB.
- The tested command used ComfyUI `--lowvram --disable-dynamic-vram --preview-method none`.

This is **not proof that the same workflow will run on a 4060 Ti**, but it moves Q5 from "theoretically possible" to "must test first".

Source:

- https://huggingface.co/elix3r/LTX-2.5-22b-distilled-GGUF

### 2.3 LTX-2.5 has a very small validated quantized Gemma option

`elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF` provides an LTX-specific Gemma 4 12B encoder with both LTX audio/video projection heads:

| Encoder | File size | Role in this plan |
|---|---:|---|
| Q2_K | ~5.96 GB | highest-priority candidate for the RTX 2060 if it has 6+ GB and sufficient runtime headroom | 
| Q4_K_M | ~8.41 GB | 8 GB-class GPU only; too large for a typical 6 GB RTX 2060 | 
| Q5_K_M | ~9.51 GB | reference-quality comparison only; not expected to fit the 2060 | 

Source:

- https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF

A separate low-VRAM LTX-2.5 project provides a `qint2` Gemma-4 LTX text encoder of approximately 5.54 GB and reports roughly 5.6 GB for the encoder itself. It also reports that its 2-bit LTX transformer is approximately 5.36 GB and that the text encoder/transformer can be sharded across GPUs in that implementation.

Source:

- https://github.com/A4ax/comfyui-LTX-2.5-Tile-train-LoRa--On-multi-Gpus-low-VRAM-18-gb-Beta

**Interpretation:** the qint2 option is particularly interesting for the RTX 2060, but a 5.6 GB model on a nominal 6 GB card is extremely close to the memory ceiling. Treat it as an explicit experiment, not an assumed working configuration.

### 2.4 LTX-2.5 transformer GGUF ladder

A current GGUF collection provides approximately these distilled transformer sizes:

| Quantization | Approx. file size | Expected role |
|---|---:|---|
| Q5_K_M | 16.88 GB | first experiment; highest-quality plausible option through aggressive low-VRAM loading/offload |
| Q4_K_M | ~15.7 GB | high-quality 16 GB target; less demanding than Q5 |
| Q3_K_M | ~12.9 GB | likely strong practical candidate |
| Q2_K | ~12.1 GB | high-memory-margin candidate |

Source examples:

- https://huggingface.co/elix3r/LTX-2.5-22b-distilled-GGUF
- https://huggingface.co/vantagewithai/LTX-2.5-GGUF/tree/main/distilled

The exact model file and commit/revision used in each experiment MUST be recorded in the result manifest.

### 2.5 LTX-2.3 has a useful fallback quantization ladder

`unsloth/LTX-2.3-GGUF` contains distilled 22B variants including approximately:

| Quantization | Approx. size |
|---|---:|
| Q2_K | 8.28 GB |
| Q3_K_S | 9.95 GB |
| Q3_K_M | 10.8 GB |
| Q4_K_S | 13.1 GB |
| Q4_K_M | 14.3 GB |
| Q5_K_M | 16.1 GB |

Source:

- https://huggingface.co/unsloth/LTX-2.3-GGUF

LTX-2.3 uses Gemma 3 rather than LTX-2.5's Gemma 4. It is therefore a valuable fallback because the Gemma 3 GGUF ecosystem is mature and includes sub-6-GB 2-bit/3-bit variants.

---

## 3. Experiment ordering

The experiments are ordered **descending by expected model quality**, not descending by probability of success.

| ID | Model | Transformer | Text encoder | 4060 Ti | 2060 | CPU fallback | Priority |
|---|---|---|---|---|---|---|---|
| E0 | Hardware/software preflight | — | — | probe | probe | yes | mandatory |
| E1 | LTX-2.5 | Q5_K_M | Gemma 4 Q2_K / qint2 | primary | encoder | yes | highest |
| E2 | LTX-2.5 | Q4_K_M | Gemma 4 Q2_K / qint2 | primary | encoder | yes | very high |
| E3 | LTX-2.5 | Q3_K_M | Gemma 4 Q2_K / qint2 | primary | encoder | yes | high |
| E4 | LTX-2.5 | Q2_K | Gemma 4 Q2_K / qint2 | primary | encoder | yes | medium-high |
| E5 | LTX-2.5 | int2 community transformer | Gemma 4 qint2 | primary | encoder/shard | yes | low-VRAM limit test |
| E6 | LTX-2.3 | Q5_K_M | Gemma 3 high-quality quant | primary | encoder/CPU | yes | fallback quality |
| E7 | LTX-2.3 | Q4_K_M | Gemma 3 Q3/Q2 | primary | encoder | yes | strong fallback |
| E8 | LTX-2.3 | Q3_K_M | Gemma 3 Q2/IQ2 | primary | encoder | yes | practical fallback |
| E9 | LTX-2.3 | Q2_K | Gemma 3 IQ2/Q2 | primary | encoder | yes | low-memory fallback |
| E10 | Best 2–3 survivors | selected | selected | selected | selected | yes | long-run/endurance |

**Do not skip directly to E5.** The point of E1–E4 is to discover how much quality can be retained before aggressively dropping precision.

---

# 4. Common test protocol

Every experiment must use the same core protocol unless it explicitly tests a different variable.

## 4.1 Fixed render settings

Use these baseline settings:

```text
Resolution: 768 × 448
FPS: 24
Frames: 49 for smoke / 97 for standard / 193 for endurance
Batch size: 1
Prompt enhancement: OFF initially
Seed: fixed per prompt
Audio: enabled
Preview: disabled
Video VAE: convolutional BF16 variant first
Spatial upscaler: enabled only in the two-stage test
```

Why 768×448:

- It is close to the project's intended ~768×432 target.
- 768 and 448 satisfy the current pipeline's 64-pixel stage-2 divisibility constraint.
- It is much smaller than the 1024×1536 and 1216×704 examples used in many upstream tests.

Do not substitute 768×416: 416 is not divisible by 64 and therefore is not a valid two-stage target for the current pipeline without changing/patching the resolution constraints.

### Frame counts

LTX's causal grid uses `8k+1` frames. At 24 FPS:

| Frames | Approx. duration |
|---:|---:|
| 49 | 2.0 s |
| 97 | 4.0 s |
| 121 | 5.0 s |
| 193 | 8.0 s |

Use 49 frames for setup/debugging, 97 for normal comparisons, 193 only for the endurance tests.

---

## 4.2 Fixed prompts

Do not change prompts between quantization variants when comparing quality.

Use four fixed prompts.

### Prompt A — peaceful visual transformation

```text
A slow cinematic shot of a quiet surreal landscape at dusk. A field of pale lavender grass surrounds a small glass observatory. The camera moves forward very slowly. The glass walls gradually begin to glow with pastel cyan and pink light, then the observatory slowly transforms into a translucent organic structure covered in luminous vines. The transformation is continuous and physically coherent, with no hard cuts. Gentle atmospheric movement, soft volumetric light, subtle reflections, peaceful mood, highly detailed but uncluttered composition.
```

### Prompt B — semantic drift

```text
A slow cinematic journey through a deserted mechanical orchard at dawn. The camera gently moves between rows of quiet metallic trees bearing translucent fruit. Over time the fruit becomes increasingly glass-like, reflections deepen into shallow pools of water, and the orchard gradually becomes partially submerged. The mechanical structures slowly develop coral-like organic surfaces until the scene feels like a luminous underwater ecosystem. The transformation is gradual and continuous with no abrupt scene cuts. Calm motion, soft light, coherent spatial layout.
```

### Prompt C — difficult temporal consistency

```text
A close cinematic view of a single floating paper sculpture rotating very slowly in the air. The sculpture folds and unfolds continuously, changing from a geometric flower into a delicate bird and then back into an abstract ribbon-like form. The camera remains smooth and stable while the object transformation remains continuous. Fine paper texture, soft shadows, subtle depth of field, controlled motion, no flicker, no sudden cuts.
```

### Prompt D — audio/video test

```text
A cinematic night scene in a quiet neon-lit city street after rain. A lone person walks slowly past reflective pavement while soft pastel signs glow through mist. A gentle ambient electronic soundtrack plays with distant city ambience. The person softly says, "It feels like the city is dreaming." The camera follows slowly from behind and then arcs gently to the side. The visual movement and audio remain temporally coherent, peaceful and cinematic.
```

For `voyage` integration testing, also use one style-only prompt variant:

```text
pastel neon line-art, slow cinematic motion, peaceful, dreamlike, restrained visual complexity
```

When using the style-only prompt, the director supplies the rest of the scene description.

---

## 4.3 Seeds

Use deterministic seeds:

```text
Prompt A: 101
Prompt B: 202
Prompt C: 303
Prompt D: 404
```

When comparing quantization variants, use the same seed wherever the implementation supports equivalent noise initialization.

Do not claim bitwise equivalence between different loaders/quantization engines. The objective is controlled qualitative comparison, not exact pixel identity.

---

## 4.4 Common environment metadata

Capture once before each experiment and store in `results/<experiment-id>/environment.txt`:

```bash
uname -a
cat /etc/os-release
nvidia-smi
nvidia-smi -q -d MEMORY,DRIVER,PCI
python3 --version
uv --version || true
ffmpeg -version | head -n 3
```

Also record:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, "VRAM GiB", p.total_memory / 2**30, "CC", f"{p.major}.{p.minor}")
PY
```

The experiment manifest must record GPU index mapping explicitly, for example:

```text
GPU 0 = RTX 4060 Ti 16GB
GPU 1 = RTX 2060 6GB
```

Do not assume this mapping without verifying it.

---

# 5. Common repository setup

## 5.1 Create a dedicated experiment checkout

Do not modify the production `voyage` environment initially.

```bash
mkdir -p ~/Projects/video/ltx2-experiments
cd ~/Projects/video/ltx2-experiments

git clone https://github.com/Lightricks/LTX-2.git upstream
cd upstream
```

Record the exact revision:

```bash
git rev-parse HEAD | tee ../upstream-commit.txt
git status --short
```

For any upstream instructions that specify a tested commit, prefer that commit for the corresponding reproducibility test.

---

## 5.2 Create an isolated Python environment

Prefer `uv`.

```bash
cd ~/Projects/video/ltx2-experiments/upstream
uv sync --frozen
```

If the exact tested ComfyUI/GGUF experiment requires a separate environment, keep it completely separate:

```bash
cd ~/Projects/video/ltx2-experiments
python3 -m venv comfy-venv
source comfy-venv/bin/activate
python -m pip install -U pip
```

Never attempt to combine the experimental ComfyUI/GGUF environment with the official LTX Python environment until the dependency versions have been recorded and compatibility has been demonstrated.

---

# 6. ComfyUI-GGUF reference setup

The current strongest validated practical path is the ComfyUI-GGUF route because it supports aggressive quantized transformers and the LTX-specific Gemma GGUF.

A current Q5 LTX-2.5 audit documents the following tested revisions:

```text
ComfyUI 0.33.0
ComfyUI commit: 2f35f4a08176d993cded35dac3332be4f7287f41
ComfyUI-GGUF: 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
PyTorch: 2.10.0+cu128
```

Source:

- https://huggingface.co/elix3r/LTX-2.5-22b-distilled-GGUF

For the first experiments, **pin this environment exactly** rather than immediately updating to the latest ComfyUI.

### Setup

```bash
cd ~/Projects/video/ltx2-experiments

git clone https://github.com/comfyanonymous/ComfyUI.git comfy
cd comfy
git checkout 2f35f4a08176d993cded35dac3332be4f7287f41
python3 -m venv venv
./venv/bin/pip install -U pip
./venv/bin/pip install -r requirements.txt

git clone https://github.com/city96/ComfyUI-GGUF.git custom_nodes/ComfyUI-GGUF
cd custom_nodes/ComfyUI-GGUF
git checkout 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
../../venv/bin/pip install -r requirements.txt
```

Use the tested LTX-2.5 GGUF compatibility patch from the Q5 audit if the pinned ComfyUI-GGUF revision still requires it.

The audited patch SHA-256 is:

```text
1c36aa38fa9c86ec503926bc6b35226e2575b3bfd27b352bff50deccb1cd958e
```

The audit describes the patch as:

- registering `gemma4` as a truthful text architecture;
- decoding three specific raw BF16 LTXAV parameters before quantized linear processing;
- not relabeling the model as Gemma 3.

Do not improvise a different patch during benchmark runs.

---

# 7. Model acquisition and manifest

Create:

```text
models/
├── ltx-2.5/
│   ├── transformer/
│   ├── text_encoder/
│   └── support/
└── ltx-2.3/
    ├── transformer/
    ├── text_encoder/
    └── support/
```

Every downloaded model must be accompanied by a manifest entry containing:

```json
{
  "name": "...",
  "repository": "...",
  "file": "...",
  "revision": "...",
  "sha256": "...",
  "bytes": 0,
  "license": "...",
  "role": "transformer|text_encoder|vae|upscaler",
  "source_url": "..."
}
```

Do not compare models without pinning their exact file hash.

---

## 7.1 LTX-2.5 Q5 transformer

Repository:

```text
elix3r/LTX-2.5-22b-distilled-GGUF
```

File:

```text
ltx-2.5-22b-distilled-transformer-Q5_K_M.gguf
```

Size:

```text
16,878,535,040 bytes
```

SHA-256:

```text
cd82849aec38d2be84805b725c803bad44bc67df9850c23a24ef011a51fd7988
```

Download:

```bash
hf download elix3r/LTX-2.5-22b-distilled-GGUF \
  ltx-2.5-22b-distilled-transformer-Q5_K_M.gguf \
  --local-dir models/ltx-2.5/transformer
```

Verify:

```bash
printf '%s  %s\n' \
  cd82849aec38d2be84805b725c803bad44bc67df9850c23a24ef011a51fd7988 \
  models/ltx-2.5/transformer/ltx-2.5-22b-distilled-transformer-Q5_K_M.gguf \
  | sha256sum -c -
```

---

## 7.2 LTX-2.5 Gemma 4 Q2_K

Repository:

```text
elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF
```

File:

```text
gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf
```

Size:

```text
5,956,930,560 bytes
```

SHA-256:

```text
a70012eeea2fbee7c9c058fca5710842ec8cb9d9e94e630f2d1a0647a8d300f6
```

Download:

```bash
hf download elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF \
  gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf \
  --local-dir models/ltx-2.5/text_encoder
```

Verify:

```bash
printf '%s  %s\n' \
  a70012eeea2fbee7c9c058fca5710842ec8cb9d9e94e630f2d1a0647a8d300f6 \
  models/ltx-2.5/text_encoder/gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf \
  | sha256sum -c -
```

This Q2 encoder includes both LTX audio/video projection heads; do **not** substitute stock Gemma 4.

---

## 7.3 LTX-2.5 Q4/Q3/Q2 transformers

Use the same model repository family for controlled comparisons. Record the exact file selected.

Suggested files:

```text
Q4_K_M  ~15.7 GB
Q3_K_M  ~12.9 GB
Q2_K    ~12.1 GB
```

Source:

https://huggingface.co/vantagewithai/LTX-2.5-GGUF/tree/main/distilled

The benchmark report must state the exact SHA-256 of every file used; do not rely on the approximate size alone.

---

## 7.4 LTX-2.5 qint2 Gemma fallback/alternative

Repository/source:

https://github.com/A4ax/comfyui-LTX-2.5-Tile-train-LoRa--On-multi-Gpus-low-VRAM-18-gb-Beta

Reported file:

```text
gemma4-12b-with-proj-ltx-2.5-qint2.safetensors
```

Reported size:

```text
~5.54 GB
```

This should be tested independently from the GGUF Q2_K encoder because the two quantization formats use different loaders/kernels and therefore have different memory/performance behavior.

---

# 8. Common instrumentation

Every experiment must record CPU, GPU, I/O, process and timing information.

## 8.1 GPU monitor

Start this in a separate terminal before each run:

```bash
mkdir -p results/E1/gpu
nvidia-smi dmon -s pucm -d 1 > results/E1/gpu/dmon.txt
```

Also:

```bash
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > results/E1/gpu/query.csv
```

Replace `E1` with the experiment ID.

For process-level samples:

```bash
while true; do
  date +%s.%N
  nvidia-smi pmon -c 1
  sleep 1
done > results/E1/gpu/pmon.log
```

Stop monitoring only after all output files are safely written.

---

## 8.2 CPU/RAM/swap

```bash
vmstat 1 > results/E1/system/vmstat.txt
```

and, when needed:

```bash
watch -n 1 'free -h; echo; ps -eo pid,pmem,rss,vsz,cmd --sort=-rss | head -n 20'
```

Record peak RSS using `/usr/bin/time` around the generation process:

```bash
/usr/bin/time -v <command> 2> results/E1/system/time.txt
```

Extract:

```text
Maximum resident set size
Elapsed wall clock time
File system inputs
File system outputs
Voluntary context switches
Involuntary context switches
```

---

## 8.3 Disk I/O

If the model is CPU/disk-offloaded, also record:

```bash
iostat -xz 1 > results/E1/system/iostat.txt
```

This is required because a configuration that fits only by continuously streaming weights from disk is technically feasible but potentially useless for `voyage`.

---

## 8.4 Kernel/driver errors

Before an experiment:

```bash
journalctl -k -n 50 > results/E1/system/kernel-before.txt
```

After an experiment:

```bash
journalctl -k -n 200 > results/E1/system/kernel-after.txt
```

Also record:

```bash
dmesg | tail -n 200
```

Any Xid error, CUDA illegal access, driver reset, GPU hang, ECC-like report, or process-kill due to OOM is a reliability failure even if the program eventually produces a file.

---

# 9. Common metrics

Every successful run must produce a machine-readable `metrics.json`.

Recommended schema:

```json
{
  "experiment_id": "E1",
  "model_family": "LTX-2.5",
  "transformer_quant": "Q5_K_M",
  "text_encoder_quant": "Q2_K",
  "transformer_source": "...",
  "text_encoder_source": "...",
  "transformer_sha256": "...",
  "text_encoder_sha256": "...",
  "resolution": [768, 448],
  "fps": 24,
  "frames": 97,
  "duration_seconds": 4.0,
  "seed": 101,
  "prompt_id": "A",
  "startup_seconds": 0,
  "text_encode_seconds": 0,
  "generation_seconds": 0,
  "decode_seconds": 0,
  "mux_seconds": 0,
  "total_seconds": 0,
  "generation_fps": 0,
  "realtime_factor": 0,
  "gpu0_peak_vram_mib": 0,
  "gpu1_peak_vram_mib": 0,
  "cpu_peak_rss_gib": 0,
  "swap_peak_gib": 0,
  "oom": false,
  "driver_error": false,
  "output_valid": true
}
```

Calculate at minimum:

### Speed

```text
generation_fps = frames / generation_seconds
wallclock_fps = frames / total_seconds
realtime_factor = total_seconds / video_duration_seconds
```

Lower `realtime_factor` is better.

### Prompt latency

Separate:

```text
model_load_seconds
text_encode_seconds
conditioning_transfer_seconds
diffusion_seconds
VAE_decode_seconds
mux_seconds
```

This distinction matters because `voyage` can amortize text encoding over many visual blocks.

### VRAM

Record for both GPUs:

```text
peak_allocated
peak_reserved
peak_used_from_nvidia_smi
minimum_free
```

### System RAM

Record:

```text
peak RSS
peak anonymous memory
peak file cache if available
peak swap used
```

### Reliability

Binary pass/fail plus reason:

```text
PASS
OOM
CUDA_ERROR
DRIVER_RESET
SEGFAULT
CORRUPTED_OUTPUT
AUDIO_FAILURE
VIDEO_FAILURE
TIMEOUT
OTHER
```

---

# 10. Quality metrics

Performance is not enough. A highly quantized model that runs 2× faster but produces unstable video should lose to a slower stable model.

## 10.1 Human quality rubric

For each output, score independently from 1–5:

| Metric | 1 | 5 |
|---|---|---|
| Prompt adherence | mostly unrelated | captures nearly every requested visual element |
| Scene coherence | broken geometry | coherent spatial layout |
| Temporal coherence | severe flicker/drift | stable objects and motion |
| Motion quality | chaotic or frozen | natural controlled motion |
| Fine detail | heavily degraded | clean fine structure |
| Color/styling | far from prompt | very faithful |
| Audio quality | unusable | clean, convincing audio |
| A/V coherence | unrelated | synchronized/semantically coherent |
| Artifact rate | severe | negligible |
| Artistic usefulness | poor | production-useful for `voyage` |

Two people should ideally score the same outputs independently. If only one evaluator is available, record uncertainty in the report rather than pretending the scores are objective.

---

## 10.2 Automated visual metrics

### Frame embeddings

Use a fixed image encoder, such as DINOv2 or another locally available embedding model, to measure:

- average cosine similarity between adjacent frames;
- similarity after a one-frame temporal offset;
- long-range semantic drift across the clip.

Do not interpret "higher adjacent similarity" as universally better. Excessive similarity can indicate frozen motion. Report it together with optical flow.

### Optical flow

Use a fixed optical-flow method to calculate:

```text
mean motion magnitude
motion variance
percentage of frames with near-zero motion
percentage of frames with extreme motion spikes
```

### Flicker metric

Compute frame-to-frame luminance/color changes after removing global motion where possible.

Flag:

```text
flicker score
largest single-frame discontinuity
scene-boundary candidates
```

### Prompt alignment

Use a fixed image-text embedding model to calculate similarity between:

```text
prompt ↔ sampled frames
```

Do not compare absolute scores across unrelated embedding models. The benchmark is for relative ranking between LTX variants.

---

## 10.3 Audio metrics

Use `ffprobe` and `ffmpeg` plus a small Python analysis script.

Record:

```text
sample rate
channel count
exact duration
RMS loudness
integrated LUFS
peak dBFS
true peak if available
clipping sample count
spectral centroid
spectral flux
silence ratio
```

For Prompt D, additionally inspect:

```text
speech exists
speech duration
speech intelligibility
voice timing relative to the requested sentence
```

Do not use speech-specific metrics for prompts without speech.

---

# 11. E1 — highest-quality plausible LTX-2.5 setup

## Goal

Test the best LTX-2.5 configuration that has credible evidence for 16 GB-class inference but may still fail on the exact 4060 Ti.

## Configuration

```text
Transformer: LTX-2.5 22B distilled Q5_K_M
Text encoder: LTX Gemma 4 12B Q2_K GGUF
Main GPU: RTX 4060 Ti 16 GB
Secondary GPU: RTX 2060, text encoder if it fits
CPU: 64 GB RAM, unlimited fallback
Video VAE: convolutional BF16
Audio VAE: BF16
Pipeline: DistilledPipeline / equivalent current LTX-2.5 ComfyUI workflow
Low-VRAM mode: enabled
Preview: disabled
Dynamic VRAM: disabled
```

## Why this is first

The Q5 transformer is the highest-precision quantized LTX-2.5 option with a current concrete 16 GB-class validation. The Q2 Gemma is the most plausible LTX-2.5 text encoder small enough to attempt on a 6 GB-class secondary GPU. The tested 16 GB audit used Q5 and completed a two-stage run at approximately 14.6 GiB peak VRAM.

## Primary placement

Desired:

```text
GPU 0 (4060 Ti): transformer + VAE
GPU 1 (2060): Gemma 4 Q2_K
CPU: embeddings / overflow / offload
```

Because the stock LTX CLI does not expose a simple text-encoder GPU selection, this experiment has two sub-stages.

### E1-A: text encoder-only placement test

First determine whether the Q2 Gemma can run on the 2060.

Launch a dedicated process restricted to GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/test_ltx25_text_encoder.py \
  --model models/ltx-2.5/text_encoder/gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf \
  --prompt "A peaceful surreal landscape with pastel neon line-art and slow cinematic motion." \
  --iterations 10
```

`test_ltx25_text_encoder.py` is an experiment helper that must:

1. load the exact LTX-specific Gemma file;
2. tokenize using the LTX tokenizer;
3. produce both video and audio conditioning;
4. measure first-load time;
5. run the same prompt ten times;
6. measure peak VRAM;
7. validate finite tensors;
8. save one conditioning cache file.

Required implementation behavior:

```python
with torch.no_grad():
    outputs = text_encoder.encode([prompt])
```

Do not use `torch.inference_mode()` with a custom qint2 engine until that engine is proven compatible. One current low-VRAM implementation explicitly requires `torch.no_grad()` for its qint2 path.

Record whether the process stays below ~5.8 GiB, whether it OOMs, and how much VRAM remains.

### E1-B: two-GPU generation

If E1-A succeeds, implement a small experiment-only bridge:

```text
2060 process
    Gemma
      ↓
CPU/shared-memory conditioning cache
      ↓
4060 process
    LTX transformer
```

Do **not** transmit the full model. Transfer only:

```text
video_context
video_mask
optional audio_context
optional audio_mask
```

The expected conditioning tensor size is tiny compared with the 22B transformer.

### E1-C: CPU fallback

If the 2060 cannot run the Q2 encoder, repeat using CPU.

This is still a valid E1 result because the experiment's main question is whether Q5 transformer quality is useful when the model is aggressively quantized and the text encoder is removed from the main GPU.

## LTX-2.5 ComfyUI setup

Use the tested Q5 workflow first.

Launch:

```bash
cd ~/Projects/video/ltx2-experiments/comfy
./venv/bin/python main.py \
  --lowvram \
  --disable-dynamic-vram \
  --preview-method none
```

Restrict the main ComfyUI process to the 4060 Ti once GPU indices are known. Prefer the application's explicit device option when available rather than relying only on `CUDA_VISIBLE_DEVICES`, because the encoder worker must independently target the other GPU.

First run the audited 608×352×49 smoke workflow before 768×448.

Then run:

```text
768×448×49
768×448×97
```

Then, only if stable:

```text
768×448×193
```

### E1 success criteria

A configuration is considered feasible if:

1. 49-frame smoke completes three consecutive times.
2. No CUDA/driver errors occur.
3. Peak 4060 Ti VRAM stays below 15.5 GiB with at least ~0.5 GiB headroom.
4. The output contains valid video and audio.
5. 97-frame generation completes at least once without memory growth.
6. Restarting the process and rerunning succeeds.
7. The quality rubric is not catastrophically degraded by Q5/Q2 quantization.

Do **not** consider it production-feasible for `voyage` merely because a one-off 49-frame run succeeds.

---

# 12. E2 — LTX-2.5 Q4_K_M + Q2 Gemma

## Goal

Determine whether one quantization step lower materially improves reliability/speed while retaining nearly Q5 quality.

## Models

```text
Transformer: LTX-2.5 distilled Q4_K_M
Text encoder: Gemma 4 Q2_K GGUF
```

Main placement remains:

```text
4060 Ti → transformer + VAEs
2060 → Gemma Q2
CPU → overflow/offload
```

## Tests

Repeat exactly E1:

```text
608×352×49 smoke
768×448×49
768×448×97
768×448×193 if stable
```

Use the same four prompts and seeds.

## Additional test

Run one prompt using the 2060 encoder and one using CPU encoder with all other settings identical.

Purpose:

```text
measure encoder device cost separately from transformer quality
```

## Required comparison

Compare E2 directly against E1:

```text
Δ generation time
Δ peak VRAM
Δ CPU RAM
Δ quality scores
Δ temporal stability
Δ startup cost
```

---

# 13. E3 — LTX-2.5 Q3_K_M + Q2 Gemma

## Goal

Find the first likely "sweet spot" between quality and memory margin.

## Models

```text
Transformer: LTX-2.5 distilled Q3_K_M
Text encoder: Gemma 4 Q2_K
```

Use identical placement and tests to E1/E2.

## Extra test: no-upscale mode

Because the two-stage refinement is relatively memory intensive, add a single-stage/lowest-memory LTX path if the current pipeline supports it with the exact checkpoint.

Run:

```text
768×448×97
```

without the optional spatial refinement stage.

Measure whether:

```text
quality loss
```

is worth the large reduction in memory/time.

This is an especially important result for `voyage`, because a long-running generator is likely to favor many modestly sized consistent frames over occasional maximum-quality frames.

---

# 14. E4 — LTX-2.5 Q2_K + Q2 Gemma

## Goal

Establish the conservative 16 GB target.

## Models

```text
Transformer: LTX-2.5 distilled Q2_K
Text encoder: Gemma 4 Q2_K
```

Expected behavior:

- substantially larger VRAM headroom;
- potentially worse fine detail;
- potentially improved throughput if the bottleneck is weight movement rather than raw GPU arithmetic;
- increased importance of quantization artifacts.

## Tests

Same standard matrix:

```text
49 / 97 / 193 frames
Prompt A / B / C / D
```

Also test 30 repeated generations with short 49-frame clips to detect allocator fragmentation and cumulative state leaks.

---

# 15. E5 — LTX-2.5 extreme 2-bit configuration

## Goal

Determine whether the 22B model can become genuinely comfortable on the available GPUs at very low precision.

This is a **feasibility boundary experiment**, not the expected final-quality winner.

## Candidate implementation

Community low-VRAM LTX-2.5 project:

https://github.com/A4ax/comfyui-LTX-2.5-Tile-train-LoRa--On-multi-Gpus-low-VRAM-18-gb-Beta

Reported optional models:

```text
Transformer: ltx-2.5-22b-distilled-int2-main-v2.safetensors (~5.36 GB)
Text encoder: gemma4-12b-with-proj-ltx-2.5-qint2.safetensors (~5.54 GB)
```

The same project reports qint2 Gemma around 5.6 GB VRAM and a 2-bit transformer around 5.36 GB, plus block-level model sharding across multiple GPUs.

## Placement experiment matrix

Run all of:

### E5-A

```text
4060 Ti → entire int2 transformer
2060 → qint2 Gemma
CPU → embeddings/offload
```

### E5-B

```text
4060 Ti → transformer
2060 + CPU → Gemma if the custom loader supports split placement
```

### E5-C

```text
4060 Ti → transformer
CPU → Gemma
```

### E5-D

Use the community project's own multi-GPU transformer sharding if it is suitable for inference.

Do not assume that sharding increases generation speed. The goal is to determine whether it changes memory capacity enough to make another quality configuration viable.

## Metrics

In addition to the standard metrics record:

```text
GPU-to-GPU transfer volume if exposed
GPU synchronization stalls
per-layer or per-block time if instrumentation exists
CPU-GPU synchronization wait time
```

The experiment succeeds even if quality is too low, provided it establishes a useful lower-bound memory/performance profile.

---

# 16. E6 — LTX-2.3 Q5_K_M fallback

## Goal

Determine whether the older LTX-2.3 stack offers a better 16 GB experience because of its more mature GGUF ecosystem and Gemma 3 encoder support.

## Models

Transformer:

```text
unsloth/LTX-2.3-GGUF
ltx-2.3-22b-distilled-Q5_K_M.gguf
```

Text encoder:

Use the best Gemma 3 LTX-2.3-compatible quantized encoder that fits the 2060, preferably the highest available Q3/Q4 candidate that leaves runtime headroom. Do not use a generic Gemma 3 checkpoint unless the LTX-2.3 connector architecture explicitly supports it.

## Important

LTX-2.3 and LTX-2.5 text features are **not interchangeable**. Generate and cache them independently.

Source for the LTX-2.3 GGUF family:

https://huggingface.co/unsloth/LTX-2.3-GGUF

## Tests

Exactly the E1 matrix:

```text
49 / 97 / 193 frames
768×448
Prompts A–D
```

Include one CPU-only encoder test and one two-GPU encoder test if practical.

---

# 17. E7 — LTX-2.3 Q4_K_M / Q3 Gemma

Use:

```text
Transformer: Q4_K_M
Text encoder: best Gemma 3 Q3/Q2 candidate that fits 2060
```

Repeat the full E6 matrix.

The purpose is not to optimize LTX-2.3 exhaustively; it is to identify whether the LTX-2.3 quality/performance curve is better than LTX-2.5 at the same VRAM budget.

---

# 18. E8 — LTX-2.3 Q3_K_M / Gemma 3 Q2

This should be the primary LTX-2.3 practical configuration.

Run:

```text
49 frames × 4 prompts × 3 repetitions
97 frames × 4 prompts × 1 repetition
193 frames × Prompt B × 1 repetition
```

Do the same with CPU text encoding for one sample to quantify the benefit of the 2060.

---

# 19. E9 — LTX-2.3 Q2_K / Gemma 3 IQ2 or Q2

This is the lowest-quality LTX-2.3 configuration in the main comparison.

The goal is to provide a lower bound for:

```text
minimum useful GPU memory
minimum viable runtime
maximum throughput
minimum quality
```

Do not spend excessive time tuning this experiment if it is obviously dominated by E8.

---

# 20. Text encoder device experiments

These experiments are important enough to run independently of transformer quantization.

## TE-1: 2060 Q2 GGUF

```text
2060 → Gemma 4 Q2_K
CPU → embeddings processor
```

Measure:

```text
cold load
first encode
warm encode ×10
peak VRAM
peak CPU RAM
```

## TE-2: 2060 qint2

```text
2060 → Gemma 4 qint2
CPU → embeddings processor
```

Compare against TE-1.

## TE-3: CPU Q2/qint2

Run encoder on CPU to establish a baseline.

## TE-4: 4060 Ti encoder

Run text encoder on the 4060 Ti temporarily, with the transformer absent, to measure the encoder's isolated performance.

This is important because if the 2060 encoder is only 2× slower than CPU, the GPU split is valuable. If it is almost identical to CPU, the secondary GPU may not be worth the added complexity.

---

# 21. Encoder result cache experiment

Because `voyage` changes visual prompts much less frequently than it changes video blocks, test whether conditioning can be generated once and reused.

For each model family:

1. Encode Prompt A once.
2. Save `video_context` and `audio_context` to a CPU-side cache.
3. Restart the transformer process.
4. Reuse the saved contexts for three generations.
5. Compare startup and generation times against raw-prompt generation.

Cache format:

```python
{
    "model_family": "LTX-2.5",
    "model_revision": "...",
    "text_encoder_hash": "...",
    "prompt_sha256": "...",
    "dtype": "bf16",
    "video_context": Tensor,
    "audio_context": Tensor,
    "masks": ...,
}
```

Never reuse embeddings generated by a different LTX model version or incompatible encoder revision.

---

# 22. VAE experiments

The LTX-2.5 repository offers two video VAE variants:

- diffusion decoder;
- convolutional decoder.

The convolutional decoder is the first choice for low-memory testing because the current GGUF audit specifically identifies it as the lower-memory choice.

Run:

### VAE-1

```text
Convolutional BF16 video VAE
BF16 audio VAE
```

### VAE-2

```text
Diffusion video VAE
BF16 audio VAE
```

Only run VAE-2 for the top two surviving transformer configurations.

Measure:

```text
VAE decode time
peak VRAM
output quality
artifact differences
```

Do not let a more expensive VAE silently contaminate transformer comparison results.

---

# 23. Two-stage versus reduced-stage experiments

For the top three LTX-2.5 configurations:

```text
Q5
Q4
Q3
```

measure:

### Mode A — standard distilled two-stage

```text
stage 1 generation
latent spatial upsample
stage 2 refinement
VAE decode
```

### Mode B — lowest-memory valid path

Use the current pipeline's supported single-stage or no-refinement path.

Compare:

```text
quality
memory
speed
```

This may become one of the most important engineering choices for `voyage`.

---

# 24. Resolution scaling experiment

For each of the top three survivors:

```text
512×320
640×384
768×448
```

Keep FPS and frame count constant.

Measure:

```text
peak VRAM
seconds/frame
quality
```

Plot:

```text
VRAM vs pixels
seconds/frame vs pixels
quality vs pixels
```

The objective is to determine whether 768×448 is genuinely sustainable or whether 640×384 gives a substantially better throughput/quality tradeoff for the infinite-video use case.

---

# 25. Duration scaling experiment

For each surviving configuration:

```text
49 frames
97 frames
193 frames
```

This is especially important because LTX may use chunked inference to bound memory.

Measure whether:

```text
peak VRAM ≈ constant
```

or:

```text
peak VRAM grows with duration
```

A configuration with constant peak memory is much more suitable for long-running `voyage` generation.

Also measure:

```text
first-frame latency
steady-state seconds/frame
seam artifacts at chunk boundaries
```

---

# 26. Long-running stability experiment

Only run this for the top three survivors.

## 26.1 Short repeated generation

Generate:

```text
30 × 49-frame clips
```

without restarting the worker.

Record every run's:

```text
peak VRAM
runtime
RSS
output validity
```

Look for:

```text
memory creep
allocator fragmentation
runtime creep
first-run-only failures
```

## 26.2 Prompt-change stress

Alternate:

```text
Prompt A
Prompt B
Prompt C
Prompt D
```

for 20 iterations.

This specifically exercises text encoder load/unload/re-encode behavior.

## 26.3 Crash recovery

Kill the process during:

1. transformer loading;
2. text encoding;
3. stage 1;
4. stage 2;
5. VAE decoding;
6. muxing.

Verify that the supervisor can identify which segment was incomplete and rerun it without corrupting previous outputs.

---

# 27. Two-GPU value experiment

For the best surviving model, compare three placements.

### Placement A

```text
4060 Ti: transformer
2060: text encoder
CPU: overflow
```

### Placement B

```text
4060 Ti: transformer + text encoder when needed
2060: idle
CPU: overflow
```

### Placement C

```text
4060 Ti: transformer
2060: text encoder
CPU: everything else possible
```

The distinction between A and C is useful if the text encoder's projection/connectors can be kept CPU-side without affecting latency.

Report:

```text
end-to-end latency
GPU utilization overlap
CPU utilization
VRAM on both cards
encoder latency
```

The 2060 is justified only if its contribution is measurable.

---

# 28. CPU/offload experiment

For the best transformer quality that fits intermittently, test:

### OFF-1

```text
No CPU offload
```

### OFF-2

```text
CPU weight offload
```

### OFF-3

```text
Disk offload
```

Disk offload should be considered a last-resort feasibility mode only.

Measure:

```text
peak VRAM
wallclock runtime
GB/s read rate
CPU utilization
```

If disk offload increases runtime enough to make the renderer impractical, mark it as "technically feasible, operationally rejected" rather than simply "failed."

---

# 29. Quantization quality experiment

For one fixed prompt, generate the same seed with:

```text
Q5
Q4
Q3
Q2
int2
```

Use one fixed 49-frame output plus one 97-frame output.

Create a visual contact sheet:

```text
Q5 | Q4 | Q3 | Q2 | int2
```

For each, display:

```text
first frame
middle frame
last frame
three motion-difference thumbnails
```

Then score:

```text
fine detail loss
geometry degradation
motion degradation
color degradation
prompt adherence loss
artifact increase
```

This is the core experiment for deciding whether Q3 or Q2 is actually worth using.

---

# 30. Audio quality experiment

Because `voyage` places unusually high importance on music, audio quality must be evaluated independently.

For the surviving LTX-2.5 configurations, generate Prompt D at least three times.

Measure:

```text
speech intelligibility
voice consistency
background/music quality
audio artifacts
A/V synchronization
stereo behavior
loudness consistency
```

Compare the generated audio against the equivalent audio produced by the planned ACE-Step backend.

Do not assume LTX's native audio should replace ACE-Step. The experiment should establish whether the integrated audio is good enough to justify simpler orchestration.

---

# 31. `voyage` integration prototype

Only after E1–E9 identify at least one stable survivor, build an experiment-only worker API.

Recommended protocol:

```text
supervisor
  ↕ JSONL over stdin/stdout
LTX worker
```

The worker should support:

```text
load_model
encode_prompt
render
render_from_embeddings
health
metrics
shutdown
```

Logs must go to stderr. Stdout must remain machine-readable.

Example request:

```json
{
  "type": "render",
  "request_id": "e1-a-0001",
  "prompt": "...",
  "seed": 101,
  "width": 768,
  "height": 448,
  "num_frames": 97,
  "fps": 24,
  "output_path": "results/E1/A/0001.mp4"
}
```

The response should include:

```json
{
  "type": "result",
  "request_id": "e1-a-0001",
  "ok": true,
  "metrics_path": "..."
}
```

---

# 32. Precomputed-conditioning worker

If the two-GPU approach proves useful, split text encoding into a dedicated worker.

```text
ltx-text-worker
  GPU 2060
  │
  ├── Gemma q2/qint2
  ├── tokenizer
  └── LTX projection/connector path as appropriate

ltx-video-worker
  GPU 4060 Ti
  │
  ├── quantized LTX transformer
  ├── VAE
  └── renderer
```

The text worker should output a portable CPU representation of the conditioning tensors.

Use `safetensors` or a structured Torch serialization with explicit dtype/device metadata.

Never send CUDA tensors over IPC.

---

# 33. Reference implementation for metrics collection

Create:

```text
experiments/
├── run_experiment.py
├── monitor_gpu.py
├── analyze_video.py
├── analyze_audio.py
├── analyze_quality.py
├── validate_output.py
└── common.py
```

`run_experiment.py` should:

1. create the result directory;
2. write an immutable experiment manifest;
3. launch monitoring processes;
4. run the exact render command;
5. capture exit status;
6. inspect output with `ffprobe`;
7. collect GPU and RAM maxima;
8. compute timing;
9. hash the output;
10. write `metrics.json`;
11. stop monitoring cleanly.

---

# 34. Output directory structure

Use:

```text
results/
└── E1/
    ├── experiment.json
    ├── environment.txt
    ├── gpu/
    │   ├── dmon.txt
    │   ├── query.csv
    │   └── pmon.log
    ├── system/
    │   ├── time.txt
    │   ├── vmstat.txt
    │   ├── iostat.txt
    │   ├── kernel-before.txt
    │   └── kernel-after.txt
    ├── A/
    │   ├── 49f/
    │   │   ├── output.mp4
    │   │   ├── metrics.json
    │   │   ├── ffprobe.json
    │   │   └── samples/
    │   ├── 97f/
    │   └── 193f/
    ├── B/
    ├── C/
    └── D/
```

Never overwrite a completed result.

---

# 35. Validating output files

After every generation:

```bash
ffprobe -v error \
  -show_entries format=duration,size,bit_rate \
  -show_entries stream=index,codec_name,codec_type,width,height,r_frame_rate,duration,sample_rate,channels \
  -of json \
  output.mp4
```

Also test decoding:

```bash
ffmpeg -v error -i output.mp4 -f null -
```

A file that exists but fails decoding is a hard failure.

---

# 36. Statistical repetition rules

Do not base conclusions on one lucky render.

### Smoke tests

At least:

```text
3 repetitions
```

for every new configuration.

### Quality tests

At least:

```text
3 seeds/prompts where practical
```

### Long-run tests

At least:

```text
30 short clips
```

for final candidates.

Report:

```text
mean
median
p90
min
max
standard deviation
```

for runtime and VRAM where enough repetitions exist.

---

# 37. Failure classification

Every failed run must be assigned one primary category:

```text
HARDWARE_OOM
CPU_RAM_OOM
SWAP_EXHAUSTION
CUDA_RUNTIME_ERROR
CUDA_ILLEGAL_ACCESS
DRIVER_RESET
MODEL_LOAD_FAILURE
TOKENIZER_FAILURE
TEXT_ENCODER_FAILURE
TRANSFORMER_FAILURE
VAE_FAILURE
AUDIO_FAILURE
OUTPUT_CORRUPTION
PERFORMANCE_TIMEOUT
QUALITY_FAILURE
DEPENDENCY_FAILURE
```

A configuration may still be considered "feasible with caveat" when the only issue is low speed. Do not call a configuration feasible if it occasionally corrupts output or crashes the GPU driver.

---

# 38. Practical feasibility thresholds

These are project-specific decision thresholds, not claims about official LTX requirements.

## Tier A — candidate for `voyage`

```text
No crashes or OOM during 30 repeated short renders
Peak 4060 VRAM < 15.5 GiB
No persistent memory creep
97-frame render succeeds reliably
Useful visual quality (human score ≥ 3.5/5)
Temporal consistency ≥ 3.5/5
Audio quality ≥ 3/5
```

Speed is project-dependent, but record a target of:

```text
≤ 20× real time at 768×448
```

as the initial upper boundary for considering LTX practical for a long-running voyage worker.

A future target of ≤10× real time is preferable.

## Tier B — technically usable

```text
stable
quality acceptable
but slower than Tier A
```

Candidate for occasional high-quality transitions rather than the primary streaming backend.

## Tier C — research-only

```text
works once
or requires extreme CPU/disk offload
or is extremely slow
```

Keep the result but do not integrate it into `voyage`.

## Tier D — infeasible

```text
cannot produce a valid 49-frame sample
or crashes repeatedly
or exceeds available memory irrecoverably
```

Do not spend optimization time on it unless a clear technical fix exists.

---

# 39. Experiment stopping rules

Stop testing a configuration early when any of the following is true:

1. It cannot complete the 49-frame smoke test after two controlled memory reductions.
2. It causes a driver reset or system instability.
3. It is strictly dominated by another tested configuration in quality, speed and memory.
4. CPU/disk offload makes the render >10× slower than the previous lower-quality candidate without an obvious quality advantage.
5. The output quality is clearly unusable by visual inspection across three independent seeds.

Do not stop simply because a run is slow if it gives valuable information about the feasibility boundary.

---

# 40. Final candidate selection protocol

After E1–E9, select at most three candidates:

```text
C1 = highest-quality stable candidate
C2 = best quality/performance balance
C3 = lowest-memory/highest-throughput candidate
```

Then run the full long-run protocol on those candidates.

---

# 41. Endurance experiment E10

For each C1–C3:

## Phase 1 — 30 short generations

```text
30 × 49 frames
4 prompt classes
rotating seeds
```

## Phase 2 — 10 medium generations

```text
10 × 97 frames
```

## Phase 3 — long generation

```text
3 × 193 frames
```

## Phase 4 — prompt churn

```text
20 prompt changes
```

## Phase 5 — restart

Restart the worker after each phase and repeat one representative render.

Record whether performance changes after warmup.

---

# 42. Crash recovery test

The production `voyage` design requires crash resilience.

For the top candidate, intentionally terminate the renderer at pseudo-random points:

```text
10% of render
30%
50%
70%
90%
```

The supervisor must:

1. identify incomplete output;
2. discard only incomplete temporary artifacts;
3. preserve prior completed segments;
4. restart the worker;
5. rerender the failed segment;
6. produce a valid final segment.

This is a feasibility requirement for the project even if raw generation quality is excellent.

---

# 43. `voyage`-specific continuous generation test

After the renderer is stable, simulate the actual usage pattern.

Generate:

```text
10 × 16–32 second visual segments
```

using gradual semantic transitions rather than independent prompts.

The director prompt sequence should resemble:

```text
mechanical orchard
→ metallic fruit becomes translucent glass
→ glass reflections deepen into pools
→ orchard partially submerges
→ architecture becomes coral
→ luminous underwater ecosystem
```

Do not jump:

```text
orchard → spaceship → dragon → castle
```

because abrupt semantic transitions are not the target use case.

Measure:

```text
segment-to-segment continuity
prompt transition success
visual style retention
scene identity drift
audio continuity
```

---

# 44. LTX-2.5 versus CausVid comparison hook

The outcome should ultimately be compared against the current CausVid plan.

For the best LTX candidate and the current CausVid candidate, compare:

| Metric | LTX | CausVid |
|---|---:|---:|
| quality | | |
| temporal coherence | | |
| prompt adherence | | |
| audio quality | | |
| seconds/frame | | |
| real-time factor | | |
| peak VRAM | | |
| peak system RAM | | |
| startup time | | |
| prompt-change latency | | |
| crash rate | | |
| implementation complexity | | |
| infinite-stream suitability | | |
| license constraints | | |

Do not decide the final backend based on model quality alone.

---

# 45. Final report format

After all experiments, generate `LTX2-REPORT.md`.

## 45.1 Executive summary

State:

```text
Best measured configuration
Best quality configuration
Fastest stable configuration
Lowest-memory configuration
Whether 2060 text encoder offload is worthwhile
Whether LTX-2.5 beats LTX-2.3 for this hardware
Whether LTX should enter voyage as primary, secondary, or experimental backend
```

Do not obscure uncertainty.

---

## 45.2 Model table

Use:

| Exp | Family | Transformer | Text encoder | Main GPU | Encoder GPU | CPU offload | Status |
|---|---|---|---|---|---|---|---|

---

## 45.3 Performance table

Use:

| Exp | Resolution | Frames | Gen time | Total time | FPS | Real-time factor | 4060 VRAM | 2060 VRAM | RAM | Swap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

---

## 45.4 Quality table

Use:

| Exp | Prompt | Prompt adherence | Temporal coherence | Motion | Detail | Audio | A/V | Artifacts |
|---|---|---:|---:|---:|---:|---:|---:|---:|

---

## 45.5 Quantization tradeoff plot

Plot:

```text
quality score vs transformer quantization
```

with:

```text
Q5 → Q4 → Q3 → Q2 → int2
```

Then:

```text
quality score vs seconds/frame
```

and:

```text
quality score vs peak VRAM
```

---

## 45.6 Two-GPU usefulness

Provide a direct table:

| Encoder placement | Text encode time | End-to-end time | 4060 VRAM | 2060 VRAM | CPU RAM | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| 4060 | | | | | | |
| 2060 | | | | | | |
| CPU | | | | | | |

The report should answer whether the second GPU actually reduces end-to-end latency enough to justify the architectural complexity.

---

## 45.7 Reliability summary

Report:

```text
successful runs / total runs
OOM count
CUDA error count
driver reset count
corrupt output count
mean successful-run duration
p95 duration
memory leak evidence
```

---

## 45.8 Recommended configuration matrix

The report should end with three concrete operating profiles:

### Profile A — quality

```text
highest stable quantization
highest quality VAE
larger render resolution if feasible
```

Use for occasional premium shots.

### Profile B — balanced

```text
best quality/performance balance
```

Use as primary candidate if LTX becomes part of `voyage`.

### Profile C — throughput

```text
lowest stable quantization
lowest stable resolution
minimum memory footprint
```

Use for long-duration generation if visual quality remains acceptable.

---

# 46. Exact implementation tasks for the experiment harness

Create an `experiments/` package with these components.

## `hardware.py`

Responsibilities:

- discover GPU names;
- discover VRAM;
- discover compute capability;
- map GPU IDs;
- record driver and CUDA information.

## `models.py`

Responsibilities:

- model manifest parsing;
- SHA verification;
- size verification;
- model compatibility metadata.

## `runner.py`

Responsibilities:

- launch process;
- set environment variables;
- capture stdout/stderr;
- enforce timeout;
- collect exit code.

## `metrics.py`

Responsibilities:

- timing;
- GPU metrics;
- CPU metrics;
- output metrics;
- aggregation.

## `quality.py`

Responsibilities:

- extract representative frames;
- calculate embeddings;
- calculate frame differences;
- calculate optical flow metrics;
- produce contact sheets.

## `audio.py`

Responsibilities:

- ffprobe analysis;
- LUFS;
- RMS;
- peak/clipping;
- duration.

## `report.py`

Responsibilities:

- load all `metrics.json`;
- produce Markdown tables;
- generate plots;
- rank candidates by objective criteria **without hiding the underlying metrics**.

---

# 47. Suggested experiment manifest

Example:

```yaml
experiment_id: E1
family: LTX-2.5
transformer:
  repo: elix3r/LTX-2.5-22b-distilled-GGUF
  file: ltx-2.5-22b-distilled-transformer-Q5_K_M.gguf
  sha256: cd82849aec38d2be84805b725c803bad44bc67df9850c23a24ef011a51fd7988
text_encoder:
  repo: elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF
  file: gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf
  sha256: a70012eeea2fbee7c9c058fca5710842ec8cb9d9e94e630f2d1a0647a8d300f6
placement:
  transformer_gpu: 0
  text_encoder_gpu: 1
  cpu_offload: true
pipeline:
  resolution: [768, 448]
  fps: 24
  frames: [49, 97, 193]
  audio: true
  preview: false
  lowvram: true
  dynamic_vram: false
prompts: [A, B, C, D]
seeds: [101, 202, 303, 404]
```

---

# 48. Exact benchmark command conventions

Whenever a command is run, record the complete command line in:

```text
results/E*/command.txt
```

Do not abbreviate commands in the report.

A recommended wrapper is:

```bash
/usr/bin/time -v \
  env \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./run-experiment ...
```

If the text encoder is a separate process, record both commands and their environment variables.

---

# 49. Environment variables worth testing

Only test one at a time after the baseline works.

### Allocator

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Debugging

For failures only:

```bash
CUDA_LAUNCH_BLOCKING=1
```

Do not benchmark with `CUDA_LAUNCH_BLOCKING=1`; it destroys the timing characteristics.

### Memory diagnostics

For selected failures:

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

This is a diagnostic experiment, not a production optimization.

---

# 50. Reproducibility rules

Every result must be reproducible from:

```text
experiment manifest
model hashes
git commits
environment.txt
command.txt
seed
prompt
```

Do not use:

```text
latest
main
unspecified model
untracked workflow
```

in final benchmark records.

If an upstream repository updates, create a new experiment ID rather than silently replacing a previous result.

---

# 51. Expected interpretation patterns

These are hypotheses to test, not conclusions.

### Hypothesis H1

Q5 may fit on the 4060 Ti only because GGUF/ComfyUI loads weights intelligently and offloads parts of the model. If so, peak VRAM may look excellent while PCIe/CPU overhead makes it too slow.

### Hypothesis H2

Q4/Q3 may provide a much better practical balance because enough transformer weights can remain resident while retaining acceptable quality.

### Hypothesis H3

Q2 may be the first configuration with enough memory headroom for longer clips and less allocator stress.

### Hypothesis H4

The 2060 may help primarily with **memory partitioning**, not compute. The text encoder is only executed when the prompt changes, so a slower second GPU may still be valuable.

### Hypothesis H5

The 2060 may simply be too small for Gemma Q2/qint2 once runtime buffers are included. In that case CPU encoding should be treated as the baseline rather than trying increasingly exotic GPU placement.

### Hypothesis H6

LTX-2.3 may have a better low-memory ecosystem but lower audiovisual quality than LTX-2.5.

### Hypothesis H7

For `voyage`, a slightly lower-quality LTX configuration may be more useful than a maximum-quality configuration because its throughput and stability permit much longer continuous generation.

---

# 52. What constitutes a successful overall outcome

The LTX-2 effort should be considered successful if at least one configuration satisfies all of:

```text
768×448 output
24 fps
native audio generation
97+ frames
stable repeated execution
no driver resets
no uncontrolled memory growth
reasonable prompt adherence
acceptable temporal coherence
measurable advantage from the second GPU or CPU fallback
```

The best result does **not** need to be real-time.

The primary question is whether LTX can be a practical renderer in an application that may run for hours or days.

---

# 53. What would make LTX-2 the preferred `voyage` renderer

After the experiments, favor LTX for primary rendering only if the best stable configuration provides a compelling combination of:

```text
quality
smooth temporal behavior
native audiovisual generation
stable long runs
predictable memory usage
reasonable segment-generation time
manageable implementation complexity
```

Otherwise keep LTX as a secondary renderer for selected segments/transitions and retain the faster causal renderer as the continuous backbone.

---

# 54. Recommended first execution order

Do not attempt all experiments at once.

Run exactly this sequence:

```text
1. E0 hardware/software preflight
2. TE-1 Q2 Gemma on 2060
3. TE-2 qint2 Gemma on 2060
4. TE-3 CPU encoder baseline
5. E1 Q5 + Q2 Gemma, 49 frames
6. E1 Q5 + CPU Gemma, 49 frames
7. E1 Q5 + successful encoder path, 97 frames
8. E2 Q4 + same encoder path
9. E3 Q3 + same encoder path
10. E4 Q2 + same encoder path
11. E5 extreme int2/qint2 feasibility
12. E6–E9 LTX-2.3 fallback ladder
13. VAE comparison on top 2–3 candidates
14. duration scaling
15. resolution scaling
16. 30-run stability
17. crash recovery
18. `voyage` continuous-transition test
19. final report
```

This sequence minimizes wasted effort because the expensive long tests are performed only after the memory boundary and basic renderer reliability are known.

---

# 55. Primary research references

### Official LTX

- LTX-2 repository: https://github.com/Lightricks/LTX-2
- LTX-2.5 model repository: https://huggingface.co/Lightricks/LTX-2.5
- LTX-2.3 model repository: https://huggingface.co/Lightricks/LTX-2.3

### LTX-2.5 Q5 GGUF audit

- https://huggingface.co/elix3r/LTX-2.5-22b-distilled-GGUF

This source is particularly valuable because it documents a reproducible 16 GB RTX 4070 Ti SUPER run, exact ComfyUI revisions, Q5 transformer hash, Q5 Gemma, low-VRAM launch flags, and measured two-stage memory/runtime.

### LTX-2.5 Gemma GGUF

- https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF

This source documents the LTX-specific Gemma 4 12B Q2/Q4/Q5 GGUFs and exact hashes.

### LTX-2.5 low-VRAM multi-GPU implementation

- https://github.com/A4ax/comfyui-LTX-2.5-Tile-train-LoRa--On-multi-Gpus-low-VRAM-18-gb-Beta

This source documents qint2 Gemma, a 2-bit LTX transformer, per-GPU Gemma placement, and very low-VRAM experiments. Treat it as experimental/community infrastructure rather than official Lightricks guidance.

### LTX-2.3 GGUF

- https://huggingface.co/unsloth/LTX-2.3-GGUF

This provides the LTX-2.3 Q2–Q8 ladder used in the fallback experiments.

### Official LTX documentation

- https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines/docs
- https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core

---

# 56. Final instruction to implementation agents

This file is a benchmark protocol, not authority over current upstream APIs.

Before implementing each experiment:

1. inspect the exact current upstream source;
2. verify that the documented CLI/node arguments still exist;
3. pin the exact revision actually tested;
4. record any deviation from this document;
5. never silently substitute a different model;
6. preserve previous experiment results;
7. do not claim feasibility from file size alone;
8. distinguish "loads", "renders once", "renders repeatedly", and "is useful for `voyage`" as separate claims.

The final report should make those distinctions explicit.

---

# 57. Decision table template for the final report

Fill this only after experiments are complete.

| Candidate | Quality | Temporal | Audio | Speed | 4060 VRAM | 2060 VRAM | CPU RAM | Stability | Complexity | Voyage role |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LTX-2.5 Q5 | | | | | | | | | | |
| LTX-2.5 Q4 | | | | | | | | | | |
| LTX-2.5 Q3 | | | | | | | | | | |
| LTX-2.5 Q2 | | | | | | | | | | |
| LTX-2.5 int2 | | | | | | | | | | |
| LTX-2.3 Q5 | | | | | | | | | | |
| LTX-2.3 Q4 | | | | | | | | | | |
| LTX-2.3 Q3 | | | | | | | | | | |
| LTX-2.3 Q2 | | | | | | | | | | |

The final report must retain raw measurements alongside any interpretation so that a later model revision can be compared without rerunning unrelated experiments.

---

# 58. Measured results (2026-09-30; RTX 4060 Ti 16 GB GPU0 + RTX 2060 6 GB GPU1)

Setup pins (experiment area `~/Projects/video/ltx2-experiments/`):
ComfyUI @2f35f4a, ComfyUI-GGUF @6ea2651 + audited gemma4 patch,
venv py3.11 torch 2.14.0+cu130, upstream LTX-2 @70ee118e in `upstream/`,
A4ax pack in `a4ax-nodes/`, e5 harness container `e5test` (pytorch 2.10/cu128
+ optimum-quanto 0.2.7, `LTX_QUANT_BITS=2`).
ComfyUI server flags `--lowvram --disable-dynamic-vram --preview-method none`.
Deviation: VHS nodes absent from pinned checkout; E1–E4 use core
SaveImage + SaveAudio, ffmpeg mux afterwards (generation path identical).

## E0 — preflight PASS
- GPU0 RTX 4060 Ti 16 GB idle, GPU1 RTX 2060 6 GB idle, driver 595.91.07.
- Disk ~267 GB free at start. Results in `results/E0/environment.txt`.

## TE-1 — Gemma4 Q2_K on 2060: FAIL (OOM)
- File `gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf` (5,956,930,560 B, sha256 `a70012ee…`).
- Server log: `1376 MB usable, 7521 MB offloaded`, then OOM
  (`allocated 2.40 GiB, requested 3.75 GiB, limit 5.60 GiB`).
- Verdict: Q2_K working set (~7.5 GB) exceeds the 2060. Heavier quants fail
  by the same argument. TE must live on the 4060 Ti or CPU.
  Full note: `results/te1/RESULT.md`.

## E1 — Q5_K_M DiT + Q2_K TE (both on 4060 Ti): ALL PASS
- Q5 file `ltx-2.5-22b-distilled-transformer-Q5_K_M.gguf`
  (16,878,535,040 B, sha256 `cd82849a…`).
- euler_ancestral, distilled 9-sigma [1.0..0.0], conv BF16 VAE, CFG 1.0.
- smoke 608x352x49 x3: 110.5s / 120s / 30.25s server-side, 49 PNG + FLAC 2.01s.
- 49 A/B/C/D @768x448: 37.3 / ~77 / ~77 / ~76s (B/C/D include ~40s CPU re-encode).
- 97 A/B/C/D @768x448: ~93 / ~92 / ~91 / ~92s, FLAC 4.01s.
- 193 B @768x448: ~130s, 193 PNG + FLAC 8.01s (= 193/24 exact).
- VRAM ~14.4 GB usable of 15.9 GB; DiT ~14.37 GB with ~1.8 GB offloaded.
- Quality (stills viewed): coherent push-in (dome, glass building);
  193-frame orchard holds structure while water rises — no morphing.
- DETERMINISM: euler_ancestral draws per-step noise from the torch global RNG.
  Fresh-process reruns bit-identical; same-process reruns diverge.
  Voyage implication: fresh process or explicit global-RNG seeding per segment.
- Full note: `results/e1/RESULTS.md`.

## E2 — Q4_K_M A/B vs Q5: PASS, Q4 dominates Q5
- Q4 file `LTX-2.5-Distilled-Q4_K_M.gguf` (15,687,639,488 B = x-linked-size exact).
- Pure-gen (encodes cached): 97f Q4 60s (confirmed 2x) vs Q5 ~92s (~35% faster);
  193f Q4 90s vs Q5 ~130s (~30% faster). 49f parity (~40s).
- Quality on par (orchard coherent to frame 193).

## E3 — Q3_K_M A/B vs Q4: PASS, new recommended working quant
- Q3 file `Q3_K_M.gguf` (12,923,897,280 B, Abiray).
- Pure-gen: 97f Q3 50s (< Q4 60s < Q5 92s); 193f Q3 90s (= Q4).
- Quality on par. Verdict: smallest, fastest, equal quality.

## E4 — Q2_K DiT (vantage) + Q2 TE matrix: PASS
- Q2 file `ltx-2.5-22b-distilled-transformer-Q2_K.gguf`
  (12,125,172,096 B = x-linked-size exact, sha256
  `a91e2a1f527a8791668de0fcc00178a0eb7bb2b47439810845fadbc72efa148d`;
  no upstream checksums published — provenance weaker than Abiray).
- Matrix all PASS, zero OOM: smoke 30s (cached); 49A 40s / B,C,D ~80s;
  97 A-D ~93–100s; 97B_rep (seed 203) 49s pure-gen; 193B 137s.
- Pure-gen 97f: Q2 49s ≈ Q3 50s < Q4 60s < Q5 92s.
- Quality on par (f193 orchard aisle over water, no degradation).
- 30x49 fragmentation (seeds 1001–1030): 30/30 success, 40–41s each, zero errors.

## E5 — int2 DiT + qint2 TE via `e5/run_e5.py`: DEFERRED 2026-09-30 (user: taking too long, move on)
- Files: int2 `ltx-2.5-22b-distilled-int2-main-v2.safetensors` (5,361,967,816 B);
  qint2 `gemma4-12b-with-proj-ltx-2.5-qint2.safetensors` (5,542,759,962 B);
  `embeddings_processor_bf16.safetensors` (6,344,536,376 B).
  Harness: A4ax `load_int4_shard` world=1 + ValidationRunner (runner loads
  TE+EP itself; transformer second — co-resident load OOMs at 15.04 GiB).
- Runs (prompt B 768x448x49 seed 202, neutral guidance unless noted):
  30-step/CFG3 default 244.6s, euler+distilled 70.8s, distilled 70.1s,
  full-neutral+distilled 28.9s, vendored-ancestral 28.8s (near-black 10 KB),
  real-upstream-ancestral 29.0s — ALL grid-of-blobs (or near-black), no content.
  Video-only 23.5s and empty-prompt 29.0s also garbage (audio branch ruled out).
- VERIFIED GOOD (so not the cause): int2 weights cos 1.0 vs Q3 GGUF on norms
  (138/150 tensors across blocks 0/24/47); TE backbone + FE (0.90 f64) + EP
  connectors (1.00, 258/258 remapped keys); aux key coverage complete;
  config multipliers 1000/1000.0 correctly applied; mask/loop plumbing;
  frames unique (no collapse); seed-responsive (seed 203 differs).
- UNRESOLVED: 12 scale_shift_table cos ~0 — but rescale probe (x0.08 on the 39
  large-a2v tables, verified 0.4034→0.0323 live) leaves velocity 3.79 unchanged,
  so the 39/9 large-a2v pattern is architecture, not corruption.
  GGUF audio dims differ (2048x1408 vs official 2048x2048) — int2 matches the
  official arch; GGUF works in its own ComfyUI path regardless.
- Sampler conclusively ruled out (deterministic Euler, vendored ancestral, and
  real upstream `euler_ancestral_denoising_loop` via stub-package `e5/uplink.py`
  all garbage; eta=1.0, s_noise=1.0, seed+10000).
- Live symptom: step-0 video velocity std 3.75 vs audio 1.01 (RF theory ~1.4);
  modalities sane (sigma/timesteps/latent/ctx all 1.0); video hidden
  1.8→11.8→57 across blocks 0–2 while audio holds ~10.
- Current lead (parked): find what over-drives the video stream in blocks 0–2
  (video MSA adaLN / timestep embed / prompt adaln / A2V interaction beyond gain).
  Next planned probe (not run): cross-path block comparison — ComfyUI
  `BasicAVTransformerBlock` (pinned ComfyUI `comfy/comfy/ldm/lightricks/av_model.py`,
  built with `comfy.ops`, `cross_attention_adaln=True`, `ff_bias=False`,
  `audio_ff_bias=True`, `apply_gated_attention=True`) vs A4ax block on identical
  inputs + identical Q3-dequant weights. Block data flow already verified
  equivalent by code reading (MSA rows 0-2, MLP rows 3-5, A2V layout, pre-AV
  snapshot; `PytorchAdaZeroFunction`/`PytorchPostSAFunction` match ComfyUI rmsnorm
  paths). File config for reference: `cross_attention_adaln=True`, `ff_bias=False`,
  `apply_gated_attention=True`, `qk_norm=rms_norm`, `attention_bias=True`,
  `timestep_scale_multiplier=1000`, `av_ca_timestep_scale_multiplier=1000.0`
  (vs ComfyUI hardcoded 1.0 — tested via `--av-ca-mult 1` + real-ancestral:
  still near-black, ruled out as single-knob fix).
  A4ax block0: 84 params, tables (9,4096)/(9,2048)/(5,2048)/(5,4096) +
  prompt tables (2,4096)/(2,2048); video FF no bias, audio FF has bias.

## E6 — LTX-2.3 Q5_K_M distilled + Gemma3 QAT (both backbones): 768x448 OOM, NOT VIABLE
- Files (unsloth/LTX-2.3-GGUF, ungated): DiT Q5 `distilled/ltx-2.3-22b-distilled-Q5_K_M.gguf`
  (16,071,115,808 B); TE backbones `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` (7,432,229,248 B)
  and `gemma-3-12b-it-qat-Q2_K.gguf` (4,768,221,568 B); connectors
  `distilled_embeddings_connectors.safetensors` (2,312,144,712 B, 4 tensors);
  VAEs `distilled_video_vae.safetensors` (1,452,256,522 B, 170 tensors) +
  `distilled_audio_vae.safetensors` (364,853,140 B, 1329 tensors).
  TE wiring = DualCLIPLoaderGGUF(gemma, connectors, 'ltxv') per the official
  unsloth flowers workflow (embedded workflow JSON extracted from the mp4).
  Pinned ComfyUI natively supports the Gemma3 LTX path (sd.py:1962, no patch).
- PASS at 608x352x49: smoke UD-Q4 140s, smoke Q2_K 121s (49 PNG + FLAC 2.01s,
  coherent glass dome — 2.3 renders correctly, pipeline is sound).
- FAIL at 768x448x49: e6_49_A OOM x2 (UD-Q4) + e6q2_49_A OOM (Q2_K) — identical
  signature every time: sampler node 15, 14.60 GiB allocated + 576 MB requested
  in the adaln_single BF16 dequant transient, ~50 MB CUDA-free. The transient is
  DiT-weight-shape-dependent, NOT TE-dependent: swapping UD-Q4 (7.43 GB) for
  Q2_K (4.77 GB) changed nothing. E6_49_B @768x448 succeeded once (81s) via
  lucky residency ordering after an error teardown — flaky, not reliable.
- Verdict: LTX-2.3 Q5 @768x448 does not fit 16 GB under this stack (needs
  14.6 + 0.6 GB vs 15.57 GB limit). Next: 2.3 Q4_K_M (14,326,856,736 B,
  saves ~1.7 GB resident — 2.5-Q4 at 15.69 GB already proved this class fits).

## E6-Q4 — LTX-2.3 Q4_K_M distilled + Gemma3 Q2_K: ALL PASS, recommended 2.3 working quant
- File: DiT Q4 `distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf`
  (14,326,856,736 B = x-linked-size exact, GGUF magic OK); same Q2_K TE,
  connectors, VAEs as E6-Q5. Workflows `e6/e6q4_*.json` (e6q4_ prefixes;
  provenance detour fixed: first Q4 run landed under e6q2_49_A/ names and was
  moved to e6q4_49_A/ — outputs stand, names now correct).
- Gate e6q4_49_A success 50s (vs Q5 OOM). Matrix zero OOM: smoke 608x352 70s;
  49 A-D 50–70s; 97 A-D ~90s each; 193 B 120s (193 PNG + FLAC 8.01s).
- Quality on par with 2.5: 49_A f25 glass dome + lavender field; 193_B f193
  mechanical orchard with glass fruit over water (model baked in letterbox
  bars); 97_D f97 neon night street with lone figure.
- Verdict: Q4 is the recommended 2.3 working quant — fits 16 GB with headroom,
  ~30% faster than Q5 would be, equal quality. 2.3 vs 2.5 comparison pending
  (E7 Q3 / E8+ as ordered).

## E9 — LTX-2.3 Q2_K distilled + Gemma3 Q2_K: ALL PASS, recommended low-memory fallback
- File: DiT Q2 `distilled/ltx-2.3-22b-distilled-Q2_K.gguf`
  (8,275,987,488 B = x-linked-size exact, GGUF magic OK; download curl died at
  ~94% but the .part was already byte-complete — `curl -C -` resumed from
  exactly the x-linked-size, proving completion). Same Q2_K TE, connectors,
  VAEs. Workflows `e6/e9_*.json`.
- Matrix zero OOM: smoke 608x352 60s (fastest smoke yet); gate e9_49_A 30s
  (encode cached); 49 B/C/D 70–71s (fresh CPU encodes); 97 A-D 80–81s;
  193 B 120s (193 PNG + FLAC 8.01s).
- Quality holds at Q2: 193_B f193 mechanical orchard over water at dusk,
  coherent to frame 193; 97_D f97 lone figure on neon night street with
  ferris wheel, cinematic — on par with Q3/Q4, no visible degradation.
- Verdict: smallest + fastest 2.3 config with equal quality — recommended
  low-memory fallback (and the 2.3 ladder is now complete: Q5 OOM-only,
  Q4/Q3/Q2 all pass with quality parity, so lower is strictly better).

## E8 — LTX-2.3 Q3_K_M distilled + Gemma3 Q2_K: ALL PASS, new recommended 2.3 quant
- File: DiT Q3 `distilled/ltx-2.3-22b-distilled-Q3_K_M.gguf`
  (10,770,199,584 B = x-linked-size exact, GGUF magic OK); same Q2_K TE,
  connectors, VAEs. Workflows `e6/e8_*.json` (e8_ prefixes; live in e6/ —
  gen_e8.py kept OUT=e6).
- Gate e8_49_A @768x448 success 40s (encode cached — faster than Q4's 50s).
  Matrix zero OOM: smoke 608x352 111s; 49 A-D 40s (cached) / ~70-71s (fresh
  CPU encodes); 97 A-D 90-91s each; 193 B 120s (193 PNG + FLAC 8.01s).
- Quality on par: 97_C f97 paper flower prompt-faithful and detailed; 193_B
  f193 orchard at dusk over water, coherent (letterbox bars again — model
  variance, also seen in Q4 runs).
- Verdict: Q3 is the new recommended 2.3 working quant — smallest, fastest,
  equal quality. Next per order: E9 (2.3 Q2 DiT).

## TE-2 — Gemma3 Q2_K + 2.3 connectors on 2060: PASS (the 2-GPU split works for 2.3)
- DualCLIPLoaderGGUF(Q2_K 4.77 GB backbone + distilled connectors), 11 distinct
  encodes via `te2/` harness (port 8190, GPU1 only): cold single 28.0s,
  warm 10x 81.5s (~8.2s/encode), peak VRAM 2187 MiB, zero OOM.
  (Contrast TE-1: Gemma4 Q2_K OOM'd on the same card at ~7.5 GB working set —
  2.3's split TE design is what makes the difference.)
- Toolchain note: Gemma3 RoPE (`llama.py:462 precompute_freqs_cis`) JIT-compiles
  a triton `cuda_utils` module via gcc, which needs Python.h — host lacked it
  (deadsnake python3.11, no -dev package). Fixed by installing the exact
  noble1 `libpython3.11` + `libpython3.11-dev` + `python3.11-dev` .debs from
  the PPA pool. Triton caches in ~/.cache/triton afterwards.
- Verdict: the 2060 CAN host the 2.3 text encoder with ~3.4 GB headroom left;
  ~8s/encode is negligible next to 50–120s generation. The 2-GPU architecture
  (4060 Ti DiT+VAE, 2060 TE) is viable for LTX-2.3.

## E7 — Q4 DiT + Gemma3 Q3 TE (UD-Q3_K_XL): PASS, no visible gain over Q2 TE
- File `gemma-3-12b-it-qat-UD-Q3_K_XL.gguf` (6,142,818,688 B = x-linked-size
  exact, GGUF magic OK); Q4 DiT + distilled connectors, prompts B.
- e6q3te_49_B 131s (fresh CPU encode of the new backbone ~80s + ~50s gen),
  e6q3te_97_B 50s (encode cached). 49 PNG + FLAC each, zero OOM.
- Quality: 49_B f25 mechanical orchard at dusk with glass fruit + water
  reflections — prompt-faithful, on par with the Q2_K-TE renders (no
  letterbox bars this time: the 193_B bars were model variance, not systematic).
- Verdict: Q3 TE buys no visible quality over Q2_K TE at these prompts while
  costing +1.4 GB disk and slower CPU encodes. Recommended 2.3 TE stays Q2_K
  (which additionally fits the 2060 per TE-2). Next per order: E8 (2.3 Q3 DiT).

## S21 — encoder result cache: PASS (in-session; cross-process is a Voyage build item)
- ComfyUI caches text-encoder output per distinct prompt text within a session,
  and the E1–E9 matrices quantify the saving exactly: under --lowvram the TE
  encodes on CPU (~40s first encode of a new text: E1 B/C/D ~77s total vs
  E1-A ~37s with the identical prompt-A text reused; TE-2 cold single 28.0s
  on the 2060; E7 fresh Q3-backbone CPU encode ~80s), while a repeat of an
  identical text costs 0.00s (cache hit — the repeat-timing method had to
  perturb the noise seed precisely because identical resubmits return
  instantly without executing).
- So per (§21.1–21.5): encode-once/reuse-many works and is worth ~40s (CPU)
  to ~8s (2060) per repeated prompt — directly relevant to `voyage`, where
  the visual prompt changes far less often than video blocks.
- NOT demonstrated: cross-process reuse (ComfyUI's cache is session-scoped;
  restart = re-encode). No .safetensors-context cache file was built — that
  is a Voyage implementation task (keyed on model_family + revision +
  encoder hash + prompt sha256 per the §21 format), not a feasibility
  question. Verdict: S21 closed — caching is proven valuable, mechanism is
  a build detail.

## S22 — VAE-2: PASS (2.5 diffusion vs conv measured same-latent; 2.3 dev == distilled)
- 2.5 diffusion VAE `vae/ltx-2.5-video-vae-bf16.safetensors` (1,472,223,346 B
  = x-linked-size exact, extent OK; 396 tensors, decoder 310 vs conv's 84 —
  carries `decoder.conv_in_x_t.weight`, so pinned ComfyUI routes it to
  `CausalDiffusionVAE` per sd.py:605; same VAEDecodeTiled node works).
  Symlinked into comfy/models/vae/.
- Steady-state 49f @768x448 (2.5-Q3, prompt B, all cached): conv 33.76s /
  peak 15096 MiB; diffusion 36.95s / peak 15512 MiB (+3.2s ≈ +9%, +416 MiB).
  First-diffusion-run 129.25s was cold-start (fresh CPU encode ~40s + Q3/VAE
  loads), NOT slow sampling — sampling runs 3.5s/it regardless of VAE (the
  earlier VAE-residency-slows-sampling theory is dead).
- Rigorous same-latent A/B: restarted the server (log backed up to
  session_server.log.pre-vae2-restart) to clear cache+RNG, ran conv-202
  (120s cold, samples latent L) then diffusion-202 back-to-back — second run
  finished in 10s (cache hit: no DiT steps, decode-only), proving shared
  latent L. mean|diffusion-conv| = 10.35/255 over 49 frames (8.8–12.9):
  same composition, different grade — diffusion renders brighter/warmer,
  conv darker/cooler. Both artifact-free.
- Verdict: conv stays the default (cheaper, no fidelity loss — the delta is
  grade, not quality); diffusion VAE is viable at +3s/+0.4 GB with a warmer
  render. VAE-1 (conv everywhere in E1–E9) did NOT contaminate transformer
  comparisons (decode is ±9%, rankings unaffected).
- 2.3 VAE-2 collapses: dev_video_vae and distilled_video_vae have IDENTICAL
  keys and IDENTICAL tensor data (data-sha16 87c20c045586e385 both; only
  __metadata__ differs) — bit-identical 49-frame output (mean diff 0.000;
  the 10s distilled run also cache-shared the latent, consistent either way).
  No A/B possible, nothing to choose. (Config: CausalVideoAutoencoder,
  causal_decoder false, timestep_conditioning false — both are conv.)
- Method corrections: (1) ComfyUI 0.33 PromptExecutor HAS a cross-prompt
  input-sensitive CacheSet (execution.py) — the "encode cached" mechanism,
  and the tool that enables same-latent VAE A/B. It is SMALL: the seed-202
  latent was evicted after ~4 intervening runs (conv-202 re-executed in
  131s) — back-to-back ordering is required. (2) SaveImage/SaveAudio
  numbering is DIR-SCAN-based, not session-global (corrects the E1-era
  lesson): post-restart runs append frames 50–98, never overwrite — frame
  index ranges must be tracked per analysis. (3) Host python3 has no numpy;
  use comfy/venv/bin/python for frame math. (4) pkill self-kill again:
  always the `[n]vidia-smi` bracket trick.

## S23 — two-stage Mode A (standard distilled) on Q5/Q4/Q3: ALL PASS
- Adapted `examples/ltx25-gguf-local-two-stage-hq.json` to the pinned stack:
  VHS CreateVideo/SaveVideo tail replaced with core SaveImage + SaveAudio
  (VHS absent — same deviation as E1). Generation path identical to audit:
  stage 1 608x352x49 distilled 8-sigma euler_ancestral CFG 1.0, 2x latent
  upscale (`LatentUpscaleModelLoader` model_name + `LTXVLatentUpsampler`
  samples/upscale_model/vae), stage 2 euler 3-step resample
  (sigmas 0.85/0.7250/0.4219/0.0) CFG 1.0, conv VAE tiled decode.
  Workflows `s23/s23_{q5,q4,q3}_49_B.json` (prompt B / seed 202, Gemma4 Q2_K
  TE — consistent with E1–E4).
- Results (server 8188/GPU0, zero OOM/errors throughout):
  Q3 roundtrip 141s (server 137.34s), Q4 81s (server 73.06s), Q5 80s
  (server 75.52s). Each: 49 PNG @1216x704 RGB + FLAC (~2s audio).
  Run samples in `results/s23/` (f01/f25/f49 + audio per quant).
- Q3 137s likely includes a fresh ~40s CPU encode (first run after S22;
  CacheSet SMALL per S22) — Q4/Q5 back-to-back reused the cached encode.
  Steady-state two-stage ≈ 73–76s (Q4/Q5); single-sample ordering is noisy
  (standing E3 note), so no Q-ranking claimed here.
- VRAM ceiling: Q5 seed-203 repeat (new prefix `s23_q5_49_B_vram`, genuine
  re-execution — identical resubmits cache-hit) with 1s nvidia-smi sampling:
  60s success, 59 samples, PEAK 15512 MiB (~15.15 GiB) — fits the 16 GB
  card under --lowvram with partial offload. (Sampler-loop lesson: a
  `while true` nvidia-smi background job holds the tool pipe — the run
  finished but the command hit the tool timeout; read the log file in a
  fresh command instead.)
- Quality (f25 viewed all three): coherent glass-orchard aisle over water
  at dusk, reflections, no morphing — visibly more detailed than the
  768x448 single-stage Mode B at ~2.6x pixels (1216x704 = 856k vs 344k).
  Different compositions per quant (expected: different weights steer the
  ancestral trajectory from the same seed).
- Verdict: Mode A two-stage is viable on ALL THREE top configs incl. Q5 —
  the audit's 4070TiS result transfers to the 4060 Ti. Cost ≈ 1.5–2x
  single-stage pure-gen time for 2.6x pixels + finer detail. Mode B stays
  the throughput default; Mode A is the quality path. Voyage implication:
  both modes share the identical node set except the upscaler + stage-2
  sampler — one worker can offer both.

## S24 resolution ladder — 2026-09-30: 512x320 / 640x384 / 768x448 x Q3/Q4/Q5 — ALL PASS

- Protocol: single-stage Mode B (E1 recipe — distilled 8-sigma
  euler_ancestral CFG 1.0, Gemma4 Q2_K TE, conv VAE tiled decode), 49f,
  prompt B / seed 202, top-three LTX-2.5 quants. Workflows
  `s24/s24_{q5,q4,q3}_{512x320,640x384,768x448}_49_B.json`
  (generator `s24/gen_s24.py`; unet_name + dims + prefix content-verified).
- Results (server 8188/GPU0, zero OOM/errors; encodes cached throughout —
  prompt B + empty negative unchanged since S23, CacheSet held):
  512x320: Q3 41s / Q4 50s / Q5 50s.
  640x384: Q3 50s / Q4 50s / Q5 61s.
  768x448: Q3 50s / Q4 60s / Q5 60s (consistent with E1–E3 B runs).
  Each: 49 PNG at native res RGB + FLAC (~2s audio). Run samples in
  `results/s24/` (Q3 f25 + audio per resolution).
- Scaling is nearly FLAT across resolutions (41→61s for 2.1x pixels) —
  at 49f the fixed costs (lowvram residency moves, tiled decode) dominate;
  pixel count is not the bottleneck. Single-sample ordering is noisy
  (standing E3 note), so no Q-ranking claimed.
- Quality (Q3 f25 viewed all three): coherent metallic/glass orchard over
  water with reflections, no morphing/artifacts — prompt-faithful at every
  rung. Visible detail progression 512x320 (softer) → 640x384 (sharper) →
  768x448 (finest: bark texture, fruit highlights). Compositions differ per
  resolution (expected: latent shape changes the noise field at same seed).
- Verdict: all three rungs viable on all three quants. Voyage implication:
  768x448 stays the quality default; 512x320 is a valid draft/preview rung
  (~40s/segment); resolution barely moves segment time at 49f, so the
  quality/throughput tradeoff sits in quant choice + frame count, not
  pixels. (Duration scaling S25 + stability S26/E10 next per #54 order.)

- S25 duration scaling verdict (2026-09-30; 2.5-Q3 @768x448 prompt B/seed
  203-205 perturbed to force re-execution, encodes cached, conv VAE,
  same 8188/GPU0 --lowvram server; 1s nvidia-smi VRAM sampler started
  BEFORE the runs via setsid — lesson: `cmd && sampler &` samples only
  post-run idle because submit blocks first; start the sampler first in
  its own call, and pkill it with the bracket trick `pkill -f
  "vram_4[9].log"` or the pattern self-matches):
  - 49f 51s / 97f 50s / 193f 90s server-side, all success, zero OOM
    (49 PNGs + 97 PNGs + 193 PNGs RGB + FLAC each; outputs flat in
    comfy/output/s25_q3_*_B_vram_*.png + audio subdir; samples in
    experiment results/s25/).
  - Scaling is SUBLINEAR: 1.9x time for 3.9x frames (51s -> 90s).
    Steady-state ~0.45-0.5 s/frame at 193f; fixed per-run costs
    (~25-30s: lowvram reshuffle + tiled decode setup) dominate 49f runs.
    Longer voyage segments are proportionally cheaper — throughput
    favors fewer, longer segments.
  - Peak VRAM approximately CONSTANT: 49f 15096 MiB (S22 same config)
    vs 97f 15738 MiB vs 193f 15642 MiB (1s sampler, peaks land in a
    6-12s episode at each run's end = tiled VAE decode transient; the
    two episodes are <1% apart). ~15.1-15.4 GiB at every duration,
    ~0.4-0.6 GiB headroom under the 16GB card. 193f fits with zero OOM
    across E1-E4 + S25.
  - No chunked inference exists in this single-stage GGUF path (one
    latent for the whole clip), so chunk-boundary seams are impossible
    by construction. Quality holds to the last frame: S25 f193
    (seed 205) is a coherent mechanical orchard over water at dusk with
    correct reflections, on par with E1-E4 f193s.
  - First-frame latency is not separable in ComfyUI (KSampler reports
    run totals only); run-level split is ~encode (0s cached, ~40s fresh
    CPU) + sampling (~3.5s/it) + tiled decode transient. Cross-check
    table (pure-gen where measured): 2.5-Q5 97s ~92s/193 130s; 2.5-Q4
    97f 60s/193f 90s; 2.5-Q3 97f 50s/193f 90s; 2.5-Q2K 97f ~49s;
    2.3-Q4/Q3/Q2 97s ~80-91s/193s 120s (2.3 pays ~30s more per 193f).
  - Verdict: duration is NOT the binding constraint — 193f runs on all
    tested quants of both families with flat-ish time and constant
    VRAM. Voyage implication: prefer 97-193f segments (amortizes the
    ~25-30s fixed cost); 49f only for drafts. (Stability S26/E10 next
    per #54 order.)

## S26 — Long-running stability (§26 + E10 core): PASS on 2.5-Q3

  50 consecutive runs, one server process (pid 695135, fresh restart
  after the S26.3 kill test), 8188/GPU0 --lowvram, 768x448, conv VAE,
  no restarts between runs, session-wide 1s nvidia-smi sampler
  (s26/vram_s26.log, 2998 samples, GPU0+GPU1 interleaved lines):
  - S26.1 short repeated generation (§26.1): 30 x 49f prompt B
    (seeds 3001-3030, s26/s26_stab_01-30.json via s26/gen_s26.py,
    loop s26/run_s26.sh -> s26_stab_results.tsv) — 30/30 success,
    40-41s every run (encodes cached), zero errors, zero OOM.
    All 30 dirs hold 49 PNGs 768x448 RGB + FLAC (verified f25 size/
    mode each; samples results/s26/stab01_f25 + stab30_f25: run 1 vs
    run 30 both coherent orchard-over-water, quality holds across
    the session).
  - S26.2 prompt-change stress (§26.2): 20 x 49f alternating A/B/C/D
    (seeds 4001-4020, s26_churn_results.tsv) — 20/20 success,
    80-81s every run (each new prompt = fresh ~40s CPU encode +
    ~40s gen), zero TE load/unload failures across 20 prompt swaps.
    All 20 dirs 49 PNGs + FLAC (samples churnA_f25 + churnD_f25:
    prompt D = coherent neon night street).
  - S26.3 crash recovery (§26.3 ComfyUI analogue): kill -9 server
    mid-sampling (193f seed 206, step 3/8) -> GPU0 back to 223 MiB
    baseline, NO partial artifacts (output dir absent — SaveImage
    writes only after full decode), all prior outputs intact.
    Restart server + resubmit same workflow -> success 191s (cold),
    193 PNGs + FLAC, f193 coherent (results/s26/crash_rerun_f193).
    ComfyUI-level recovery = kill leaves nothing behind, restart +
    rerun works; supervisor duties (identify/rerun/preserve) are a
    Voyage build item.
  - VRAM: session peak 15578 MiB (~15.2 GiB, fits 16GB); quintile
    maxes 15546/15578/15546/15546/15578 and p90 flat 15514-15546
    across ~50 min — NO creep, NO fragmentation growth. Server RSS
    11.7GB at 58 min elapsed.
  - Method notes: node id for the DiT loader is "1" (UnetLoaderGGUF),
    not "10" (gen_s26.py first failed KeyError 'unet_name' on the
    wrong id); s26/submit_only.py (POST /prompt, print id, exit)
    enables kill-mid-run tests; s26 server stdout went nowhere
    (session_server_s26.log holds only the pid line — the harness
    reads status from history API, so no log-based error check was
    possible; all 50 statuses came from history + on-disk outputs).
  - E10 mapping: S26.1+S26.2 ARE E10 Phase 1 (30x49) + Phase 4
    (20 prompt churn) for candidate C1 (2.5-Q3); E4 frag loop was
    the same 30x49 on 2.5-Q2K (30/30, 40-41s). Remaining E10
    phases (10x97, 3x193 per candidate, C2-C3) are extrapolations
    from existing data (S25 durations + zero-OOM record), not
    re-runs — full E10 x 3 candidates is ~6h, deferred to Voyage
    soak (§68) on the built worker.
  - Verdict: the recommended config is STABLE for long unattended
    runs — flat timings, flat VRAM, clean crash behavior. No
    memory-leak or TE-swap stopper for voyage continuous
    generation. (S27 2-GPU value next per #54 order.)

## S27 — Two-GPU value (§27): placements A/B/C measured, 2060 NOT justified for 2.5, marginal for 2.3

  Two fresh-prompt runs (new prompt E, highland loch — vocabulary disjoint
  from A-D, so both encodes are full fresh, CacheSet cannot recall),
  768x448x49, 8188/GPU0 server (--lowvram, placement B / status quo),
  with a 1s resource sampler (s27/sample_s27.sh -> s27/trace.tsv, 241
  samples: ts/gpu0_util/gpu0_mem/gpu1_mem/cpu_busy). Workflows
  s27/s27_49_E_25.json (2.5-Q3, single CLIPLoaderGGUF TE) +
  s27/s27_49_E_23.json (2.3-Q3, DualCLIPLoaderGGUF TE) via s27/gen_s27.py
  (content-verified: DiT names, TE names, s27_ prefixes).
  - Run 1 (2.5-Q3): 80s roundtrip, success, 49 PNGs + FLAC.
    Trace phases razor-sharp: rel 6-46 (~41s) CPU ~73% + GPU 0% =
    fresh Gemma4-Q2_K CPU encode; rel 47-77 (~31s) GPU 90-100% +
    CPU ~8-10% = sampling + tiled decode (peak 15482 MiB).
    So 2.5 fresh-prompt cost = ~41s encode + ~31s gen.
  - Run 2 (2.3-Q3): 151s roundtrip, success, 49 PNGs + FLAC.
    rel 111-196 (~86s) CPU ~75-80% + GPU 0% = cold-backbone
    Gemma3-Q2_K+connectors load + CPU encode (backbone was not
    resident — server held the 2.5 stack); rel 198-208 GPU mem
    888->11320 MiB = 2.5-stack evict + 2.3-stack swap-in (~10s);
    rel 209-238 (~30s) GPU 92-100% + CPU ~13-20% = sampling +
    decode (peak 13624 MiB = 13.3 GiB, 2.3-Q3 resident is smaller).
  - GPU1 (2060): 7 MiB, 0% for the entire 252s span — fully idle,
    as designed (single-device server).
  - Quality spot-check: results/s27/25_E_f25 (stone tower on island,
    mist/water/reeds/dawn — prompt-faithful) + 23_E_f25 (ruined
    tower, mountain reflection in loch — prompt-faithful, different
    composition as expected across models).
  - Placement verdicts (§27 A/B/C):
    A (4060Ti DiT, 2060 TE): IMPOSSIBLE for 2.5 — TE-1 proved the
    Gemma4 Q2_K working set (~7.5GB) exceeds the 2060 (6GB) at the
    smallest available quant; there is no smaller Gemma4 quant to
    try. POSSIBLE for 2.3 — TE-2 proved Gemma3 Q2_K + connectors
    fits (peak 2187 MiB) at ~8.2s/warm-encode on the 2060.
    B (4060Ti DiT+TE when needed, 2060 idle): this is the status
    quo MEASURED above — TE CPU-encodes under --lowvram
    (~41s 2.5 / ~30s-warm 2.3 fresh, 0.00s cached). B-literal
    (TE forced onto the 4060Ti) needs a non-lowvram server mode =
    S28 OFF-1/OFF-2 territory, not tested here.
    C (A + CPU overflow): == A for latency — both GGUF TEs are
    single files with no separable projection/connector stage, so
    the §27 A-vs-C distinction does not apply; encode is already
    fully off the DiT card in A.
  - Value math (the §27 question: is the 2060 justified?):
    2.5: NO — A/C impossible, so the 2060 contributes nothing to
    generation latency. B-status-quo stands (S28 may move the
    encode onto the 4060Ti itself).
    2.3: MARGINAL — A saves ~(warm-CPU-encode ~30s minus 2060 ~8s)
    ~= ~22s per FRESH prompt, 0 per cached prompt, against added
    architecture (second resident process + conditioning transfer
    the ComfyUI path cannot even do — single-device server, no
    conditioning save/load; a Voyage worker would hand-place
    tensors per device trivially).
    Overlap (the §27 report item that matters most): CPU sits at
    ~8-20% during sampling — 80%+ headroom. In a pipelined Voyage
    worker the NEXT prompt's CPU encode runs DURING the current
    segment's GPU sampling, making even the CPU encode ~free in
    wallclock. That retires the 2060 from encode duty on latency
    grounds for BOTH families; its value is elsewhere (director
    worker, SFX worker, or a second concurrent generation
    stream — not measured here).
  - Method notes: SaveImage/SaveAudio filename_prefix "name/frames"
    lands FLAT as comfy/output/<name>/frames_NNNNN_.png (the /frames
    suffix becomes part of the filename, not a subdir — earlier
    "flat outputs" lesson re-confirmed after a false-missing scare);
    start the sampler via setsid in its OWN call BEFORE submit
    (S25 lesson holds); pkill with the [s]ample bracket trick;
    trace ts column is host-clock seconds (relative deltas only).
    (S28 offload next per #54 order; S29 quant + S30 audio are
    offline-composable from existing frames/FLACs.)

## S28 verdict (2026-09-30): CPU/offload modes on C1 (2.5-Q3 @768x448x49 prompt B) — OFF-2 stands, OFF-3 viable, OFF-1 fragile

  - Mapping (§28 -> pinned ComfyUI flags, model_management.py + cli_args.py):
    OFF-1 "no CPU offload" = --highvram (models stay GPU-resident);
    OFF-2 "CPU weight offload" = --lowvram --disable-dynamic-vram
    (status quo, layer streaming); OFF-3 "disk offload" has NO pinned
    knob (no disk-offload flag exists for DiT in ComfyUI) so the
    closest last-resort proxy ran: --novram --disable-smart-memory
    (maximum RAM offload, everything streams). True disk offload
    remains untested by construction — the OFF-3 verdict below is
    for the max-RAM-offload proxy, which is the stricter Voyage
    question anyway (can a bigger quant fit via streaming?).
  - Method: fresh server per mode (cold CacheSet, so every run
    includes a fresh ~40s CPU Gemma4 encode + cold model load —
    totals comparable, sampling-only not isolated); seeds 207/208/
    209 (s28/gen_s28.py clones the S25 49f base, content-verified);
    1s sampler (s27/sample_s27.sh) per run; all success, zero OOM.
  - Results (49 PNGs 768x448 RGB + FLAC each; samples results/s28/):
    OFF-1: 70s roundtrip, peak 15894 MiB (~15.52 GiB — ~50MB under
      the 15.57 limit, FRAGILE), gpu>80% 36/80 samples, CPU max
      40.7% (encode runs GPU-side under highvram). Surprise: fully
      resident FITS — but with no headroom for fragmentation; one
      unlucky transient away from the E6-Q5 signature.
    OFF-2: 120s cold (fresh encode+load; warm steady-state ~50s per
      S25), peak 15094 MiB, CPU max 77.8% (classic CPU-encode
      signature). The known quantity.
    OFF-3: 141s cold, peak 2104 MiB (!), CPU max 73.6%,
      gpu>80% 39/145. Max-offload costs only ~20s over OFF-2 cold
      and holds GPU0 essentially EMPTY — a bigger quant (2.3-Q5's
      16.07GB, or 2.5-Q4/Q5) could stream here where residency
      fails. Quality f25 coherent orchard, on par with OFF-1/OFF-2.
  - Read rate (the §28 GB/s item): no disk I/O exists in any mode
    (proxy is RAM<->VRAM streaming, not disk) — item recorded as
    N/A for the ComfyUI path; a Voyage worker doing true disk
    offload would measure it there. Not a gap: the proxy answers
    the design question (streaming cost ~+15% wallclock,VRAM ~2GB).
  - Quality: f25 all three modes prompt-faithful coherent orchard
    (results/s28/off1_f25.png off2_f25.png off3_f25.png); offload
    mode does not touch pixels (same weights, same math, only the
    residency schedule differs).
  - Verdicts: OFF-2 (--lowvram) STANDS as the Voyage default (best
    understood, warm ~50s/segment, 0.5GB headroom). OFF-3 proxy
    (max-RAM-offload) is VIABLE not last-resort: use it as the
    overflow lane for quants that fail residency (2.3-Q5 @768x448
    is the first candidate — its E6 OOM was a 576MB dequant
    transient at 14.60 GiB resident, exactly what streaming
    sidesteps). OFF-1 (--highvram) marked FRAGILE-feasible: fits
    today at 15.52/15.57, fastest cold (70s, GPU-side encode), but
    do not ship on it — no margin. TE-on-4060Ti (S27's deferred
    B-literal) is implicitly answered: OFF-1 already keeps
    everything resident incl TE and still fits, so a Voyage worker
    CAN hold TE on-card; it just buys ~40s cold / 0s warm (cached)
    at the cost of all headroom — not worth it, CPU-encode-during-
    sampling (S27) stays the design point.
  - Method notes: trace column order is ts/util/mem/gpu1/cpu — an
    earlier pass read r[0] (timestamps) as util and reported
    80/80 and 145/145 >80%; corrected to 36/80, 29/125, 39/145.
    Always index by header name. OFF-3 server flags were
    --novram --disable-smart-memory --preview-method none (dynamic
    VRAM left enabled — novram implies it per cli_args.py:315).
    (S29 quant + S30 audio next per #54 order, both offline from
    existing frames/FLACs; then LTX2-REPORT.md.)

## S29 verdict (2026-09-30): quant contact sheet on C-set (2.5 Q5/Q4/Q3/Q2K @768x448x49 prompt B f25) — lower is strictly better, no cliff

  - Source frames (same latent: all four workflows carry
    RandomNoise noise_seed 202; same prompt B text; same dims):
    comfy/output/e1_49_B, e2_49_B, e3_49_B, e4_49_B
    (frames_00025_.png) -> results/s29/s29_{Q5,Q4,Q3,Q2K}_f25.png
    (all 768x448 RGB).
  - Eyeball: all four render the SAME scene — same orchard
    layout, same trees, same reflections. Quant changes fine
    detail/grain, not composition. No structural degradation at
    any rung incl Q2K; Q5 marginally warmer fruit tones, Q3/Q2K
    marginally cooler. Nothing here would change a candidate pick.
  - Numeric (mean|abs diff| vs Q5, /255): Q4 16.95, Q3 20.30,
    Q2K 20.64. Large per-pixel but composition-identical =
    quant noise redistributes fine texture over the 8 distilled
    steps while conditioning + initial latent hold the layout.
    Q3->Q2K adds ~0.3/255 beyond Q3->Q5: no extra cliff at Q2K.
  - Verdict: S29 CONFIRMS the ladder ranking from E1-E4 timing
    data (Q3 recommended balanced, Q2K smallest/fastest viable).
    Quality is not the differentiator; VRAM/speed are.
    (S30 audio evaluation next, offline from existing FLACs;
    then LTX2-REPORT.md.)

## S30 verdict (2026-09-30): audio evaluation on real FLACs — all candidates ship sound, none disqualified

  - Corpus (results/s30/, copied from frame dirs): Q5/Q4/Q3/Q2K
    49f prompt B (s30_Q{5,4,3,2K}_49B.flac) + Q3 97f/193f
    (s30_Q3_97B/193B.flac). All 48 kHz stereo FLAC.
  - Duration match: 49f -> 2.010s, 97f -> 4.010s, 193f ->
    8.010s in every file. Exact scaling with frame count;
    ~30 ms under frames/24 is a fixed encoder offset, not
    truncation (no abrupt tail: last-250 ms mean/max =
    Q3-49f -15.5/-5.3 dB, Q3-193f -20.8/-10.4 dB,
    Q2K-49f -15.3/-4.6 dB — healthy signal to the last sample).
  - Silence: silencedetect (-50 dB, min 0.2 s) reports ZERO
    gaps in all six files — continuous signal, no dropouts.
  - LUFS integrated: Q5 -12.0 / Q4 -12.3 / Q3 -13.2 /
    Q2K -13.3 (49f); Q3-97f -15.7; Q3-193f -20.1. Level
    declines slightly down the quant ladder (~1.3 LU Q5->Q2K)
    and with clip length (longer beds run quieter). Reported
    as measured — no pass/fail thresholds invented.
  - Audio-VRAM: N/A as a separate offline number — audio VAE
    decode runs inside every measured session peak (S22-S28),
    and the campaign recorded zero audio-attributed OOMs.
  - Audibility verdict per candidate: Q5/Q4/Q3/Q2K all
    signal-present, continuous, healthy levels — audio does
    not disqualify or differentiate any quant. (No listening
    test; levels/silence/continuity only.)
    (Experiment arc complete per #54 order. Next:
    LTX2-REPORT.md comparison report.)

## Phase 0 — Integration spikes (Voyage ltx25/ltx23 backends, 2026-10-01)

### Spike A: Mode-A-121f VRAM probe — PASS
Graph `s0/s0_121_B.json` (from `s0/gen_s0.py`, S23 recipe at length 121:
stage-1 608x352x121 8-sigma euler_ancestral CFG 1.0, 2x latent
upscale, stage-2 1216x704x121 3-step euler, 2.5-Q3 DiT, Gemma4-Q2K
TE, conv VAE tiled decode, prompt B/seed 202, server 8188/GPU0
--lowvram): success 231 s server-side, 121 PNGs 1216x704 RGB +
FLAC, zero OOM. 1 s VRAM sampler (276 samples): peak **14933 MiB
(~14.58 GiB)** — fits 16 GB with ~1.4 GiB headroom. Mid-frame
f61 (`results/s0_spikeA_f61.png`): coherent glass orchard over
water at dusk, prompt-faithful, no artifacts — quality holds at
121 f in Mode A. VERDICT: locks `segment_frames=96` in 121-frame
windows with 25-frame carry for both workers (no 48-novel
fallback needed).

### Spike B: 3-segment continuation handover — PASS (Mechanism 1)
Chain in the experiment tree, same prompt B, server 8188/GPU0
--lowvram: seg1 `s0_121_B` (seed 202, fresh) -> seg2
`s0_chain_seg2` (seed 303, prefix = seg1 frames 97-121) ->
seg3 `s0_chain_seg3` (seed 404, prefix = seg2 frames 97-121).
Each chained graph = Spike A graph + 25x LoadImage (prefix
frames must be copied to `comfy/input/` ROOT as
`s0<seg>_frames_*.png` — 0.33.0 validates LoadImage against
the input root, not subdirs) + `BatchImagesNode` (the node
name is `BatchImagesNode`, not `BatchImages`; autogrow keys
`images.image{i}`) + `LTXVImgToVideoInplace` (strength 1.0,
explicit `bypass: False` required) rewiring
`10.video_latent` from node 8 to node 56. Builder
`s0/gen_s0chain.py` (prompt format, string keys seg2/seg3).
seg2 success ~230 s, seg3 success 110 s; each 121 PNGs
1216x704 RGB + FLAC, zero OOM. Boundary metrics (mean|diff|
/255, gray): handover A prefix-fidelity avg 23.71, seam
f25->f26 ratio **0.93x**; handover B prefix-fidelity avg
25.07, seam ratio **1.02x** — both far under the 3x qual
gate. Eyeball (`results/s0B_A_seg1f121.png`,
`s0B_A_seg2f01/f25/f26.png`, `s0B_B_seg3f26.png`): same
orchard/aisle/water/dusk across all five, prefix holds
layout, seams show no jump. VERDICT: Mechanism 1
(25-frame frozen prefix via noise_mask, strength 1.0) is
the workers' continuation mechanism; Mechanism 2
(`LTXVAddGuide`) not needed. Builders document the
LoadImage-root convention for the worker's prefix staging.

### Live ltx25 worker test in voyage-ltx image — FULL PASS
Ephemeral driver `/tmp/ltxtest_ltx25.py` (never committed) drove
`voyage/workers/video_ltx25.py` in-container (voyage-ltx image,
GPU0, both GPUs idle first): INIT READY (load 3.8 s) -> FRESH
seg1 (seed 303, 1216x704@24, 121 frames committed, prefix_discarded
0) -> CONT seg2 (seed 404, 96 novel committed, prefix_discarded
25) -> EVICT clean. Two bugs fixed along the way: (1)
`PromptExecutor` needs full `cache_args`
(lru+ram+ram_inactive, pinned main.py:329-349 formulas — not
just `{"ram": ...}`); (2) `imageio` is absent from the image by
design, so both workers use an ffmpeg-direct `_save_mp4` (temp
PNGs via PIL + libx264/yuv420p + non-empty verify, no rebuild
needed). Outputs `/tmp/ltxtest/`: seg1.mp4 (121 f/5.04 s) +
seg2.mp4 (96 f/4.0 s) both h264 1216x704@24, joint audio.wav
(pcm_s16le 48 kHz stereo 3.97 s), peak ~14.8 GiB, zero
OOM/errors. Seam seg1-f121 -> seg2-f01 mean|diff| **10.2/255**,
eyeball continuous (same dusk orchard, water channel,
pink-purple sky). VERDICT: worker bootstrap, in-process
ComfyUI execution, prefix continuation, joint-audio commit and
evict all proven live — Phase 2 plumbing is sound; same test
still to run for video_ltx23.py.

### ltx23 live worker test — FULL PASS (video_ltx23.py)

Ephemeral driver `/tmp/ltxtest_ltx23.py` (never committed, mirror
of the ltx25 driver) drove `voyage/workers/video_ltx23.py`
in-container (voyage-ltx image, GPU0, both GPUs idle first):
INIT READY (backend ltx23, load 4.5 s, 15.2/15.6 GiB free;
"LTX23 ComfyUI stack ready (Q3 DiT, Gemma3+connectors TE,
distilled VAE)") -> FRESH seg1 (seed 303, 1216x704@24, 121
frames committed, prefix_discarded 0, denoise 134.1 s, save
24.8 s) -> CONT seg2 (seed 404, 96 novel committed,
prefix_discarded 25) -> EVICT clean. TE load shows a long
`clip missing: [vision_model.*, multi_modal_projector.*]`
list — benign (the Gemma3 QAT backbone is text-only, vision
tower absent by design; same class as the E6 token_embd
dequant note). One worker bug fixed along the way (applies to
BOTH workers): `_save_mp4` assumed the output parent dir
exists — the ltx23 driver never created `/tmp/ltxtest23/`,
so ffmpeg died with "No such file or directory" AFTER both
sampling stages completed. Fix: `output_path.parent.mkdir
(parents=True, exist_ok=True)` at the single commit choke
point in `generate_blocks` (covers segment mp4 + audio.wav +
tail + tape). Outputs `/tmp/ltxtest23/`: seg1.mp4 (121 f/5.04
s) + seg2.mp4 (96 f/4.0 s) both h264 1216x704@24, joint
audio.wav (pcm_s16le 48 kHz stereo), zero OOM/errors.
Handover, measured properly (prefix pair, not motion pair):
prefix fidelity seg2-f01..25 vs seg1-f97..121 avg **0.109**
(consistent with Spike B's 0.093 — the prefix drifts
slightly through the two stages but holds layout),
in-segment consecutive baseline 0.0161, seam seg2-f25->f26
**0.0148 ratio 0.92x vs baseline — PASS** (same as Spike B
0.93x, far under the 3x qual gate). Eyeball seg1-f121 vs
seg2-f01: identical glass pavilion at dusk (string lights,
round ottomans, white SUV left, trees, dusk sky), same
layout, no morphing/artifacts. (A first seam metric of
0.055 was motion-across-24-frames on the wrong pair
seg1-f121 vs seg2-f01 — the prefix-fidelity/seam-ratio
script above is the correct methodology; keep it for Phase
5.) VERDICT: ltx23 worker plumbing proven live to the same
bar as ltx25 — DualCLIP TE load, prefix continuation,
joint-audio commit and evict all sound. Both Phase-2
workers are now live-verified; remaining integration work
is Phase 3 (supervisor bypass), Phase 4 surface/docs, and
Phase 5 qual legs.

## Phase 5 qual legs (2026-10-01): end-to-end via Voyage — BOTH PASS

Same 8s shape each (`run.sh generate --backend <ltx25|ltx23>
--duration 8s --style 'glass orchard at dusk, calm' --name
qual-<backend> --seed 303 --director deterministic --no-sfx`;
runs in gitignored `Voyage/output/qual-ltx25/` +
`Voyage/output/qual-ltx23/`): 2 segments per run (fresh 121f
+ continued 96f = 217f @24fps) through the supervisor drive
+ joint-audio bypass + commit validation + validate verb +
floors-lift finalize.

Leg-1 ltx25: seg0 video 157.5s + audio 0.0s (joint, `no take
(joint)`, 8 beats @95BPM), seg1 video 129.1s + audio 0.0s
(4 beats @60BPM); validate VALID (2 segments, 217 frames);
finalize → final.mp4 h264 1280x720@32fps 287f 8.97s +
aac48k stereo. Eyeball: prompt-faithful dusk orchard, no
artifacts; 3 separately-extracted frames have distinct
md5s — real motion, not frozen (a first double-extraction
gave byte-identical PNGs through my own ffmpeg `-ss`
placement error; corrected methodology is separate
invocations).

Leg-2 ltx23: seg1 video 128.8s + audio 0.0s (joint),
validate VALID (2 segments, 217 frames), finalize 8.978s
→ final.mp4 same 1280x720@32 287f 8.97s + aac48k (3.9MiB).
(`validate --run` takes the /app-relative path
`output/qual-ltx23`, not `Voyage/output/qual-ltx23` —
path-doubling lesson.)

VERDICT: both backends proven end to end — supervisor
drive, joint-audio bypass (audio stage ~0s), commit
validation, and floors-lift finalize all sound. This closes
the LTX-2/2.5 experiment arc E0–E9 + S21–S30 + Phase 0–5;
remaining Voyage work is the floors-as-minimum hardening
(`plan_augmentation` treats 1280x720@32 as a floor, never a
ceiling) and routine operation.
