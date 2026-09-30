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

# 44. LTX-2.5 versus LongLive comparison hook

The outcome should ultimately be compared against the current LongLive 2.0 plan.

For the best LTX candidate and the current LongLive candidate, compare:

| Metric | LTX | LongLive |
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
