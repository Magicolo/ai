# INSTALL — environments and prerequisites

Three container images cover the three compute profiles. The host needs
only the `docker` CLI (plus an NVIDIA driver + nvidia-container-toolkit
for GPU work). Nothing is ever `pip install`ed on the host.

## Supervisor environment (`voyage:latest`, `Dockerfile`)

`python:3.12-slim` + ffmpeg + `pydantic/pytest/mypy/ruff`. Runs the
supervisor CLI and all **fake** backends (testsrc video, sine audio,
deterministic director) — enough for init/run/validate/finalize,
benchmarks, soak tests, and the full pytest suite. No GPU libraries:
the supervisor package never imports torch/transformers/diffusers.

```bash
./scripts/build.sh   # docker build -t voyage:latest .
./scripts/gates.sh   # ruff + format-check + mypy strict + pytest, in-container
```

## LongLive environment (`voyage-video:latest`, `worker/Dockerfile.video`)

CUDA 12.8 + Python 3.10 + torch 2.8/cu128 + torchao 0.13 + flash-attn +
LongLive@`6b36d20` + ACE-Step-1.5@`ca1e85fe` + torchaudio + transformers
4.57.6 (pinned: 5.x breaks LongLive). Runs the `video_longlive` and
`audio_acestep` workers with `--gpus all`. ~16 GB VRAM minimum
(4060 Ti class); the 2060 cannot hold the ACE DiT (see TROUBLESHOOTING).

```bash
./scripts/build-video.sh   # docker build -f worker/Dockerfile.video .
```

## ACE-Step environment

No separate image: ACE-Step lives in `voyage-video:latest` (same CUDA
stack as LongLive). Video and audio time-share the GPU sequentially —
the supervisor evicts the video session, renders audio, evicts audio,
and rebuilds video from `recovery.pt` (see ARCHITECTURE).

## Director environment (unified image, no separate build)

No separate image: the director decider runs in its own venv inside
`voyage-video:latest` (`/opt/venvs/director`: CUDA torch + transformers
5.17 + GPTQModel for 4-bit AWQ) on cuda:1, spawned by the supervisor via
`VOYAGE_DIRECTOR_PYTHON`. Default placement is cuda:1
(`Qwen3-4B-AWQ`, ~2.6 GiB); `--director-device cpu` opts back into the
legacy bf16 Qwen3-8B CPU path (~16 GB host RAM, slim-image compatible).
The Qwen3.5-9B VLM inspector still serves from CPU system RAM (~19 GB).

```bash
./scripts/build-video.sh
```

## CUDA requirements

- NVIDIA driver supporting CUDA 12.8, nvidia-container-toolkit installed.
- Pass `--gpus all`: `VOYAGE_GPUS=1 ./scripts/run.sh …`.
- `voyage doctor` reports driver/GPU/ffmpeg facts plus torch-CUDA,
  disk-free, and a models presence summary before you start. Full
  weight-file checks stay behind `models verify`; remaining doctor gaps
  (compute capability, CUDA runtime, FlashAttention/Triton, checkpoint
  compat, permissions, worker interpreters, ACE-Step runtime
  availability) are listed in
  TROUBLESHOOTING.

## ffmpeg

Baked into every image. Used for fake-backend renders, take slicing,
segment assembly, and finalization. Host ffmpeg is never used.

## Environment variables (`scripts/run.sh`)

| Variable        | Default                  | Meaning                                    |
|-----------------|--------------------------|--------------------------------------------|
| `VOYAGE_IMAGE`  | `voyage:latest`          | image to run (`voyage-video:latest` = GPU) |
| `VOYAGE_GPUS`   | `0`                      | `1` adds `--gpus all`                      |
| `VOYAGE_MODELS` | `~/.cache/voyage-models` | host dir mounted at `/models`              |

`PYTHONDONTWRITEBYTECODE=1` is exported by all scripts so container runs
never leave root-owned `__pycache__` in the bind mount.

## Model downloads

```bash
./scripts/run.sh models download longlive2-bf16   # video (~48 GB)
./scripts/run.sh models download ltxv-2b         # LTXV video (~7 GB)
./scripts/run.sh models download causvid        # CausVid DMD + Wan2.1-1.3B base (~28 GB)
./scripts/run.sh models download director-qwen8b  # director (~16 GB)
./scripts/run.sh models download audio-acestep    # audio checkpoints
./scripts/run.sh models download inspector-qwen35 # VLM (~19 GB, optional)
./scripts/run.sh models verify                    # presence + size checks
```

Exact repo IDs, revisions, and links: `docs/MODELS.md`. All repos are
ungated — no Hugging Face token needed.
