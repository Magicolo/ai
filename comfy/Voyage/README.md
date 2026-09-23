# Voyage — autonomous infinite audiovisual voyage

Voyage generates an endless stylized video with a matching soundtrack,
one **segment** at a time, forever (or until paused/stopped). A supervisor
process owns run state and drives three worker subprocesses over a typed
JSONL-RPC protocol:

- **director** — picks the next concept/shot (deterministic built-in, or
  Qwen3-8B + MiniLM novelty embeddings);
- **video** — renders frames (fake testsrc built-in, or LongLive 2.0
  causal stream on CUDA);
- **audio** — renders a slow loop of music takes (fake sine built-in, or
  ACE-Step 1.5 on CUDA) plus video-synced SFX.

Every segment commits transactionally (`DONE` marker → checksums → state
advance), so a crash at any point heals on retry. Full design:
`Voyage/DESIGN.md`. Operator docs: `Voyage/docs/`.

## Quick install

All execution happens inside Docker — the host needs only the `docker` CLI.
No `pip install` on the host, ever.

```bash
./scripts/build.sh          # slim CPU image (supervisor + fake backends)
./scripts/build-video.sh    # CUDA image: torch + LongLive + ACE-Step
./scripts/build-director.sh # CPU image: Qwen3-8B director + VLM inspector
./scripts/gates.sh          # ruff + format-check + mypy strict + pytest
```

Model weights live outside images in `~/.cache/voyage-models` (override
with `VOYAGE_MODELS`). See `docs/INSTALL.md` and `docs/MODELS.md`.

## Model prerequisites

Fake backends need nothing. Real backends need one download each
(all ungated Hugging Face repos, pinned revisions in `docs/MODELS.md`):

```bash
./scripts/run.sh models download longlive2-bf16   # ~48 GB video weights
./scripts/run.sh models download director-qwen8b  # ~16 GB director LLM
./scripts/run.sh models download audio-acestep    # ACE-Step checkpoints
./scripts/run.sh models download inspector-qwen35 # ~19 GB VLM (optional)
./scripts/run.sh models verify                    # check every weight file
```

## Basic run

```bash
# New run (style charter = the permanent visual identity):
./scripts/run.sh init --output /tmp/vdemo --run-id vdemo \
  --style "pastel neon line-art, peaceful" --force

# Generate 2 segments, then check them:
./scripts/run.sh run --run /tmp/vdemo --segments 2
./scripts/run.sh validate --run /tmp/vdemo

# Ongoing status, then finalize to one MP4:
./scripts/run.sh status --run /tmp/vdemo
./scripts/run.sh finalize --run /tmp/vdemo --output /tmp/vdemo/final.mp4
```

Omit `--segments` to run until `voyage pause` / `voyage stop` / SIGINT.
GPU run: `VOYAGE_IMAGE=voyage-video:latest VOYAGE_GPUS=1 ./scripts/run.sh …`
Fast iteration: add `--draft` (640×352, 1 block/segment, 45 s takes).

## Layout

- `voyage/` — supervisor package (config, state, RPC, workers, media,
  CLI). Never imports torch/transformers/diffusers (DESIGN §83).
- `voyage/workers/` — subprocess entry points (fake + GPU + director).
- `worker/` — CUDA/director Dockerfiles.
- `tests/` — pytest suite, runs in-container (`./scripts/test.sh`).
- `scripts/` — thin docker wrappers (`VOYAGE_IMAGE`/`VOYAGE_GPUS`/`VOYAGE_MODELS`).
- `docs/` — operator and reference documentation (§87 tree).

## Docs index

- `docs/INSTALL.md` — environments, CUDA, ffmpeg, env vars, downloads.
- `docs/MODELS.md` — exact model IDs, revisions, links.
- `docs/BACKENDS.md` — worker adapter interface, experimental backends.
- `docs/ARCHITECTURE.md` — process layout and ownership.
- `docs/STATE_AND_RECOVERY.md` — persistence invariants, crash scenarios.
- `docs/PROMPTING.md` — style charter, novelty, staged prompts.
- `docs/AUDIO.md` — ACE-Step slow loop, continuation, final mix.
- `docs/OPERATIONS.md` — runbook: run/pause/resume/stop/recover/finalize.
- `docs/TROUBLESHOOTING.md` — OOM, CUDA, disk-full, corruption, …
- `docs/BENCHMARKING.md` — benchmark/soak protocol and reports.
- `docs/UPSTREAM_LONG_LIVE_PATCHES.md` — LongLive runtime patches.
