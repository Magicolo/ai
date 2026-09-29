# Voyage — autonomous infinite audiovisual voyage

Voyage generates an endless stylized video with a matching soundtrack,
one **segment** at a time, forever (or until paused/stopped). A supervisor
process owns run state and drives three worker subprocesses over a typed
JSONL-RPC protocol:

- **director** — picks the next concept/shot (deterministic built-in, or
  Qwen3-8B + MiniLM novelty embeddings);
- **video** — renders frames (fake testsrc built-in, LongLive 2.0
  causal stream, LTXV 2B tail-chained extensions, or CausVid DMD causal
  rollouts — all on CUDA);
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
./scripts/run.sh models download ltxv-2b         # ~7 GB LTXV video weights
./scripts/run.sh models download causvid        # ~28 GB CausVid DMD + Wan2.1-1.3B base
./scripts/run.sh models download director-qwen8b  # ~16 GB director LLM
./scripts/run.sh models download audio-acestep    # ACE-Step checkpoints
./scripts/run.sh models download inspector-qwen35 # ~19 GB VLM (optional)
./scripts/run.sh models verify                    # check every weight file
```

## Basic run

One-shot fixed-duration video (init → run → validate → finalize in one call;
aborts if validation fails unless `--skip-bad`):

```bash
VOYAGE_GPUS=1 ./scripts/run.sh generate --backend ltxv --duration 5s \
  --style "pastel neon line-art, peaceful"
# -> ./output/voyage/final.mp4 (run dir defaults to ./output/<run-id>)
# --backend causvid renders 832x480 @ 16 fps (CausVid DMD + Wan2.1-1.3B base);
# --backend longlive2 renders 1280x704 @ 24 fps; --backend fake needs no GPU.
```

Step-by-step (for pause/resume and unbounded runs):

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

`generate` flags: `--backend ltxv|longlive2|causvid|fake` (default ltxv),
`--duration 5s` (e.g. `5s`, `90`, `1m30s`, `2m`, `1h`, `1h2m3.5s`; rounds
up to whole segments), `--draft`, `--director qwen|deterministic`,
`--blocks`, `--take-seconds`, `--quantization fp8|bf16`,
`--beats-per-segment`, `--drift-every-n`, `--final-video`, `--skip-bad`,
`--verbose`/`--no-color`.

Bare `voyage` (no verb) launches the interactive launcher TUI: every
`generate` setting with its default in one required-first list, live
validation with a derived segments/frames plan, then Generate runs the
same pipeline with per-segment prompts and Stop/Back/Quit. Needs a TTY;
pick `fake` on GPU-less boxes for CPU smoke runs.

```bash
./scripts/run.sh status --run <dir>            # §59 sections + slowest stage
./scripts/run.sh inspect scoreboard --run <dir> # per-segment frames/stages/metrics
./scripts/run.sh benchmark video --run <dir> --warmup 1 --measured 3
./scripts/run.sh soak --run <dir> --segments 3 # stability trend + validate
```

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
- `docs/UPSTREAM_CAUSVID_NOTES.md` — CausVid integration notes.
- `docs/UPSTREAM_LTXV_NOTES.md` — LTXV integration notes.
