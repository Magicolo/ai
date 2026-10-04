# Voyage — autonomous infinite audiovisual voyage

Voyage generates an endless stylized video with a matching soundtrack,
one **segment** at a time, forever (or until paused/stopped). A supervisor
process owns run state and drives three worker subprocesses over a typed
JSONL-RPC protocol:

- **director** — picks the next concept/shot (deterministic built-in, or
  Qwen3-8B + MiniLM novelty embeddings);
- **video** — renders frames (fake testsrc built-in,
  LTXV 2B tail-chained extensions, or CausVid DMD causal
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
./scripts/build-video.sh    # CUDA image: torch + ACE-Step + director venv (Qwen decider on cuda:1)
./scripts/gates.sh          # ruff + format-check + mypy strict + pytest
```

Model weights live outside images in `~/.cache/voyage-models` (override
with `VOYAGE_MODELS`). See `docs/INSTALL.md` and `docs/MODELS.md`.

## Model prerequisites

Fake backends need nothing. `configure` verifies — and downloads when
missing — only the stacks the effective config needs (pinned revisions
in `docs/MODELS.md`); `--no-download` verifies without fetching:

- ltx25 ~38 GB Q3 + TE + VAEs, joint A/V (TE/VAEs gated; default)
- ltxv-2b ~7 GB video weights; causvid ~28 GB DMD + Wan2.1-1.3B base;
  ltx23 ~20 GB Q3 + TE + VAEs, joint A/V
- director-qwen8b ~16 GB LLM; director-qwen4b-awq ~2.6 GB GPU decider;
  director-qwen35-gguf ~3 GB llama-server sidecar GGUF
- audio-acestep checkpoints; sfx-mmaudio ~13 GB (CC-BY-NC-4.0);
  film ~66 MB (MIT + Apache-2.0); realesrgan-anime ~18 MB (BSD-3-Clause);
  inspector-qwen35 ~19 GB VLM (optional)

## Basic run

Two verbs: `configure` plans a run into `output/<NAME>/manifest.json`,
`generate` reconciles the run directory to that plan and runs it
(segments → validate → finalize in one call; aborts if validation
fails unless `--skip-bad`):

```bash
VOYAGE_GPUS=1 ./scripts/run.sh configure vdemo \
  --backend ltx25 --segments 2 \
  --style "pastel neon line-art, peaceful"
VOYAGE_GPUS=1 ./scripts/run.sh generate vdemo
# -> ./output/vdemo/final.mp4
# ltx25 (the default) renders joint video+audio native 1216x704 @ 24 fps;
# finalize lifts video to >=1216x704 @ >=24 fps via the augmentation floors
# (see docs/AUGMENT.md; --no-augment keeps native geometry) and dubs the
# MMAudio SFX bed under the native soundtrack. --backend fake needs no GPU.
# Re-running `generate vdemo` renders more segments after `configure vdemo
# --segments 4` grows the plan, resumes crashed runs, and re-finalizes;
# a validated run whose final.mp4 already covers the timeline does nothing.
```

`configure` flags: `--backend ltxv|causvid|ltx25|ltx23|fake` (default ltx25),
`--segments N` or `--duration 5s` (e.g. `5s`, `90`, `1m30s`, `2m`, `1h`,
`1h2m3.5s`; duration rounds up to whole segments and is stored as a
count), `--seed` (omit for a fresh random seed, printed at init),
`--director qwen|deterministic|llama`, `--blocks`, `--take-seconds`,
`--quantization fp8|bf16`, `--beats-per-segment`, `--drift-every-n`,
`--final-video`, `--skip-bad`, `--min-fps`/`--min-resolution`/`--no-augment`
(finalize floors, see docs/AUGMENT.md),
`--music-caption`/`--video-caption`/`--sfx-caption` (caption pins, see
docs/SFX.md), `--no-download`, `--verbose`/`--no-color`. `generate`
takes the run NAME (plus `--verbose`/`--no-color`) with an optional
plan-extension shorthand: `--segments N` / `--duration D` additively
extends the stored plan before rendering.

```bash
./scripts/run.sh configure vdemo --segments 4  # grow the plan, then:
./scripts/run.sh generate vdemo                # ...renders segments 2-3
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
- `docs/SFX.md` — finalize-time video-synced effects (captions, windows).
- `docs/AUGMENT.md` — finalize presentation floors (≥24 fps, ≥1216×704), chunked runner.
- `docs/OPERATIONS.md` — runbook: configure/generate/recover.
- `docs/TROUBLESHOOTING.md` — OOM, CUDA, disk-full, corruption, …
- `docs/BENCHMARKING.md` — worker benchmark ops and soak protocol.
- `docs/UPSTREAM_CAUSVID_NOTES.md` — CausVid integration notes.
- `docs/UPSTREAM_LTXV_NOTES.md` — LTXV integration notes.
