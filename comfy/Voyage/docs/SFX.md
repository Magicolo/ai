# SFX — finalize-time video-synced effects

The SFX pass dubs director-captioned sound effects under the music-only
final **after** `finalize_run` publishes it. It never touches segments:
windows condition on the shipped pixels, join with manual fades, and
`amix` lays the bed under the music at −6 dB (Zoomy parity). Stems
persist under `audio/sfx/` with an `sfx.jsonl` ledger; `validate_run`
extends read-only.

## Three-caption doctrine

Every director decision carries three caption families, all derived from
the same concept + style charter so music and effects track the story
(`voyage/director.py:44`):

1. per-block **video stages** (the shot prompts);
2. a **music_caption** (tempo/texture/environment for ACE-Step takes);
3. an **sfx_caption** (concrete sound descriptions — objects, actions,
   environments — for MMAudio windows).

All three must evolve gradually as the general prompt drifts
(`voyage/director.py:126-127`); the deterministic fallback tracks the
concept the same way. Previous captions feed back into the next
decision, so a mid-run style change repaints music (source-timeline
repaint + 1 s WAV crossfade) and re-captions future SFX windows without
a restart.

## Backends and ladder

| Backend | Device | What it renders |
|---------|--------|-----------------|
| `fake` (default) | CPU | built-in noise — old runs stay byte-identical, no weights |
| `mmaudio` | `cuda:0` | MMAudio 44 kHz windows from the segment captions |

CUDA video backends pair the MMAudio stack on `cuda:0` by default; `fake`
keeps the fake-noise backend on CPU so CPU-only test runs never touch
weights (`voyage/config.py:157-176`, `_sfx_preset`). The SFX worker is a
subprocess behind the JSONL loop — no torch imports in the supervisor.

Weight ladder (2026-09-29, `docs/MODELS.md:130-131`):

- `small_44k` fits the 6 GB 2060 (4.6 GiB peak);
- `medium_44k` OOMs the 2060;
- `large_44k_v2` (default) needs the 4060 (6.2 GiB peak).

Weights are CC-BY-NC-4.0 (`models download sfx-mmaudio`, ~13 GB:
3 variants + VAE/sync/CLIP/vocoder under `<models>/mmaudio/`).
Under-provisioning the SFX stack is the common finalize OOM — drop
`model_size` before touching anything else (see TROUBLESHOOTING).

## Finalize pass (windows / fades / amix / stems)

`voyage/sfx_finalize.py:38-54`:

- native window `SFX_WINDOW_SECONDS = 8.0`, overlap
  `SFX_WINDOW_OVERLAP = 1.0` (killed by manual fades on join — never
  `acrossfade`, same two incidents as the music path);
- shortest renderable window `MIN_SFX_WINDOW_SECONDS = 1.0` (below
  ~0.64 s the sync branch yields fewer than 16 frames; the planner
  merges a starved tail into its predecessor so counts stay exact);
- bed level `SFX_VOLUME = 0.5` (−6 dB under the music);
- ledger `sfx.jsonl`, stems dir `audio/sfx/`
  (`SFX_LEDGER_NAME`, `SFX_STEMS_DIRNAME`).

`plan_sfx_windows` tiles `[0, timeline)` with step `window − overlap`;
the last window clamps to the timeline. A window fully inside one
segment carries that segment's SFX caption; a junction window joins both
sides with `"; "` so the render hears the cut. Each window renders to
staging (`window.flac`, `voyage/workers/sfx_mmaudio.py:383`), converts
to the requested WAV shape, then joins use manual fades + delay and the
final `amix` mixes the bed under the music. Every assembly step verifies
non-empty outputs. The ledger appends one row per window
(`window_id`, `start`, `duration`, `caption`, `seed`, …) with
flush + fsync + fsync_dir (takes-philosophy, immutable, versioned).

Config (`voyage/config.py:361`, `[sfx]` TOML):

```toml
[sfx]
backend = "fake"        # "mmaudio" (CUDA) | "fake" (built-in, CPU-only)
device = "cpu"          # CUDA backends pair "cuda:0"
models_dir = "/models"
model_size = "large_44k_v2"  # small_44k | medium_44k | large_44k_v2
```

## `voyage sfx` verb and caption pins

```bash
./scripts/run.sh sfx --run <dir> [--video <file.mp4>] [--output <file.mp4>]
```

Dubs SFX onto an existing video without re-finalizing
(`cmd_sfx`, `voyage/cli.py:1200`): defaults to the run's `final.mp4`,
publishes `final-sfx.mp4` beside the input (the input is never
modified). Caption source is per-segment director captions unless
`--sfx-caption` overrides — required for runs committed before SFX
captions existed.

Pins (all in-memory, never written to `voyage.toml` unless noted):

- `--sfx-caption` (finalize + `sfx` verb): one caption for the whole
  timeline; default is per-segment director captions.
- `--music-caption` / `--video-caption` (`run`/`generate`): pin the
  music/video families; default is director-driven evolution.
- `--no-sfx` (finalize path only): skip the SFX pass even when `[sfx]`
  is configured. The standalone `sfx` verb has no `--no-sfx` — the
  verb IS the pass.
- `--sfx-backend fake|mmaudio`, `--sfx-device`, `--sfx-model-size`
  (`small_44k|medium_44k|large_44k_v2`), `--sfx-workers 1|2`
  (2 = shard `small_44k` across cuda:0+cuda:1, needs 2 visible GPUs).

`generate` downloads the SFX stack only when the finalize pass will
run (`fake` needs nothing and stays offline); `--no-download` fails
instead of fetching.

## CUDA pairing

Video-augment chunks run on `cuda:0` while the MMAudio SFX stack (when
present) renders on `cuda:1` — the two stages share nothing but chunk
boundaries, so a 2-GPU box runs them side by side and a 1-GPU box runs
serially on `cuda:0` (`voyage/augment.py:14-18`). The SFX worker
defaults to `cuda:0` (`voyage/workers/sfx_mmaudio.py:40`); the
`--sfx-device` override and `[sfx] device` TOML move it.

`run.sh` selects `voyage-video:latest` + `--gpus all` for any CUDA
backend in `[video]`/`[audio]`/`[sfx]` (`scripts/run.sh:69-70`,
`ltxv|longlive2|causvid|acestep|mmaudio`); `fake`/`fake`/`fake` stays
slim. A CUDA backend in an image without torch fails fast with the
`voyage-video` pointer (`_require_cuda_stack`, `voyage/cli.py:1386`),
never a late worker error.

## Failure modes

- **OOM at finalize**: drop `model_size` down the ladder first
  (`large` → `small`), then check co-residency — the supervisor evicts
  video before audio/SFX and rebuilds from `recovery.pt`; driving
  workers manually without `evict_gpu` re-creates the OOM.
- **Missing weights**: `models download sfx-mmaudio` + `models verify`;
  the VLM/director resolution pattern applies (missing snapshot
  re-downloads into the volume on demand instead of silently falling
  back).
- **Pre-SFX runs**: finalize needs `--sfx-caption` (one caption for the
  whole timeline) because those segments carry no SFX family.
- **Short timelines**: below 1.0 s the planner merges the tail instead
  of rendering a starved window — counts stay exact, no crash.
