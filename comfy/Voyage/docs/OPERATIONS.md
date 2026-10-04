# OPERATIONS — runbook

```bash
./scripts/run.sh configure vdemo --backend ltxv --segments 3 --style "..."
./scripts/run.sh generate vdemo
```

`configure` plans (writes `output/<name>/manifest.json`); `generate`
renders exactly `manifest.segments − state.committed_segments`, then
validates and finalizes. Re-running `generate` resumes crashes,
renders grown plans, and re-finalizes; a validated run whose
`final.mp4` already covers the timeline does nothing ("nothing to
do"). Ctrl-C rests PAUSED (state re-read at each boundary).

## Console output

`generate` renders per-segment progress: a header with
the destination + phase, the video-geometry line (backend, geometry, fps,
blocks, scene-cut flag), the full video prompt(s), the music caption with
the beat grid (beats @ BPM) plus the SFX caption the finalize-time pass
will condition on, animated spinners with live elapsed
timers per stage (inspect/director/video/audio/validate/commit), and a
commit summary with take ids + take action, prefetch state, and per-stage seconds. Two verbosity levels, console
only (logs/metrics stay plain):

```bash
./scripts/run.sh generate vdemo                 # compact default
./scripts/run.sh generate vdemo --verbose       # + seeds, transitions, take reasons
./scripts/run.sh generate vdemo --no-color      # plain (also honors NO_COLOR)
```

Colors/animation engage only on a real TTY with `rich` installed
(`rich>=13.7` is a core dependency, pinned into the worker images that
install with `--no-deps`); pipes and tests get identical plain words.

## Two verbs (no TUI, no other verbs)

`configure` owns every run option (backend, segments/duration, style,
seed, director, generation overrides, sfx/augment policy); `generate`
takes the run NAME, with `--segments`/`--duration` as an approved
shorthand that additively extends the stored plan before rendering. The launcher TUI and the `run` / `status` /
`pause` / `resume` / `stop` / `validate` / `finalize` / `sfx` /
`benchmark` / `soak` / `inspect` / `models` / `doctor` verbs are all
removed — `generate` runs the whole pipeline (segments → validate →
finalize) internally. Bare `voyage` (no verb) prints help, exit 2.

## Plan (`configure`) then render (`generate`)

```bash
./scripts/run.sh configure vdemo --backend ltxv --segments 3 --style "..."
./scripts/run.sh generate vdemo
```

`configure` writes `output/vdemo/manifest.json` (full effective config
+ planned segment count + finalize policy); `generate` renders exactly
`manifest.segments − state.committed_segments`, then validates and
finalizes under manifest policy. Duration is human-readable (`5s`,
`90`, `1m30s`, `2m`, `1h`, `1h2m3.5s`; fractional/combined/bare/
whitespace-padded all parse) and converts to a segment count at
configure time (rounds **up**, so the video never runs short).
Backend presets set geometry/device automatically (`ltxv`: 768×512 on
`cuda:0`; `causvid`: 832×480 @ 16 fps on `cuda:0`; `fake`: CPU smoke
runs) and pair the audio backend too (`ltxv`/`causvid` get real
ACE-Step music on `cuda:0`; `fake` keeps the sine test tone). Add
`--low-definition` for the lowest native reasonable resolution of the
effective backend instead (fresh creates default to high;
`--high-definition` selects it explicitly; both together is an error;
tier changes on committed runs are refused). Extra
run flags (`--director`, `--blocks`, `--take-seconds`,
`--quantization`) live on `configure`. Validation failure aborts
before finalize unless `--skip-bad`; on a GPU box with no visible GPU
a warning is printed (the worker will fail at init). `run.sh` selects
the container automatically: a CUDA backend (`ltxv`, `causvid`,
`acestep` — from `--backend` or the run's `manifest.json`) switches
to `voyage-video:latest` with `--gpus all` and pins `-w /app`, unless
`VOYAGE_IMAGE`/`VOYAGE_GPUS` are set explicitly. A CUDA backend in an
image without torch fails fast with a pointer to `voyage-video`
instead of a cryptic worker error. Before the banner, `generate`
gates on `ffmpeg`/`ffprobe` presence and free disk (run dir + each
model stack's mount, nearest existing ancestor when the mount is not
created yet), then verifies — and downloads when missing — only the
models the effective config needs (the video backend's own stack,
paired ACE-Step audio, qwen director unless deterministic, MMAudio
SFX only when the finalize pass will run; `fake` needs nothing and
stays offline). Missing stacks download in parallel with one spinner
each (plain lines without a TTY); `--no-download` fails instead of
fetching.

## Stopping mid-run

`generate` commits until the manifest count, honoring the state.json
control plane at each segment boundary; Ctrl-C (SIGINT) rests PAUSED.
Re-running `generate` continues from `next_segment_number` — there is
no separate resume verb.

## Crash recovery

Do nothing special: `generate` again. Uncommitted segment dirs are
deleted as torn work, DONE-without-advance re-commits, worker crashes
restart within budget, disk-full rests at PAUSED_DISK_FULL until space
is freed. Validation runs inline before every finalize.
Full scenario table: `docs/STATE_AND_RECOVERY.md`.

## Finalization

`generate`'s finalize step collects DONE segments only, verifies
checksums/ranges/alignment, and publishes atomically (sources never
mutated). Use `configure --skip-bad` to finalize around corrupt
segments with warnings. Presentation floors apply at finalize: output
is ≥24 fps and ≥1216×704 by default (`--min-fps`/`--min-resolution`/
`--no-augment` on `configure`, stored in `[augment]` — see
`docs/AUGMENT.md`). The SFX pass dubs director-captioned effects
under the music afterwards unless `--no-sfx` (pins
`--sfx-caption`/`--music-caption`/`--video-caption` on `configure` —
see `docs/SFX.md`).

## Long-run monitoring

- `state.json` — counters, status, concepts: the run at a glance.
- `logs/metrics.jsonl` — every commit's stages, gauges, and verdicts
  (tail it live; `logs/` rotates daily, 30-day prune).
- `output/<name>/final.mp4` — the shipped artifact (plus the `-audio`
  twin when the SFX pass runs).
- Every commit logs a `resource_gauges` event (RSS peak, disk free,
  worker VRAM).
- Logs rotate daily (`metrics.jsonl`, `*-worker.log`; 30-day prune).
  History readers span the rotation: `iter_metric_files(run_dir)`
  (`voyage/logrotate.py`) yields dated siblings oldest-first with the
  live file last.
- Worker `benchmark` ops + soak trending: see `docs/BENCHMARKING.md`.
