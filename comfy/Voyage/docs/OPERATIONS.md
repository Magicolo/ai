# OPERATIONS — runbook

```bash
./scripts/run.sh doctor
./scripts/run.sh models verify
./scripts/run.sh generate --backend ltxv --duration 5s --style "..."
./scripts/run.sh init --output <dir> --run-id <id> --style "..." --force
./scripts/run.sh run --run <dir> [--segments N] [--draft] [--quantization fp8|bf16]
./scripts/run.sh status --run <dir>
./scripts/run.sh pause --run <dir>     # safe pause at next boundary
./scripts/run.sh resume --run <dir>    # continue after pause/stop
./scripts/run.sh stop --run <dir> [--finalize]
./scripts/run.sh validate --run <dir>
./scripts/run.sh finalize --run <dir> --output <file.mp4> [--skip-bad]
```

Run without `--segments` for the autonomous voyage (runs until
pause/stop/SIGINT; state re-read at each boundary; SIGINT rests PAUSED).
After `stop`, status rests at STOP_REQUESTED until `resume`.

## Console output

`run` / `generate` / `soak` render per-segment progress: a header with
the destination + phase, the full video prompt(s) and the music caption
with the beat grid (beats @ BPM), animated spinners with live elapsed
timers per stage (inspect/director/video/audio/validate/commit), and a
commit summary with per-stage seconds. Two verbosity levels, console
only (logs/metrics stay plain):

```bash
./scripts/run.sh run --run <dir> --segments 2                 # compact default
./scripts/run.sh run --run <dir> --segments 2 --verbose       # + seeds, transitions, take reasons
./scripts/run.sh run --run <dir> --segments 2 --no-color      # plain (also honors NO_COLOR)
```

Colors/animation engage only on a real TTY with `rich` installed
(`rich>=13.7` is a core dependency, pinned into the worker images that
install with `--no-deps`); pipes and tests get identical plain words.
`--verbose` / `--no-color` also parse on `status`, `validate`,
`finalize`, `benchmark`, `soak`, and `inspect`.

## Interactive launcher TUI (bare `voyage`)

With no verb, `voyage` launches a Textual form that shows every
`generate` setting with its default in one compact required-first
list, validates live with a derived segments/frames plan, and runs
the generation in a worker thread with per-segment video + audio
prompts streamed into the run view — then Back/Quit actions on
completion:

```bash
./scripts/run.sh            # needs a TTY (run.sh allocates -it when attached)
```

Non-TTY invocations and images without `textual` get guidance + exit 2
(all CLI verbs stay usable non-interactively). `textual>=8.0` is a core
dependency (slim image via `pyproject.toml`, worker images via explicit
`pip install`). The worker thread's stdout/stderr are redirected into
the run log so stray prints can't fight Textual's screen
(`voyage/tui.py:780`). Stop writes STOP_REQUESTED to the run-dir state
— the same control plane as `voyage stop` — so the run exits at the
next segment boundary and still validates + finalizes. Note: the
default backend is `ltxv` (CUDA); bare `./scripts/run.sh` on a GPU box
already launches the video image with `--gpus all` (host GPU probe, no
variables needed), so ltxv just works — on a GPU-less box it stays slim,
so pick `fake` for CPU smoke runs. The run view
paints synchronously on Generate (headline + `▶ starting` line before
the worker boots), ticks an elapsed timer on the headline every second
while the event loop is alive, and streams segment lines as they
commit; a nonzero exit returns to the form with the worker's last line
appended (e.g. the CUDA-stack reason), never a stuck view.

### Form fields (one list, required first)

`voyage/tui.py:379-470`; verbose help in `FIELD_HELP`,
`voyage/tui_state.py`:

- Style (required) — human-owned string, baked into the run config
  and every prompt; compact multiline editor (height 3), focused on
  mount (`AUTO_FOCUS`, `voyage/tui.py:258`) so typing lands with no
  click.
- Name (required) — the run name; doubles as the output folder
  (`output/<name>`) and final video (`output/<name>/final.mp4`) —
  there are no separate output/final-video fields.
- Duration (required) — target length (`5s`, `90`, `1m30s`); rounds
  up to whole segments.
- Backend — video preset (geometry + device + audio pairing),
  with a visible `▾` affordance; `ltxv`/`longlive2` need the CUDA
  worker image + a GPU.
- Director — `qwen` drifts the story every Nth segment,
  `deterministic` holds the style.
- Quantization — DiT weight precision; `bf16` keeps highlights clean
  at ~+4.5 GB VRAM.
- Blocks, Take seconds, Beats, Drift, Seed (empty = preset
  defaults: 1 block; 45 s takes above the 20 s audio-ahead window;
  4 beats doubling to hold >=60 BPM; drift every segment; seed 0)
  plus draft / force / skip-bad / verbose / no-color checkboxes.

Invalid fields get flagged (`field-invalid`: red-tinted background on
text fields, red border on dropdowns) and errors render live in
`#errors-line`;
Generate is blocked until they clear, with the derived
segments/frames plan in `#plan-line`. Dropdowns keep their bordered
chrome at a fixed 3 rows — an explicit height blanks the value line on
Textual 8 (verified by SVG-text test), so the border color alone
carries focus/invalid and spacing never shifts.

### Focus-driven help panel

The form sits beside a help panel (`#form-columns` /
`#help-panel`, `voyage/tui.py:544-548`) showing an overview plus the
focused field's `FIELD_HELP` text and its current error
(`#help-body`, updated on `DescendantFocus`). On terminals narrower
than 100 cells the panel stacks below the form (`NARROW_WIDTH`,
`voyage/tui.py:110`).

### Key map (mouse optional)

The whole flow is keyboard-driven; the mouse works but is never
required (buttons mirror every action). The map is also shown
in-app (`#key-hints` under the form, `#run-keys` in the run view):

| Keys                | Action                                  |
|---------------------|-----------------------------------------|
| `ctrl+g`            | Generate from the form                  |
| `ctrl+x`            | Stop from the run view (`ctrl+s` would  |
|                     | freeze the terminal via XOFF)           |
| `b`                 | Back to the form (run view, when idle)  |
| `ctrl+q`            | Quit (two-press confirm while running)  |
| `Tab` / `Shift+Tab` | Move focus between fields               |
| `Enter` / `Space`   | Activate button / open dropdown         |
| `↑` / `↓`           | Move within dropdowns                   |

### Remembered settings

Every successful Generate rewrites `~/.config/voyage/tui-last.toml`
with the submitted form (including `name` — legacy `run_id` files
migrate silently); the next launch prefills from it
(`LAST_SETTINGS_PATH`, `voyage/tui_state.py`). Both directions
never raise: a missing/unparseable file falls back per-field to
defaults while valid fields survive, and a failed save is silently
ignored — persistence can never break the UI
(`save/load_last_settings`).

### GPU warning

CUDA backends (`ltxv`, `longlive2`) show a one-line warning under the
backend dropdown (`#gpu-warning` via `tui_state.gpu_warning`): they
need the CUDA worker image (`VOYAGE_IMAGE=voyage-video`) plus a GPU
(`--gpus all`). Other backends show nothing.

### Two-press quit

`ctrl+q` / Quit while a run is active arms a confirm ("press Quit
again to confirm") instead of exiting; the second press quits. When
idle, Quit exits immediately.

## One-shot fixed-duration video (`generate`)

`generate` chains init → run → validate → finalize in one call with sane
defaults (`--backend ltxv`, `--run-id voyage`, `--seed 0`, run dir
`./output/<run-id>`, final video `<run>/final.mp4` unless `--final-video`):

```bash
./scripts/run.sh generate --backend ltxv --duration 5s --style "..."
```

Duration is human-readable (`90`, `90s`, `2m`, `1m30s`, `1h`; combined forms
like `1h2m3.5s` allowed). Segment count rounds **up**, so the video is never
shorter than requested. Backend presets set geometry/device automatically
(`ltxv`: 768×512 on `cuda:0`; `longlive2`: default geometry on `cuda:0`;
`fake`: CPU smoke runs) and pair the audio backend too (`ltxv`/`longlive2`
get real ACE-Step music on `cuda:0`; `fake` keeps the sine test tone). Extra run flags (`--draft`, `--director`,
`--blocks`, `--take-seconds`, `--quantization`) pass through. Validation
failure aborts before finalize unless `--skip-bad`; on a GPU box with no
visible GPU a warning is printed (the worker will fail at init). `run.sh`
selects the container automatically: a CUDA backend (`ltxv`, `longlive2`,
`acestep` — from `--backend` or the run's `voyage.toml`) switches to
`voyage-video:latest` with `--gpus all` and pins `-w /app`, unless
`VOYAGE_IMAGE`/`VOYAGE_GPUS` are set explicitly. Bare `run.sh` (the TUI,
backend picked interactively) counts as CUDA-needing when the host has a
GPU (`nvidia-smi -L` probe), so ltxv works with no explicit variables;
without a GPU it stays slim. A CUDA backend in an image
without torch fails fast with a pointer to `voyage-video` instead of a
cryptic worker error.

## Start / pause / resume / stop

`run` flips state to RUNNING and commits until the count or a request.
`pause` sets PAUSE_REQUESTED (honored at the boundary, rests PAUSED).
`stop` sets STOP_REQUESTED (rests STOP_REQUESTED; `--finalize` runs the
finalizer inline). `resume` clears the request so the next `run`
continues from `next_segment_number`. Pause-mid-run is pinned by test.

## Crash recovery

Do nothing special: `run` again. Partial dirs heal by reuse, DONE-without
advance re-commits, worker crashes restart within budget, disk-full rests
at PAUSED_DISK_FULL until space is freed. Then `validate` to confirm.
Full scenario table: `docs/STATE_AND_RECOVERY.md`.

## Finalization

`finalize` collects DONE segments only, verifies checksums/ranges/
alignment, and publishes atomically (sources never mutated). Use
`--skip-bad` to finalize around corrupt segments with warnings.

## Long-run monitoring

- `status` — §59 sections: uptime, video backend/render spec, world,
  audio buffer, workers, slowest stages, free storage.
- `inspect scoreboard` — per-segment table: frames, stage seconds,
  visual metrics with deltas, view paths.
- `inspect metrics` — event count + last 5 `metrics.jsonl` events.
- Every commit logs a `resource_gauges` event (RSS peak, disk free,
  worker VRAM) — the input to `soak` trend reports.
- Logs rotate daily (`metrics.jsonl`, `*-worker.log`; 30-day prune).
- `benchmark` / `soak` harnesses: see `docs/BENCHMARKING.md`.
