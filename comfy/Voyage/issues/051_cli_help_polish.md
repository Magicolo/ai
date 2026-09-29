# 051 — CLI `--help` leaves required values anonymous; `models_target` bare; three error-path papercuts

- Status: resolved (2026-09-25, CLI track — CLI-side subset; TUI/OPERATIONS-duration + models-info deferred, see log)
- Severity: medium-low (discoverability + one real path bug: `stop --finalize`
  reintroduces the doubling fix)
- Area: CLI polish — `voyage/cli.py:101-132,213-216,489-494,675-694,
  749-766,1106-1311`, `voyage/console.py:145-147`
- Rank rationale: mostly papercuts, but (1) below re-opens a fixed data-loss-class
  bug for relative `--run`.

## Technical description

## Why this is an issue

The CLI is the only interface for headless and scripted runs, so anonymous
required values and late validation errors tax every new user: a typo'd
models target fails deep in `cmd_models` instead of at parse time, an empty
`benchmark --run` passes argparse only to exit 2 at runtime, and a relative
`stop --finalize` writes the output path to a doubled location — silently
re-opening a data-loss-class path bug that was already fixed once for
`generate`. Individually papercuts, together they erode trust in the tool;
each fix is a one-liner (`help=`, `choices=`, resolve through `_run_dir_arg`).


(a) Anonymous requireds: `init --output/--run-id/--style/--seed/--force`,
`run --run/--segments`, `finalize --run/--output`, `soak --run/--segments` show
argparse defaults with empty help strings — no hint that `--output` is a run
dir, `--style` is the permanent charter, `--run` must be the run dir.
`models [models_target]` has no help, no `choices=` — typo targets fail late in
`cmd_models` (`unknown models target`, exit 2 to stdout) instead of argparse
`invalid choice … (choose from …)` to stderr. `benchmark --run` defaults to `""`
with help "required for video/audio" but argparse accepts empty → runtime
`requires --run` exit 2. `inspect {concepts,…}` lists positionals before `--run`
in usage, inviting `voyage inspect scoreboard --run X` vs
`voyage inspect --run X scoreboard` confusion (only the latter parses).

(b) Duration help understates the parser (`cli.py:675-694,1190-1195`,
`tui_state.py:56`, `docs/OPERATIONS.md:128-133`): parser accepts `1h`, fractional
`2.5m/1.5h`, combined `1h2m3.5s`, bare `90`, whitespace — all verified live —
but help/error strings say only `'5s','90','1m30s','2m'`. `'-5s'` fails the regex
→ `invalid duration '-5s' (examples…)` instead of `duration must be positive`
(the `total<=0` branch is unreachable for negatives). Neither help nor TUI
`FIELD_HELP["duration"]` states the round-up overshoot (see 024's 5s→6s fake plans).

(c) Three error-path papercuts:
1. `cmd_stop --finalize` builds `Path(args.run)/"final.mp4"` from the *raw*
   `--run` string (`cli.py:489-494`), while `cmd_run/cmd_finalize` resolve via
   `_run_dir_arg()` (absolute, fixed 2026-09-24 after the live doubling bug).
   Relative `--run` + `--finalize` reintroduces the doubling for the output path
   only.
2. `_require_cuda_stack()` prints `video backend {backend!r} needs…` even when
   only the *audio* backend is CUDA (`needs_cuda = video in CUDA or audio in
   CUDA` but message uses `config.video.backend` — an acestep-audio + fake-video
   run on a CPU image blames `fake`).
3. Streams/exits: `console.error()` → stderr, `ok/info/line` → stdout (progress
   splits under capture); `unknown models target` + `known: …` → stdout exit 2,
   while siblings (`refusing to init…`, `download failed`) → stderr;
   `cmd_models: info` prints a pointer instead of real info.

## Evidence

```
$ PYTHONPATH=. python3 -c "from voyage.cli import parse_duration; ..."
'5s' -> 5.0 | '90' -> 90.0 | '1m30s' -> 90.0 | '2m' -> 120.0 | '1h' -> 3600.0
'2.5m' -> 150.0 | '1h2m3.5s' -> 3723.5 | ' 90 ' -> 90.0
'-5s' -> ValueError: invalid duration '-5s' (examples: '5s', '90', '1m30s', '2m')
'0s' -> ValueError: duration must be positive, got '0s'
$ sed -n '489,494p' Voyage/voyage/cli.py   # cmd_stop --finalize path (still raw)
    code = _set_status(_run_dir_arg(args.run), "STOP_REQUESTED")
    if code == 0 and args.finalize:
        args.output = str(Path(args.run) / "final.mp4")
        return cmd_finalize(args)
$ sed -n '762,766p' Voyage/voyage/cli.py   # CUDA message (still video-only)
    needs_cuda = config.video.backend in _CUDA_BACKENDS or ...
    ...
    print(_cuda_stack_error(config.video.backend), file=sys.stderr)
```

`python3 -m voyage.cli <verb> --help` per verb (sweep); duration matrix above
re-ran live 2026-09-25.

## Reproduction

Helps above; `voyage stop --run output/v --finalize` → tries
`output/v/output/v/final.mp4` context.

## Source references

- Files/lines above.

## Resolution candidates

One-line `help=` per argument; `choices=` for `models_target`; `required=True`
split for benchmark video/audio vs end-to-end (or two sub-parsers); unify help to
`e.g. '5s','90','1m30s','2m','1h','1h2m3.5s' (rounds up to whole segments)` in
CLI + TUI + OPERATIONS; extend regex to signed numbers so negatives hit the
positivity branch; resolve `args.output` through `_run_dir_arg` in `cmd_stop`;
interpolate the actual offending backend(s) in the CUDA message; usage errors to
stderr; flesh out `models info` or drop the stub.

## Investigation / progress / resolution log

- 2026-09-25: found by docs sweep; duration matrix executed live.
- 2026-09-25 (repair): re-ran the duration matrix live (output pasted in
  Evidence — fractional/combined/bare/whitespace all pass; `-5s` hits the
  regex branch, `total<=0` reachable only via `0s`). Re-verified
  `cmd_stop:489-494` (output path still raw — bug stands) and the CUDA
  message (:762-766, still `config.video.backend` — bug stands); corrected
  Area line refs (`:744-766` → `:749-766`, added `:213-216` for
  `models_target`). Added `## Why this is an issue`.
- 2026-09-25 (CLI track): FIXED the CLI-side subset in `voyage/cli.py`:
  (a) `help=` added to every anonymous required (init --output/--run-id/
  --style/--seed/--force; run/generate/pause/resume/stop/validate/finalize/
  soak/inspect --run; soak --segments; stop --finalize; benchmark
  --warmup/--measured/--segments; models action/target/dir) and
  `choices=[longlive2-bf16, ltxv-2b, causvid, director-qwen8b, audio-acestep,
  inspector-qwen35]` on `models_target` (typos now fail at parse with
  `invalid choice` on stderr); (b) duration help unified to
  `_DURATION_EXAMPLES` (`'5s','90','1m30s','2m','1h','1h2m3.5s'` + rounds-up
  note, shared by the error string and `--duration` help) and the regex
  extended to signed numbers so `-5s` hits `duration must be positive`
  instead of the regex branch; (c1) `cmd_stop --finalize` builds the output
  via `resolve_run_dir(args.run)` (:588); (c2) `_cuda_offenders()` (:858)
  qualifies video/audio so audio-only CUDA blames e.g. `audio 'acestep'`,
  and `_cuda_stack_error` says plain `backend ...` (:~850); (c3)
  `unknown models target` + `known:` go to stderr (:~316-322). Deferred as
  out of scope (tui_state.py/config.py/OPERATIONS-duration not touched):
  TUI `FIELD_HELP["duration"]` + OPERATIONS duration unification, `Literal`
  on `DirectorConfig.backend`, `models info` stub (kept), inspect
  positional-before-`--run` usage order (argparse definition order), and
  the console stdout/stderr split (by design: error→stderr,
  progress→stdout). Tests: `tests/test_cli_hardening.py` (models choices
  parse + programmatic-stderr, duration forms + `-5s` positivity, stop
  finalize no-doubling via monkeypatched `cmd_finalize`, audio-blame).
  Gates green.
- 2026-09-29 (this track, scope leftovers only): FIXED the deferred trio.
  `voyage/tui_state.py:69-70` `FIELD_HELP["duration"]` unified to
  `'5s','90','1m30s','2m','1h','1h2m3.5s'` + rounds-up note (matches
  `cli._DURATION_EXAMPLES`); `docs/OPERATIONS.md` duration unified in both
  spots (TUI field + generate paragraph: fractional/combined/bare/
  whitespace + overshoot); `voyage/cli.py` `models info` fleshed out
  (bundles + pins pointer + CausVid CC BY-NC-SA 4.0 + verify hint).
  Out of scope kept: `Literal` on `DirectorConfig.backend`, inspect usage
  order, console split (by design). Tests: 1 new in
  `tests/test_cli_hardening.py` (`models info` bundles/license/verify).
  Scoped gates green.
