# 218 — Deleted CLI verbs (`benchmark`/`soak`/`models`) still prescribed in docs and runtime strings (HIGH)

## Technical description
`BENCHMARKING.md:12-17` teaches `run.sh benchmark video|audio|end-to-end`
and `run.sh soak`; `INSTALL.md:85-98` teaches
`run.sh models download <stack>` / `models verify`. The two-verb CLI deleted
all of them. `voyage/cli.py:1-11` says
`run/status/pause/resume/stop/validate/finalize/sfx/benchmark/soak/inspect/models/doctor
+ TUI are gone`; `build_parser` registers exactly 2 subparsers
(`voyage/cli.py:239,329` — `configure`, `generate`; verified live
2026-10-07).

## Rationale
Not cosmetic staleness — the primary bench/soak/model runbook. An operator
following it gets `exit 2` after a full image-selection + `docker run`.

## Live evidence
```
$ grep -n "benchmark\|soak" docs/BENCHMARKING.md | head
12:./scripts/run.sh benchmark video --run <dir> [--warmup 1 --measured 3]
15:./scripts/run.sh benchmark end-to-end [--segments 2]
17:./scripts/run.sh soak --run <dir> --segments N

$ grep -n "add_parser" voyage/cli.py
239:    conf = sub.add_parser(   # configure
329:    gen = sub.add_parser(    # generate

$ ./scripts/run.sh --help 2>&1 | head -8
usage: voyage [-h] {configure,generate} ...
```
`run.sh` dry-run even masks the failure: `VOYAGE_DRY_RUN=1 ./scripts/run.sh
benchmark video …` prints `image=voyage:latest / gpus=none` exit 0 — the
verb error only surfaces inside `docker run`.

## Repro
`VOYAGE_DRY_RUN=1 ./scripts/run.sh benchmark video --run output/vdemo` →
dry-run OK; real run → `voyage: error: argument {configure,generate}:
invalid choice: 'benchmark' (exit 2)`.

## Source refs
- `Voyage/docs/BENCHMARKING.md:8-22,69-87`
- `Voyage/docs/INSTALL.md:82-99`
- `Voyage/docs/OPERATIONS.md:131`
- `Voyage/voyage/cli.py:1-11,239,329`
- `Voyage/scripts/run.sh:38-103` (sniff still names the dead verbs)

## Online sources
- `Voyage/DESIGN.md:8525-8528` (two-verb deletion as-built); CLI best
  practice "bare verb prints help exit 2" is honored, the docs are not.

## Fix candidates
1. Rewrite `BENCHMARKING.md` to the library path
   (`python -m pytest -m gpu`, `tests/test_qualification.py::summarize_run`,
   `voyage/boundary_metrics.py`) or delete it with a pointer.
2. Rewrite `INSTALL.md:82-99` to `configure [--no-download]` ensure-path
   (`voyage/models_ensure.py`, `model_registry.download_model`), not
   `run.sh models …`.
3. Add a `run.sh` fail-fast: reject first-arg not in
   `{configure,generate}` before image selection (keeps dry-run honest).

## Log
- Track E sweep, 2026-10-07. Parser count verified live by orchestrator.
  Read-only; nothing fixed.

## Consolidated from 296_deleted_models_verb_still_prescribed_everywhere (2026-10-07)

296 (HIGH, pass-2 TUI-remnant sweep) is the code + `BACKENDS`/`MODELS`/`SFX` residue of the same two-verb deletion: #218 above covers `BENCHMARKING.md:8-22` + `INSTALL.md:82-99` + `OPERATIONS.md:131` + the `run.sh` sniff only; none of the sites below are cited in 218.

### Technical description

`voyage/cli.py:1-11` + parser probe prove `models` is gone (only `configure`/`generate`). Yet 12 live strings tell the operator to run it. Following them yields `invalid choice: 'models' (exit 2)` after a full container start. User-facing breakage, not cosmetic.

### Live evidence

```
$ rg -n "voyage models (download|verify)" voyage tests docs README.md scripts
voyage/model_registry.py:4: Downloads are explicit (`voyage models download`) —
voyage/model_registry.py:522: volumes provisioned outside `voyage models download`
voyage/model_registry.py:547: (provision with `voyage models download`, ...
voyage/model_registry.py:1531: Backs the `models download causvid` CLI target
voyage/model_registry.py:1550: `models download ltx25` CLI target
voyage/model_registry.py:1569: `models download ltx23` CLI target
voyage/model_registry.py:1585: `models download film` CLI target
voyage/model_registry.py:1600: `models download rife` CLI target
voyage/model_registry.py:1614: Backs the `models download realesrgan-anime` CLI target
voyage/workers/video_ltx25.py:771: run `voyage models download` first
voyage/workers/video_ltx23.py:686: run `voyage models download` first
voyage/workers/video_ltxv.py:362: run `voyage models download` first
voyage/workers/video_causvid.py:511,514: run `voyage models download` first (x2)
voyage/workers/augment_worker.py:1120: re-provision via `voyage models download`
voyage/audio/mmaudio_sfx.py:219: run `voyage models download sfx-mmaudio`
voyage/audio/acestep.py:8: populated by `voyage models download audio-acestep`
voyage/llama_server.py:238: provision with `voyage models download director-qwen35-gguf`
voyage/doctor.py:12: behind `voyage models verify`
voyage/registry_records.py:659: `voyage models download rife/film/realesrgan-anime`
docs/BACKENDS.md:77,97,125,139,148: Needs `models download ltxv-2b|causvid|ltx25|ltx23|inspector-qwen35`
docs/MODELS.md:10,21,36,48,58,70,78,90,104,116,128,141,153: every header is (`models download <stack>`)
docs/SFX.md:46,158: (`models download sfx-mmaudio` ...)
tests/test_causvid_worker.py:663: match=r"voyage models download" (pins the dead string)
```

### Repro

Parser list → no `models`; then `./scripts/run.sh generate vdemo` on a box missing weights → stderr says `run voyage models download ...` → running it → `voyage: error: argument {configure,generate}: invalid choice: 'models'`.

### Source refs

As above; `Voyage/DESIGN.md:8525-8528` (two-verb deletion); in-tree replacement is `configure [--no-download]` ensure-path (`voyage/models_ensure.py`, `model_registry.download_model`).

### Online sources

None (in-tree two-verb deletion note is the anchor).

### Fix candidates (296's, complementing 218's)

1. Reword all runtime strings to `configure [--no-download]` / `models_ensure.ensure_models` (single helper for the sentence).
2. Reword `BACKENDS.md`/`MODELS.md`/`SFX.md` headers to `configure`-time ensure (`download_model(models_dir, "<spec>")` programmatic entry).
3. Re-pin `test_causvid_worker.py:663` to the new string.

### Log

- 2026-10-07: filed from read-only pass-2 TUI-remnant sweep; no code touched; consolidated into 218 the same day.
