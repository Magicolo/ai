# 037 — Script bugs: `qualify.sh` literal `'$run_dir'`; `run.sh` greps TOML; `/tmp:/tmp` over-share

- Status: open
- Severity: medium (wrong-image selection; broken qual summaries; host-tmp exposure)
- Area: scripts — `Voyage/scripts/qualify.sh:15-33`, `Voyage/scripts/run.sh:40-44,70-71`
- Rank rationale: `run.sh` backend sniff silently picks the slim image for a CUDA
  run; `qualify.sh` summary step emits junk with exit 0.

## Technical description

```bash
# qualify.sh:31-33 — single quotes inside double-quoted bash -c: no expansion
python -c "import json; ... summarize_run('$run_dir') ..."
# python receives summarize_run('$run_dir') — FileNotFoundError / empty summary, exit 0 with junk
# run.sh:42-43 — backend sniff:
requested_backend="$(grep -E '^backend *= *"' "$run_dir/voyage.toml" | head -n1 | sed -E 's/.*"(.*)".*/\1/')"
```

Both verified live by orchestrator 2026-09-25 (`sed -n` output):

```
./scripts/run.sh benchmark video --run "$run_dir" --warmup 1 --measured 3
...
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/app" voyage:latest \
  python -c "import json; from tests.test_qualification import summarize_run; \
print(json.dumps(summarize_run('$run_dir'), indent=2))"
```

and

```
requested_backend="$(grep -E '^backend *= *"' "$run_dir/voyage.toml" \
    | head -n 1 | sed -E 's/.*"(.*)".*/\1/')"
```

`grep '^backend'` matches any top-level `backend=` (video/audio/director sections
all have one: `config.py:25,69,144,296,309,324`); indented `[video]` keys are
missed or the wrong one wins → wrong image (`voyage:latest` vs
`voyage-video:latest`) / missing `--gpus all`. Shellcheck flags SC2016 on the
`python -c` line. `run.sh:71` mounts `-v /tmp:/tmp` (whole host tmp shared;
container temp drivers can clobber host files) and `mkdir -p "$models"` with
unvalidated `VOYAGE_MODELS`. (`set -euo pipefail` everywhere, quoted
`"$@"`/`"$PWD"`/`"$models"`/`"$run_dir"`, `bash -n` clean — verified; the basics
are right.)

## Why this is an issue

The backend sniff silently picks the slim image for a CUDA run, so a one-character TOML layout difference (indented key under `[video]`) produces a confusing downstream failure instead of a clear config error — the user debugs the container, not the grep. The qualify summary emitting junk with exit 0 is worse: green CI on garbage numbers that then get committed as benchmark evidence. And sharing all of host `/tmp` with the container turns every temp driver into a potential clobber of unrelated host files. Scripts are the front door; bugs here tax every user.

## Evidence

`sed` outputs above.

## Reproduction

1. `./scripts/qualify.sh "./output/it's-broken"` (quote/space in path breaks or
   injects).
2. `printf '[video]\nbackend = "ltxv"\n' > run/voyage.toml` → `run.sh` misses it
   (anchors `^backend`).

## Source references

- `Voyage/scripts/qualify.sh:31-33`; `Voyage/scripts/run.sh:40-44,70-71`.

## Resolution candidates

1. Pass run dir via env (`RUN_DIR="$run_dir" python -c
   '...summarize_run(os.environ["RUN_DIR"])...'`) or argv; better: a real script
   instead of `python -c` interpolation.
2. Parse TOML with python (`tomllib`) instead of grep/sed (or match
   `^\s*backend\s*=` as a stopgap).
3. Mount a scoped tmp (`-v /tmp/voyage:/tmp/voyage`); validate `VOYAGE_MODELS`.

## Investigation / progress / resolution log

- 2026-09-25: found by standards + supply sweeps (convergent); both snippets
  re-verified live.
- 2026-09-25: repair pass — added `## Why this is an issue`; `run.sh:40-44`
  grep-sniff + `qualify.sh:31-33` literal still current live (note: `run.sh` now
  also honors an explicit `--backend` flag first — the grep fallback path is
  where the bug lives).
- Open: implement + shellcheck in gates.
- 2026-09-25 (fix): relevance re-verified live — `qualify.sh:31-33` literal
  `'$run_dir'` and `run.sh:48-49` grep/sed sniff both still present, so the
  issue was live. Fixed in `Voyage/scripts/qualify.sh:31-35` (run dir now
  travels via `-e RUN_DIR="$run_dir"` + single-quoted `python -c` reading
  `os.environ["RUN_DIR"]` — no shell interpolation of the path) and
  `Voyage/scripts/run.sh:51-58` (grep/sed replaced with a stdlib `tomllib`
  sniff reading only the `[video]` backend; unparseable/missing key falls
  back to empty via `|| true`, never a launcher failure). Verified:
  `bash -n` clean on all 7 scripts; `VOYAGE_DRY_RUN=1` dry-runs — a trap
  TOML with `[audio] backend="fake"` first selects `voyage-video:latest`
  (old grep returned `fake` → wrong slim image), broken TOML still exits 0
  with slim defaults, and a hostile `RUN_DIR` (`it's-broken";touch PWNED;…`)
  passes through byte-exact with no `PWNED` file created. shellcheck
  unavailable (absent on host, not in the voyage image — noted, not run).
  NOT done (left open): candidate 3 — scoped `/tmp` mount (`run.sh:98`
  still `-v /tmp:/tmp`) and `VOYAGE_MODELS` validation; changing the tmp
  mount risks breaking container temp drivers, needs its own task.

## Resolution

- Status: fixed (partial — candidates 1+2 done; candidate 3 `/tmp` scoping +
  `VOYAGE_MODELS` validation deferred, see log above).
- Files: `Voyage/scripts/qualify.sh:31-35`, `Voyage/scripts/run.sh:51-58`.
