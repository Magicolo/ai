# 069 — `scripts/run.sh` silently drops the `--run=<path>` form (wrong image selected)

- Status: open
- Severity: medium (CUDA run silently uses `voyage:latest`)
- Area: scripts — `Voyage/scripts/run.sh:21-36`
- Rank rationale: pass-2 finding; asymmetric with `--backend=` which works —
  distinct from 037's TOML-sniff bug.

## Technical description

The sniff loop handles `--backend=*` (line 29) but for `--run` only the
separate-arg form (`elif [[ "$arg" == --backend || "$arg" == --run ]]`, line 31).
Verified live by orchestrator 2026-09-25 (`sed -n '21,36p'`):

```bash
for arg in "$@"; do
  if [ "$prev_arg" = "--backend" ] || [ "$prev_arg" = "--run" ]; then
    ...
  elif [[ "$arg" == --backend=* ]]; then
    requested_backend="${arg#--backend=}"
  elif [[ "$arg" == --backend || "$arg" == --run ]]; then
    prev_arg="$arg"
  else
    prev_arg=""
  fi
done
```

`--run=/tmp/vdemo` falls to `else prev_arg=""`, so `run_dir` stays unset, the
`voyage.toml` backend sniff (:40-44) is skipped, and a CUDA run silently uses
`voyage:latest` instead of `voyage-video:latest`. Sweep simulation: `run
--run=/tmp/vdemo --segments 2` → `run_dir=UNSET backend=UNSET`.

## Why this is an issue

A common, documented CLI form silently selects the wrong image: the CUDA run
proceeds on the slim `voyage:latest` and fails late and confusingly instead of
at argument parse. The asymmetry with `--backend=` (which works in both forms)
makes this a trap — users who learn the `=` form once will hit it on `--run`
with no warning.

## Evidence

`sed` output above.

## Reproduction

`bash -c` the loop with `set -- run --run=/tmp/vdemo --segments 2`; or prefix a
CUDA `voyage.toml` run dir and compare image selection between `--run DIR` and
`--run=DIR`.

## Source references

- `Voyage/scripts/run.sh:21-44`.

## Resolution candidates

Add `elif [[ "$arg" == --run=* ]]; then run_dir="${arg#--run=}"` mirroring line
29-30; add a shell-level test or a comment-level contract. Consider parsing with
python (see 037) to kill the whole class.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests/scripts sweep; loop re-verified live.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`run.sh:21-36` sniff loop, `:40-44` toml sniff — match;
  only `--backend=*` has an `=` branch, `--run` does not).
- Open: implement + shell test.
