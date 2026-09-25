# 079 — `--segments 0`/negative silently a no-op with exit 0

- Status: open
- Severity: low (zero work reported as success)
- Area: CLI — `voyage/cli.py:1131-1136` (`run`), `:1301` (`soak`),
  `run_segments` (`while count is None or len(committed) < count`)
- Rank rationale: pass-2 CLI finding; segment-count validation previously
  uncovered.

## Technical description

Argparse accepts `0`/`-5`; `run_segments(0)` commits nothing and `cmd_run`
prints `run finished · 0 segment(s) committed`, return 0. Same for
`soak --segments -1`.

## Why this is an issue

Zero work is reported as success with exit 0: a `--segments 0` typo in an
automated pipeline silently produces an empty run that downstream
validate/finalize treat as trivially fine. Input errors should fail loudly at
the boundary (stderr + exit 2), not masquerade as a finished voyage.

## Evidence

Argparse probe (re-run 2026-09-25, stdlib only):

```
$ python3 -c "...add_argument('--segments', type=int, default=None); parse_args(['--segments', '0']); parse_args(['--segments', '-5'])..."
Namespace(segments=0)
Namespace(segments=-5)
```

Both parse fine; `run_segments` (`supervisor.py:390-412`,
`while count is None or len(committed) < count`) commits nothing and `cmd_run`
(`cli.py:345-350`) prints the success line with return 0.

## Reproduction

`voyage run --run <dir> --segments 0` → exit 0, zero work, success message.

## Source references

- Files/lines above.

## Resolution candidates

Reject `--segments <= 0` in `cmd_run`/`cmd_soak` (stderr + return 2); same for
`benchmark --segments/--warmup/--measured` (also unvalidated ints,
`cli.py:917-918,943` — pairs with 060's worker-side crash).

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`cli.py:1129-1136` run `--segments`, `:1301-1305` soak
  `--segments`, `supervisor.py:390-412` `run_segments`, `cli.py:345-350`
  success line — all match); re-ran argparse probe (0/-5 parse fine).
- Open: implement + tests (CLI-side and worker-side together).
