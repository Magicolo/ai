# 057 — `cmd_init` skips `_run_dir_arg` normalization (relative-path doubling only half-fixed)

- Status: open (known-open, `TASK.md:1698-1701` §30.2)
- Severity: medium (correctness — documented `output/<run>/output/<run>/…`
  doubling re-triggerable)
- Area: CLI paths — `voyage/cli.py:60-66,101-102,791-794`, `rpc.py:99-112`
- Rank rationale: `generate` was fixed; `init`+`run` directly still isn't, and
  workers spawn with `cwd=run_dir`.

## Technical description

Workers spawn with `cwd=run_dir` (`rpc.py:105`, supervisor passes `run_dir`
through). `_run_dir_arg` (`Path(value).resolve()`) normalizes
`run/validate/finalize/...`, and `cmd_generate` resolves once (`:794`) — but
`cmd_init` still does `run_dir = Path(args.output)` raw (`sed -n '101,115p'` vs
`sed -n '60,66p'`). A relative `--output` + later relative `--run` re-triggers
the documented doubling. Related stale item from the same TASK note: `cli.py`
25/24 duration math (verify against current `_frames_per_segment` before closing).

## Why this is an issue

The relative-path doubling (`output/<run>/output/<run>/…`) already caused a
live circuit-breaker failure once, and the fix is currently half-applied:
`generate` resolves once, but `init` + every direct `run` invocation still
goes through raw paths. Any user following the documented relative-path
workflow re-triggers missing-segment failures that look like backend bugs,
wasting GPU time on runs that can never commit. Worse, TASK.md now claims
all subcommands are immune — so nobody is looking. Blast radius: every
relative-path `init`/`run`; fix is one line plus a regression test.

## Evidence

```
$ sed -n '60,66p;101,102p' Voyage/voyage/cli.py
def _run_dir_arg(value: str) -> Path:
    ...
    return Path(value).resolve()
def cmd_init(args: argparse.Namespace) -> int:
    run_dir = Path(args.output)          # raw — NOT _run_dir_arg
$ rg -n "_run_dir_arg" Voyage/voyage/cli.py | head -4
60:def _run_dir_arg(value: str) -> Path:
284:    run_dir = _run_dir_arg(args.run)   # cmd_run
397:    run_dir = _run_dir_arg(args.run)   # cmd_status
609:    run_dir = _run_dir_arg(args.run)   # cmd_validate
```

## Reproduction

`init` with a relative dir, then `run` with a relative `--run`; assert
`voyage.toml` models/payload paths (expect doubling).

## Source references

- `voyage/cli.py:60-66,101-102,791-794`; `voyage/rpc.py:99-112`;
  `Voyage/TASK.md:1698-1711` (§30.2 — note the "Resolved" paragraph claims
  `_run_dir_arg` makes "all subcommands immune", which `cmd_init:101` still
  contradicts).

## Resolution candidates

`cmd_init` → `run_dir = _run_dir_arg(args.output)` (one line); regression test:
init with relative dir, assert absolute paths. Re-check the 25/24 duration-math
note against present code; close or file separately.

## Investigation / progress / resolution log

- 2026-09-25: confirmed present by supply-chain sweep against TASK §30.2.
- 2026-09-25 (repair): re-verified live — `cmd_init:101` still
  `Path(args.output)` raw while `cmd_run:284`/`cmd_status:397`/
  `cmd_validate:609` all use `_run_dir_arg` (Evidence pasted). TASK §30.2
  "Resolved" text now over-claims immunity — flagged, not edited (TASK is
  outside scope). Added `## Why this is an issue`.
- Open: one-line fix + regression test.
