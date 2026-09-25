# 049 — `status` drifted from §59; scoreboard/`status`/soak blind after rotation

- Status: open
- Severity: medium (stale contract + silent history loss the day after a run ends)
- Area: observability — `cli.py:396-466`, `scoreboard.py:37-60`,
  `logrotate.py:35-88`
- Rank rationale: `logrotate.py:5` claims "`metrics.jsonl` stays the reader
  contract" — true for the path, false for history completeness.

## Technical description

(a) Status drift (`voyage/cli.py:396-466` vs `DESIGN.md:3047-3085` §59 +
`docs/OPERATIONS.md:196-202`): spec shows `Render / Timeline / Current block /
GPU: 16.2GB / Novelty: accepted / Workers: READY / video/audio backends`. Code
prints `Render: WxH @ fps / Timeline: Ns (N frames) / Blocks per segment / GPU:
first nvidia-smi line or unavailable / Workers: idle (<backend> backend; workers
run during run) / Stages (last commit …) / Free: X GiB`. Gaps: `Novelty`,
`Current block`, GPU-mem breakdown never shown; `Workers: idle` is always idle
(informationally empty, contradicts §59 `READY`); OPERATIONS promises "slowest
stages" but code prints last-commit stages verbatim; `Uptime` is `now -
manifest.created_at` (wall since init — long-paused runs report days of "uptime").

(b) Rotation blindness: `append_line()` rotates `metrics.jsonl` daily + prunes
>30d (correct), but every reader opens only the live file:
`scoreboard._stages_by_segment()` → `run_dir/logs/metrics.jsonl` only;
`cli._last_commit_stages()` returns `None` "when rolled past the last commit —
status must degrade, never fail" (silently drops the Stages section the day after
a run ends); `cmd_soak` / `cmd_benchmark end-to-end` average only live events —
post-rotation soak reports silently exclude older segments.

## Why this is an issue

`status` is the operator's primary window into a run, and the scoreboard is
the calibration record — when they silently diverge from the spec (§59) or
go blind the day after rotation, users make wrong calls (believing workers
are broken because "idle", believing a run produced no stage data because
the section vanished) and multi-day performance trends become unmeasurable.
Blast radius: every run older than one day for the rotation half; every
`status` invocation for the drift half. The rotation fix is small (glob
dated siblings in three readers); the drift fix is a spec-or-code alignment
decision.

## Evidence

```
$ sed -n '441,448p' Voyage/voyage/cli.py
    print("Workers")
    for name in ("video", "audio", "director"):
        ...
        print(f"  {name}: idle ({backend} backend; workers run during `voyage run`)")
$ sed -n '206,207p' Voyage/docs/OPERATIONS.md
- `status` — §59 sections: uptime, video backend/render spec, world,
  audio buffer, workers, slowest stages, free storage.
$ sed -n '372,376p' Voyage/voyage/cli.py
def _last_commit_stages(run_dir: Path) -> tuple[str, dict[str, object]] | None:
...
    # (returns None "when rolled past the last commit —
    #  status must degrade, never fail")
```

`sed -n '413,456p' Voyage/voyage/cli.py`; `python3 -m voyage.cli status --help`;
rotate once (`rotate_log()` renames to `metrics-YYYY-MM-DD.jsonl`), then
`inspect scoreboard`/`status` lose all stages.

## Reproduction

Commands above; rotate-then-read sequence.

## Source references

- `voyage/cli.py:372-394,396-466,997-1025`; `voyage/scoreboard.py:37-60`;
  `voyage/logrotate.py:5,35-88`; `DESIGN.md:3047-3085` (§59, spec example
  with `Novelty: accepted` at :3068 and `Current block` at :3061);
  `docs/OPERATIONS.md:206-207` (status section promise incl. slowest stages).

## Resolution candidates

1. Either update §59/OPERATIONS to as-built or add novelty + slowest-stage
   (`max(stages)` + label) + rename `Uptime` to `Age` with a `RUNNING`-gated elapsed.
2. Glob `metrics*.jsonl` sorted by day in the three readers (small, keeps
   contract), or document "post-rotation history requires concatenating siblings"
   + add `inspect metrics --all`.
3. Extend 003's drift metric into scoreboard/stages so alignment trends survive
   rotation too.

## Investigation / progress / resolution log

- 2026-09-25: found by docs sweep.
- 2026-09-25 (repair): re-verified refs live — `cli.py:372` (`_last_commit_stages`),
  `:396` (`cmd_status`), `:411` (`Uptime`), `:441-448` (`Workers … idle`);
  `scoreboard.py:37-60` (live-file-only reader); `logrotate.py:35-88` all
  current. Fixed stale OPERATIONS ref `:196-202` → `:206-207` (the status
  promise moved). Added `## Why this is an issue` + real Evidence output.
- Open: fix readers + update spec/docs to match.
