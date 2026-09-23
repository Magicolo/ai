# OPERATIONS — runbook

```bash
./scripts/run.sh doctor
./scripts/run.sh models verify
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
