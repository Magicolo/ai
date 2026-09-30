# 055 — `inspect metrics` is rotation-blind (only history reader that ignores `iter_metric_files`)

**Severity:** MEDIUM

**File:line:** `voyage/cli.py:1632-1644` (branch `args.inspect_target == "metrics"`); contrast rotation-aware readers `voyage/cli.py:616-666` (`_read_all_metric_events`, `_last_commit_stages`), `voyage/scoreboard.py:47-74` (`_stages_by_segment`), `voyage/logrotate.py:107-135` (`iter_metric_files`)

**Overlaps with:** 029 (inspect/metrics rotation/fps divergence — same reader family; not a duplicate) / 061 (status omits health — sibling observability gap; not a duplicate)
- **Description:** Every history reader was converted to `iter_metric_files()` (issue 049) — except `inspect metrics`, which opens `run_dir/logs/metrics.jsonl` directly, counts only live lines, and prints the last 5 of the live file. After a daily rotation the command under-reports the event count and hides all pre-rotation history, while `inspect scoreboard` on the same dir sees both.
- **Rationale:** `docs/OPERATIONS.md:220-235` promises history readers span the rotation (`iter_metric_files`, live file last, so scoreboard / status last-commit / soak / benchmark keep pre-rotation history) and lists `inspect metrics` two lines above as a first-class monitor. The omission makes the command lie the day after a rotation (count drops, recent history appears to vanish). The structure-sweep (ses_f10013fe0, item 10a) independently reports the same gap plus the divergent `fps=0` fallbacks merged into 061 — deduplicated here.
- **Evidence (re-verified live 2026-09-30):**
```python
# voyage/cli.py:1632-1644 (live; sweep cited 1579-1591, drifted by concurrent growth)
if args.inspect_target == "metrics":
    metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    if not metrics_path.exists():
        print("no metrics yet")
        return 0
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    print(f"{len(lines)} metric events")
    for line in lines[-5:]:
```
`grep -rn iter_metric_files voyage/` shows `cli.py` (in `_read_all_metric_events`, `_last_commit_stages`), `scoreboard.py:55`, `logrotate.py` def — zero references in the `inspect_target == "metrics"` branch. (Sweep cited `cli.py:1579-1591`; live tree reads `1632-1644` after concurrent growth; same branch.)
- **Repro:**
  1. `mkdir -p /tmp/vdemo/logs && echo '{"event":"segment_committed","segment_id":"000000"}' > /tmp/vdemo/logs/metrics-2026-09-28.jsonl && echo '{"event":"segment_committed","segment_id":"000001"}' > /tmp/vdemo/logs/metrics.jsonl`
  2. `./scripts/run.sh inspect metrics --run /tmp/vdemo` → reports `1 metric events`, segment `000000` invisible. `inspect scoreboard` on the same dir sees both.
- **Fix candidates:** Reuse `_read_all_metric_events(run_dir)` in the metrics branch; print `N metric events across K files`; keep last-5 semantics over the merged list. Add a regression test with one rotated sibling + live file.
- **Refs:** `docs/OPERATIONS.md:220-235`; `voyage/logrotate.py:107-135`; JSONL rotation guidance — rotated siblings must be queryable as one stream (dated, sortable names; glob in order).
