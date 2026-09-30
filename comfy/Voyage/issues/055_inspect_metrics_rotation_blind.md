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

## Progress log (2026-09-30, resolution track — overlap check FIRST per brief)

- Compared 055 against 029 claim by claim on the live tree: same branch
  (`args.inspect_target == "metrics"`, live `voyage/cli.py:1700-1712`
  after concurrent growth), same direct-`open` of the live file, same
  contrast helpers (`_read_all_metric_events`, `iter_metric_files`),
  same repro shape (rotated sibling + live file; sibling reader sees
  both), same fix (route through `_read_all_metric_events`, report the
  file count). 029's part (b) (fps divergence) is extra scope 055 never
  claims; 055 has no leg outside 029's part (a) — no different reader,
  no different file, no different failure mode.
- Live evidence for the subsumption: `grep -rn iter_metric_files
  voyage/` shows `cli.py` (`_read_all_metric_events`,
  `_last_commit_stages`), `scoreboard.py`, `logrotate.py` — zero
  references in the metrics branch pre-fix, exactly as both issues
  state; post-fix the branch calls `_read_all_metric_events` and the
  regression test `test_inspect_metrics_spans_rotated_siblings`
  (in `tests/test_inspect_metrics_fps_029.py`) replays 055's own repro
  (segment `000000` rotated + `000001` live → `2 metric events across
  2 files`, both listed).
- No code change in 055's name: fixed once under 029 (single fix, both
  files reference it).

## Resolution: SUPERSEDED by 029

- Verdict: premise fully subsumed — 055's rotation-blindness claim is
  029 part (a) verbatim (same branch, same helper gap, same repro,
  same fix). Marked SUPERSEDED, not fixed-twice; the fix + regression
  tests live under 029's resolution above.
- Residuals: none — close 055 when 029 lands (or keep as a pointer;
  do not implement separately).
