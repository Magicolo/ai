# 216 — Double-rotate archives are invisible to every rotation-aware reader AND never pruned (HIGH)

## Technical description
`rotate_log` collision path (`voyage/logrotate.py:158-162`) and
`_unique_rotated` (`:337-354`) emit
`metrics-<written>-<today>[ -<n>].jsonl`. `iter_metric_files` (`:196-224`),
`_prune_siblings` (`:227-253`), and therefore `read_all_metric_events`,
`last_commit_stages`, `scoreboard._stages_by_segment` accept only the
single-date shape enforced by
`_ROTATED_SUFFIX = ^(?P<stem>.+)-(?P<day>\d{4}-\d{2}-\d{2})$` (`:90`) +
`stem == "metrics"`.

## Rationale
The exact files produced under clock-skew / double-rotate / repeated
same-day size rolls are the files no reader will ever open and no pruner
will ever delete. Scoreboard/stages/soak silently drop segments; disk grows
without bound despite the "30-day prune" contract in
`docs/OPERATIONS.md:121-130`. Silent history loss + disk leak.

## Live evidence
```
$ python3 -c "
import re; pat=re.compile(r'^(?P<stem>.+)-(?P<day>\d{4}-\d{2}-\d{2})\$');
for n in ['metrics-2026-01-01','metrics-2026-01-01-2026-10-07','metrics-2026-01-01-2026-10-07-2']:
    m=pat.match(n); print(n,'->',(m.groupdict() if m else None),'accepted:',bool(m and m.group('stem')=='metrics'))"
metrics-2026-01-01 -> {'stem': 'metrics', 'day': '2026-01-01'} accepted: True
metrics-2026-01-01-2026-10-07 -> {'stem': 'metrics-2026-01-01', 'day': '2026-10-07'} accepted: False
metrics-2026-01-01-2026-10-07-2 -> None accepted: False
```
Live probe with a planted double-rotate sibling: `iter_metric_files`
returned `['metrics-2026-01-01.jsonl','metrics.jsonl']` — the
`…-2026-01-01-2026-01-02.jsonl` decoy was absent.

## Repro
Create `logs/metrics-2026-01-01-2026-10-07.jsonl` with a
`segment_committed` event → `scoreboard_rows` / `last_commit_stages` never
see it; wait 31+ days → file still present (`_prune_siblings` glob
`metrics-*.jsonl` matches the path but the stem check `continue`s).

## Source refs
- `Voyage/voyage/logrotate.py:90,118-119,141-169,227-253,337-354,196-224,445-496`
- `Voyage/voyage/scoreboard.py:67-99`
- `Voyage/docs/OPERATIONS.md:121-130`

## Online sources
- logrotate `copytruncate` race literature (copy-then-truncate gap
  drops/duplicates lines — dev.to 2026-08-12; Grafana Promtail docs
  recommend rename-and-create over copytruncate; `man logrotate`: "a very
  small time slice … some logging data might be lost") — same bug class:
  rotation produces files the tailer never reads.

## Fix candidates
1. Widen the matcher to `metrics(-\d{4}-\d{2}-\d{2})+\.jsonl` (+ optional
   `-\d+`), sort by full name, keep live-last.
2. Or stop emitting compound names: `_unique_rotated` counter-only
   (`metrics-<day>-<n>.jsonl`), which the widened matcher also covers.
3. Regression test: plant single + double + counter sibling, assert all
   three surface oldest-first and prune beyond `keep_days`.

## Log
- Track E sweep, 2026-10-07. Read-only; nothing fixed.

## Evaluation
- Claim CONFIRMED live 2026-10-07: the `_ROTATED_SUFFIX` regex test from the
  issue reproduces exactly, and a host probe (`PYTHONPATH=Voyage`,
  stdlib-only) planting single + double (`metrics-2026-01-01-2026-01-02`)
  + counter siblings showed `iter_metric_files` returning only
  `[single, live]` — the compound files were invisible, and `_prune_siblings`
  skips them by the same stem check. Single-date behavior was the only
  working shape.

## Progress log
- 2026-10-07 (Group J): fixed in `voyage/logrotate.py` — new
  `_ROTATED_MULTI_SUFFIX` (`base + (-YYYY-MM-DD)+ + optional -N`, lazy base)
  with `_rotated_content_day(stem, base)` returning the FIRST date group
  (content age; every group calendar-validated), now used by both
  `iter_metric_files` (sort still by full name, live still last) and
  `_prune_siblings`; `_unique_rotated` emits counter-only
  (`<stem>-<day>-<N>`) and `rotate_log`'s collision path routes through it
  (also gaining the missing counter loop — a third collision used to
  overwrite). Old compound names stay readable; single-date emission and
  matching are unchanged. Regression tests added to
  `tests/test_observability.py` (compound surface + order + prune by content
  day + counter-only emission). Gates: ruff check + format clean, mypy
  strict clean, scoped pytest 67 passed (observability + ledger_rotation +
  scoreboard + boundary_metrics + qualification).

## Resolution (2026-10-07)
- RESOLVED. Files: `Voyage/voyage/logrotate.py`,
  `Voyage/tests/test_observability.py`. Name-order note: `-` (0x2D) sorts
  before `.` (0x2E), so compounds precede their same-day single — all shapes
  surface, order among same-content-day siblings is deterministic. Nothing
  deliberately left open.
