# 253 — `scoreboard.py` documents a deleted CLI target; `partial_segment_ids` has zero production callers

Severity: MEDIUM (track E-08).

## Technical description

`voyage/scoreboard.py:13` says "The CLI `inspect scoreboard` target renders the compact
table"; `partial_segment_ids:102-118` says "the CLI renders this list as a trailing
`partial: […]` line". `inspect` was deleted with the two-verb migration
(`voyage/cli.py:6-11`). Grep shows `partial_segment_ids` referenced only in tests
(`test_scoreboard.py`, `test_observability_rank2.py:228`); `scoreboard_rows` likewise
test-only.

## Rationale

The per-segment table (frames, stage seconds incl. rotated siblings, metric deltas,
`baseline_segment_id`, `errors` cell) is the fast-iteration view (§59). After deletion
the only documented run-inspection tool is `boundary_metrics --prompts`, which covers
continuity + prompts but not stages/takes/deltas. Operators lose the table with no pointer
to its replacement.

## Live evidence

```
$ grep -rn "partial_segment_ids\|scoreboard_rows" voyage/ --include="*.py" | grep -v tests
voyage/scoreboard.py:def partial_segment_ids … def scoreboard_rows …  # definitions only
$ grep -rn "inspect scoreboard" voyage/ docs/ | head
voyage/scoreboard.py:13: The CLI `inspect scoreboard` target …
```

Repro: `./scripts/run.sh inspect scoreboard --run output/x` → `invalid choice: 'inspect'
(exit 2)`; `python -m voyage.boundary_metrics --run …` covers a different surface.

## Source refs

`voyage/scoreboard.py:1-21,67-118,121-195`; `voyage/cli.py:1-11`;
`voyage/logrotate.py:196-224` (rotation-aware reader the scoreboard depends on).

## Online sources

- None (in-tree §59 fast-iteration view contract).

## Fix candidates

- Either re-expose a read-only inspection entry (`python -m voyage.scoreboard --run …`
  mirroring `boundary_metrics.main`, sharing `iter_metric_files`), or officially retire
  `scoreboard.py` to test-only with a `boundary_metrics` pointer and delete the CLI
  sentences.

## Log

- 2026-10-07: filed from read-only Track E sweep; no code touched.

## Evaluation
- Claim CONFIRMED live 2026-10-07: `voyage/cli.py` exposes only
  `configure` + `generate` (two-verb migration), while `scoreboard.py:13`
  and `partial_segment_ids` docstring still reference the deleted
  `inspect scoreboard` verb; `partial_segment_ids`/`scoreboard_rows` have no
  production callers (definitions + tests only). The table (stages incl.
  rotated siblings, deltas, partial trailer) has no replacement —
  `boundary_metrics --prompts` covers continuity + prompts only.

## Progress log
- 2026-10-07 (Group J): chose re-exposure over retirement (preserves the
  §59 operator view; retirement would delete value to save ~70 lines).
  Added a read-only `python -m voyage.scoreboard --run ...` entry mirroring
  `boundary_metrics.main` (stdlib argparse, `sys.stdout.write` — no
  `print`, per the T201 contract — exit 0/2, `--json` for raw rows,
  `partial: [...]` trailer; shares `iter_metric_files` via the existing
  `_stages_by_segment` path) plus `format_scoreboard_table` and the
  `__main__` runner; both stale CLI sentences reworded to the new entry.
  Tests in `tests/test_scoreboard.py` use the Supervisor-free
  `_write_committed_segment` helper (text + JSON + read-only pins). Live
  proof in-container: text and JSON render with the partial trailer, exit 0.
  Gates: ruff + format + mypy strict clean, scoped pytest green.

## Resolution (2026-10-07)
- RESOLVED. Files: `Voyage/voyage/scoreboard.py`,
  `Voyage/tests/test_scoreboard.py`. Nothing deliberately left open.
