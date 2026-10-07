# 271 — Scoreboard masks new-metric appearance as 0.0 delta

Severity: LOW (track C-13b, split from the exit-code twin #242).

## Technical description

Scoreboard deltas use `previous.get(key, current[key])` (`voyage/scoreboard.py:170`): a
metric key appearing for the first time (e.g. a new §43 metric backfilled from some
segment on) renders delta `0.0` — indistinguishable from "unchanged" — and
`baseline_segment_id` still points at the previous segment as if a comparison happened.

## Rationale

Silent information loss at an observability boundary the project otherwise instruments
carefully (per-row `errors` cells, `baseline_segment_id`, distinct exit 2 for misuse).
Small, cheap, and repro'd live.

## Live evidence

```python
# live stdlib demo of the masking:
current = {'motion_energy': 0.5, 'brand_new_metric': 0.9}
previous = {'motion_energy': 0.4}
# per scoreboard.py:169-171 → deltas == {'motion_energy': 0.1, 'brand_new_metric': 0.0}
# brand_new_metric 0.0 reads as "unchanged" — information lost.
```

Repro: snippet above.

## Source refs

`voyage/scoreboard.py:163-171`.

## Online sources

- Scoreboard's own `errors`-cell philosophy (`voyage/scoreboard.py:153-160` records
  unusable cells instead of hiding them).

## Fix candidates

- Emit `None` (or a `"new"` marker) for deltas whose key is absent in the baseline,
  keeping `0.0` for true zero-change — mirrors the existing `errors`-cell explicitness.

## Log

- 2026-10-07: filed from read-only Track C sweep; no code touched.

## Evaluation (2026-10-07)

Claim PARTIALLY STALE on re-verification: the cited `previous.get(key, current[key])`
at `voyage/scoreboard.py:170` no longer exists. Current
`voyage/scoreboard.py:321-334` already emits `None` for a key absent from the
baseline (`baseline = previous.get(key)` → `deltas[key] = None`) and the module
docstring (`voyage/scoreboard.py:35-36`) already documents "New metric keys delta
as null (not 0.0)". The masking defect is therefore already fixed in-tree (prior
batch, untested). Scope downgraded accordingly: no `scoreboard.py` behavior change
needed. What remains: (a) no regression test pinned the None-delta path, and
(b) the first-segment row still deltas `0.0` via `dict.fromkeys` — kept
deliberately (its `baseline_segment_id` is `None`, so unlike the reported case it
is distinguishable from a true comparison; the issue's own fix candidate keeps
`0.0` for non-comparisons of this shape).

## Progress log

- 2026-10-07: added `test_scoreboard_new_metric_key_deltas_none` (+
  `_write_visual_segment` helper) to `tests/test_scoreboard.py`: two committed
  segments, second carrying a new `palette_distance` key → asserts
  `deltas == {"motion_energy": 0.1, "palette_distance": None}` and
  `baseline_segment_id == "000000"`. Scoped pytest: 8 passed. ruff check +
  format --check + mypy strict clean on `voyage/scoreboard.py` +
  `tests/test_scoreboard.py` (in-container `voyage:latest`).

## Resolution (2026-10-07)

RESOLVED (fix pre-existing in-tree; regression test added here). Nothing left open.
