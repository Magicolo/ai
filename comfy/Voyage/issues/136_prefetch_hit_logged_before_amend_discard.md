# 136 — Prefetch `hit` is logged before the amendments check discards it: inspector-on hit-rate is inflated

- **Severity:** LOW-MEDIUM (observability — the prefetch hit-rate the soak report uses to judge the feature counts discarded proposals as hits whenever the inspector amends)
- **File:line:** `Voyage/voyage/supervisor.py:774-798` (`_take_prefetch` logs `director_prefetch_hit` at `:798` and returns the raw proposal) → `Voyage/voyage/supervisor.py:1478-1481` (`_propose_segment`: `prefetched_raw = self._take_prefetch(...)` then `if amendments: prefetched_raw = None`)
- **Area:** supervisor prefetch/invalidation edge (below both previous windows; 030 covered shutdown hang, 033 covers hit-rate aggregation shape, 107 covers the pipeline-docs omission — none covers the metric miscount)

## Description

`_take_prefetch` decides hit vs miss at consumption time and logs it immediately:

```python
# supervisor.py:791-792
self._log_metric({"event": "director_prefetch_hit", "segment_id": segment_id})
return raw
```

The caller then applies the inspector rule — prefetched proposals postdate fresh inspect amendments (computed during the previous segment's render window), so they are unusable when amendments exist:

```python
# supervisor.py:1470-1473
prefetched_raw = self._take_prefetch(number, segment_id)
if amendments:
    prefetched_raw = None
prefetch_hit = prefetched_raw is not None
```

The local `prefetch_hit` boolean is correct (False when discarded), but the *emitted event* already says `hit`. `summarize_prefetch_outcome` (`supervisor.py:179-196`) aggregates raw `director_prefetch_hit/miss` events into the `prefetch_hit_rate` the soak report consumes — so every inspector-amended segment with a ready prefetch counts as a hit for a proposal that was thrown away and never entered the accept loop (the loop then calls the worker live at `:922-930`). With the inspector on, the reported hit-rate measures "prefetch was ready", not "prefetch was used"; the two diverge exactly when the inspector is doing work.

## Rationale

Hit-rate is the decision metric for prefetch restructuring (033's own comment: "the hit rate the soak report needs before any prefetch restructuring is considered — measure first"). An inflated numerator green-lights keeping (or expanding) a speculative window whose proposals are systematically discarded on amended segments. The fix is a recognized category: prefetch telemetry distinguishes *cancelled/invalidated* work from hits — e.g. cashew memory-provider metrics carry `record_prefetch_cancelled` ("prefetch work invalidated before its next expensive stage") as a third outcome beside hit/miss, precisely so invalidation doesn't masquerade as usefulness.

## Evidence (verified live 2026-09-30, host reads)

- `sed -n '774,798p' voyage/supervisor.py` — the hit log at `:798` fires before the function returns; there is no amendments parameter, no post-return hook, no way for the caller to retract it.
- `sed -n '1478,1481p' voyage/supervisor.py` — the discard (`if amendments: prefetched_raw = None`) happens strictly after the event is emitted; `prefetch_hit` (local) and the logged event disagree on exactly these segments.
- `summarize_prefetch_outcome` (`:179-196`) counts `event == "director_prefetch_hit"` with no amendments awareness — the inflation flows straight into the reported rate.
- Overlap check: 030 is the shutdown hang (`_call`/`stop_workers`), 033 is the aggregation helper shape (`None` on zero events), 107 is the ARCHITECTURE pipeline-docs omission (names the invalidation edge as missing *documentation* — this file is the *metric* it corrupts). None names the hit-before-discard ordering.

## Repro

Static (deterministic): enable `[experimental] visual_inspector`, run two segments where segment N+1's inspect returns non-empty amendments while the prefetch for N+1 is ready → `logs/metrics.jsonl` shows `director_prefetch_hit` for N+1, while the commit log shows the accept loop calling `decide` live (first candidate *not* from prefetch) and `prefetch_hit: False` in the plan dict. `summarize_prefetch_outcome` over that window reports a hit that was never used.

## Fix candidates

1. Emit the hit only when the proposal survives: move the amendments check before consumption (peek at amendments first, or pass an `invalidated: bool` into `_take_prefetch` so it logs `director_prefetch_invalidated` instead of `hit` on the discard path).
2. Add the third event (`director_prefetch_invalidated` with `segment_id`) and exclude it from the hit-rate numerator (report `hit / (hit + miss)`, invalidated separately) — mirrors the `prefetch_cancelled` precedent; keeps the signal (prefetch was ready *and* wasted) instead of deleting it.
3. Test: prefetch-ready + amendments → assert no `hit` event and one `invalidated` event; prefetch-ready + no amendments → one `hit`; `summarize_prefetch_outcome` pins all three shapes.

## Refs

- `Voyage/voyage/supervisor.py:179-196,709-792,913-930,1467-1473`; DESIGN §44 (inspector), §20 (prefetch).
- Adjacent, not overlapping: 030 (prefetch shutdown hang — different method, different failure); 033-adjacent `summarize_prefetch_outcome` shape (aggregation — this file is the *emission* ordering that feeds it); 107 (pipeline-docs omission — documentation, not the metric).
- Web rationale: hermes-cashew `metrics.py` `record_prefetch_cancelled` ("prefetch work invalidated before its next expensive stage") — the invalidated-as-third-outcome precedent.
