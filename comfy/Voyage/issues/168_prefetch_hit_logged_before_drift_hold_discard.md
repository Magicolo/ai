# 168 — Prefetch `hit` is logged before the drift-hold discard: `--drift-every-n > 1` inflates hit-rate on held segments

- Severity: LOW-MEDIUM (observability — same class as 136, different discard site; the prefetch hit-rate counts proposals that never entered the accept loop)
- Area: supervisor prefetch/invalidation edge — drift-cadence path (below both previous windows; 030 covered shutdown hang, 033 the aggregation shape, 136 the amendments discard — none covers the drift-hold discard)
- Files (as-read 2026-09-30):
  - `voyage/supervisor.py:774-799` (`_take_prefetch` logs `director_prefetch_hit` at `:798`)
  - `voyage/supervisor.py:1478-1483` (`_propose_segment`: consume → `prefetch_hit = prefetched_raw is not None`)
  - `voyage/supervisor.py:899-921` (`_accept_director_decision`: drift-hold early return at `:900-917`, before `prefetch_pending` at `:921`)
  - `voyage/supervisor.py:179-196` (`summarize_prefetch_outcome` aggregates raw hit/miss events)
  - `Voyage/DESIGN.md` rhythm-cut entry (~`:6652-6658`: prefetch consumed "when it targets the right segment and no fresh inspect amendments exist")

## Technical description

`_take_prefetch` logs the hit at consumption time (`supervisor.py:797-799`):

```python
self._log_metric({"event": "director_prefetch_hit", "segment_id": segment_id})
return raw
```

`_propose_segment` consumes the future, then computes the local flag (`:1478-1483`):

```python
prefetched_raw = self._take_prefetch(number, segment_id)
if amendments:
    prefetched_raw = None
prefetch_hit = prefetched_raw is not None
```

The amendments discard (`:1479-1480`) is 136's site. But the drift-hold discard happens one call deeper: `_accept_director_decision` (`:899-917`) returns the deterministic hold *before* the prefetch is ever consulted — `prefetch_pending` is only computed at `:921`, after the early return:

```python
drift_every = max(1, config.voyage.drift_every_n_segments)
if state.next_segment_number % drift_every != 0:
    hold = DeterministicDirector(style_spec.prompt).propose(...)
    ...
    return hold          # prefetched_raw never becomes a candidate
```

So on every held segment with a ready prefetch: the `director_prefetch_hit` event is already emitted, `ProposedSegment.prefetch_hit` is `True` (`:1537`), the plan/console record a hit — and the accept loop never ran at all (no live `decide` either; the hold path returns before `:922-930`). `summarize_prefetch_outcome` (`:179-196`) counts the raw event with no drift awareness, so the reported rate measures "prefetch was ready", not "prefetch was used", diverging exactly on held segments.

Reachability: only with `drift_every_n_segments > 1` (`config.py:462` defaults to `1`, where `number % 1 == 0` always and the hold never fires). The flag exists (`--drift-every-n` on run/generate, `config.py:734-795` plumbing), so any cadence experiment that leans on the hit-rate to judge prefetch usefulness reads an inflated number.

DESIGN drift (docs side of the same defect): the §140 rhythm-cut entry documents the prefetch as consumed "when it targets the right segment and no fresh inspect amendments exist" — the drift-hold invalidation is a second, undocumented discard condition, and the `hit/miss` metric pair has no `invalidated` outcome for either.

## Why this is an issue

Hit-rate is the decision metric for prefetch restructuring (033's own comment: "the hit rate the soak report needs before any prefetch restructuring is considered — measure first"). With cadence holds on, the numerator is inflated by segments where prefetch could never have helped (the hold path ignores proposals by construction). An operator comparing cadence `1` vs `4` sees hit-rate *rise* under holds — the opposite of the feature's real usefulness — and may keep or expand a speculative window whose proposals are systematically thrown away on 3 of 4 segments.

## Live evidence

Host reads 2026-09-30 (no GPU needed; ordering is static):

```
$ sed -n '774,799p' voyage/supervisor.py
    798:        self._log_metric({"event": "director_prefetch_hit", "segment_id": segment_id})
    799:        return raw

$ sed -n '899,921p' voyage/supervisor.py
    900:        if state.next_segment_number % drift_every != 0:
    ...
    917:            return hold
    921:        prefetch_pending = prefetched_raw is not None and not amendments

$ sed -n '1478,1483p' voyage/supervisor.py
    1478:        prefetched_raw = self._take_prefetch(number, segment_id)
    1479:        if amendments:
    1480:            prefetched_raw = None
    1481:        prefetch_hit = prefetched_raw is not None
```

- The hit log (`:798`) fires before the function returns; there is no drift parameter, no post-return hook, no retraction path — identical mechanics to 136's amendments finding, one call site deeper for drift.
- The hold return (`:917`) precedes any reference to `prefetched_raw` in the method body (`:921` is the first); the parameter is accepted but unread on this path.
- `drift_every_n_segments: int = 1` (`config.py:462`) — default masks the site; any `--drift-every-n 2+` run exposes it.

## Minimal repro

Static (deterministic): set `drift_every_n_segments=2`, arrange a ready prefetch for an odd segment number (odd `next_segment_number` → hold). Commit → `logs/metrics.jsonl` shows `director_prefetch_hit` for that segment, while the plan dict shows `drift_hold: True` and the accept loop never called `decide` (no `worker_restart`, no live decision latency in `stage_seconds["director"]` beyond the hold). `summarize_prefetch_outcome` over the window counts a hit that was never usable.

## Fix candidates

1. (Preferred) Fold drift into the same third-outcome fix 136 proposes: pass hold-state into `_take_prefetch` (or check cadence before consuming) and log `director_prefetch_invalidated` instead of `hit` when the proposal cannot be used — one fix covers both discard sites, and the soak report gets `hit / (hit + miss)` with invalidated reported separately.
2. Cheapest correct variant: compute `drift_hold` *before* `_take_prefetch` in `_propose_segment` and skip consumption on held segments (log `miss` or nothing — the future stays pending for... no: the future targets `number`, the next commit targets `number+1`, so a skipped future is stale anyway; log `invalidated` and drop it explicitly).
3. Test: drift-hold + ready prefetch → assert no `hit` event and one `invalidated` event; `summarize_prefetch_outcome` pins hit/miss/invalidated shapes (extends 136's candidate-3 test).
4. Docs: amend the §140 prefetch paragraph to name both discard conditions (amendments + drift-hold).

## References

- In-tree: `voyage/supervisor.py:179-196,774-799,899-938,1475-1483,1506-1508,1532-1538`; `voyage/config.py:462`; `DESIGN.md` §20 (prefetch) + §140 rhythm-cut entry.
- Neighbor issues — not a duplicate of 136 (136 is the *amendments* discard at `:1479-1480`; this is the *drift-hold* discard at `:900-917` — different trigger, different layer, same inflated metric; fix once, covering both): 030 (shutdown hang), 033 (aggregation shape), 107 (pipeline-docs omission).
- External: hermes-cashew `metrics.py` `record_prefetch_cancelled` ("prefetch work invalidated before its next expensive stage") — the invalidated-as-third-outcome precedent (same citation as 136; the two files should resolve to one implementation).

## Investigation log

- 2026-09-30: filed by the 168-177 tails sweep; re-verified live via Read/Grep (concurrent uncommitted edits noted in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py`, `voyage/config.py`, `voyage/persistence.py`, `voyage/rpc.py`, `voyage/supervisor.py` — citations are as-read values above).
