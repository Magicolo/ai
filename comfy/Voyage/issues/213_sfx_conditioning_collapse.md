# 213 — SFX `existing` dict collapses conditioning_source (proxy vs shipped) (HIGH)

## Technical description
`_render_single_track` dedupes the ledger by `window_id` alone, discarding
the `conditioning_source` dimension that `_stem_cache_hit` and `validate`
both honor:

```python
existing = {record["window_id"]: record for record in load_sfx_ledger(ledger)}
# voyage/sfx_finalize.py:883 (verified live 2026-10-07)
```

vs `validate` grouping by `(conditioning_source, window_id)`
(`_validate_one_ledger`, ~line 458) and
`_stem_cache_hit(record, window, ..., conditioning_source)` checking source
equality.

## Rationale
After a parallel (proxy) + sequential (shipped) finalize, one window has two
records. Only the last survives in `existing`. If the survivor is the wrong
source → every re-finalize misses and re-renders a GPU window needlessly
(516 renders on a 30-min dual-pan run: 258 windows × 2). If the survivor is
the right source but the ledger also holds the other source's newer line,
the wrong-source record is invisible yet still counts in validate — latent
inconsistency between "what validate sees" and "what the renderer reuses".

## Live evidence
```
$ grep -n "existing = {" voyage/sfx_finalize.py
883: existing = {record["window_id"]: record for record in load_sfx_ledger(ledger)}
$ python3 -c "...
plan_sfx_windows(1800.0, [(0,1800.0,'c')]) → 30min windows: 258 dual renders: 516"
```

## Repro
Ledger with two lines, same `window_id`, different `conditioning_source`;
call the `_render_single_track` lookup path (or unit-test the dict
construction) — one record is lost.

## Source refs
- `Voyage/voyage/sfx_finalize.py:883`, `:248-313`, `~:458`
- In-tree contracts (`validate_sfx_ledger` groups per source;
  `_stem_cache_hit` docstring: "shipped-pixel stems and proxy-pixel stems
  are never interchangeable")

## Online sources
- Cache-key doctrine: dedupe key must equal the validity key (in-tree
  contracts above are the authority).

## Fix candidates
1. Key `existing` by `(conditioning_source, window_id)`.
2. Or two dicts per source.
3. Pin with a regression test carrying both sources for one window.

## Log
- Track D sweep, 2026-10-07. Line verified live by orchestrator.
  Read-only; nothing fixed.

## Evaluation (2026-10-07, live re-probe)
- **Confirmed.** `voyage/sfx_finalize.py:883` built `existing` keyed by
  `record["window_id"]` alone, while `_validate_one_ledger` groups by
  `(conditioning_source, window_id)` and `_stem_cache_hit` refuses
  cross-source hits. Reproduced the mechanism in code: ledger
  `[shipped w0000, proxy w0000]` → lookup kept only the proxy line →
  shipped pass missed and re-rendered.
- **Refined (new):** the orphan-stem adoption block had the same collapse
  in the other direction — it skipped adoption only when the window_id
  was present at all, so after keying `existing` by tuple a proxy-led
  ledger would ADOPT the proxy stem file as a shipped record (cross-source
  false hit without rendering). Found via the neighbor pin
  `test_proxy_then_shipped_rerenders_without_false_hits`, which failed
  (`calls == []`, adoption stole the render) after the bare rekey.

## Progress log (2026-10-07)
- Added `_sfx_ledger_key(record)` (`(conditioning_source, window_id)`,
  legacy lines read as shipped) and rekeyed all four `existing` sites
  (construction, adoption skip, adoption store, render lookup).
- Hardened orphan adoption with `claimed_window_ids`: a window_id the
  ledger attributes to any source is never adopted under another source.
- Updated the two `resume reads key by window_id` docstrings to the tuple
  key. `_stem_cache_hit`'s source check stays as defense in depth.
- Tests: `test_sfx_ledger_key_splits_conditioning_source` (pure) +
  `test_proxy_then_shipped_ledger_still_hits_shipped` (both sources one
  window, shipped pass renders zero windows) in
  `tests/test_sfx_finalize.py`.
- Verification (in-container `voyage:latest`): ruff check + format clean,
  mypy clean on `voyage/sfx_finalize.py`; scoped pytest 78 passed
  (`test_sfx_finalize`, new 213/214 tests, `test_sfx_parallel`
  incl. the proxy-then-shipped pin, bounds/contract/dual-pan/timeline
  suites). One failure in the wider sweep,
  `test_sfx_dual_pan::test_skip_key_carries_dual_pan`, is foreign (a
  concurrent agent's uncommitted `cli_core.py` skip-key `,master=0`
  change) — untouched per §9.

## Resolution (2026-10-07)
- **RESOLVED.** Dedupe key is now `(conditioning_source, window_id)`
  everywhere the renderer looks up or adopts, matching validate's
  grouping; cross-source false hits are impossible at lookup AND at
  adoption. Files: `voyage/sfx_finalize.py`,
  `tests/test_sfx_finalize.py`. No commit (per mandate).
