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
