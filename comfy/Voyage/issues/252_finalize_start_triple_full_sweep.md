# 252 — Finalize-start does 3–4 full sweeps (hash + PIL-open + rglob-stat every file)

Severity: MEDIUM (track D-08).

## Technical description

`run_durable_model_pass` → `sweep_chunk_mp4s` (all ledgers + PNG completeness) →
`prune_orphan_plan_dirs` (re-derives live dirs + `sha256_file` every joint video +
`_newest_modification_time` rglob-stats every file) → `_poll_to_completion` per-pass
scans → `build_parallel_tasks` PIL-opens every chunk PNG header.

## Rationale

Each sweep is O(segments × chunks × files). `prune_orphan_plan_dirs` alone hashes every
joint video (GBs) and stats every file under every non-live plan dir.
`build_parallel_tasks` calls `chunk_frames_match_size` (PIL open per PNG) per ledgered
chunk just to build the queue — 256×8×32 ≈ 65k header opens before any GPU work. These
run serially at every finalize start, and `_poll_to_completion` re-enumerates per pass
(up to 10).

## Live evidence

```
$ grep -n "sweep_chunk_mp4s\|prune_orphan_plan_dirs\|_poll_to_completion" voyage/augment_finalize.py | head
685-723
$ grep -n "rglob\|chunk_frames_match_size" voyage/augment_drain.py voyage/augment_parallel.py | head
drain:359 (_newest_modification_time rglob); parallel:184,186
```

Repro: `strace -c`-style counter (or `LD_PRELOAD`-free Python stat counter via `pathlib`
shim) during finalize-start on a 10-seg fixture — stat/open counts scale with plans ×
chunks, not with missing work.

## Source refs

`voyage/augment_finalize.py:685-701`; `voyage/augment_drain.py:372-479`
(`_newest_modification_time` rglob); `voyage/augment_parallel.py:177-219`;
`voyage/augment_sidecar.py:632-760`.

## Online sources

- In-tree budget comments (`UPSCALE_TILE_BUDGET_PIXELS`, chunk-32 design notes show the
  project measures these floors — sweeps have no equivalent budget).

## Fix candidates

- Single enumeration shared by sweep/GC/queue-build; cache PIL sizes per dir+mtime; hash
  joint videos once per process; skip GC when `augment/` mtime unchanged.

## Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.
