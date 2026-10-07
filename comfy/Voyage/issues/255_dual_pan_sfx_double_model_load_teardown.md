# 255 — Dual-pan SFX loads + tears down the MMAudio stack twice

Severity: MEDIUM (track D-10).

## Technical description

`render_sfx_bed(dual_pan=True)` calls `_render_single_track` twice sequentially; each call
constructs workers, `worker.start()` (model load, GBs H2D), renders, `worker.stop()`
(evict). Track R differs only by seed offset + stem suffix.

## Rationale

2× model-load latency + 2× H2D traffic + 2× cold-cache warmup for what is one model
rendering two seed streams. Documented ladder cost is per-window forward; load cost is
pure overhead (~2× on short timelines: the 6-window 38 s case pays two full large-model
loads for 12 renders). Also doubles the failure window for transient load OOMs.

## Live evidence

```
$ grep -n "_render_single_track\|worker.start\|worker.stop" voyage/sfx_finalize.py | head
771,806 (two calls); worker.start once per _render_single_track
```

Repro: counter on `SubprocessWorker.start` during a 2-window dual-pan fake run — 2
starts where 1 suffices.

## Source refs

`voyage/sfx_finalize.py:771-836` (two `_render_single_track` calls);
worker start/stop `voyage/workers/sfx_mmaudio.py:149-182,497-504`.

## Online sources

- torch best practice (resident nets reused across calls; `_ESRGAN_CACHE`/`_RIFE_CACHE`
  issue-047 rationale in-tree states exactly this).

## Fix candidates

- Hoist worker pool to `render_sfx_bed` and share across tracks; or render both tracks'
  windows in one pool pass keyed by (window, track).

## Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.

## Evaluation (2026-10-07)
- Claim CURRENT on re-read: `render_sfx_bed(dual_pan=True)` still constructed and
  tore down a worker pool per track. Fix candidate (hoist + share pool) adopted.

## Progress log
- Batch-6 Group Q hoisted the pool in `voyage/sfx_finalize.py`: new
  `_sfx_pool_layout` / `_start_sfx_workers` helpers and a `shared_workers` path so
  both tracks render through one pool (one `start()`/model load, one `stop()`).
  New `tests/test_issue_255_dual_pan_single_pool.py` pins single-start behavior.
  Scoped gates green (ruff + format + mypy strict + pytest).

## Resolution (2026-10-07)
- RESOLVED. One model load serves both pan tracks; 2× load latency + 2× H2D
  traffic eliminated, and the transient-load-OOM failure window halved.
