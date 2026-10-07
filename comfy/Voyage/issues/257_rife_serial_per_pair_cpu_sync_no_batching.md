# 257 — RIFE interp is strictly serial per pair with a `.cpu()` sync per mid

Severity: MEDIUM (track D-12).

## Technical description

`interpolate_rife_mids` loops pairs one by one; each mid does `.float().cpu().squeeze(0)`
inside the loop. No `pair_batch`, no OOM-halving (docstring: "OOM propagates — a single
pair is already minimal").

## Rationale

m=4 on a 32f chunk = 31 pairs × 3 forwards = 93 launches + 93 D2H syncs. Kernel-launch +
sync latency dominates at RIFE's 0.05 s/forward scale (in-tree: RIFE ~16.8× faster than
FILM per forward — the loop overhead is now the bottleneck, not the model). FILM already
proved the batching pattern with bit-exact `pair_batch=1` default; RIFE has no equivalent
knob, so throughput cannot scale with free VRAM (0.65 GiB peak leaves GBs idle on both
cards).

## Live evidence

```
$ grep -n "pair_batch\|_run_pair_window\|for pair_index" voyage/workers/augment_worker.py | sed -n '1,12p'
FILM: pair_batch + _run_pair_window; RIFE: bare 'for pair_index in range(pair_count)'
```

Repro: time 31-pair RIFE sweep vs batched-stacked equivalent on synthetic tensors
(structure-only, no weights) — launch-count delta is exact.

## Source refs

`voyage/workers/augment_worker.py:2082-2107` vs FILM `interpolate_mids`
(`1648-1736`: `pair_batch`, `_run_pair_window` halving, `on_pair` batching).

## Online sources

- ComfyUI-VFI guidance (batch_size trades memory for speed).
- RIFE-vs-FILM reports (RIFE 5–10× faster per forward, VRAM 4–6 GB typical — batching is
  the standard lever).

## Fix candidates

- Stack N pairs per forward with OOM-halving mirroring `_run_pair_window`; keep
  `pair_batch=1` default for bit-exactness (same contract as FILM).

## Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.

## Evaluation (2026-10-07)
- Claim CURRENT on re-read: `interpolate_rife_mids` still looped pairs one by one
  with a per-mid `.cpu()` sync and no batching knob. Fix candidate adopted
  (stacked pairs with OOM-halving, `pair_batch=1` default).

## Progress log
- Batch-6 Group Q added RIFE pair-batching in
  `voyage/workers/augment_worker.py`, mirroring the FILM `_run_pair_window`
  pattern with OOM-halving and a bit-exact `pair_batch=1` default. New
  `tests/test_issue_257_rife_batching.py` pins batching + default exactness.
  Scoped gates green (ruff + format + mypy strict + pytest).

## Resolution (2026-10-07)
- RESOLVED. RIFE throughput can now scale with free VRAM via `pair_batch` while
  the default stays bit-exact with the old serial behavior.
