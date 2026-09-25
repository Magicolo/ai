# 027 — ConceptStore is O(history²) per commit (full re-read + full-matrix rewrite + Python-loop cosine)

- Status: open
- Severity: medium-high (supervisor CPU wall on long voyages; fsync storms)
- Area: performance/durability — `voyage/concepts.py`, commit path
- Rank rationale: per-segment cost O(history) ⇒ lifetime O(N²) I/O + math; at
  1000 segments ≈ 1000 full-file rewrites of the vector matrix.

## Technical description

- `voyage/concepts.py:78-98` (`__init__` re-reads all `concepts.jsonl` per segment).
- `voyage/concepts.py:133-139` (`_load_vectors`: `np.load` full matrix per
  `check_novel`).
- `voyage/concepts.py:141-172` (`_append_vector`: `np.load → tolist → append →
  asarray → np.save` full matrix per segment).
- `voyage/concepts.py:174-193` (`check_novel`: pure-Python loop + per-record
  `cosine_similarity`).
- `voyage/supervisor.py:1030-1034` (new `ConceptStore` per commit), `:671-694`
  (embed RPC + check + append per attempt, up to `novelty_max_attempts`).

## Why this is an issue

Per-segment cost that grows with history punishes precisely the workload voyage exists for: long autonomous runs get slower the longer they run, with ~1000 full-file matrix rewrites plus fsync storms hammering the commit path at 1000 segments. Supervisor CPU burned in a Python-loop cosine is CPU stolen from orchestration on a box where the GPU is already the bottleneck. Left alone, this turns voyage length — the product's core promise — into a performance cliff.

## Evidence

`rg -n "ConceptStore\(|_load_vectors|_append_vector|check_novel"
Voyage/voyage/supervisor.py Voyage/voyage/concepts.py`.

Verified live 2026-09-25:

```
voyage/supervisor.py:673:  accepted, last_score = store.check_novel(...)
voyage/supervisor.py:1030: store = ConceptStore(
voyage/concepts.py:133:  def _load_vectors ... 141: def _append_vector
voyage/concepts.py:174:  def check_novel ... 182: stored = self._load_vectors()
voyage/concepts.py:203:  embedding_index = self._append_vector(vector) ...
```

## Reproduction

Microbench `append` in a loop with 500 existing rows; profile a 200-segment fake
run's commit overhead vs history length.

## Source references

- Files/lines above.

## Resolution candidates

1. Resident store across `run_segments` (single writer already — see 004 for the
   multi-writer caveat).
2. Append via `np.lib.format` resize or memory-mapped growth instead of full
   rewrite.
3. Vectorized cosine (`matrix @ vec / norms`) instead of the Python loop.

Payoff: commit overhead flat vs history length; removes fsync storms.

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; ConceptStore /
  supervisor refs re-verified live, current; pasted rg output into Evidence.
- Open: resident store + vectorized check + bench.
