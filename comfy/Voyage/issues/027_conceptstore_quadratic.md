# 027 — ConceptStore is O(history²) per commit (full re-read + full-matrix rewrite + Python-loop cosine)

- Status: resolved (fixed 2026-09-25: matrix cache + vectorized cosine + single-load append + tests; resident store stays follow-up)
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
- 2026-09-26 (resolution, FIXED — `concepts.py`-local slice): the
  resident-store half needs `supervisor.py` (fresh `ConceptStore` per
  commit at `supervisor.py:1321`, check+append per attempt at `:860`),
  which is out of scope — so this slice removes the per-segment
  Python-scale costs without touching the supervisor: (1) in-memory
  matrix cache (`ConceptStore._matrix_cache`,
  `voyage/concepts.py:91-96,189-202`) — the single-writer contract
  (DESIGN §72) makes it safe: only our own appends mutate the file
  and they refresh the cache; a fresh instance per commit (the
  supervisor pattern) always re-reads, so the 059 fail-loud dangling
  check still fires on a deleted matrix (pinned by test);
  (2) `_append_vector` (`:221-268`) loads the file once (was two
  `np.load` calls) and appends via C-speed `concatenate` (was
  `tolist → append → asarray`, a Python-float roundtrip of every
  stored row per segment) plus an empty-vector guard;
  (3) `check_novel` (`:270-295`) scores via `_max_cosine_to_accepted`
  (`:77-107`), a vectorized twin of the `cosine_similarity` loop with
  identical semantics (0.0 floor, zero-norm/dimension-mismatch → 0.0).
  The per-commit full-matrix `.npy` rewrite remains (format
  constraint) — now one C `memcpy` + atomic rename instead of a
  Python loop; a resident store or mmap growth across `run_segments`
  is the supervisor-side follow-up. Also adapted `_write_index` to
  the concurrent `atomic.py` `JsonValue` tightening
  (`dict[str, JsonValue]` annotation — behavior identical).
  Tests: `tests/test_issue_027_concepts_perf.py` (9 tests: loop
  equivalence incl. orthogonal/duplicate, zero/dimension edges,
  sequential indices, empty-vector rejection, cache sharing +
  refresh, fresh-instance dangling, token fallback).
  Gates: ruff + format + mypy strict clean on `concepts.py`; new
  tests pass in-container (see 014 log for full-tree gate state).
