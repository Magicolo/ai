# 059 — Novelty silently disabled when vectors are lost; `validate` covers no concept state

- Status: open
- Severity: low-medium (silent correctness drift — duplicates accepted as novel)
- Area: `voyage/concepts.py:174-193`, `voyage/cli.py:555-604`
- Rank rationale: disk cleanup / partial restore silently turns off novelty
  enforcement with no error anywhere.

## Technical description

```python
if vector is not None:
    stored = self._load_vectors()   # [] when concept_vectors.npy deleted
    best = 0.0
    for record in self._records:
        if not record.accepted or record.embedding_index < 0: continue
        if record.embedding_index < len(stored):   # all skipped when stored==[]
            best = max(best, cosine_similarity(...))
    return (best < self._threshold, best)  # (True, 0.0) — duplicate accepted as novel
```

Deleting/losing `concept_vectors.npy` makes every duplicate score `0.0` →
accepted; nothing reports it — `validate_run` checks checksums/metrics/numbering/
orphans/timeline but never `novelty/concepts.jsonl`, `concept_vectors.npy`
row-count vs `concept_index.json`, or index-vs-records consistency. Same for
`concept_index.json` staleness after a crash between record-append (fsynced) and
index rewrite (atomic) — harmless today (index derived, vectors read directly)
but unvalidated.

## Why this is an issue

Novelty enforcement is the mechanism that keeps an infinite voyage from
looping the same destination forever — and a routine disk cleanup (or a
partial restore) silently disables it with no error anywhere: every duplicate
scores 0.0 and is accepted as novel, so the run happily generates repetitive
content while reporting success. `validate` green-lights the damaged state,
so the corruption compounds across segments. Blast radius is creative
correctness over unbounded future output; the fix (fail loud or degrade with
a metric + validate coverage) is small and testable.

## Evidence

```
$ sed -n '174,190p' Voyage/voyage/concepts.py
    def check_novel(self, text: str, vector: list[float] | None = None) -> tuple[bool, float]:
        ...
        if vector is not None:
            stored = self._load_vectors()
            best = 0.0
            for record in self._records:
                if not record.accepted or record.embedding_index < 0:
                    continue
                if record.embedding_index < len(stored):
                    best = max(best, cosine_similarity(vector, stored[record.embedding_index]))
            return (best < self._threshold, best)  # stored==[] → (True, 0.0)
$ sed -n '555,556p;595,603p' Voyage/voyage/cli.py   # validate: checksums/metrics/
def validate_run(run_dir: Path) -> list[str]:       # numbering/timeline/orphans —
    orphans = ( ... segments_root.rglob("*.partial") ... )   # no concept state
```

Source quotes above (all re-verified live 2026-09-25).

## Reproduction

Commit N segments, `rm novelty/concept_vectors.npy`, propose an identical
destination with embeddings on → `check_novel` returns `(True, 0.0)`;
`validate_run` → `[]`.

## Source references

- Files/lines above.

## Resolution candidates

When `vector is not None` but any accepted record has `embedding_index >=
len(stored)`, return fallback or `StateError` (fail loud, or degrade to token-set
with a `novelty_degraded` metric); add `validate` checks: `len(vectors) >=
max(index)+1`, `index keys ⊆ accepted ids`, no `embedding_index=-1` with vectors
present unexpectedly.

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep.
- 2026-09-25 (repair): re-verified live — `check_novel` at
  `concepts.py:174-190` matches the quoted logic exactly; `validate_run` at
  `cli.py:555-606` checks checksums/metrics/numbering/timeline/`*.partial`
  orphans with zero concept-state coverage (Evidence pasted). No staleness.
  Added `## Why this is an issue`.
- Open: fail-loud/degrade + validate checks + tests.
- 2026-09-25 (full resolution — FIXED, fail-loud variant):
  `voyage/concepts.py` `check_novel` and `_append_vector` now raise
  `StateError` listing the dangling record ids when accepted records
  reference missing vector rows (shared `_dangling_vector_records`
  helper; token-only/legacy records with `embedding_index == -1` are
  unaffected); new `validate_concepts(directory)` covers vectors
  row-count vs max accepted index, index keys ⊆ accepted ids, and
  index-vs-record row agreement (mixed -1 deliberately not an error —
  legacy migrations coexist); `cli.validate_run` runs it whenever
  `novelty/` or legacy `concepts.jsonl` exists. Tests in
  `tests/test_concept_integrity.py` (fail-loud on deleted vectors for
  both entry points, token-only unaffected, all three validator rules,
  run-level `validate_run` flag). Note for the supervisor track:
  `check_novel` raising `StateError` inside `_accept_director_decision`
  aborts the commit loudly — that is the intended fail-loud behavior,
  no supervisor change required.
