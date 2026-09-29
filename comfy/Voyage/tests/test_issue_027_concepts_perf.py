"""Issue 027: ConceptStore per-commit cost no longer scales in Python per row.

The store re-read all of `concepts.jsonl` per segment, re-loaded the full
vector matrix per `check_novel`, rewrote it via a `tolist → append →
asarray` Python-float roundtrip per segment, and scored cosine in a pure
Python loop. The resident-store half of the fix needs the supervisor
(out of scope — see the 027 log); this covers the `concepts.py`-local
slice: an in-memory matrix cache, a single disk load per append, a
C-speed `concatenate`, and a vectorized cosine that matches the old
loop exactly (0.0 floor, zero-norm and dimension-mismatch semantics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voyage import concepts
from voyage.concepts import ConceptStore, cosine_similarity
from voyage.errors import StateError


def _vector(seed: int, dimension: int = 8) -> list[float]:
    generator = np.random.default_rng(seed)
    return [float(value) for value in generator.normal(size=dimension)]


def _seeded_store(directory: Path, rows: int = 6, dimension: int = 8) -> ConceptStore:
    store = ConceptStore(directory)
    for index in range(rows):
        store.append(f"lantern valley {index}", accepted=True, vector=_vector(index, dimension))
    return store


def test_vectorized_check_matches_python_loop(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    stored = store._load_vectors()
    for seed in (100, 101, 102):
        query = _vector(seed)
        expected = 0.0
        for record in store.records():
            if record.accepted and 0 <= record.embedding_index < len(stored):
                expected = max(expected, cosine_similarity(query, stored[record.embedding_index]))
        accepted, best = store.check_novel(f"probe valley {seed}", query)
        assert best == pytest.approx(expected, abs=1e-6)
        assert accepted == (expected < 0.85)


def test_vectorized_check_matches_loop_on_orthogonal_and_duplicate(
    tmp_path: Path,
) -> None:
    store = ConceptStore(tmp_path)
    store.append("red lantern", accepted=True, vector=[1.0, 0.0, 0.0])
    store.append("blue lantern", accepted=True, vector=[0.0, 1.0, 0.0])
    accepted, best = store.check_novel("red lantern", [1.0, 0.0, 0.0])
    assert accepted is False
    assert best == pytest.approx(1.0)
    accepted, best = store.check_novel("green lantern", [0.0, 0.0, 1.0])
    assert accepted is True
    assert best == pytest.approx(0.0)


def test_zero_norm_and_mismatched_queries_score_zero(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    accepted, best = store.check_novel("empty echo", [0.0] * 8)
    assert accepted is True
    assert best == 0.0
    accepted, best = store.check_novel("short echo", [1.0, 0.0])
    assert accepted is True
    assert best == 0.0


def test_append_keeps_sequential_rows_and_indices(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path)
    for index in range(5):
        record = store.append(f"valley {index}", accepted=True, vector=_vector(index))
        assert record.embedding_index == index
    assert len(store._load_matrix()) == 5
    accepted, best = store.check_novel("valley 0", _vector(0))
    assert accepted is False
    assert best == pytest.approx(1.0)


def test_append_rejects_empty_vector(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        store.append("hollow echo", accepted=True, vector=[])


def test_repeated_checks_share_one_cached_matrix(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    first = store._load_matrix()
    second = store._load_matrix()
    assert first is second
    store.append("new valley", accepted=True, vector=_vector(99))
    refreshed = store._load_matrix()
    assert len(refreshed) == len(first) + 1
    accepted, _best = store.check_novel("new valley", _vector(99))
    assert accepted is False


def test_fresh_instance_still_fails_loud_on_missing_rows(tmp_path: Path) -> None:
    """Issue 059 survives the cache: supervisor builds a fresh store per
    commit, so a deleted matrix is re-read (cache miss) and dangles."""
    _seeded_store(tmp_path)
    (tmp_path / "concept_vectors.npy").unlink()
    reopened = ConceptStore(tmp_path)
    with pytest.raises(StateError, match="missing rows"):
        reopened.check_novel("late echo", _vector(7))


def test_token_fallback_path_unchanged(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path)
    store.append("red lantern", accepted=True)
    accepted, best = store.check_novel("red lantern")
    assert accepted is False
    assert best == pytest.approx(1.0)


def test_concepts_module_helpers_intact() -> None:
    assert concepts.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert concepts.cosine_similarity([], []) == 0.0
