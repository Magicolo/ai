"""Run-file pruning: legacy `legacy_path` readers tolerate a missing file (DESIGN §21).

`cmd_init` no longer writes the root `concepts.jsonl` duplicate, so fresh
runs have no legacy file at all. The deprecated `ConceptStore(legacy_path=...)`
migration path must keep working when the legacy file is absent — it simply
starts empty with no migration and no warning.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from voyage.concepts import ConceptStore


def test_concept_store_with_missing_legacy_path_starts_empty(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-legacy.jsonl"
    assert not missing.exists()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store = ConceptStore(tmp_path / "novelty", legacy_path=missing)
    assert len(store) == 0
    accepted, _ = store.check_novel("glowing neon lattice")
    assert accepted


def test_concept_store_without_legacy_path_starts_empty(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty")
    assert len(store) == 0
    accepted, _ = store.check_novel("glowing neon lattice")
    assert accepted
