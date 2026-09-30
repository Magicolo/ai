"""Issue 059 tests: novelty must fail loud on lost vectors; validate covers concepts.

Fail-loud contract: `check_novel` / `append` with an embedding raise
StateError when accepted records reference missing vector rows, instead
of scoring every duplicate 0.0 (accepted as novel). `validate_concepts`
pins the index/vectors/records consistency rules `validate_run` enforces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.cli import validate_run
from voyage.concepts import ConceptStore, validate_concepts
from voyage.config import load_config
from voyage.errors import StateError
from voyage.supervisor import Supervisor


def test_check_novel_fails_loud_on_deleted_vectors(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty")
    store.propose("glowing neon lattice", vector=[1.0, 0.0, 0.0], segment=0)
    (tmp_path / "novelty" / "concept_vectors.npy").unlink()
    reopened = ConceptStore(tmp_path / "novelty")
    with pytest.raises(StateError, match="concept vectors missing rows"):
        reopened.check_novel("glowing neon lattice", vector=[1.0, 0.0, 0.0])


def test_append_fails_loud_on_deleted_vectors(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty")
    store.propose("glowing neon lattice", vector=[1.0, 0.0, 0.0], segment=0)
    (tmp_path / "novelty" / "concept_vectors.npy").unlink()
    reopened = ConceptStore(tmp_path / "novelty")
    with pytest.raises(StateError, match="concept vectors missing rows"):
        reopened.append("silent copper dunes", accepted=True, vector=[0.0, 1.0, 0.0])


def test_token_only_history_needs_no_vectors(tmp_path: Path) -> None:
    """Legacy/migrated token-set records must keep working vector-free."""
    store = ConceptStore(tmp_path / "novelty")
    store.append("glowing neon lattice", accepted=True)
    reopened = ConceptStore(tmp_path / "novelty")
    accepted, _ = reopened.check_novel("glowing neon lattice")
    assert not accepted
    assert validate_concepts(tmp_path / "novelty") == []


def test_validate_concepts_accepts_clean_roundtrip(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty")
    record, _ = store.propose("glowing neon lattice", vector=[1.0, 0.0, 0.0], segment=0)
    assert record.embedding_index == 0
    assert validate_concepts(tmp_path / "novelty") == []


def test_validate_concepts_reports_missing_rows(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty")
    store.propose("glowing neon lattice", vector=[1.0, 0.0, 0.0], segment=0)
    (tmp_path / "novelty" / "concept_vectors.npy").unlink()
    errors = validate_concepts(tmp_path / "novelty")
    assert any("missing rows" in error for error in errors)


def test_validate_concepts_reports_unknown_index_key(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty")
    store.propose("glowing neon lattice", vector=[1.0, 0.0, 0.0], segment=0)
    index_path = tmp_path / "novelty" / "concept_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["concept-999999"] = 0
    index_path.write_text(json.dumps(index), encoding="utf-8")
    errors = validate_concepts(tmp_path / "novelty")
    assert any("concept-999999" in error for error in errors)


def test_validate_concepts_reports_row_mismatch(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty")
    record, _ = store.propose("glowing neon lattice", vector=[1.0, 0.0, 0.0], segment=0)
    index_path = tmp_path / "novelty" / "concept_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index[record.id] = record.embedding_index + 5
    index_path.write_text(json.dumps(index), encoding="utf-8")
    errors = validate_concepts(tmp_path / "novelty")
    assert any("mismatch" in error and record.id in error for error in errors)


def test_validate_run_flags_lost_vectors(tmp_path: Path) -> None:
    """A run whose vectors vanished must not validate clean."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="concepts")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    Supervisor(run_dir, config).run_segments(1)
    novelty_dir = run_dir / "novelty"
    novelty_dir.mkdir(exist_ok=True)
    store = ConceptStore(novelty_dir)
    store.propose("glowing neon lattice", vector=[1.0, 0.0, 0.0], segment=0)
    (novelty_dir / "concept_vectors.npy").unlink()
    errors = validate_run(run_dir)
    assert any("concept_vectors" in error for error in errors)


def test_load_jsonl_roundtrips_records_and_tolerates_missing(tmp_path: Path) -> None:
    """Test-pin for the `load_jsonl` helper (issue 046 disposition)."""
    missing = tmp_path / "absent.jsonl"
    assert ConceptStore.load_jsonl(missing) == []
    concepts_file = tmp_path / "concepts.jsonl"
    concepts_file.write_text(
        '{"id": "concept-000000"}\n\n{"id": "concept-000001"}\n', encoding="utf-8"
    )
    loaded = ConceptStore.load_jsonl(concepts_file)
    assert [entry["id"] for entry in loaded] == ["concept-000000", "concept-000001"]
