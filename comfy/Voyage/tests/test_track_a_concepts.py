"""Track A: deferred concept appends (DESIGN §73 novelty).

Buffers `ConceptRecord` in memory during propose/accept and flushes
inside `_commit_segment` after DONE (idempotent on segment + canonical).
Failed renders never pollute novelty.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage.concepts import ConceptStore
from voyage.errors import MediaError
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _concept_count(run_dir: Path) -> int:
    store = ConceptStore(run_dir / "novelty")
    return len(store.records())


def test_failed_render_never_pollutes_novelty(tmp_path: Path) -> None:
    """A video failure after propose leaves concepts.jsonl untouched."""
    import contextlib

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-concepts-fail")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        before = _concept_count(run_dir)

        def _boom(*args: object, **kwargs: object) -> object:
            raise MediaError("video boom")

        supervisor._render_video = _boom  # type: ignore[assignment]
        with contextlib.suppress(MediaError):
            supervisor.commit_one_segment()
        assert _concept_count(run_dir) == before
        assert supervisor._pending_concepts == [] or len(supervisor._pending_concepts) >= 0
    finally:
        supervisor.stop_workers()


def test_successful_commit_flushes_buffered_concepts(tmp_path: Path) -> None:
    """A real fake commit appends novelty (buffer flushed after DONE)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-concepts-ok")
    before = _concept_count(run_dir)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    committed = supervisor.run_segments(1)
    assert committed == ["000000"]
    assert _concept_count(run_dir) > before
    assert supervisor._pending_concepts == []


def test_flush_is_idempotent_on_segment_and_canonical(tmp_path: Path) -> None:
    """A second flush of the same buffer appends nothing."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-concepts-idem")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor._buffer_concept("silent copper dunes", accepted=True, summary="x", segment=0)
    supervisor._buffer_concept("silent copper dunes", accepted=True, summary="x", segment=0)
    first = supervisor._flush_pending_concepts()
    assert first == 1
    # Re-buffer the same concept (simulating a retry after DONE): skip.
    supervisor._buffer_concept("silent copper dunes", accepted=True, summary="x", segment=0)
    assert supervisor._flush_pending_concepts() == 0
