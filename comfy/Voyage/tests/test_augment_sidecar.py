"""Independent augment sidecar ledger: durable chunks + resume (TDD, CPU-only).

Covers the Phase 1 contract: plan-hash isolation, append/load round trip,
exact-match cache hits (no silent downgrade), stage filtering for the
independent upscale/interp pollers, missing-chunk computation, and
partial pruning. No torch/GPU/ffmpeg — pure ledger logic over tmp_path.
"""

from __future__ import annotations

from pathlib import Path

from voyage.augment_sidecar import (
    AUGMENT_DIRNAME,
    CHUNKS_LEDGER_FILENAME,
    ChunkKey,
    append_chunk_record,
    chunk_cache_hit,
    completed_stages,
    load_chunk_ledger,
    missing_chunk_indexes,
    plan_hash_for,
    prune_stale_partials,
    sidecar_dir,
)


def _key(index: int = 0) -> ChunkKey:
    return ChunkKey(
        chunk_index=index,
        start_frame=index * 4,
        source_frames=4,
        expected_frames=13,
        upscale_factor=2,
        multiplier=4,
        crf=15,
        preset="veryfast",
        source_key="sources-abc",
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
    )


def test_sidecar_dir_lives_under_run_augment(tmp_path: Path) -> None:
    assert sidecar_dir(tmp_path, "deadbeef") == tmp_path / AUGMENT_DIRNAME / "deadbeef"
    assert CHUNKS_LEDGER_FILENAME == "chunks.jsonl"


def test_plan_hash_stable_and_recipe_sensitive() -> None:
    first = plan_hash_for(
        source_key="sources-abc",
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        multiplier=4,
        crf=15,
        preset="veryfast",
    )
    second = plan_hash_for(
        source_key="sources-abc",
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        multiplier=4,
        crf=15,
        preset="veryfast",
    )
    assert first == second and len(first) == 16
    changed = plan_hash_for(
        source_key="sources-abc",
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        multiplier=4,
        crf=18,
        preset="veryfast",
    )
    assert changed != first


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    ledger = tmp_path / AUGMENT_DIRNAME / "plan" / CHUNKS_LEDGER_FILENAME
    append_chunk_record(ledger, _key(0), stage="upscaled", path="augment/plan/upscaled_00.mp4")
    records = load_chunk_ledger(ledger)
    assert len(records) == 1
    assert records[0]["chunk_index"] == 0
    assert records[0]["stage"] == "upscaled"


def test_missing_ledger_loads_empty(tmp_path: Path) -> None:
    assert load_chunk_ledger(tmp_path / "nope.jsonl") == []


def test_cache_hit_requires_exact_key(tmp_path: Path) -> None:
    ledger = tmp_path / CHUNKS_LEDGER_FILENAME
    append_chunk_record(ledger, _key(1), stage="upscaled", path="augment/plan/upscaled_01.mp4")
    records = load_chunk_ledger(ledger)
    assert chunk_cache_hit(records, _key(1), stage="upscaled") is True
    altered = _key(1)
    object.__setattr__(altered, "weights_key", "weights-changed")
    assert chunk_cache_hit(records, altered, stage="upscaled") is False
    assert chunk_cache_hit(records, _key(1), stage="interpolated") is False
    assert chunk_cache_hit(records, _key(2), stage="upscaled") is False


def test_completed_stages_last_wins(tmp_path: Path) -> None:
    ledger = tmp_path / CHUNKS_LEDGER_FILENAME
    append_chunk_record(ledger, _key(0), stage="upscaled", path="augment/plan/upscaled_00.mp4")
    append_chunk_record(ledger, _key(0), stage="interpolated", path="augment/plan/chunk_00.mp4")
    stages = completed_stages(load_chunk_ledger(ledger))
    assert stages[0] == {"upscaled", "interpolated"}


def test_missing_chunk_indexes(tmp_path: Path) -> None:
    ledger = tmp_path / CHUNKS_LEDGER_FILENAME
    append_chunk_record(ledger, _key(0), stage="upscaled", path="augment/plan/upscaled_00.mp4")
    append_chunk_record(ledger, _key(1), stage="upscaled", path="augment/plan/upscaled_01.mp4")
    append_chunk_record(ledger, _key(0), stage="interpolated", path="augment/plan/chunk_00.mp4")
    records = load_chunk_ledger(ledger)
    assert missing_chunk_indexes(records, [0, 1, 2], stage="upscaled") == [2]
    assert missing_chunk_indexes(records, [0, 1, 2], stage="interpolated") == [1, 2]


def test_prune_stale_partials(tmp_path: Path) -> None:
    target = tmp_path / "plan"
    target.mkdir(parents=True)
    (target / "chunk_00.mp4.partial").write_bytes(b"junk")
    (target / "upscaled_01.mp4.partial").write_bytes(b"junk")
    (target / "chunk_01.mp4").write_bytes(b"keep")
    crashed_dir = target / "upscaled_02.partial"
    crashed_dir.mkdir()
    (crashed_dir / "frame_000001.png").write_bytes(b"junk")
    assert prune_stale_partials(target) == 3
    assert (target / "chunk_01.mp4").exists()
    assert not crashed_dir.exists()
    assert prune_stale_partials(target) == 0
