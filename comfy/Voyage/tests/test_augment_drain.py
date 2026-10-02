"""Finalize drain over ledgered interp chunks (issue: independent augment workers).

DESIGN §57: the drain consumes what the pollers produced — ledgered
`interpolated_<NN>/` PNG dirs — encodes each to a chunk mp4 and concats
them in plan order. Stub `encode_fn` / `concat_fn` seams keep these tests
stdlib-only (no ffmpeg, no torch); production passes the real chunk
encoder and stream-copy concat.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import augment_sidecar as sidecar
from voyage.augment_drain import drain_interpolated_plan
from voyage.errors import MediaError


def _key(index: int, *, expected: int) -> sidecar.ChunkKey:
    return sidecar.ChunkKey(
        chunk_index=index,
        start_frame=index * 2,
        source_frames=2,
        expected_frames=expected,
        upscale_factor=2,
        multiplier=4,
        crf=15,
        preset="veryfast",
        source_key="seg",
        weights_key="w",
        out_width=1536,
        out_height=1024,
        out_fps=96,
    )


def _plan_with_chunks(tmp_path: Path, *, indexes: list[int], png_counts: dict[int, int]) -> Path:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    ledger = plan_dir / sidecar.CHUNKS_LEDGER_FILENAME
    for index in indexes:
        expected = (png_counts[index] - 1) * 4 + 1
        key = _key(index, expected=expected)
        sidecar.append_chunk_record(ledger, key, stage="upscaled", path=f"up_{index}")
        sidecar.append_chunk_record(
            ledger, key, stage="interpolated", path=f"interpolated_{index:02d}"
        )
        out_dir = plan_dir / f"interpolated_{index:02d}"
        out_dir.mkdir()
        for frame in range(expected):
            (out_dir / f"frame_{frame:06d}.png").write_bytes(b"png")
    return plan_dir


def test_drain_encodes_each_chunk_and_concats_in_order(tmp_path: Path) -> None:
    plan_dir = _plan_with_chunks(tmp_path, indexes=[0, 1], png_counts={0: 2, 1: 3})
    encoded: list[tuple[Path, Path, float]] = []
    concatenated: list[list[Path]] = []

    def encode_fn(png_dir: Path, dest: Path, fps: float) -> Path:
        encoded.append((png_dir, dest, fps))
        dest.write_bytes(b"chunk")
        return dest

    def concat_fn(chunks: list[Path], dest: Path) -> Path:
        concatenated.append(list(chunks))
        dest.write_bytes(b"full")
        return dest

    result = drain_interpolated_plan(
        plan_dir, out_fps=96.0, encode_fn=encode_fn, concat_fn=concat_fn
    )
    assert [call[0].name for call in encoded] == ["interpolated_00", "interpolated_01"]
    assert all(fps == 96.0 for _, _, fps in encoded)
    # Temp renders keep the real suffix (SFX `<name>.partial.wav`
    # convention): ffmpeg infers the format from it, so a `.partial`
    # terminator here would break the production encode.
    assert [call[1].name for call in encoded] == [
        "chunk_00.partial.mp4",
        "chunk_01.partial.mp4",
    ]
    assert concatenated == [
        [plan_dir / "chunk_00.mp4", plan_dir / "chunk_01.mp4"],
    ]
    assert result.chunk_mp4s == (plan_dir / "chunk_00.mp4", plan_dir / "chunk_01.mp4")
    assert result.intermediate_mp4 == plan_dir / "model_intermediate.mp4"
    assert result.chunks_drained == 2


def test_redrain_skips_existing_chunk_mp4s(tmp_path: Path) -> None:
    plan_dir = _plan_with_chunks(tmp_path, indexes=[0, 1], png_counts={0: 2, 1: 2})
    (plan_dir / "chunk_00.mp4").write_bytes(b"kept")
    encoded: list[Path] = []

    def encode_fn(png_dir: Path, dest: Path, fps: float) -> Path:
        encoded.append(png_dir)
        dest.write_bytes(b"chunk")
        return dest

    def concat_fn(chunks: list[Path], dest: Path) -> Path:
        dest.write_bytes(b"full")
        return dest

    result = drain_interpolated_plan(
        plan_dir, out_fps=96.0, encode_fn=encode_fn, concat_fn=concat_fn
    )
    assert encoded == [plan_dir / "interpolated_01"]
    assert (plan_dir / "chunk_00.mp4").read_bytes() == b"kept"
    assert result.chunks_drained == 2


def test_missing_interpolated_record_fails_loud(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    ledger = plan_dir / sidecar.CHUNKS_LEDGER_FILENAME
    key = _key(0, expected=5)
    sidecar.append_chunk_record(ledger, key, stage="upscaled", path="up_0")

    with pytest.raises(MediaError, match="0"):
        drain_interpolated_plan(plan_dir, out_fps=96.0)


def test_png_count_mismatch_fails_loud(tmp_path: Path) -> None:
    plan_dir = _plan_with_chunks(tmp_path, indexes=[0], png_counts={0: 2})
    extra = plan_dir / "interpolated_00" / "frame_000005.png"
    extra.write_bytes(b"stray")

    def encode_fn(png_dir: Path, dest: Path, fps: float) -> Path:
        dest.write_bytes(b"chunk")
        return dest

    with pytest.raises(MediaError, match="interpolated_00"):
        drain_interpolated_plan(plan_dir, out_fps=96.0, encode_fn=encode_fn)


def test_empty_chunk_mp4_is_reencoded(tmp_path: Path) -> None:
    plan_dir = _plan_with_chunks(tmp_path, indexes=[0], png_counts={0: 2})
    (plan_dir / "chunk_00.mp4").write_bytes(b"")
    encoded: list[Path] = []

    def encode_fn(png_dir: Path, dest: Path, fps: float) -> Path:
        encoded.append(png_dir)
        dest.write_bytes(b"chunk")
        return dest

    def concat_fn(chunks: list[Path], dest: Path) -> Path:
        dest.write_bytes(b"full")
        return dest

    drain_interpolated_plan(plan_dir, out_fps=96.0, encode_fn=encode_fn, concat_fn=concat_fn)
    assert encoded == [plan_dir / "interpolated_00"]
