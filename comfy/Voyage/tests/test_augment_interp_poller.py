"""Independent interp poller: FILM over upscaled chunks + resume (CPU-only, stubbed GPU).

Pins the Phase 3 contract with fake segments: the interp poller consumes
the `upscaled_<idx>/` PNG dirs (and `upscaled` ledger records) published
by the upscale poller in the same plan dir, interpolates each chunk to
`(n-1)*m+1` frames, and records `interpolated` ledger entries. Chunks
without upscaled work wait; completed chunks are skipped on re-poll;
crashed partials are re-rendered. Every rendered chunk is immediately
encoded to its durable `chunk_NN.mp4` and both PNG dirs are pruned
(per-segment cleanup); chunks already mp4-complete skip before any PNG
gate. Stub `interp_fn` / `chunk_encode_fn` seams keep these tests
stdlib-only (no torch/GPU/ffmpeg).
"""

from __future__ import annotations

import json
from pathlib import Path

from voyage.augment_interp_poller import interp_poll_once
from voyage.augment_sidecar import (
    STAGE_CHUNK_MP4,
    STAGE_INTERPOLATED,
    STAGE_UPSCALED,
    load_chunk_ledger,
)
from voyage.augment_upscale_poller import upscale_poll_once


def _make_segment(
    run_dir: Path,
    segment_id: str = "000000",
    *,
    frames: int = 8,
    checksum: str = "abc123",
) -> Path:
    segment_dir = run_dir / "segments" / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "video.mp4").write_bytes(b"fake-video")
    manifest = {
        "format": 1,
        "transition": {},
        "prompt_plan": {},
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": frames},
        "checksums": {"video.mp4": checksum},
    }
    (segment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    return segment_dir


def _stub_decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for position in range(count):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(b"fake-frame")
        written.append(frame)
    return written


def _stub_upscale(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
    assert dest_dir.is_dir()
    return list(frame_paths)


def _stub_interp(frame_paths: list[Path], dest_dir: Path, multiplier: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected = (len(frame_paths) - 1) * multiplier + 1
    written = []
    for position in range(expected):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(b"fake-interp")
        written.append(frame)
    return written


def _stub_encode(png_dir: Path, dest: Path, fps: float) -> Path:
    dest.write_bytes(b"fake-chunk")
    return dest


def _upscale_first(run_dir: Path, frames: int = 8) -> None:
    _make_segment(run_dir, frames=frames)
    result = upscale_poll_once(
        run_dir,
        weights_path=Path("/models/realesrgan/realesr-animevideov3.pth"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=4,
        decode_fn=_stub_decode,
        upscale_fn=_stub_upscale,
    )
    assert result.chunks_done == 2 if frames == 8 else result.chunks_done >= 1


def _interp_kwargs(**overrides):  # type: ignore[no-untyped-def]
    params = {
        "weights_path": Path("/models/frame_interpolation/film_net_fp16.safetensors"),
        "weights_key": "weights-abc",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "chunk_frames": 4,
        "multiplier": 4,
        "interp_fn": _stub_interp,
        "chunk_encode_fn": _stub_encode,
    }
    params.update(overrides)
    return params


def test_waits_without_upscaled_work(tmp_path: Path) -> None:
    _make_segment(tmp_path, frames=8)
    result = interp_poll_once(tmp_path, **_interp_kwargs())
    assert result.segments_seen == 1
    assert result.chunks_done == 0
    assert result.chunks_waiting == 2
    assert (tmp_path / "augment").exists() is False


def test_poll_interpolates_upscaled_chunks(tmp_path: Path) -> None:
    _upscale_first(tmp_path, frames=8)
    result = interp_poll_once(tmp_path, **_interp_kwargs())
    assert result.segments_seen == 1
    assert result.chunks_done == 2
    assert result.chunks_waiting == 0
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    records = load_chunk_ledger(plan_dir / "chunks.jsonl")
    interp_records = [record for record in records if record["stage"] == STAGE_INTERPOLATED]
    assert len(interp_records) == 2
    assert all(record["expected_frames"] == 13 for record in interp_records)
    assert all(record["multiplier"] == 4 for record in interp_records)
    assert all(record["out_fps"] == 96 for record in interp_records)
    assert sorted(record["chunk_index"] for record in interp_records) == [0, 1]
    upscaled = [record for record in records if record["stage"] == STAGE_UPSCALED]
    assert len(upscaled) == 2
    mp4_records = [record for record in records if record["stage"] == STAGE_CHUNK_MP4]
    assert len(mp4_records) == 2
    assert sorted(record["chunk_index"] for record in mp4_records) == [0, 1]
    for index in (0, 1):
        assert (plan_dir / f"chunk_{index:02d}.mp4").stat().st_size > 0
        # Per-segment cleanup: both PNG dirs are pruned once the durable
        # mp4 supersedes them.
        assert not (plan_dir / f"interpolated_{index:02d}").exists()
        assert not (plan_dir / f"upscaled_{index:02d}").exists()


def test_repoll_skips_completed_chunks(tmp_path: Path) -> None:
    _upscale_first(tmp_path, frames=8)
    calls: list[int] = []

    def counting_interp(frame_paths: list[Path], dest_dir: Path, multiplier: int) -> list[Path]:
        calls.append(len(frame_paths))
        return _stub_interp(frame_paths, dest_dir, multiplier)

    first = interp_poll_once(tmp_path, **_interp_kwargs(interp_fn=counting_interp))
    assert first.chunks_done == 2
    second = interp_poll_once(tmp_path, **_interp_kwargs(interp_fn=counting_interp))
    assert second.chunks_done == 0
    assert second.chunks_skipped == 2
    assert calls == [4, 4]


def test_crashed_partial_is_rerendered(tmp_path: Path) -> None:
    _upscale_first(tmp_path, frames=8)
    result = interp_poll_once(tmp_path, **_interp_kwargs())
    assert result.chunks_done == 2
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    ledger = plan_dir / "chunks.jsonl"
    # Simulate a crash mid-render for chunk 1: drop its interp + mp4
    # records and its mp4, restore its upscaled inputs (pruned by the
    # first poll's cleanup), leave a partial behind.
    kept = [
        record
        for record in load_chunk_ledger(ledger)
        if not (
            record["chunk_index"] == 1 and record["stage"] in (STAGE_INTERPOLATED, STAGE_CHUNK_MP4)
        )
    ]
    ledger.write_text("\n".join(json.dumps(record) for record in kept) + "\n", encoding="utf-8")
    (plan_dir / "chunk_01.mp4").unlink()
    upscaled = plan_dir / "upscaled_01"
    upscaled.mkdir()
    for position in range(4):
        (upscaled / f"frame_{position + 1:06d}.png").write_bytes(b"fake-frame")
    crashed = plan_dir / "interpolated_01.partial"
    crashed.mkdir()
    (crashed / "frame_000001.png").write_bytes(b"junk")
    rerun = interp_poll_once(tmp_path, **_interp_kwargs())
    assert rerun.chunks_done == 1
    assert rerun.chunks_skipped == 1
    assert rerun.partials_pruned == 1
    records = load_chunk_ledger(ledger)
    assert len([record for record in records if record["stage"] == STAGE_INTERPOLATED]) == 2
    assert len([record for record in records if record["stage"] == STAGE_CHUNK_MP4]) == 2


def test_kill_before_ensure_encodes_without_rerender(tmp_path: Path) -> None:
    """A kill between the interp record and the per-chunk ensure heals via encode.

    The interp PNGs are intact, so the next poll must NOT re-render —
    it encodes the missing mp4, records it, and prunes the PNGs.
    """
    _upscale_first(tmp_path, frames=8)
    result = interp_poll_once(tmp_path, **_interp_kwargs())
    assert result.chunks_done == 2
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    ledger = plan_dir / "chunks.jsonl"
    # Kill landed after the chunk-1 interp record append but before its
    # ensure: PNGs intact on both legs, mp4 record + mp4 missing.
    kept = [
        record
        for record in load_chunk_ledger(ledger)
        if not (record["stage"] == STAGE_CHUNK_MP4 and record["chunk_index"] == 1)
    ]
    ledger.write_text("\n".join(json.dumps(record) for record in kept) + "\n", encoding="utf-8")
    (plan_dir / "chunk_01.mp4").unlink()
    upscaled = plan_dir / "upscaled_01"
    upscaled.mkdir()
    for position in range(4):
        (upscaled / f"frame_{position + 1:06d}.png").write_bytes(b"fake-frame")
    interpolated = plan_dir / "interpolated_01"
    interpolated.mkdir()
    for position in range(13):
        (interpolated / f"frame_{position + 1:06d}.png").write_bytes(b"fake-interp")
    calls: list[int] = []

    def counting_interp(frame_paths: list[Path], dest_dir: Path, multiplier: int) -> list[Path]:
        calls.append(len(frame_paths))
        return _stub_interp(frame_paths, dest_dir, multiplier)

    rerun = interp_poll_once(tmp_path, **_interp_kwargs(interp_fn=counting_interp))
    assert rerun.chunks_done == 0
    assert calls == []
    assert rerun.chunks_skipped == 2
    records = load_chunk_ledger(ledger)
    assert len([record for record in records if record["stage"] == STAGE_CHUNK_MP4]) == 2
    assert (plan_dir / "chunk_01.mp4").stat().st_size > 0
    assert not (plan_dir / "interpolated_01").exists()
    assert not (plan_dir / "upscaled_01").exists()


def test_mp4_complete_chunks_skip_without_pngs(tmp_path: Path) -> None:
    """Steady state: pruned PNGs + durable mp4s skip with nothing waiting."""
    _upscale_first(tmp_path, frames=8)
    first = interp_poll_once(tmp_path, **_interp_kwargs())
    assert first.chunks_done == 2
    rerun = interp_poll_once(tmp_path, **_interp_kwargs())
    assert rerun.chunks_done == 0
    assert rerun.chunks_skipped == 2
    assert rerun.chunks_waiting == 0
    assert rerun.frames_skipped == 8


def test_single_frame_chunk_passes_through(tmp_path: Path) -> None:
    _make_segment(tmp_path, frames=1)
    upscale_result = upscale_poll_once(
        tmp_path,
        weights_path=Path("/models/realesrgan/realesr-animevideov3.pth"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=4,
        decode_fn=_stub_decode,
        upscale_fn=_stub_upscale,
    )
    assert upscale_result.chunks_done == 1
    result = interp_poll_once(tmp_path, **_interp_kwargs())
    assert result.chunks_done == 1
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    records = load_chunk_ledger(plan_dir / "chunks.jsonl")
    interp_records = [record for record in records if record["stage"] == STAGE_INTERPOLATED]
    assert len(interp_records) == 1
    assert interp_records[0]["expected_frames"] == 1


def test_interp_result_counts_source_frames_not_output_frames(tmp_path: Path) -> None:
    """Interp frame counts stay in source frames (multiplier excluded)."""
    from voyage.augment_upscale_poller import upscale_poll_once

    _make_segment(tmp_path, frames=10)
    upscale_poll_once(
        tmp_path,
        weights_path=Path("/models/realesrgan/realesr-animevideov3.pth"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=4,
        decode_fn=_stub_decode,
        upscale_fn=_stub_upscale,
    )
    seen: list[tuple[str, int]] = []

    def _collect(segment_id: str, frames: int) -> None:
        seen.append((segment_id, frames))

    result = interp_poll_once(tmp_path, **_interp_kwargs(on_chunk_frames=_collect))
    # 10 source frames over chunk_frames=4 → 4/4/2 in, never the ×4 outputs.
    assert result.chunks_done == 3
    assert result.frames_done == 10
    assert seen == [("000000", 4), ("000000", 4), ("000000", 2)]
    rerun = interp_poll_once(tmp_path, **_interp_kwargs())
    assert rerun.frames_done == 0
    assert rerun.frames_skipped == 10
