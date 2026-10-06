"""Independent upscale poller: per-segment plans + resume (CPU-only, stubbed GPU).

Pins the Phase 2 contract with fake segments and stub decode/upscale
seams: committed segments are discovered via DONE + manifest, each gets
an isolated plan dir, completed chunks are skipped on re-poll, crashed
partials are re-rendered, and unusable segments are skipped without
touching generation. No torch/GPU/ffmpeg.
"""

from __future__ import annotations

import json
from pathlib import Path

from voyage.augment_sidecar import STAGE_UPSCALED, load_chunk_ledger
from voyage.augment_upscale_poller import (
    committed_segment_sources,
    upscale_poll_once,
)


def _make_segment(
    run_dir: Path,
    segment_id: str = "000000",
    *,
    frames: int = 8,
    checksum: str = "abc123",
    with_done: bool = True,
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
    if with_done:
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
    # Same contract as the real seam: PNG paths in, PNG paths out.
    assert dest_dir.is_dir()
    return list(frame_paths)


def _poll_kwargs(**overrides):  # type: ignore[no-untyped-def]
    params = {
        "weights_path": Path("/models/realesrgan/realesr-animevideov3.pth"),
        "weights_key": "weights-abc",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "chunk_frames": 4,
        "decode_fn": _stub_decode,
        "upscale_fn": _stub_upscale,
    }
    params.update(overrides)
    return params


def test_discovers_committed_segments(tmp_path: Path) -> None:
    _make_segment(tmp_path, "000000")
    _make_segment(tmp_path, "000001", with_done=False)
    sources, skipped = committed_segment_sources(tmp_path)
    assert [source.segment_id for source in sources] == ["000000"]
    assert sources[0].total_frames == 8
    assert sources[0].source_key == "abc123"
    assert skipped == 1


def test_poll_renders_chunks_and_ledger(tmp_path: Path) -> None:
    _make_segment(tmp_path, frames=8)
    result = upscale_poll_once(tmp_path, **_poll_kwargs())
    assert result.segments_seen == 1
    assert result.chunks_done == 2
    assert result.chunks_skipped == 0
    plan_dirs = list((tmp_path / "augment").iterdir())
    assert len(plan_dirs) == 1
    records = load_chunk_ledger(plan_dirs[0] / "chunks.jsonl")
    assert len(records) == 2
    assert all(record["stage"] == STAGE_UPSCALED for record in records)
    assert sorted(record["chunk_index"] for record in records) == [0, 1]


def test_repoll_skips_completed_chunks(tmp_path: Path) -> None:
    _make_segment(tmp_path, frames=8)
    calls: list[tuple[int, int]] = []

    def counting_decode(video: Path, dest: Path, start: int, count: int) -> list[Path]:
        calls.append((start, count))
        return _stub_decode(video, dest, start, count)

    first = upscale_poll_once(tmp_path, **_poll_kwargs(decode_fn=counting_decode))
    assert first.chunks_done == 2
    second = upscale_poll_once(tmp_path, **_poll_kwargs(decode_fn=counting_decode))
    assert second.chunks_done == 0
    assert second.chunks_skipped == 2
    assert len(calls) == 2


def test_crashed_partial_is_rerendered(tmp_path: Path) -> None:
    segment_dir = _make_segment(tmp_path, frames=4)
    result = upscale_poll_once(tmp_path, **_poll_kwargs())
    assert result.chunks_done == 1
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    ledger = plan_dir / "chunks.jsonl"
    # Simulate a crash between publish and ledger append: output exists,
    # ledger record and output both removed, partial left behind.
    ledger.write_text("", encoding="utf-8")
    output_dir = plan_dir / "upscaled_00"
    for frame in output_dir.glob("*.png"):
        frame.unlink()
    output_dir.rmdir()
    crashed = plan_dir / "upscaled_00.partial"
    crashed.mkdir()
    (crashed / "frame_000001.png").write_bytes(b"junk")
    assert segment_dir.exists()
    rerun = upscale_poll_once(tmp_path, **_poll_kwargs())
    assert rerun.chunks_done == 1
    assert rerun.partials_pruned == 1
    assert len(load_chunk_ledger(ledger)) == 1


def test_unusable_segments_skipped(tmp_path: Path) -> None:
    _make_segment(tmp_path, "000000", frames=4)
    bad_dir = tmp_path / "segments" / "000001"
    bad_dir.mkdir(parents=True)
    (bad_dir / "video.mp4").write_bytes(b"fake-video")
    (bad_dir / "DONE").write_text("done\n", encoding="utf-8")
    result = upscale_poll_once(tmp_path, **_poll_kwargs())
    assert result.segments_seen == 1
    assert result.segments_skipped == 1
    assert result.chunks_done == 1


def test_poll_result_counts_source_frames(tmp_path: Path) -> None:
    """Rendered + skipped chunks report exact source-frame counts."""
    _make_segment(tmp_path, "000000", frames=10)
    seen: list[tuple[str, int]] = []

    def _collect(segment_id: str, frames: int) -> None:
        seen.append((segment_id, frames))

    result = upscale_poll_once(tmp_path, **_poll_kwargs(on_chunk_frames=_collect))
    # 10 source frames over chunk_frames=4 → windows of 4/4/2.
    assert result.chunks_done == 3
    assert result.frames_done == 10
    assert result.frames_skipped == 0
    assert seen == [("000000", 4), ("000000", 4), ("000000", 2)]
    rerun = upscale_poll_once(tmp_path, **_poll_kwargs())
    assert rerun.chunks_done == 0
    assert rerun.frames_done == 0
    assert rerun.frames_skipped == 10


def _tiny_esrgan(work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    path = work / "esrgan.pth"
    path.write_bytes(b"tiny-esrgan-weights")
    return path


def _keyed_dirs(run_dir: Path, weights_key: str) -> list[Path]:
    """Plan dirs holding at least one record with `weights_key`."""
    found = []
    for plan in sorted((run_dir / "augment").iterdir()):
        if not plan.is_dir():
            continue
        ledger = plan / "chunks.jsonl"
        if not ledger.is_file():
            continue
        if any(r.get("weights_key") == weights_key for r in load_chunk_ledger(ledger)):
            found.append(plan)
    return found


def test_adopt_upscaled_chunks_across_weights_keys(tmp_path: Path) -> None:
    """Cross-backend resume: same esrgan leg, new interp leg → adopt, no render."""
    from voyage.hashing import sha256_file

    _make_segment(tmp_path, frames=8)
    esrgan = _tiny_esrgan(tmp_path / "weights")
    esrgan_sha = sha256_file(esrgan)
    donor_key = f"donor-interp|{esrgan_sha}"
    current_key = f"current-interp|{esrgan_sha}"
    first = upscale_poll_once(tmp_path, **_poll_kwargs(weights_path=esrgan, weights_key=donor_key))
    assert first.chunks_done == 2

    def _must_not_run(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
        raise AssertionError("adopted chunks must not re-render")

    second = upscale_poll_once(
        tmp_path,
        **_poll_kwargs(weights_path=esrgan, weights_key=current_key, upscale_fn=_must_not_run),
    )
    assert second.chunks_done == 2
    assert second.chunks_skipped == 0
    plans = _keyed_dirs(tmp_path, current_key)
    assert len(plans) == 1
    for chunk in ("upscaled_00", "upscaled_01"):
        pngs = sorted((plans[0] / chunk).glob("frame_*.png"))
        assert len(pngs) == 4
        assert all(p.stat().st_size > 0 for p in pngs)


def test_incomplete_donor_falls_through_to_render(tmp_path: Path) -> None:
    """A donor missing PNGs adopts nothing for that chunk (renders instead)."""
    from voyage.hashing import sha256_file

    _make_segment(tmp_path, frames=8)
    esrgan = _tiny_esrgan(tmp_path / "weights")
    esrgan_sha = sha256_file(esrgan)
    donor_key = f"donor-interp|{esrgan_sha}"
    current_key = f"current-interp|{esrgan_sha}"
    first = upscale_poll_once(tmp_path, **_poll_kwargs(weights_path=esrgan, weights_key=donor_key))
    assert first.chunks_done == 2
    donor_plans = _keyed_dirs(tmp_path, donor_key)
    assert len(donor_plans) == 1
    victim = sorted((donor_plans[0] / "upscaled_00").glob("frame_*.png"))[0]
    victim.unlink()

    calls: list[int] = []

    def _recording_upscale(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
        calls.append(len(frame_paths))
        return _stub_upscale(frame_paths, dest_dir)

    second = upscale_poll_once(
        tmp_path,
        **_poll_kwargs(weights_path=esrgan, weights_key=current_key, upscale_fn=_recording_upscale),
    )
    assert second.chunks_done == 2
    assert calls == [4]


def test_unknown_sha_donor_is_ignored(tmp_path: Path) -> None:
    """Donor records with unparseable keys never adopt (unknown provenance)."""
    from voyage.hashing import sha256_file

    _make_segment(tmp_path, frames=8)
    first = upscale_poll_once(tmp_path, **_poll_kwargs(weights_key="weights-abc"))
    assert first.chunks_done == 2
    esrgan = _tiny_esrgan(tmp_path / "weights")
    esrgan_sha = sha256_file(esrgan)
    current_key = f"current-interp|{esrgan_sha}"

    calls: list[int] = []

    def _recording_upscale(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
        calls.append(len(frame_paths))
        return _stub_upscale(frame_paths, dest_dir)

    second = upscale_poll_once(
        tmp_path,
        **_poll_kwargs(weights_path=esrgan, weights_key=current_key, upscale_fn=_recording_upscale),
    )
    assert second.chunks_done == 2
    assert calls == [4, 4]


def test_esrgan_sha_from_weights_key_shapes() -> None:
    """Unit: both wild key shapes parse; anything else is None (never adopt)."""
    from voyage.augment_sidecar import _esrgan_sha_from_weights_key

    assert _esrgan_sha_from_weights_key("aaa|bbb") == "bbb"
    assert _esrgan_sha_from_weights_key("rife|aaa|bbb") == "bbb"
    assert _esrgan_sha_from_weights_key("film|aaa|bbb") == "bbb"
    assert _esrgan_sha_from_weights_key("weights-abc") is None
    assert _esrgan_sha_from_weights_key("") is None
    assert _esrgan_sha_from_weights_key(None) is None
    assert _esrgan_sha_from_weights_key("a|b|c|d") is None
    assert _esrgan_sha_from_weights_key("rife||bbb") is None
