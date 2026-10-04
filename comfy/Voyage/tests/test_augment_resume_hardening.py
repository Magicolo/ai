"""Resume hardening: torn ledgers + ledger-intact output loss (TDD, CPU-only).

Covers the interruption/resume/crash/corruption contract the user asked
for: the augment sidecar must always pick up where it left off, even when
the crash lands between operations the existing tests never simulate
(ledger record intact, output gone).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voyage.augment_interp_poller import interp_poll_once
from voyage.augment_sidecar import load_chunk_ledger
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


def _upscale_kwargs(**overrides: Any) -> Any:
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


def _interp_kwargs(**overrides: Any) -> Any:
    params = {
        "weights_path": Path("/models/frame_interpolation/film_net_fp16.safetensors"),
        "weights_key": "weights-abc",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "chunk_frames": 4,
        "multiplier": 4,
        "interp_fn": _stub_interp,
    }
    params.update(overrides)
    return params


def test_torn_ledger_tail_is_skipped(tmp_path: Path) -> None:
    _make_segment(tmp_path, frames=4)
    first = upscale_poll_once(tmp_path, **_upscale_kwargs())
    assert first.chunks_done == 1
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    ledger = plan_dir / "chunks.jsonl"
    before = load_chunk_ledger(ledger)
    assert len(before) == 1
    # Simulate a crash mid-append: torn last line, no trailing newline.
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write('{"chunk_index": 0, "stage": "upscal')
    records = load_chunk_ledger(ledger)
    assert len(records) == 1
    assert records[0]["stage"] == "upscaled"


def test_upscale_rerenders_when_output_deleted_but_ledger_intact(tmp_path: Path) -> None:
    _make_segment(tmp_path, frames=8)
    first = upscale_poll_once(tmp_path, **_upscale_kwargs())
    assert first.chunks_done == 2
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    doomed = plan_dir / "upscaled_01"
    for frame in doomed.glob("*.png"):
        frame.unlink()
    doomed.rmdir()
    assert not doomed.exists()
    rerun = upscale_poll_once(tmp_path, **_upscale_kwargs())
    assert rerun.chunks_done == 1
    assert rerun.chunks_skipped == 1
    assert len(list(doomed.glob("frame_*.png"))) == 4


def test_upscale_rerenders_when_output_incomplete_but_ledger_intact(tmp_path: Path) -> None:
    _make_segment(tmp_path, frames=4)
    first = upscale_poll_once(tmp_path, **_upscale_kwargs())
    assert first.chunks_done == 1
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    output_dir = plan_dir / "upscaled_00"
    frames = sorted(output_dir.glob("frame_*.png"))
    assert len(frames) == 4
    for extra in frames[2:]:
        extra.unlink()
    assert len(list(output_dir.glob("frame_*.png"))) == 2
    rerun = upscale_poll_once(tmp_path, **_upscale_kwargs())
    assert rerun.chunks_done == 1
    assert len(list(output_dir.glob("frame_*.png"))) == 4


def test_interp_rerenders_when_output_deleted_but_ledger_intact(tmp_path: Path) -> None:
    _make_segment(tmp_path, frames=8)
    assert upscale_poll_once(tmp_path, **_upscale_kwargs()).chunks_done == 2
    assert interp_poll_once(tmp_path, **_interp_kwargs()).chunks_done == 2
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    doomed = plan_dir / "interpolated_01"
    for frame in doomed.glob("*.png"):
        frame.unlink()
    doomed.rmdir()
    rerun = interp_poll_once(tmp_path, **_interp_kwargs())
    assert rerun.chunks_done == 1
    assert rerun.chunks_skipped == 1
    assert len(list(doomed.glob("frame_*.png"))) == 13


def test_interp_waits_when_upscaled_output_deleted_but_ledger_intact(
    tmp_path: Path,
) -> None:
    _make_segment(tmp_path, frames=8)
    assert upscale_poll_once(tmp_path, **_upscale_kwargs()).chunks_done == 2
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    doomed = plan_dir / "upscaled_01"
    for frame in doomed.glob("*.png"):
        frame.unlink()
    doomed.rmdir()
    # Must not crash fail-loud: the healthy chunk proceeds while the
    # healed chunk waits for the upscale poller, which then heals it.
    waiting = interp_poll_once(tmp_path, **_interp_kwargs())
    assert waiting.chunks_done == 1
    assert waiting.chunks_waiting >= 1
    healed = upscale_poll_once(tmp_path, **_upscale_kwargs())
    assert healed.chunks_done == 1
    resumed = interp_poll_once(tmp_path, **_interp_kwargs())
    assert resumed.chunks_done == 1
