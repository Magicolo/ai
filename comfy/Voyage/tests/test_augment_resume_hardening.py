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

import pytest

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


def _craft_png(width: int, height: int) -> bytes:
    """Minimal valid PNG (stdlib only) so geometry tests run without torch."""
    import struct
    import zlib

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        body = struct.pack(">I", len(data)) + ctype + data
        return body + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + b"\x00" * (width * 3)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(scanline * height))
        + _chunk(b"IEND", b"")
    )


def test_chunk_frames_match_size_accepts_uniform(tmp_path: Path) -> None:
    from voyage.augment import chunk_frames_match_size

    out_dir = tmp_path / "upscaled_00"
    out_dir.mkdir()
    for position in range(3):
        (out_dir / f"frame_{position + 1:06d}.png").write_bytes(_craft_png(8, 6))
    assert chunk_frames_match_size(out_dir, (8, 6)) is True


def test_chunk_frames_match_size_rejects_mixed(tmp_path: Path) -> None:
    pytest.importorskip("PIL", reason="geometry gate needs Pillow (absent from slim)")
    from voyage.augment import chunk_frames_match_size

    out_dir = tmp_path / "upscaled_01"
    out_dir.mkdir()
    (out_dir / "frame_000001.png").write_bytes(_craft_png(8, 6))
    (out_dir / "frame_000002.png").write_bytes(_craft_png(4, 3))
    assert chunk_frames_match_size(out_dir, (8, 6)) is False


def test_upscale_rerenders_when_output_geometry_wrong_but_ledger_intact(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL", reason="geometry gate needs Pillow (absent from slim)")

    def _sized_upscale(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for position in range(len(frame_paths)):
            frame = dest_dir / f"frame_{position + 1:06d}.png"
            frame.write_bytes(_craft_png(8, 6))
            written.append(frame)
        return written

    _make_segment(tmp_path, frames=8)
    kwargs = _upscale_kwargs(out_width=8, out_height=6, upscale_fn=_sized_upscale)
    first = upscale_poll_once(tmp_path, **kwargs)
    assert first.chunks_done == 2
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    (plan_dir / "upscaled_01" / "frame_000001.png").write_bytes(_craft_png(4, 3))
    healed = upscale_poll_once(tmp_path, **kwargs)
    assert healed.chunks_done == 1
    assert healed.chunks_skipped == 1


def _heal_sized_upscale(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
    """Write real 8x6 PNGs (record geometry matches files via 8x6 overrides)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for position in range(len(frame_paths)):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(_craft_png(8, 6))
        written.append(frame)
    return written


def test_heal_strips_count_missing_records_and_reheals(tmp_path: Path) -> None:
    """Kaolin case: deleted output dir + intact ledger → heal → clean → re-render."""
    import shutil

    from voyage.augment_sidecar import heal_augment_ledgers
    from voyage.cli_validate import _check_sidecar_plan_consistency

    _make_segment(tmp_path, frames=8)
    kwargs = _upscale_kwargs(out_width=8, out_height=6, upscale_fn=_heal_sized_upscale)
    first = upscale_poll_once(tmp_path, **kwargs)
    assert first.chunks_done == 2
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    shutil.rmtree(plan_dir / "upscaled_01")
    assert _check_sidecar_plan_consistency(tmp_path) != []
    stripped = heal_augment_ledgers(tmp_path)
    assert stripped >= 1
    assert _check_sidecar_plan_consistency(tmp_path) == []
    healed = upscale_poll_once(tmp_path, **kwargs)
    assert healed.chunks_done == 1


def test_heal_keeps_healthy_outputs(tmp_path: Path) -> None:
    """Uniform outputs at record geometry are never stripped."""
    from voyage.augment_sidecar import heal_augment_ledgers
    from voyage.cli_validate import _check_sidecar_plan_consistency

    _make_segment(tmp_path, frames=8)
    kwargs = _upscale_kwargs(out_width=8, out_height=6, upscale_fn=_heal_sized_upscale)
    first = upscale_poll_once(tmp_path, **kwargs)
    assert first.chunks_done == 2
    assert heal_augment_ledgers(tmp_path) == 0
    assert _check_sidecar_plan_consistency(tmp_path) == []


def test_heal_strips_geometry_mixed_outputs(tmp_path: Path) -> None:
    """Count-complete but geometry-mixed chunk → heal strips → re-render heals."""
    pytest.importorskip("PIL", reason="geometry gate needs Pillow (absent from slim)")

    from voyage.augment_sidecar import heal_augment_ledgers
    from voyage.cli_validate import _check_sidecar_plan_consistency

    _make_segment(tmp_path, frames=8)
    kwargs = _upscale_kwargs(out_width=8, out_height=6, upscale_fn=_heal_sized_upscale)
    first = upscale_poll_once(tmp_path, **kwargs)
    assert first.chunks_done == 2
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    (plan_dir / "upscaled_01" / "frame_000001.png").write_bytes(_craft_png(4, 3))
    stripped = heal_augment_ledgers(tmp_path)
    assert stripped >= 1
    assert _check_sidecar_plan_consistency(tmp_path) == []
    healed = upscale_poll_once(tmp_path, **kwargs)
    assert healed.chunks_done == 1
