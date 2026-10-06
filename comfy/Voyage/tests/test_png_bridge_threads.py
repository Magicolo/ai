"""Threaded PNG bridge: round-trip fidelity, order, and error semantics.

Covers the RIFE-finalize PNG-bound fix (`_PNG_IO_WORKERS` pool +
`_PNG_COMPRESS_LEVEL` fast level in `voyage.augment`): the pool must not
reorder frames, the fast level must stay lossless, and every fail-loud
contract of the serial bridge must hold (empty input, bad tensor shape
with its position, mixed geometry naming the offender).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.augment import load_png_frames_as_tensors, write_tensors_as_png_frames
from voyage.errors import MediaError

torch = pytest.importorskip("torch", reason="PNG bridge needs torch (absent from slim image)")
pytest.importorskip("PIL", reason="PNG bridge needs Pillow (absent from slim image)")


def _random_frames(count: int, width: int = 64, height: int = 48) -> list[Any]:
    return [torch.rand(3, height, width, dtype=torch.float32) for _ in range(count)]


def test_round_trip_is_pixel_exact_and_ordered(tmp_path: Path) -> None:
    frames = _random_frames(9)
    written = write_tensors_as_png_frames(frames, tmp_path / "out")
    assert [path.name for path in written] == [f"frame_{index:06d}.png" for index in range(9)]
    loaded = load_png_frames_as_tensors(written)
    assert len(loaded) == len(frames)
    for original, revived in zip(frames, loaded, strict=True):
        assert torch.equal(
            (original.clamp(0.0, 1.0).mul(255.0).round().byte()),
            (revived.clamp(0.0, 1.0).mul(255.0).round().byte()),
        )


def test_write_rejects_empty_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_tensors_as_png_frames([], tmp_path / "out")


def test_load_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        load_png_frames_as_tensors([])


def test_write_names_bad_shape_position(tmp_path: Path) -> None:
    frames: list[Any] = [torch.zeros(3, 8, 8), torch.zeros(4, 8, 8)]
    with pytest.raises(MediaError, match=r"enhanced frame 1 must be \(3, H, W\)"):
        write_tensors_as_png_frames(frames, tmp_path / "out")


def test_load_names_geometry_offender(tmp_path: Path) -> None:
    first = tmp_path / "frame_000000.png"
    second = tmp_path / "frame_000001.png"
    write_tensors_as_png_frames([torch.zeros(3, 8, 8)], tmp_path / "wide")
    narrow = write_tensors_as_png_frames([torch.zeros(3, 6, 8)], tmp_path / "narrow")
    first.write_bytes((tmp_path / "wide" / "frame_000000.png").read_bytes())
    second.write_bytes(narrow[0].read_bytes())
    with pytest.raises(MediaError, match="frame size mismatch"):
        load_png_frames_as_tensors([first, second])


def test_load_rejects_non_path() -> None:
    with pytest.raises(TypeError):
        load_png_frames_as_tensors(["frame_000000.png"])  # type: ignore[list-item]
