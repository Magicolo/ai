"""Issue 256: PNG bridge copies + RAM-bounded pool.

The load bridge dropped the `tobytes`/`frombuffer`/`.copy()` chain for a
single `numpy.asarray` + in-place divide; the save bridge fuses scale +
round in place on the fresh clamped tensor. Both stay pixel-exact, and
the pool width is additionally bounded by transient RAM.

Budget pins run torch-free (slim gates image); bridge tests need
PIL + torch (run in voyage-video).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.augment import (
    _PNG_IO_WORKERS,
    _PNG_TRANSIENT_BUDGET_BYTES,
    _png_io_workers,
    _tensor_transient_bytes,
)


def test_pool_width_caps_at_frame_count() -> None:
    """Never more workers than frames (legacy count bound preserved)."""
    assert _png_io_workers(1) == 1
    assert _png_io_workers(3) == 3
    assert _png_io_workers(100) == _PNG_IO_WORKERS


def test_pool_width_bounds_transient_ram() -> None:
    """Huge frames serialize to one worker; unknown geometry keeps legacy."""
    assert _png_io_workers(32, 0) == min(32, _PNG_IO_WORKERS)
    assert _png_io_workers(32, _PNG_TRANSIENT_BUDGET_BYTES + 1) == 1
    assert _png_io_workers(32, _PNG_TRANSIENT_BUDGET_BYTES // 4) == 4


def test_transient_bytes_counts_float_frames() -> None:
    """~2 full f32 frames per bridge frame; garbage reads as unknown."""

    class _TorchLike:
        def numel(self) -> int:
            return 12

    assert _tensor_transient_bytes(_TorchLike()) == 12 * 4 * 2

    class _NumpyLike:
        size = 12

    assert _tensor_transient_bytes(_NumpyLike()) == 12 * 4 * 2
    assert _tensor_transient_bytes(object()) == 0
    assert _tensor_transient_bytes(None) == 0


def _bridge_frames(torch: Any, count: int) -> list[Any]:
    """Deterministic gradient frames in [0, 1] plus out-of-range probes."""
    frames = []
    for offset in range(count):
        rows = torch.linspace(0.0, 1.0, 32).unsqueeze(0).expand(32, 32)
        tilt = (rows + offset / max(count, 1)) % 1.0
        frames.append(torch.stack([tilt, rows, 1.0 - rows]))
    frames.append(torch.full((3, 32, 32), 2.0))
    frames.append(torch.full((3, 32, 32), -1.0))
    return frames


def test_bridge_round_trip_is_pixel_exact(tmp_path: Path) -> None:
    """Save + load returns the exact PNG bytes (clamped to [0, 1])."""
    torch = pytest.importorskip("torch", reason="PNG bridge needs torch (absent from slim image)")
    pytest.importorskip("PIL.Image", reason="PNG bridge needs Pillow (absent from slim image)")
    from voyage.augment import load_png_frames_as_tensors, write_tensors_as_png_frames

    frames = _bridge_frames(torch, 6)
    dest = tmp_path / "frames"
    written = write_tensors_as_png_frames(frames, dest)
    assert len(written) == len(frames)
    loaded = load_png_frames_as_tensors(written)
    assert len(loaded) == len(frames)
    for original, round_tripped in zip(frames, loaded, strict=True):
        expected = original.clamp(0.0, 1.0).mul(255.0).round().byte()
        assert torch.equal(round_tripped.mul(255.0).round().byte(), expected)


def test_load_path_divides_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The decode scales with the in-place divide (no 4th transient buffer)."""
    torch = pytest.importorskip("torch", reason="PNG bridge needs torch (absent from slim image)")
    pytest.importorskip("PIL.Image", reason="PNG bridge needs Pillow")
    from voyage.augment import write_tensors_as_png_frames

    written = write_tensors_as_png_frames([torch.zeros(3, 16, 16)], tmp_path / "src")

    def _no_out_of_place_div(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("out-of-place div allocates a 4th buffer")

    monkeypatch.setattr(torch.Tensor, "div", _no_out_of_place_div)
    from voyage.augment import load_png_frames_as_tensors

    loaded = load_png_frames_as_tensors(written)
    assert len(loaded) == 1
    assert bool((loaded[0] == 0.0).all())


def test_bridge_perf_smoke(tmp_path: Path) -> None:
    """16 mid-size frames bridge well inside a generous wall bound (tripwire)."""
    import time

    torch = pytest.importorskip("torch", reason="PNG bridge needs torch (absent from slim image)")
    pytest.importorskip("PIL.Image", reason="PNG bridge needs Pillow (absent from slim image)")
    from voyage.augment import load_png_frames_as_tensors, write_tensors_as_png_frames

    frames = [torch.rand(3, 256, 256) for _ in range(16)]
    started = time.monotonic()
    written = write_tensors_as_png_frames(frames, tmp_path / "perf")
    loaded = load_png_frames_as_tensors(written)
    wall = time.monotonic() - started
    assert len(loaded) == 16
    assert wall < 120.0
