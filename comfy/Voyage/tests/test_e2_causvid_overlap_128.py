"""CausVid tail-slice re-encode never builds an empty window (128).

CPU-only: `_vae_encode_slice` is exercised with a numpy-backed fake
video tensor (basic slicing semantics are identical in torch here) and
a stub VAE wrapper — no GPU, no upstream checkout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from voyage.workers.video_causvid import (
    _vae_encode_slice,
    reencode_window_frames,
)


class _FakeTensor:
    """Minimal video-tensor double: slicing + scale + layout + dtype."""

    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        self._array = array
        self.device = "cpu"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._array.shape)

    def __getitem__(self, key: Any) -> _FakeTensor:
        return _FakeTensor(np.asarray(self._array[key]))

    def __mul__(self, other: float) -> _FakeTensor:
        return _FakeTensor(self._array * other)

    def __sub__(self, other: float) -> _FakeTensor:
        return _FakeTensor(self._array - other)

    def __rtruediv__(self, other: float) -> _FakeTensor:
        return _FakeTensor(other / self._array)

    def transpose(self, dim0: int, dim1: int) -> _FakeTensor:
        axes = list(range(self._array.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return _FakeTensor(np.transpose(self._array, axes))

    def to(self, dtype: Any, **kwargs: Any) -> _FakeTensor:
        del dtype, kwargs
        return self


class _FakeVaeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_window_mean: float | None = None

    def encode(self, tensor: _FakeTensor, scale: Any) -> _FakeTensor:
        del scale
        self.calls += 1
        assert tensor.shape[1] > 0, "VAE must never receive an empty window"
        self.last_window_mean = float(tensor._array.mean())
        return _FakeTensor(np.zeros((1, 16, 1, 4, 4), dtype=np.float32))


class _FakeVae:
    def __init__(self) -> None:
        self.mean = _FakeTensor(np.zeros(16, dtype=np.float32))
        self.std = _FakeTensor(np.ones(16, dtype=np.float32))
        self.model = _FakeVaeModel()
        self.device = "cpu"


def _video(frames: int = 81) -> _FakeTensor:
    return _FakeTensor(np.zeros((1, frames, 16, 4, 4), dtype=np.float32))


def test_overlap_one_slice_is_the_last_frame_not_empty() -> None:
    """128: `video[:, -1:0, :]` is empty — the slice must take `[:, -1:, :]`.

    Marker check: the window carries the last frame's values.
    """
    video = _video()
    video._array[0, -1] = 7.0
    vae = _FakeVae()
    _vae_encode_slice(vae, video, 1, "bfloat16")
    assert vae.model.calls == 1
    # Scaled [0,1] → [-1,1]: 7.0 → 13.0 proves the last frame was taken.
    assert vae.model.last_window_mean == pytest.approx(13.0)


def test_overlap_one_agrees_with_window_size_accounting() -> None:
    """The slice and `reencode_window_frames` agree a 1-frame window exists."""
    assert reencode_window_frames(1) == 1


def test_overlap_three_slice_unchanged() -> None:
    """The common path still encodes exactly one latent frame."""
    vae = _FakeVae()
    _vae_encode_slice(vae, _video(), 3, "bfloat16")
    assert vae.model.calls == 1


def test_non_positive_overlap_rejected_at_slice() -> None:
    with pytest.raises(ValueError, match="positive"):
        _vae_encode_slice(_FakeVae(), _video(), 0, "bfloat16")
