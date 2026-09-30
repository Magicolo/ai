"""LongLive VAE-decode uint8 conversion clips highlight overshoot (125).

The conversion contract all three video backends share (issue 065):
VAE output in nominal [0, 1] overshoots on highlights, and a bare
float→uint8 cast on CPU wraps modulo 256 (1.01 × 255 = 257.55 → ~1,
near-black). The numpy half runs everywhere; the torch half is gated
on an installed torch (absent from the slim gates image — it runs on a
GPU box / the video image, CPU tensors only, no CUDA needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from voyage.workers.video_common import clip_array_to_uint8
from voyage.workers.video_longlive import clamp_to_uint8


def test_bare_astype_wraps_overshoot_to_near_black() -> None:
    """Pin the failure mechanism: bare `.astype(uint8)` wraps 257 → ~1."""
    overshoot = np.array([1.01], dtype=np.float64)
    assert int((overshoot * 255.0).astype("uint8")[0]) < 10


def test_clip_array_to_uint8_saturates_highlights() -> None:
    """The shared helper (ltxv path) saturates instead of wrapping."""
    frames = np.array([0.0, 0.5, 1.0, 1.01, 1.05, 2.0], dtype=np.float64)
    converted = clip_array_to_uint8(frames)
    assert converted.dtype == np.uint8
    assert list(converted) == [0, 127, 255, 255, 255, 255]


def test_clip_array_to_uint8_floors_undershoot() -> None:
    frames = np.array([-0.02, -1.0, 0.0], dtype=np.float64)
    assert list(clip_array_to_uint8(frames)) == [0, 0, 0]


def test_torch_cpu_cast_clamped_saturates() -> None:
    """125: the longlive decode helper clamps before `.to(torch.uint8)`."""
    torch = pytest.importorskip("torch", reason="torch absent from slim image")
    scaled = torch.tensor([0.0, 127.5, 255.0, 257.55, 300.0])
    assert clamp_to_uint8(scaled, torch.uint8).tolist() == [0, 127, 255, 255, 255]


def test_clamp_helper_preserves_dtype_and_negatives() -> None:
    torch = pytest.importorskip("torch", reason="torch absent from slim image")
    converted = clamp_to_uint8(torch.tensor([-5.0, 0.0, 100.7]), torch.uint8)
    assert str(converted.dtype) == "torch.uint8"
    assert converted.tolist() == [0, 0, 100]


class _NoClampTensor:
    """Slim-image test-double shape: in-range values, `.to()` only."""

    def to(self, dtype: object) -> str:
        return f"converted-to-{dtype}"


def test_clamp_helper_passes_through_tensors_without_clamp() -> None:
    """Doubles carrying in-range values convert directly (no clamp_ needed)."""
    assert clamp_to_uint8(_NoClampTensor(), "uint8") == "converted-to-uint8"


def test_torch_cpu_bare_cast_wraps_without_clamp() -> None:
    """Documents why the clamp is load-bearing (CPU wrap semantics)."""
    torch = pytest.importorskip("torch", reason="torch absent from slim image")
    wrapped = torch.tensor([257.55]).to(torch.uint8).tolist()
    assert wrapped == [1], f"expected CPU wrap 257.55 → 1, got {wrapped}"
