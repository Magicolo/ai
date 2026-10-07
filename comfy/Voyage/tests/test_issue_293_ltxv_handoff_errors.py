"""Issue 293: LTXV tensor-handoff fallback no longer swallows OOM.

`_tail_clip_to_handoff_frames` caught everything — including
`torch.cuda.OutOfMemoryError`/`MemoryError` — and silently degraded to
the lossy chain-mp4 tail. The fallback now covers only layout-drift
shapes (`AttributeError`, `TypeError`, `ValueError`); resource failures
propagate to the worker's OOM-retry path (which cleans up chain mp4s
and re-raises).

All duck-typed (slim gates image).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from voyage.workers.video_ltxv import _tail_clip_to_handoff_frames


class _OomTailTensor:
    """Torch-style clip whose host transfer raises `MemoryError` (OOM)."""

    shape = (1, 3, 25, 8, 8)

    def __getitem__(self, key: object) -> _OomTailTensor:
        del key
        return self

    def dim(self) -> int:
        return 5

    def permute(self, *order: int) -> _OomTailTensor:
        del order
        return self

    def float(self) -> _OomTailTensor:
        return self

    def cpu(self) -> _OomTailTensor:
        raise MemoryError("CUDA out of memory: simulated pressure")

    def numpy(self) -> np.ndarray[Any, Any]:
        raise AssertionError("unreachable: cpu() must raise first")


def test_handoff_propagates_memory_error() -> None:
    """OOM on the GPU→CPU bottling is never a quiet mp4 downgrade."""
    with pytest.raises(MemoryError, match="out of memory"):
        _tail_clip_to_handoff_frames(_OomTailTensor())


class _TypeErrorTail:
    """Clip failing with a layout-drift shape (fallback territory)."""

    def __getitem__(self, key: object) -> Any:
        raise TypeError(f"bad index {key!r}")


class _ValueErrorTail:
    """Clip failing inside validation (fallback territory)."""

    shape = (1, 3, 25, 8, 8)

    def __getitem__(self, key: object) -> _ValueErrorTail:
        del key
        return self

    def dim(self) -> int:
        raise ValueError("no dim on this stub")


def test_handoff_still_falls_back_on_layout_drift() -> None:
    """Type/value failures keep today's chain-mp4 fallback (no behavior break)."""
    assert _tail_clip_to_handoff_frames(_TypeErrorTail()) is None
    assert _tail_clip_to_handoff_frames(_ValueErrorTail()) is None
