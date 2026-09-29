"""Issue 028 tensor handoff: pure tail helpers + shape gates (slim-testable).

The in-memory inter-block handoff (tensor instead of lossy mp4 roundtrip)
bottles through `validate_tail_length` (fail-loud short anchors) and
`tail_frames_for_conditioning` ((T, H, W, C) uint8 shape gates). Both are
GPU-free pure functions — covered here without a session.
"""

from __future__ import annotations

import numpy as np
import pytest

from voyage.workers.video_ltxv import (
    CONDITIONING_TAIL_FRAMES,
    SEGMENT_TARGET_FRAMES,
    _tail_clip_to_handoff_frames,
    split_prefix_novel,
    tail_frames_for_conditioning,
    validate_tail_length,
)


def test_validate_tail_length_accepts_exact() -> None:
    assert validate_tail_length(25, 25) == 25


def test_validate_tail_length_rejects_short_anchor() -> None:
    with pytest.raises(ValueError, match="refusing a short anchor"):
        validate_tail_length(24, 25)


def test_validate_tail_length_rejects_long_anchor() -> None:
    with pytest.raises(ValueError, match="refusing a short anchor"):
        validate_tail_length(26, 25)


def test_conditioning_tail_constants_match_stream_a() -> None:
    assert SEGMENT_TARGET_FRAMES == 121
    assert CONDITIONING_TAIL_FRAMES == 25
    prefix, novel = split_prefix_novel(SEGMENT_TARGET_FRAMES, CONDITIONING_TAIL_FRAMES)
    assert (prefix, novel) == (25, 96)


def test_tail_frames_accepts_conditioning_tail() -> None:
    frames = np.zeros((25, 512, 768, 3), dtype=np.uint8)
    assert tail_frames_for_conditioning(frames) is frames


def test_tail_frames_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\(T, H, W, 3\)"):
        tail_frames_for_conditioning(np.zeros((512, 768, 3), dtype=np.uint8))


def test_tail_frames_rejects_wrong_channels() -> None:
    with pytest.raises(ValueError, match=r"\(T, H, W, 3\)"):
        tail_frames_for_conditioning(np.zeros((25, 512, 768, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"\(T, H, W, 3\)"):
        tail_frames_for_conditioning(np.zeros((25, 512, 768, 1), dtype=np.uint8))


def test_tail_frames_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        tail_frames_for_conditioning(np.zeros((0, 512, 768, 3), dtype=np.uint8))


def test_tail_frames_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="must be uint8"):
        tail_frames_for_conditioning(np.zeros((25, 512, 768, 3), dtype=np.float32))


class _FakeTailTensor:
    """Torch-style (B,C,T,H,W) clip: permute/float/cpu/numpy chain like _save_mp4."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array, dtype=np.float32)
        self.shape = self._array.shape

    def dim(self) -> int:
        return len(self.shape)

    def permute(self, *order: int) -> _FakeTailTensor:
        return _FakeTailTensor(np.transpose(self._array, order))

    def float(self) -> _FakeTailTensor:
        return self

    def cpu(self) -> _FakeTailTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def __getitem__(self, key: object) -> _FakeTailTensor:
        return _FakeTailTensor(self._array[key])  # type: ignore[index]


def test_handoff_helper_bottles_channel_first_tail() -> None:
    clip = _FakeTailTensor(np.full((1, 3, 4, 2, 2), 0.5, dtype=np.float32))
    frames = _tail_clip_to_handoff_frames(clip)
    assert frames is not None
    assert frames.shape == (4, 2, 2, 3)
    assert frames.dtype == np.uint8
    assert int(frames.min()) == 127
    assert int(frames.max()) == 127


def test_handoff_helper_falls_back_on_stub_tensors() -> None:
    """Stubs without the torch accessor chain (e.g. _FakeBlockTensor doubles)
    bottle nothing — the next block falls back to the chain mp4 path."""

    class _StubTail:
        shape = (1, 3, 25, 8, 8)

        def __getitem__(self, key: object) -> _StubTail:
            del key
            return self

    assert _tail_clip_to_handoff_frames(_StubTail()) is None
