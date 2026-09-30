"""CausVid session honors its `device` for every placement (124).

CPU-only: sessions are built via `__new__` with fake pipeline/torch
handles (the `test_causvid_worker._test_session` idiom, self-contained
here so no other group's file is touched).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from voyage.workers.video_causvid import CausvidSession
from voyage.workers.video_common import cuda_device_index as _cuda_device_index
from voyage.workers.video_common import torch_device_arg


class _FakeTensor:
    def __init__(self, array: Any) -> None:
        self._array = np.asarray(array)

    def to(self, *args: Any, **kwargs: Any) -> _FakeTensor:
        return self


class _FakeCuda:
    def empty_cache(self) -> None:
        pass


class _FakeTorch:
    bfloat16 = "bfloat16"

    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class _FakeTextEncoder:
    """T5 stub recording every `.to()` placement."""

    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str) -> _FakeTextEncoder:
        self.devices.append(device)
        return self

    def __call__(self, prompts: Any) -> dict[str, Any]:
        return {"prompt_embeds": _FakeTensor(np.zeros((1, 2), dtype=np.float32))}


class _FakePipeline:
    def __init__(self) -> None:
        self.text_encoder = _FakeTextEncoder()


def _session_on(device: str) -> tuple[CausvidSession, _FakeTextEncoder]:
    pipeline = _FakePipeline()
    session = CausvidSession.__new__(CausvidSession)
    session._torch = _FakeTorch()
    session._device = device
    session._pipeline = pipeline
    return session, pipeline.text_encoder


def test_encode_conditionals_shuttles_t5_on_session_device() -> None:
    """124: a `cuda:1` session must encode on cuda:1, never bare `cuda`."""
    session, encoder = _session_on("cuda:1")
    session._encode_conditionals(["a neon voyage"])
    assert encoder.devices[0] == "cuda:1", (
        f"T5 must move to the session device, saw {encoder.devices}"
    )
    assert encoder.devices[-1] == "cpu"
    assert "cuda" not in [device for device in encoder.devices if device != "cpu"], (
        f"bare 'cuda' (device 0) must never be used, saw {encoder.devices}"
    )


def test_encode_conditionals_cuda0_session_unchanged() -> None:
    session, encoder = _session_on("cuda:0")
    session._encode_conditionals(["a neon voyage"])
    assert encoder.devices[0] == "cuda:0"
    assert encoder.devices[-1] == "cpu"


def test_cuda_device_index_parses_session_devices() -> None:
    assert _cuda_device_index("cuda:0") == 0
    assert _cuda_device_index("cuda:1") == 1
    assert _cuda_device_index("cuda:12") == 12


def test_cuda_device_index_rejects_non_cuda() -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        _cuda_device_index("cpu")
    with pytest.raises(ValueError, match="CUDA device"):
        _cuda_device_index("cuda:x")


def test_torch_device_arg_empty_on_zero_explicit_on_one() -> None:
    """124: the default path keeps the pinned zero-arg fake contract."""
    assert torch_device_arg(0) == ()
    assert torch_device_arg(1) == (1,)
