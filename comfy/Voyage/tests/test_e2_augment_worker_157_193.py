"""Augment worker-owned remainders: OOM-split cache relief + device signal (157/193).

CPU-only with stubbed `torch` (the worker imports torch function-level,
so `sys.modules` stubs intercept exactly like the real lazy import).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from voyage.workers import augment_worker


def _torch_stub(*, cuda_available: bool, calls: list[str]) -> types.ModuleType:
    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        empty_cache=lambda: calls.append("empty_cache"),
    )

    def _device(name: str) -> types.SimpleNamespace:
        resolved = "cpu" if (name.startswith("cuda") and not cuda_available) else name
        return types.SimpleNamespace(type=resolved.split(":")[0], name=resolved)

    stub = types.ModuleType("torch")
    stub.cuda = cuda  # type: ignore[attr-defined]
    stub.device = _device  # type: ignore[attr-defined]
    return stub


def test_resolve_device_fallback_warns_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """193: a planned-cuda/executed-CPU mismatch is visible exactly once."""
    monkeypatch.setattr(augment_worker, "_DEVICE_FALLBACK_WARNED", False)
    monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda_available=False, calls=[]))
    assert augment_worker._resolve_device("cuda:0").name == "cpu"
    first = capsys.readouterr().err
    assert "cuda:0" in first and "cpu" in first.lower()
    assert augment_worker._resolve_device("cuda:1").name == "cpu"
    assert capsys.readouterr().err == ""


def test_resolve_device_cuda_path_stays_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(augment_worker, "_DEVICE_FALLBACK_WARNED", False)
    monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda_available=True, calls=[]))
    assert augment_worker._resolve_device("cuda:0").name == "cuda:0"
    assert capsys.readouterr().err == ""


class _StubBatch:
    """Splittable batch double: shape + slice halving only."""

    def __init__(self, size: int) -> None:
        self._size = size
        self.shape = (size, 2, 2)

    def __getitem__(self, index: Any) -> _StubBatch:
        if isinstance(index, slice):
            return _StubBatch(len(range(*index.indices(self._size))))
        return _StubBatch(1)


class _StubOutputs:
    def __init__(self, size: int) -> None:
        self.shape = (size,)

    def __getitem__(self, index: int) -> int:
        return index


def test_run_stacked_relieves_cache_on_cpu_oom_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """157: the split path relieves the allocator even without CUDA.

    `empty_cache` is a no-op without CUDA — the `is_available` guard
    buys nothing and skips relief on CPU-OOM recursion.
    """
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda_available=False, calls=calls))
    attempts: list[int] = []

    def _forward(batch: _StubBatch) -> _StubOutputs:
        attempts.append(batch._size)
        if batch._size > 1:
            raise RuntimeError("CUDA out of memory")
        return _StubOutputs(batch._size)

    assert augment_worker._run_stacked(_forward, _StubBatch(2)) == [0, 0]
    assert attempts == [2, 1, 1]
    assert calls == ["empty_cache"]
