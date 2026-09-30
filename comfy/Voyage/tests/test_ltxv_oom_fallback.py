"""LTXV OOM fallback: broad catch, gc-before-empty_cache, memory logging (issue 049).

CPU-only: `_generate_block` is driven through stub sessions — the first
`_run_multiscale` call raises a scripted OOM shape, the retry returns a
sentinel. The `ltx_video.inference` import inside `_generate_block` is
stubbed (the slim gates image has no ltx_video), and torch itself is faked
(the slim image has no torch either).
"""

from __future__ import annotations

import gc
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

from voyage.workers import video_ltxv


def _stub_ltx_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `ltx_video` + `ltx_video.inference` (fresh path needs padding only)."""
    parent = types.ModuleType("ltx_video")
    monkeypatch.setitem(sys.modules, "ltx_video", parent)
    inference = types.ModuleType("ltx_video.inference")
    inference.calculate_padding = lambda *args: (0, 0, 0, 0)  # type: ignore[attr-defined]
    inference.prepare_conditioning = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ltx_video.inference", inference)


def _record_gc(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    """Record `gc.collect` calls (still running the real collection)."""
    real_collect = gc.collect

    def _collect(*args: Any, **kwargs: Any) -> Any:
        events.append("gc.collect")
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(gc, "collect", _collect)


def _stub_session(
    fail_factory: Callable[[Any], BaseException],
    *,
    already_fallback: bool = False,
    cuda_available: bool = True,
) -> tuple[Any, list[str], Any]:
    """Stub LTXV session: scripted first-call failure, sentinel on retry.

    `fail_factory` builds the first-call failure from the fake torch
    namespace, so torch-shaped failures are the same class object the
    worker's `except` matches (separately-created same-named classes would
    never match — the pre-fix suite caught that artifact, not the worker).
    """
    events: list[str] = []
    sentinel = object()
    oom_cls = type("OutOfMemoryError", (RuntimeError,), {})
    calls = {"run": 0}

    def _allocated() -> int:
        events.append("cuda.memory_allocated")
        return 3 * 1024**3

    def _reserved() -> int:
        events.append("cuda.memory_reserved")
        return 4 * 1024**3

    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        empty_cache=lambda: events.append("cuda.empty_cache"),
        synchronize=lambda: events.append("cuda.synchronize"),
        memory_allocated=_allocated,
        memory_reserved=_reserved,
    )

    class _Generator:
        def manual_seed(self, seed: int) -> _Generator:
            events.append(f"generator.manual_seed({seed})")
            return self

    torch_ns = types.SimpleNamespace(
        OutOfMemoryError=oom_cls,
        Generator=lambda device: _Generator(),
        cuda=cuda,
    )

    def _run_multiscale(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        calls["run"] += 1
        if calls["run"] == 1:
            raise fail_factory(torch_ns)
        return sentinel

    session = types.SimpleNamespace()
    session._torch = torch_ns
    session._device = "cuda:0"
    session._negative = ("neg-embeds", "neg-mask")
    session._encode = lambda text: (f"embeds:{text}", f"mask:{text}")
    session._run_multiscale = _run_multiscale
    session._fp8_fallback = already_fallback

    def _quantize() -> None:
        # Mirror the real method: record the call and arm the fallback flag.
        events.append("quantize")
        session._fp8_fallback = True

    session._quantize_fp8_fallback = _quantize
    return session, events, sentinel


def _generate(session: Any) -> Any:
    """One fresh (unconditioned) block through the real `_generate_block`."""
    return video_ltxv.LTXVSession._generate_block(
        session, "amber dunes", 7, 768, 512, 121, 24, None
    )


def test_is_oom_matches_class_and_message_shapes() -> None:
    oom_cls = type("OutOfMemoryError", (RuntimeError,), {})
    assert video_ltxv.is_oom(oom_cls("alloc failed"))
    assert video_ltxv.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB."))
    assert video_ltxv.is_oom(RuntimeError("cuda Out Of Memory"))
    assert not video_ltxv.is_oom(RuntimeError("boom"))
    assert not video_ltxv.is_oom(ValueError("bad shape"))


def test_runtime_error_oom_triggers_fp8_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError-shaped OOMs must reach the fp8 fallback (the 049 gap)."""
    _stub_ltx_inference(monkeypatch)
    session, events, sentinel = _stub_session(
        lambda _torch: RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB.")
    )
    assert _generate(session) is sentinel
    assert session._fp8_fallback is True
    assert "quantize" in events


def test_torch_oom_still_triggers_fp8_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-existing torch-OOM path keeps working (regression guard)."""
    _stub_ltx_inference(monkeypatch)
    session, events, sentinel = _stub_session(
        lambda torch_ns: torch_ns.OutOfMemoryError("CUDA out of memory")
    )
    assert _generate(session) is sentinel
    assert session._fp8_fallback is True
    assert "quantize" in events


def test_non_oom_runtime_error_propagates_without_quantize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The broadened catch must not swallow genuine failures."""
    _stub_ltx_inference(monkeypatch)
    session, events, _sentinel = _stub_session(lambda _torch: RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _generate(session)
    assert "quantize" not in events
    assert session._fp8_fallback is False


def test_armed_fallback_reraises_oom_without_requantize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One fallback only: the second OOM propagates (both shapes)."""
    _stub_ltx_inference(monkeypatch)
    session, events, _sentinel = _stub_session(
        lambda _torch: RuntimeError("CUDA out of memory. Tried to allocate 1 GiB."),
        already_fallback=True,
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        _generate(session)
    assert "quantize" not in events


def test_fallback_collects_gc_before_empty_cache_and_logs_memory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """gc runs before empty_cache; synchronize + memory figures are logged."""
    _stub_ltx_inference(monkeypatch)
    session, events, sentinel = _stub_session(
        lambda torch_ns: torch_ns.OutOfMemoryError("CUDA out of memory")
    )
    _record_gc(monkeypatch, events)
    assert _generate(session) is sentinel
    assert events.index("gc.collect") < events.index("cuda.empty_cache")
    assert events.index("cuda.empty_cache") < events.index("cuda.synchronize")
    assert events.index("cuda.synchronize") < events.index("quantize")
    assert "cuda.memory_allocated" in events
    assert "cuda.memory_reserved" in events
    captured = capsys.readouterr()
    assert "fp8" in captured.err.lower()
    assert "gib" in captured.err.lower()
