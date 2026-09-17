"""Tests for the third-party compatibility shims."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from zoomy import vendor_compat

if TYPE_CHECKING:
    import pytest


def test_compat_is_idempotent() -> None:
    """Repeated calls settle the patch set without failing."""
    vendor_compat.apply_transformers5_compat()
    before = set(vendor_compat._APPLIED_PATCHES)  # noqa: SLF001
    vendor_compat.apply_transformers5_compat()
    assert before == vendor_compat._APPLIED_PATCHES  # noqa: SLF001


def _pristine_init_context(*_args: object, **_kwargs: object) -> list[object]:
    """Stand-in for the unpatched init context (wrapped, never invoked)."""
    return []


class _PristineContextDescriptor:
    """Class attribute standing in for the unpatched init context."""

    def __get__(self, _instance: object, _owner: type | None = None) -> object:
        return _pristine_init_context


class _SlowAttributeAccess(type):
    """Stretch every attribute fetch so storming threads overlap mid-window."""

    def __getattribute__(cls, name: str) -> object:
        time.sleep(0.001)
        return super().__getattribute__(name)


class _FakePreTrainedModel(metaclass=_SlowAttributeAccess):
    """Minimal stand-in for transformers' model base (patch point only)."""

    get_init_context = _PristineContextDescriptor()

    def _move_missing_keys_from_meta_to_device(self, *_args: object, **_kwargs: object) -> None:
        """Never invoked in tests; only its presence is patched over."""


def _stub_transformer_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the transformers/torch imports at fakes (never the real stack)."""
    monkeypatch.setitem(sys.modules, "transformers", _FakeTransformersModule())
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))


class _FakeTransformersModule(ModuleType):
    """Module stand-in exposing a transformers-5 version and a model base."""

    def __init__(self) -> None:
        super().__init__("transformers")
        self.__version__ = "5.0.0"
        self.PreTrainedModel = _FakePreTrainedModel


def _wrap_layers(model_class: type[_FakePreTrainedModel]) -> int:
    """Count how many compat wrappers coat the init context."""
    current: object = model_class.__dict__["get_init_context"]
    if isinstance(current, classmethod):
        current = current.__func__
    layers = 0
    while getattr(current, "__zoomy_no_meta__", False):
        layers += 1
        closure = getattr(current, "__closure__", None) or ()
        wrapped = [cell.cell_contents for cell in closure if callable(cell.cell_contents)]
        if not wrapped:
            break
        current = wrapped[0]
    return layers


def _apply_once(start_barrier: threading.Barrier, errors: list[BaseException]) -> None:
    """Release on the start barrier, apply compat once, record any failure."""
    try:
        start_barrier.wait(timeout=10)
        vendor_compat.apply_transformers5_compat()
    except BaseException as failure:  # noqa: BLE001
        errors.append(failure)


def test_concurrent_first_calls_wrap_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Racing first calls must serialize: one wrapper layer, no exception.

    Statistical by nature (it needs threads inside the check-then-act
    window), so every round storms the window: a start barrier releases all
    threads at once, and the fake model stretches each attribute fetch to a
    millisecond, forcing captures to land after rival installs. The fixed
    code is deterministic — the lock serializes every round — only the
    pre-fix red relies on overlap, which the storm makes near-certain.
    """
    _stub_transformer_modules(monkeypatch)
    monkeypatch.setattr(vendor_compat, "_APPLIED_PATCHES", set())
    problems: list[str] = []
    for round_number in range(60):
        vendor_compat._APPLIED_PATCHES.clear()  # noqa: SLF001
        _FakePreTrainedModel.get_init_context = _PristineContextDescriptor()
        errors: list[BaseException] = []
        start_barrier = threading.Barrier(9)
        threads = [
            threading.Thread(target=_apply_once, args=(start_barrier, errors)) for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        start_barrier.wait(timeout=10)
        for thread in threads:
            thread.join()
        problems.extend(f"round {round_number}: raised {error!r}" for error in errors)
        layers = _wrap_layers(_FakePreTrainedModel)
        if layers != 1:
            problems.append(f"round {round_number}: {layers} wrapper layers")
    assert problems == []


def test_progress_stub_counts_steps() -> None:
    """The comfy ProgressBar stand-in tracks updates like the original."""
    vendor_directory = Path(__file__).resolve().parent.parent / "vendor"
    sys.path.insert(0, str(vendor_directory))
    try:
        from comfy.utils import ProgressBar  # noqa: PLC0415
    finally:
        sys.path.remove(str(vendor_directory))
    bar = ProgressBar(10)
    bar.update(3)
    assert bar.done == 3
    bar.update_absolute(7)
    assert bar.done == 7
    bar.update_absolute(2, total=20)
    assert (bar.done, bar.total) == (2, 20)
