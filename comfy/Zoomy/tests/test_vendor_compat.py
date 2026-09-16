"""Tests for the third-party compatibility shims."""

from __future__ import annotations

import sys
from pathlib import Path

from zoomy import vendor_compat


def test_compat_is_idempotent() -> None:
    """Repeated calls settle the patch set without failing."""
    vendor_compat.apply_transformers5_compat()
    before = set(vendor_compat._APPLIED_PATCHES)  # noqa: SLF001
    vendor_compat.apply_transformers5_compat()
    assert before == vendor_compat._APPLIED_PATCHES  # noqa: SLF001


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
