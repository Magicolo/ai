"""Tests for RIFE interpolation batching above the heavy imports.

Bisection shape and pair chaining run against a stub model, so no RIFE
weights are needed: depth 2 turns each pair into 5 frames, and pairs chain
without duplicating their shared boundary frame.
"""

from __future__ import annotations

import pytest

from zoomy.errors import EngineExecutionError
from zoomy.interpolation import INTERPOLATION_BISECTION_DEPTH, bisect, interpolate_arrays


class _StubRifeModel:
    """Bisect by averaging markers: every midpoint names its parents."""

    def inference_image_list(self, pair: list[str]) -> list[str]:
        """Return one midpoint marker for the pair."""
        first, second = pair
        return [f"mid({first},{second})"]


def test_bisection_depth_derives_the_x4_multiplier() -> None:
    """Two bisection levels double twice: the x4 multiplier, by construction."""
    assert INTERPOLATION_BISECTION_DEPTH == 2


def test_bisect_expands_one_pair_to_five_frames() -> None:
    """Depth 2 yields [a, m1, m, m2, b]: three forwards per pair."""
    model = _StubRifeModel()
    assert bisect(model, "a", "b", 2) == [
        "a",
        "mid(a,mid(a,b))",
        "mid(a,b)",
        "mid(mid(a,b),b)",
        "b",
    ]


def test_interpolate_arrays_chains_pairs_without_duplicating_joints() -> None:
    """Three sources give 9 frames: 5 + 4, the shared joint kept once."""
    model = _StubRifeModel()
    assert interpolate_arrays(model, ["a", "b", "c"], lambda: None) == [
        "a",
        "mid(a,mid(a,b))",
        "mid(a,b)",
        "mid(mid(a,b),b)",
        "b",
        "mid(b,mid(b,c))",
        "mid(b,c)",
        "mid(mid(b,c),c)",
        "c",
    ]


def test_interpolate_arrays_reports_interrupts() -> None:
    """An interrupt between pairs aborts before the next bisection."""
    from zoomy.errors import RenderInterruptedError  # noqa: PLC0415

    calls = 0

    def interrupting() -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RenderInterruptedError("stopped")

    with pytest.raises(RenderInterruptedError):
        interpolate_arrays(_StubRifeModel(), ["a", "b", "c"], interrupting)


def test_rife_load_failure_names_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing RIFE backend fails as a stage error, not an import crash."""
    import sys  # noqa: PLC0415
    from types import ModuleType  # noqa: PLC0415

    from zoomy.interpolation import load_rife_model  # noqa: PLC0415

    broken = ModuleType("ccvfi")
    monkeypatch.setitem(sys.modules, "ccvfi", broken)
    with pytest.raises(EngineExecutionError, match="RIFE model load failed"):
        load_rife_model()
