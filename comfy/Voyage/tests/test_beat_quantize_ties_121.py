"""Issue 121 fixed: `quantize_take_seconds` never plans short (ceil).

Ceil tie rule: the snap uses `math.ceil`, so exact-half ratios round UP
(`45 s / 18 s = 2.5 → 3 → 54 s`). Over-coverage is trimmed by the
finalize slice walk; shortfall (the old round-half-to-even wart) chained
extra takes with no diagnostic. CPU-only.
"""

from __future__ import annotations

import pytest

from voyage.audio.beat import quantize_take_seconds


def test_exact_multiple_unaffected() -> None:
    assert quantize_take_seconds(45.0, 15.0) == pytest.approx(45.0)
    assert quantize_take_seconds(4.0, 2.0) == pytest.approx(4.0)


def test_minimum_is_one_segment() -> None:
    assert quantize_take_seconds(1.0, 4.0) == pytest.approx(4.0)


def test_exact_half_ratio_ceils_up() -> None:
    """Issue 121 fix: 45/18 = 2.5 ceils to 3 → 54.0 s, never short."""
    assert quantize_take_seconds(45.0, 18.0) == pytest.approx(54.0)


def test_exact_half_ratio_other_direction_ceils_up() -> None:
    """3.5 ceils to 4 (lengthens, benign — same value, new rule)."""
    assert quantize_take_seconds(45.0, 45.0 / 3.5) == pytest.approx(4 * (45.0 / 3.5))


def test_quantized_take_never_plans_short() -> None:
    """No-shorten invariant: quantized take covers the requested length."""
    assert quantize_take_seconds(45.0, 18.0) >= 45.0
    assert quantize_take_seconds(45.0, 2.0) >= 45.0
    assert quantize_take_seconds(45.0, 4.0) >= 45.0
