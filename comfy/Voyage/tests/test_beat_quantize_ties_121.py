"""Issue 121 characterization: `quantize_take_seconds` tie behavior.

CHARACTERIZATION (not TDD-fix): this pins the current round-half-to-even
behavior, warts included, because the fix (never plan short — ceil or
round-half-up) would change the deliberately pinned expectations at
`tests/test_rhythm.py:66-67` (`(45, 2) -> 44`, `(45, 4) -> 44`, with
comments documenting the rounding) and the sole caller
(`voyage/audio/planner.py:240`) — both outside this group’s file scope.
Any future rounding change must update these pins deliberately. CPU-only.
"""

from __future__ import annotations

import pytest

from voyage.audio.beat import quantize_take_seconds


def test_exact_multiple_unaffected() -> None:
    assert quantize_take_seconds(45.0, 15.0) == pytest.approx(45.0)
    assert quantize_take_seconds(4.0, 2.0) == pytest.approx(4.0)


def test_minimum_is_one_segment() -> None:
    assert quantize_take_seconds(1.0, 4.0) == pytest.approx(4.0)


def test_exact_half_ratio_rounds_to_even_down() -> None:
    """Known wart (issue 121): 45/18 = 2.5 rounds to 2 → 36.0 s, 9 s short
    of the requested take. Over-coverage is trimmed downstream; shortfall
    adds chained takes. Pinned so the change is deliberate, not silent."""
    assert quantize_take_seconds(45.0, 18.0) == pytest.approx(36.0)


def test_exact_half_ratio_rounds_to_even_up() -> None:
    """Same wart, other direction: 3.5 rounds to 4 (lengthens, benign)."""
    assert quantize_take_seconds(45.0, 45.0 / 3.5) == pytest.approx(4 * (45.0 / 3.5))
