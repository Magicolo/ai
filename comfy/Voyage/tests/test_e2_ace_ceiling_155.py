"""ACE take durations have an upper bound like the SFX sibling (155).

CPU-only: pure validator tests. Takes are 30-60 s by design; anything
past the ceiling is a caller bug (misplaced milliseconds, benchmark
typo) that must fail in 1 ms of validation, not 10 minutes of DiT.
"""

from __future__ import annotations

import pytest

from voyage.audio.acestep import MAX_TAKE_SECONDS, validate_duration_seconds


def test_absurd_take_rejected_before_any_side_effect() -> None:
    with pytest.raises(ValueError, match="duration_seconds"):
        validate_duration_seconds(3600.0)


def test_ceiling_boundary_accepts_max_rejects_above() -> None:
    validate_duration_seconds(MAX_TAKE_SECONDS)
    with pytest.raises(ValueError, match="duration_seconds"):
        validate_duration_seconds(MAX_TAKE_SECONDS + 0.5)


def test_legitimate_takes_still_pass() -> None:
    for seconds in (1.0, 15.0, 30.0, 45.0, 60.0):
        validate_duration_seconds(seconds)


def test_bottom_half_unchanged() -> None:
    for seconds in (0.0, -3.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="duration_seconds"):
            validate_duration_seconds(seconds)


def test_sibling_sfx_validator_agrees_on_absurd() -> None:
    """Both audio validators reject the same absurd input (parity)."""
    from voyage.audio.mmaudio_sfx import (
        validate_duration_seconds as validate_sfx_duration,
    )

    with pytest.raises(ValueError, match="duration_seconds"):
        validate_sfx_duration(3600.0)
