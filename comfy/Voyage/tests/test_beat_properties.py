"""Property tests for rhythm-grid math (`voyage.audio.beat`).

Why properties here: `beats_for_segment`/`quantize_take_seconds` are pure
cores (§12) feeding segment planning — a wrong beat count compounds into
every take ledger downstream. Examples pin the known operating points
(see `test_rhythm.py`, which also carries the nan/inf rejection pins);
the invariants below must hold for every finite positive input:
BPM identity, minimal power-of-two doubling, grid alignment, and
determinism. Generators stay finite-positive (non-finite inputs are
rejected by contract — pinned deterministically at the bottom).
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis", reason="property tests need Hypothesis (pinned dev extra)")

from hypothesis import given
from hypothesis import strategies as strategies
from hypothesis.strategies import DataObject

from tests.conftest import bounded_counts
from voyage.audio.beat import MIN_BPM, beats_for_segment, quantize_take_seconds

# Segment/take durations in practice: fractions of a second to a minute.
# Lower bound stays strictly positive — zero/negative/non-finite inputs
# raise ValueError by contract (pinned below, tested in test_rhythm.py).
positive_durations = strategies.floats(min_value=1e-300, max_value=1e300, allow_nan=False)
# Grid alignment needs a float-exact ratio check, so this property draws a
# moderate range (extreme magnitudes are pinned separately below — a
# 1e300/1e-300 ratio overflows float division and would test `round(inf)`,
# not the grid).
moderate_durations = strategies.floats(min_value=0.001, max_value=1000.0, allow_nan=False)


@given(strategies.data())
def test_bpm_identity_holds_for_every_duration(data: DataObject) -> None:
    """`bpm` is always exactly `beats * 60 / segment_seconds`."""
    drawn_seconds = data.draw(positive_durations)
    drawn_base = data.draw(strategies.integers(min_value=1, max_value=64))
    drawn_floor = data.draw(strategies.floats(min_value=1.0, max_value=240.0, allow_nan=False))
    drawn_beats, drawn_bpm = beats_for_segment(drawn_seconds, drawn_base, drawn_floor)
    assert drawn_bpm == pytest.approx(drawn_beats * 60.0 / drawn_seconds)
    assert drawn_bpm >= drawn_floor


@given(strategies.data())
def test_beats_double_minimally_to_reach_the_floor(data: DataObject) -> None:
    """`beats` is the smallest doubling of `base_beats` reaching `min_bpm`."""
    drawn_seconds = data.draw(positive_durations)
    drawn_base = data.draw(strategies.integers(min_value=1, max_value=64))
    drawn_floor = data.draw(strategies.floats(min_value=1.0, max_value=240.0, allow_nan=False))
    drawn_beats, _ = beats_for_segment(drawn_seconds, drawn_base, drawn_floor)
    assert drawn_beats >= drawn_base
    assert (drawn_beats // drawn_base).bit_count() == 1
    if drawn_beats > drawn_base:
        halved_bpm = (drawn_beats // 2) * 60.0 / drawn_seconds
        assert halved_bpm < drawn_floor


@given(strategies.data())
def test_beat_math_is_deterministic(data: DataObject) -> None:
    drawn_seconds = data.draw(positive_durations)
    drawn_base = data.draw(strategies.integers(min_value=1, max_value=64))
    first = beats_for_segment(drawn_seconds, drawn_base)
    second = beats_for_segment(drawn_seconds, drawn_base)
    assert first == second
    assert quantize_take_seconds(drawn_seconds, 2.0) == quantize_take_seconds(drawn_seconds, 2.0)


@given(strategies.data())
def test_quantized_take_lands_on_the_segment_grid(data: DataObject) -> None:
    """The take is a whole number of segments (at least one)."""
    drawn_take = data.draw(moderate_durations)
    drawn_segment = data.draw(moderate_durations)
    quantized = quantize_take_seconds(drawn_take, drawn_segment)
    assert quantized >= drawn_segment
    assert quantized / drawn_segment == pytest.approx(round(quantized / drawn_segment))


@given(strategies.data())
def test_quantized_take_never_below_one_segment(data: DataObject) -> None:
    drawn_count = data.draw(bounded_counts)
    quantized = quantize_take_seconds(1.0 + drawn_count, 4.0)
    assert quantized >= 4.0


def test_negative_zero_durations_rejected_like_zero() -> None:
    """Pin: `-0.0 <= 0` is true, so it must raise exactly like `0.0`."""
    with pytest.raises(ValueError):
        beats_for_segment(-0.0, 4)
    with pytest.raises(ValueError):
        quantize_take_seconds(-0.0, 4.0)
    with pytest.raises(ValueError):
        quantize_take_seconds(45.0, -0.0)


def test_extreme_finite_durations_stay_in_contract() -> None:
    """Pin: denormal/tiny inputs are valid (positive); huge ones terminate."""
    tiny_beats, tiny_bpm = beats_for_segment(1e-300, 4)
    assert tiny_beats == 4
    assert tiny_bpm == pytest.approx(4 * 60.0 / 1e-300)
    huge_beats, huge_bpm = beats_for_segment(1e300, 4)
    assert huge_beats >= 4
    assert huge_bpm >= MIN_BPM
    assert quantize_take_seconds(1e-300, 1e-300) == pytest.approx(1e-300)


def test_negative_base_beats_rejected() -> None:
    """Pin: the `0` case lives in test_rhythm.py; negatives raise too."""
    with pytest.raises(ValueError):
        beats_for_segment(2.0, -4)
