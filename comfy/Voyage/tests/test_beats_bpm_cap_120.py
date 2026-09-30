"""Issue 120: `beats_per_segment` has no upper bound — short segments x fine
grids imply BPMs the ACE worker rejects (`1..300`), failing takes fatally
minutes into a GPU run.

Fix: opt-in `max_bpm` ceiling in `beats_for_segment` (halve toward one
beat, else `ValueError` naming the knob) + `min_bpm > 0`. Default behavior
is unchanged (test_rhythm pins `(0.5, 4) -> (4, 480)` deliberately), so the
production call sites opt in explicitly. CPU-only.
"""

from __future__ import annotations

import pytest

from voyage.audio.beat import beats_for_segment


def test_default_has_no_ceiling() -> None:
    assert beats_for_segment(0.5, 4) == (4, 480.0)


def test_max_bpm_halves_fine_grids() -> None:
    beats, bpm = beats_for_segment(29 / 24, 8, max_bpm=300)
    assert beats == 4
    assert bpm == pytest.approx(4 * 60.0 / (29 / 24))
    assert bpm <= 300


def test_max_bpm_leaves_fitting_grids_alone() -> None:
    assert beats_for_segment(4.0, 4, max_bpm=300) == (4, 60.0)
    assert beats_for_segment(29 / 24, 4, max_bpm=300)[0] == 4


def test_impossible_grid_raises_naming_the_knob() -> None:
    with pytest.raises(ValueError, match="beats_per_segment"):
        beats_for_segment(0.1, 4, max_bpm=300)


def test_degenerate_min_bpm_rejected() -> None:
    with pytest.raises(ValueError):
        beats_for_segment(4.0, 4, 0)
    with pytest.raises(ValueError):
        beats_for_segment(4.0, 4, -5)


def test_non_finite_max_bpm_rejected() -> None:
    with pytest.raises(ValueError):
        beats_for_segment(4.0, 4, float("nan"))
    with pytest.raises(ValueError):
        beats_for_segment(4.0, 4, float("inf"))


def test_non_positive_max_bpm_rejected() -> None:
    with pytest.raises(ValueError):
        beats_for_segment(4.0, 4, 0)
