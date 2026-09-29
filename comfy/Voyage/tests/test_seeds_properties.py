"""Property tests for deterministic seed derivation (`voyage.seeds`).

Why properties here: `derive_seed` is a pure core (§12) — a handful of
examples cannot cover the label space, while the invariants (range,
determinism, stream separation, unambiguous encoding) must hold for
every input. Generators are domain-constrained (bounded integers,
short texts — seeds derive from small run/segment/block numbers in
practice), `st.data()` draws keep signatures narrow, and the boundary
cases that generators hit only probabilistically are pinned as
deterministic example tests below.

Runs only when Hypothesis is installed (the gate image does not carry
it yet — `tests/conftest.py` skips this module there); the pins at
the bottom document the contract even without it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis", reason="property tests need Hypothesis (absent from gate image)")

from hypothesis import given
from hypothesis import strategies as strategies
from hypothesis.strategies import DataObject

from voyage.seeds import audio_seed, derive_seed, director_seed, video_seed

# `derive_seed` reduces the digest modulo 2**31, so every output is a
# non-negative signed 31-bit integer (downstream RNGs take uint32).
MAXIMUM_SEED_VALUE = 2**31 - 1

run_seeds = strategies.integers(min_value=-(2**63), max_value=2**63)
small_counts = strategies.integers(min_value=0, max_value=999999)
# `st.text()` already excludes surrogate code points by default, so labels
# are always encodable without an explicit alphabet restriction.
short_labels = strategies.text(max_size=24)
label_lists = strategies.lists(
    strategies.one_of(short_labels, strategies.integers(min_value=-9999, max_value=9999)),
    max_size=4,
)


@given(strategies.data())
def test_derive_seed_stays_within_signed_31_bit_range(data: DataObject) -> None:
    drawn_seed = data.draw(run_seeds)
    derived = derive_seed(drawn_seed, *data.draw(label_lists))
    assert 0 <= derived <= MAXIMUM_SEED_VALUE


@given(strategies.data())
def test_derive_seed_is_deterministic(data: DataObject) -> None:
    drawn_seed = data.draw(run_seeds)
    drawn_labels = data.draw(label_lists)
    assert derive_seed(drawn_seed, *drawn_labels) == derive_seed(drawn_seed, *drawn_labels)


@given(strategies.data())
def test_concern_streams_stay_separated(data: DataObject) -> None:
    """Video/audio/director streams from the same numbers must differ.

    A 31-bit hash collision is possible in principle but never observed;
    the fixed colon-ambiguity case below is the realistic merge risk.
    """
    drawn_seed = data.draw(run_seeds)
    drawn_count = data.draw(small_counts)
    assert video_seed(drawn_seed, drawn_count, drawn_count) != audio_seed(
        drawn_seed, drawn_count, drawn_count
    )
    assert video_seed(drawn_seed, drawn_count, drawn_count) != director_seed(
        drawn_seed, drawn_count
    )


@given(strategies.data())
def test_colon_split_position_stays_injective(data: DataObject) -> None:
    """Length-prefixing keeps ("a:b", "c") and ("a", "b:c") apart.

    Plain colon-joining mapped both label tuples to the identical key
    string, silently merging two RNG streams (issue 070). Splitting an
    arbitrary colon-bearing label at any position must change the seed.
    """
    drawn_seed = data.draw(run_seeds)
    left = data.draw(short_labels)
    right = data.draw(short_labels)
    joint = left + ":" + right
    assert derive_seed(drawn_seed, joint) != derive_seed(drawn_seed, left, right)


@given(strategies.data())
def test_wrappers_match_derive_seed_with_concern_label(data: DataObject) -> None:
    drawn_seed = data.draw(run_seeds)
    drawn_first = data.draw(small_counts)
    drawn_second = data.draw(small_counts)
    assert video_seed(drawn_seed, drawn_first, drawn_second) == derive_seed(
        drawn_seed, "video", drawn_first, drawn_second
    )
    assert audio_seed(drawn_seed, drawn_first, drawn_second) == derive_seed(
        drawn_seed, "audio", drawn_first, drawn_second
    )
    assert director_seed(drawn_seed, drawn_first) == derive_seed(
        drawn_seed, "director", drawn_first
    )


def test_derive_seed_with_no_labels_stays_in_range() -> None:
    assert 0 <= derive_seed(7) <= MAXIMUM_SEED_VALUE


def test_derive_seed_accepts_boundary_run_seeds() -> None:
    for boundary_seed in (0, MAXIMUM_SEED_VALUE, -1, 2**63, -(2**63)):
        assert 0 <= derive_seed(boundary_seed, "video", 1, 2) <= MAXIMUM_SEED_VALUE


def test_derive_seed_accepts_unicode_and_empty_labels() -> None:
    assert 0 <= derive_seed(7, "néon récif 海", "🌊") <= MAXIMUM_SEED_VALUE
    assert 0 <= derive_seed(7, "") <= MAXIMUM_SEED_VALUE


def test_integer_label_merges_with_its_string_spelling() -> None:
    """Characterization: labels are normalized with `str()` before the
    length prefix, so integer 1 and string "1" deliberately share the
    stream. Pinned so a future encoding change fails loudly instead of
    silently reshuffling every derived stream."""
    assert derive_seed(7, 1) == derive_seed(7, "1")


def test_wrappers_stay_in_range_on_boundary_counts() -> None:
    assert 0 <= video_seed(0, 999999, 999999) <= MAXIMUM_SEED_VALUE
    assert 0 <= audio_seed(0, 999999, 999999) <= MAXIMUM_SEED_VALUE
    assert 0 <= director_seed(0, 999999) <= MAXIMUM_SEED_VALUE
