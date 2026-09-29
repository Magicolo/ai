"""Property tests for concept similarity math (`voyage.concepts`).

Why properties here: `token_set_similarity` is the novelty fallback when
no embedding backend is available, and `canonicalize` feeds the embedding
index — both are pure cores (§12) whose contracts (bounded, symmetric,
idempotent) must hold for every input pair, not just the pinned examples.
`cosine_similarity` shares the file and gets the same treatment. Only
`tokenize`'s word pattern is trusted, never the outputs.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("hypothesis", reason="property tests need Hypothesis (pinned dev extra)")

from hypothesis import given
from hypothesis import strategies as strategies
from hypothesis.strategies import DataObject

from tests.conftest import short_texts
from voyage.concepts import canonicalize, cosine_similarity, token_set_similarity

# Squared sums must stay inside float range (8 × max² < 1.8e308), so the
# ceiling is 1e100 — larger magnitudes overflow the dot product to inf and
# would test `inf/inf`, not the cosine.
finite_floats = strategies.floats(
    min_value=-1e100, max_value=1e100, allow_nan=False, allow_infinity=False
)


@given(strategies.data())
def test_similarity_stays_bounded_and_symmetric(data: DataObject) -> None:
    first_text = data.draw(short_texts)
    second_text = data.draw(short_texts)
    forward = token_set_similarity(first_text, second_text)
    backward = token_set_similarity(second_text, first_text)
    assert 0.0 <= forward <= 1.0
    assert forward == backward


@given(strategies.data())
def test_similarity_is_reflexive(data: DataObject) -> None:
    """Any text fully overlaps with itself — including the empty string."""
    drawn_text = data.draw(short_texts)
    assert token_set_similarity(drawn_text, drawn_text) == 1.0


@given(strategies.data())
def test_canonicalize_is_sorted_idempotent_lowercase(data: DataObject) -> None:
    drawn_text = data.draw(short_texts)
    once_canonical = canonicalize(drawn_text)
    assert once_canonical == once_canonical.lower()
    assert once_canonical.split(" ") == sorted(once_canonical.split(" "))
    assert canonicalize(once_canonical) == once_canonical


@given(strategies.data())
def test_cosine_is_symmetric_and_self_unit(data: DataObject) -> None:
    drawn_vector = data.draw(strategies.lists(finite_floats, min_size=1, max_size=8))
    drawn_norm = math.sqrt(sum(component * component for component in drawn_vector))
    if drawn_norm == 0.0:
        # Float floor (see the underflow pin below): an all-zero vector, or
        # a nonzero one whose squares underflow to zero, reads as zero.
        assert cosine_similarity(drawn_vector, drawn_vector) == 0.0
    else:
        assert cosine_similarity(drawn_vector, drawn_vector) == pytest.approx(1.0)
    other_vector = data.draw(strategies.lists(finite_floats, min_size=1, max_size=8))
    if len(other_vector) == len(drawn_vector):
        assert cosine_similarity(drawn_vector, other_vector) == pytest.approx(
            cosine_similarity(other_vector, drawn_vector)
        )


def test_empty_against_empty_is_identical() -> None:
    """Pin: the singular no-token case is defined as identical, not 0/0."""
    assert token_set_similarity("", "") == 1.0


def test_empty_against_nonempty_is_disjoint() -> None:
    """Pin: one-sided emptiness can share no tokens."""
    assert token_set_similarity("", "neon reef") == 0.0
    assert token_set_similarity("neon reef", "") == 0.0


def test_similarity_ignores_case_and_order() -> None:
    """Pin: token sets, not strings — case/order/duplicates do not matter."""
    assert token_set_similarity("Neon Reef", "reef neon") == 1.0
    assert token_set_similarity("neon neon reef", "reef neon") == 1.0


def test_cosine_rejects_mismatched_and_empty_vectors() -> None:
    """Pin: shape errors are 0.0 by contract, never an exception."""
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_underflow_reads_as_zero() -> None:
    """Pin: `[4.4e-197]` is nonzero, but its square underflows float64.

    Found by the self-unit property above (Hypothesis minimal example):
    `4.4e-197 ** 2 ≈ 2e-393` is below the smallest denormal, so the norm
    computes as 0.0 and the function returns 0.0. The novelty policy reads
    this as "no signal" — same as the zero vector. Whether that conflation
    is acceptable is a policy call for the concepts owner, not this test.
    """
    assert cosine_similarity([4.4373185127303166e-197], [4.4373185127303166e-197]) == 0.0
