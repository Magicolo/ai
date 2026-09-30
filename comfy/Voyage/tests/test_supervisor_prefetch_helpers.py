"""Prefetch-outcome aggregation agreement (issue 081 extraction from `supervisor`).

`summarize_prefetch_outcome` is the verbatim pure reader moved to
`voyage.supervisor_prefetch` so `supervisor.py` (above the §12
~500-line signal) shrinks toward it. Behavior contract: identical
mapping to the pre-split function — hits/misses counted, rate None
with no prefetch events, `director_prefetch_invalidated` deliberately
excluded — and the facade re-export is the same object (single
source, not a copy).
"""

from __future__ import annotations

from typing import Any

import voyage.supervisor as supervisor
import voyage.supervisor_prefetch as supervisor_prefetch
from voyage.supervisor_prefetch import summarize_prefetch_outcome


def _events(*names: str) -> list[dict[str, Any]]:
    return [{"event": name, "segment_id": f"{index:06d}"} for index, name in enumerate(names)]


def test_facade_reexport_is_single_sourced() -> None:
    """The facade name is the new home object, not a copy (issue 080)."""
    assert supervisor.summarize_prefetch_outcome is supervisor_prefetch.summarize_prefetch_outcome


def test_empty_events_yield_no_rate() -> None:
    """No prefetch events read as zeros with a None rate (never 0/0)."""
    assert summarize_prefetch_outcome([]) == {
        "prefetch_hits": 0,
        "prefetch_misses": 0,
        "prefetch_hit_rate": None,
    }


def test_mixed_events_count_and_rate() -> None:
    """Hits/misses count; unrelated events are ignored."""
    summary = summarize_prefetch_outcome(
        _events(
            "director_prefetch_miss",
            "director_prefetch_hit",
            "director_prefetch_hit",
            "segment_committed",
        )
    )
    assert summary["prefetch_hits"] == 2
    assert summary["prefetch_misses"] == 1
    assert summary["prefetch_hit_rate"] == 2 / 3


def test_invalidated_events_are_excluded() -> None:
    """Ready-but-discarded proposals are neither hit nor miss (136/168)."""
    summary = summarize_prefetch_outcome(
        _events(
            "director_prefetch_hit",
            "director_prefetch_miss",
            "director_prefetch_invalidated",
            "director_prefetch_invalidated",
        )
    )
    assert summary == {"prefetch_hits": 1, "prefetch_misses": 1, "prefetch_hit_rate": 0.5}


def test_all_misses_rate_is_zero() -> None:
    """All-miss stream rates 0.0 (not None)."""
    summary = summarize_prefetch_outcome(
        _events("director_prefetch_miss", "director_prefetch_miss")
    )
    assert summary["prefetch_hit_rate"] == 0.0
