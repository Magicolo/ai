"""Issue 033 measurement: prefetch hit/miss aggregation (slim-testable).

The commit path already emits `director_prefetch_hit/miss` per segment;
`summarize_prefetch_outcome` turns those events into the hit rate the soak
report needs before any prefetch restructuring. Pure reader — synthetic
events only, no supervisor needed.
"""

from __future__ import annotations

from typing import Any

from voyage.supervisor import summarize_prefetch_outcome


def _events(*names: str) -> list[dict[str, Any]]:
    return [{"event": name, "segment_id": f"{index:06d}"} for index, name in enumerate(names)]


def test_empty_events_yield_no_rate() -> None:
    summary = summarize_prefetch_outcome([])
    assert summary == {"prefetch_hits": 0, "prefetch_misses": 0, "prefetch_hit_rate": None}


def test_mixed_events_count_and_rate() -> None:
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


def test_unrelated_events_are_ignored() -> None:
    summary = summarize_prefetch_outcome(
        _events("segment_committed", "resource_gauges", "video_resumed")
    )
    assert summary == {"prefetch_hits": 0, "prefetch_misses": 0, "prefetch_hit_rate": None}


def test_all_hits_rate_is_one() -> None:
    summary = summarize_prefetch_outcome(_events("director_prefetch_hit", "director_prefetch_hit"))
    assert summary["prefetch_hit_rate"] == 1.0


def test_all_misses_rate_is_zero() -> None:
    summary = summarize_prefetch_outcome(
        _events("director_prefetch_miss", "director_prefetch_miss")
    )
    assert summary["prefetch_hit_rate"] == 0.0
