"""Prefetch-outcome aggregation for the supervisor (DESIGN §§73, 68).

Split from `voyage.supervisor` (issue 081): the soak report's
director-prefetch hit/miss aggregation as an importable pure reader
with no supervisor state. `voyage.supervisor` re-exports the name
below so existing importers keep working; new code imports from here
directly.

- `summarize_prefetch_outcome`: pure reader over metrics/log events
  (issue 033 measurement first; issues 136 + 168 third outcome).
"""

from __future__ import annotations

from typing import Any


def summarize_prefetch_outcome(events: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate director-prefetch hit/miss events (issue 033).

    Pure reader over metrics/log events — the commit path already emits
    `director_prefetch_hit/miss` per segment; this turns them into the hit
    rate the soak report needs before any prefetch restructuring is
    considered (candidate 3: measure first). `prefetch_hit_rate` is None
    with no prefetch events (never 0/0). Lives beside the emitter (not in
    the CLI) so the aggregation and the event names cannot drift apart;
    the soak report renders the returned mapping as-is.

    Third outcome (issues 136 + 168): `director_prefetch_invalidated`
    events (ready proposals discarded by amendments or drift-hold) are
    deliberately NOT counted here — neither hit nor miss — so the rate
    stays `hit / (hit + miss)` by construction and the return shape stays
    frozen. Count `invalidated` separately from the raw event stream when
    the soak report needs the waste signal.
    """
    hits = sum(1 for event in events if event.get("event") == "director_prefetch_hit")
    misses = sum(1 for event in events if event.get("event") == "director_prefetch_miss")
    total = hits + misses
    return {
        "prefetch_hits": hits,
        "prefetch_misses": misses,
        "prefetch_hit_rate": (hits / total) if total else None,
    }
