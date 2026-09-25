"""Benchmark + soak helpers: timing stats, §104 reports, gauge summaries.

Pure functions over plain data — the worker `benchmark` ops produce the
numbers, the CLI renders them, and the soak harness trends them.
"""

from __future__ import annotations

from typing import Any


def timing_stats(seconds: list[float]) -> dict[str, float | int]:
    """Mean/min/max over measured wall times (warmup already excluded)."""
    if not seconds:
        raise ValueError("timing_stats requires at least one measurement")
    return {
        "count": len(seconds),
        "mean": sum(seconds) / len(seconds),
        "min": min(seconds),
        "max": max(seconds),
    }


def format_report(
    title: str,
    setup: dict[str, object],
    metrics: dict[str, object],
) -> str:
    """Render a §104-style report: setup block then measured metrics."""
    lines = [f"benchmark {title}"]
    lines.append("setup:")
    for key, value in setup.items():
        lines.append(f"  {key}: {value}")
    lines.append("measured:")
    for key, value in metrics.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def summarize_gauges(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Trend summary over per-segment `resource_gauges` events (§68).

    Why float | None: missing gauges previously surfaced as the string
    "unknown" in numeric fields, breaking soak/CLI aggregation for the
    whole report (issue 071). None keeps the fields typed as numbers
    while still signalling absence.
    """
    rss = [float(event["rss_peak_mb"]) for event in events if "rss_peak_mb" in event]
    disk = [float(event["disk_free_gib"]) for event in events if "disk_free_gib" in event]
    return {
        "segments": len(events),
        "rss_first_mb": rss[0] if rss else None,
        "rss_last_mb": rss[-1] if rss else None,
        "rss_delta_mb": (rss[-1] - rss[0]) if rss else None,
        "disk_first_gib": disk[0] if disk else None,
        "disk_last_gib": disk[-1] if disk else None,
    }
