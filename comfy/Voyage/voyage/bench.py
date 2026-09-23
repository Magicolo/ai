"""Benchmark + soak helpers: timing stats, §104 reports, gauge summaries.

Pure functions over plain data — the worker `benchmark` ops produce the
numbers, the CLI renders them, and the soak harness trends them.
"""

from __future__ import annotations

from typing import Any


def timing_stats(seconds: list[float]) -> dict[str, float | int]:
    """Mean/min/max over measured wall times (warmup already excluded)."""
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
    """Trend summary over per-segment `resource_gauges` events (§68)."""
    rss = [float(event["rss_peak_mb"]) for event in events if "rss_peak_mb" in event]
    disk = [float(event["disk_free_gib"]) for event in events if "disk_free_gib" in event]
    return {
        "segments": len(events),
        "rss_first_mb": rss[0] if rss else "unknown",
        "rss_last_mb": rss[-1] if rss else "unknown",
        "rss_delta_mb": (rss[-1] - rss[0]) if rss else "unknown",
        "disk_first_gib": disk[0] if disk else "unknown",
        "disk_last_gib": disk[-1] if disk else "unknown",
    }
