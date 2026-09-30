"""Benchmark + soak helpers: timing stats, §104 reports, gauge summaries.

Pure functions over plain data — the worker `benchmark` ops produce the
numbers, the CLI renders them, and the soak harness trends them.
"""

from __future__ import annotations

import math
from typing import Any

_VRAM_WORKERS = ("video", "audio", "director")
"""Workers whose `*_vram_free_gib` gauges the supervisor samples (§68)."""


def _finite_float(value: Any) -> float | None:
    """Numeric value as float, or None when absent/unusable.

    Why the guard: missing gauges previously surfaced as the string
    "unknown" in numeric fields, breaking aggregation for the whole
    report (issue 071). Bool is excluded explicitly (it subclasses int).
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


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


def timing_stats_ex(seconds: list[float]) -> dict[str, float | int]:
    """Extended timing stats with percentiles (051-benchmark-half).

    Why a successor, not an edit: `timing_stats` return shape is frozen
    (existing benchmark tests pin the exact four keys). New consumers
    read p50/p95/std here; legacy callers keep the four-key contract.
    p50 is the median (mean of the two middles on even counts); p95 is
    nearest-rank; std is the population standard deviation.
    """
    if not seconds:
        raise ValueError("timing_stats_ex requires at least one measurement")
    ordered = sorted(seconds)
    count = len(ordered)
    mean = sum(ordered) / count
    mid = count // 2
    p50 = ordered[mid] if count % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2.0
    rank = math.ceil(0.95 * count) - 1
    p95 = ordered[max(0, min(rank, count - 1))]
    variance = sum((value - mean) ** 2 for value in ordered) / count
    return {
        "count": count,
        "mean": mean,
        "min": ordered[0],
        "max": ordered[-1],
        "p50": p50,
        "p95": p95,
        "std": math.sqrt(variance),
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


def report_document(
    title: str,
    setup: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    """JSON-serializable benchmark/soak report document (issue 060).

    Why a second shape, not a format change: `format_report` output is
    stdout prose (existing tests pin its lines). This dict is the machine
    artifact the CLI tees to `logs/benchmark-<target>-<ts>.json` so reruns
    stay comparable without hand-copying stdout. Callers must keep values
    JSON-serializable (plain setup/metric dicts already are).
    """
    return {"title": title, "setup": dict(setup), "measured": dict(metrics)}


def summarize_gauges(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Trend summary over per-segment `resource_gauges` events (§68).

    Why float | None: missing gauges previously surfaced as the string
    "unknown" in numeric fields, breaking soak/CLI aggregation for the
    whole report (issue 071). None keeps the fields typed as numbers
    while still signalling absence.
    """
    rss = [
        sample
        for event in events
        if (sample := _finite_float(event.get("rss_peak_mb"))) is not None
    ]
    disk = [
        sample
        for event in events
        if (sample := _finite_float(event.get("disk_free_gib"))) is not None
    ]
    summary: dict[str, Any] = {
        "segments": len(events),
        "rss_first_mb": rss[0] if rss else None,
        "rss_last_mb": rss[-1] if rss else None,
        "rss_delta_mb": (rss[-1] - rss[0]) if rss else None,
        "disk_first_gib": disk[0] if disk else None,
        "disk_last_gib": disk[-1] if disk else None,
    }
    reporting: list[str] = []
    for worker in _VRAM_WORKERS:
        free = [
            sample
            for event in events
            if (sample := _finite_float(event.get(f"{worker}_vram_free_gib"))) is not None
        ]
        total = [
            sample
            for event in events
            if (sample := _finite_float(event.get(f"{worker}_vram_total_gib"))) is not None
        ]
        if free:
            reporting.append(worker)
        summary[f"{worker}_vram_free_first_gib"] = free[0] if free else None
        summary[f"{worker}_vram_free_last_gib"] = free[-1] if free else None
        summary[f"{worker}_vram_free_min_gib"] = min(free) if free else None
        summary[f"{worker}_vram_total_gib"] = total[-1] if total else None
    summary["vram_workers_reporting"] = reporting
    return summary
