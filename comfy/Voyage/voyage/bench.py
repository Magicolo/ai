"""Benchmark + soak helpers: timing stats, DESIGN §104 reports, gauge summaries.

Pure functions over plain data — the worker `benchmark` ops produce the
numbers, the CLI renders them, and the soak harness trends them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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
    setup: Mapping[str, object],
    metrics: Mapping[str, object],
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
    setup: Mapping[str, object],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    """JSON-serializable benchmark/soak report document (issue 060).

    Why a second shape, not a format change: `format_report` output is
    stdout prose (existing tests pin its lines). This dict is the machine
    artifact the CLI tees to `logs/benchmark-<target>-<ts>.json` so reruns
    stay comparable without hand-copying terminal output. Callers must keep values
    JSON-serializable (plain setup/metric dicts already are).
    """
    return {"title": title, "setup": dict(setup), "measured": dict(metrics)}


def sfx_benchmark_setup(
    *,
    model_size: str,
    sfx_workers: int,
    device: str,
    warmup: int,
    measured: int,
) -> dict[str, object]:
    """Setup block for the future `benchmark sfx` branch (issues 154/163).

    Records the two knobs the numbers must discriminate (model-size
    ladder × `--sfx-workers` sharding) plus the harness counts — the CLI
    branch threads this into `format_report`/`report_document` exactly
    like the video/audio branches thread theirs.
    """
    return {
        "backend": "mmaudio",
        "model_size": model_size,
        "sfx_workers": sfx_workers,
        "device": device,
        "warmup": warmup,
        "measured": measured,
    }


def augment_benchmark_setup(
    *,
    chunk_frames: int,
    upscale_factor: int,
    crf: int,
    preset: str,
    device: str,
    warmup: int,
    measured: int,
) -> dict[str, object]:
    """Setup block for the future `benchmark augment` branch (issue 154).

    Records the chunk + quality knobs (chunk size, upscale factor, CRF,
    preset) the augment ladder must discriminate — same shape contract
    as `sfx_benchmark_setup` above.
    """
    return {
        "backend": "augment",
        "chunk_frames": chunk_frames,
        "upscale_factor": upscale_factor,
        "crf": crf,
        "preset": preset,
        "device": device,
        "warmup": warmup,
        "measured": measured,
    }


def summarize_sfx_windows(windows: list[dict[str, object]]) -> dict[str, object]:
    """Soak-section aggregation over plain SFX window records (issue 163).

    Pure over caller-supplied records (`wall_seconds` + `audio_seconds`
    per window) so the soak SFX section stays a post-run pass with no
    extra renders: the CLI collects the records from stems + ledger and
    merges this dict into the soak metrics. Empty input is zero, never
    a ZeroDivisionError (the `validate_benchmark_counts` lesson).
    """
    walls = [
        sample
        for record in windows
        if (sample := _finite_float(record.get("wall_seconds"))) is not None
    ]
    audio = [
        sample
        for record in windows
        if (sample := _finite_float(record.get("audio_seconds"))) is not None
    ]
    total_wall = sum(walls)
    total_audio = sum(audio)
    return {
        "windows": len(windows),
        "mean_wall_seconds": (total_wall / len(walls)) if walls else 0.0,
        "total_audio_seconds": total_audio,
        "audio_seconds_per_wall_second": (total_audio / total_wall) if total_wall else 0.0,
    }


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
