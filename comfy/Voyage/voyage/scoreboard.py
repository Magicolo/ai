"""Segment scoreboard view (fast-iteration slice 3, DESIGN §59).

Why this module exists: `voyage status` shows the live tail, but
iteration needs a per-segment table — frames, per-stage seconds,
deterministic visual metrics with deltas, director destination/phase,
and view paths — across the whole run, including segments committed
before a daily log rotation (DESIGN §60, via `logrotate`).

`scoreboard_rows` reads one run directory and returns one dict per
committed segment: frame counts, per-stage seconds, deterministic visual
metrics (when the experimental inspector ran), deltas against the previous
segment, the director destination/phase, take ids, and the viewable media
paths. The CLI `inspect scoreboard` target renders the compact table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.atomic import JsonValue
from voyage.logrotate import iter_metric_files
from voyage.segment_manifest import load_audio_state, load_metrics, load_transition

METRIC_KEYS = [
    "motion_energy",
    "visual_complexity",
    "semantic_change_rate",
    "palette_distance",
    "style_similarity",
    "scene_boundary_strength",
]
"""The six §43 deterministic metrics, in summarize_segment order."""

_DELTA_ROUND_DIGITS = 3
"""Decimal places for per-segment metric deltas (enough to see drift)."""


def _finite_float(value: JsonValue) -> float | None:
    """Numeric cell as float, or None when the cell is absent/unusable.

    `JsonValue` (issue 035), not bare `Any`: scoreboard cells come from
    JSON-parsed metrics, so every input is JSON-shaped by construction —
    the guard below still rejects non-numeric members (strings, bools,
    nulls, containers, non-finite floats) instead of trusting the shape.
    Bool is excluded (it subclasses int); non-finite floats are dropped
    like the hardened `_slowest_stage` sibling in cli.py.
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _stages_by_segment(run_dir: Path) -> dict[str, dict[str, float]]:
    """Per-stage seconds keyed by segment id from logs/metrics*.jsonl.

    Reads the live file plus rotated siblings oldest-first (issue 049):
    after a daily rotation the live file alone would silently drop every
    older segment's stages. Later files win on duplicate segment ids.
    """
    stages: dict[str, dict[str, float]] = {}
    for events_path in iter_metric_files(run_dir):
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event: JsonValue = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("event") != "segment_committed":
                continue
            segment_id = event.get("segment_id")
            raw_stages = event.get("stages")
            if isinstance(segment_id, str) and isinstance(raw_stages, dict):
                cleaned: dict[str, float] = {}
                for key, value in raw_stages.items():
                    sample = _finite_float(value)
                    if sample is not None:
                        cleaned[str(key)] = sample
                stages[segment_id] = cleaned
    return stages


def partial_segment_ids(run_dir: Path) -> list[str]:
    """Sorted ids of segment dirs without a DONE marker (062).

    Why a helper, not a silent skip: non-DONE dirs are stalled partial
    commits, invisible in the very table meant for iteration. The row
    builder keeps skipping them (DONE-gating invariant,
    docs/STATE_AND_RECOVERY.md); the CLI renders this list as a trailing
    `partial: [...]` line so stalls stay visible.
    """
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    if not segments_root.is_dir():
        return []
    return sorted(
        segment.name
        for segment in segments_root.iterdir()
        if segment.is_dir() and not (segment / paths.DONE_MARKER).exists()
    )


def scoreboard_rows(run_dir: Path) -> list[dict[str, Any]]:
    """One scoreboard row per committed segment (DESIGN fast-iteration §140)."""
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    stages = _stages_by_segment(run_dir)
    rows: list[dict[str, Any]] = []
    previous: dict[str, float] | None = None
    previous_id: str | None = None
    if not segments_root.is_dir():
        return rows
    for segment in sorted(p for p in segments_root.iterdir() if p.is_dir()):
        if not (segment / paths.DONE_MARKER).exists():
            continue
        metrics = load_metrics(segment)
        transition = load_transition(segment)
        audio_state = load_audio_state(segment)
        video = metrics.get("video")
        raw_frames = video.get("frames") if isinstance(video, dict) else None
        if isinstance(raw_frames, bool):
            frames: int | None = None
        elif isinstance(raw_frames, int) and raw_frames >= 0:
            frames = raw_frames
        else:
            frames = None
        visual = metrics.get("visual")
        current: dict[str, float] | None = None
        errors: list[str] = []
        if isinstance(visual, dict) and isinstance(visual.get("metrics"), dict):
            raw_metrics = visual["metrics"]
            cleaned_metrics: dict[str, float] = {}
            for key in METRIC_KEYS:
                if key not in raw_metrics:
                    continue
                sample = _finite_float(raw_metrics[key])
                if sample is None:
                    errors.append(f"metric {key}: non-numeric value skipped")
                else:
                    cleaned_metrics[key] = sample
            current = cleaned_metrics or None
            if current is None and any(key in raw_metrics for key in METRIC_KEYS):
                errors.append("metrics: no usable cells in visual.metrics")
        deltas: dict[str, float] | None = None
        baseline_segment_id: str | None = None
        if current is not None:
            if previous is None:
                deltas = dict.fromkeys(current, 0.0)
            else:
                baseline_segment_id = previous_id
                deltas = {}
                for key in current:
                    baseline = previous.get(key, current[key])
                    deltas[key] = round(current[key] - baseline, _DELTA_ROUND_DIGITS)
        destination = transition.get("destination")
        video_path = segment / "video.mp4"
        audio_path = segment / "audio.wav"
        row = {
            "segment_id": segment.name,
            "done": True,
            "frames": frames,
            "stages": stages.get(segment.name, {}),
            "metrics": current,
            "deltas": deltas,
            "baseline_segment_id": baseline_segment_id,
            "errors": errors,
            "destination": destination.get("canonical_name")
            if isinstance(destination, dict)
            else None,
            "phase": transition.get("phase"),
            "take_ids": audio_state.get("take_ids"),
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "video_exists": video_path.exists(),
            "audio_exists": audio_path.exists(),
        }
        rows.append(row)
        if current is not None:
            previous = current
            previous_id = segment.name
    return rows
