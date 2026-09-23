"""Segment scoreboard view (fast-iteration slice 3).

`scoreboard_rows` reads one run directory and returns one dict per
committed segment: frame counts, per-stage seconds, deterministic visual
metrics (when the experimental inspector ran), deltas against the previous
segment, the director destination/phase, take ids, and the viewable media
paths. The CLI `inspect scoreboard` target renders the compact table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voyage import paths

METRIC_KEYS = [
    "motion_energy",
    "visual_complexity",
    "semantic_change_rate",
    "palette_distance",
    "style_similarity",
    "scene_boundary_strength",
]
"""The six §43 deterministic metrics, in summarize_segment order."""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _stages_by_segment(run_dir: Path) -> dict[str, dict[str, float]]:
    """Per-stage seconds keyed by segment id from logs/metrics.jsonl."""
    stages: dict[str, dict[str, float]] = {}
    events_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return stages
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("event") != "segment_committed":
            continue
        segment_id = event.get("segment_id")
        raw_stages = event.get("stages")
        if isinstance(segment_id, str) and isinstance(raw_stages, dict):
            stages[segment_id] = {str(key): float(value) for key, value in raw_stages.items()}
    return stages


def scoreboard_rows(run_dir: Path) -> list[dict[str, Any]]:
    """One scoreboard row per committed segment (DESIGN fast-iteration §140)."""
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    stages = _stages_by_segment(run_dir)
    rows: list[dict[str, Any]] = []
    previous: dict[str, float] | None = None
    if not segments_root.is_dir():
        return rows
    for segment in sorted(p for p in segments_root.iterdir() if p.is_dir()):
        if not (segment / paths.DONE_MARKER).exists():
            continue
        metrics = _read_json(segment / "metrics.json") or {}
        transition = _read_json(segment / "transition.json") or {}
        audio_state = _read_json(segment / "audio_state.json") or {}
        video = metrics.get("video")
        frames = video.get("frames") if isinstance(video, dict) else None
        visual = metrics.get("visual")
        current: dict[str, float] | None = None
        if isinstance(visual, dict) and isinstance(visual.get("metrics"), dict):
            current = {
                key: float(visual["metrics"][key])
                for key in METRIC_KEYS
                if key in visual["metrics"]
            }
        deltas: dict[str, float] | None = None
        if current is not None:
            if previous is None:
                deltas = dict.fromkeys(current, 0.0)
            else:
                deltas = {
                    key: round(current[key] - previous.get(key, current[key]), 3) for key in current
                }
        destination = transition.get("destination")
        row = {
            "segment_id": segment.name,
            "done": True,
            "frames": frames,
            "stages": stages.get(segment.name, {}),
            "metrics": current,
            "deltas": deltas,
            "destination": destination.get("canonical_name")
            if isinstance(destination, dict)
            else None,
            "phase": transition.get("phase"),
            "take_ids": audio_state.get("take_ids"),
            "video_path": str(segment / "video.mp4"),
            "audio_path": str(segment / "audio.wav"),
        }
        rows.append(row)
        if current is not None:
            previous = current
    return rows
