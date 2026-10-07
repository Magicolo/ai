"""Segment scoreboard view (fast-iteration slice 3, DESIGN §59).

Why this module exists: `voyage status` shows the live tail, but
iteration needs a per-segment table — frames, per-stage seconds,
deterministic visual metrics with deltas, director destination/phase,
and view paths — across the whole run, including segments committed
before a daily log rotation (DESIGN §60, via `logrotate`).

Entrypoints (read-only, never write into the run):

- `python -m voyage.scoreboard --run Voyage/output/<name>` renders the
  compact text table (one line per committed row + `partial: [...]`
  trailer for stalled non-DONE dirs + `status: ...` line for the run
  lifecycle including stale-RUNNING detection).
- `python -m voyage.scoreboard --run ... --json` prints the raw rows
  document plus `partial`, `torn` (torn metric lines, loud accounting
  via `logrotate.parse_metric_lines`), and `status` (see
  `run_status_summary`).
- `python -m voyage.boundary_metrics --run ... --prompts` is the
  seam-continuity + prompt-adherence companion over the same committed
  segments ( DESIGN §§22.5, 137A, 18): scoreboard answers "what
  committed", boundary_metrics answers "do the joints continue".

All-deferred audio choice (documented): `audio_path`/`audio_exists`
columns are dropped (no per-segment audio artifact anymore; old runs
may still carry `audio.wav` on disk, ignored here) and no `audio.wav`
path is constructed. `take_ids` is still read from `audio_state`
(harmless): new deferred commits carry an empty list there, while the
finalize takes themselves live under `run/audio/takes.jsonl`.

Adopted segments (Track E): orphan-adopted DONE dirs (crash window where
DONE landed but `state.json` never advanced) index as rows with
`adopted: true` and `stages: {}` when no `segment_committed` exists for
them — the video is real, the timing is unknown, and the delta baseline
skips them transparently. New metric keys delta as null (not 0.0): a key
with no baseline has no delta to report.

`prompt_enhanced` key (Track E): new writers emit `segment_id`
(canonical); readers accept the legacy `segment` key too via
`prompt_enhanced_segment_id` — history scans never break on the rename.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from voyage import paths
from voyage.atomic import JsonValue
from voyage.logrotate import (
    count_torn_metric_lines,
    is_stale_running,
    iter_metric_files,
    parse_metric_lines,
    read_all_metric_events_counted,
)
from voyage.logrotate import (
    prompt_enhanced_segment_id as _prompt_enhanced_segment_id,
)
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


def adopted_segment_ids(run_dir: Path) -> set[str]:
    """Ids with a `segment_adopted` event (Track E orphan adoption, DESIGN §59).

    Crash-window orphans (DONE durable, state never advanced) re-enter via
    checksum adoption; the commit path logs `segment_adopted` there. This
    scans live + rotated siblings (torn lines skipped, later files win —
    same rotation tolerance as `_stages_by_segment`) and returns the
    adopted id set. `prompt_enhanced` rename note: adopted ids come from
    `segment_id` only (that event never used the legacy `segment` key).
    """
    adopted: set[str] = set()
    for events_path in iter_metric_files(run_dir):
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        events, _torn = parse_metric_lines(lines)
        for event in events:
            if not isinstance(event, dict) or event.get("event") != "segment_adopted":
                continue
            segment_id = event.get("segment_id")
            if isinstance(segment_id, str) and segment_id:
                adopted.add(segment_id)
    return adopted


def torn_metric_lines(run_dir: Path) -> int:
    """Torn metric lines across live + rotated siblings (Track E/C).

    Loud accounting via `logrotate.count_torn_metric_lines`: a crash
    mid-append leaves a torn tail the loaders skip — the count surfaces in
    `--json` and the `validate_scoreboard` hook so log health is visible
    instead of silent. Never raises (missing logs read as zero).
    """
    return count_torn_metric_lines(run_dir)


def run_status_summary(run_dir: Path) -> dict[str, JsonValue]:
    """Read-only run lifecycle summary for `--run` status (Track E, DESIGN §59).

    Returns `{"status": ..., "stale_running": bool, "committed": int,
    "partial": [...], "torn": int}` without writing anything: `status`
    comes from `state.json` (missing/torn reads as `"unknown"`), staleness
    from `logrotate.is_stale_running` over the state-file mtime, `partial`
    from `partial_segment_ids`, `torn` from
    `read_all_metric_events_counted` (issue 231, loud accounting). The text
    table renders `status: ...` from this; `--json` embeds the whole dict.
    """
    try:
        from voyage.persistence import read_state
    except ImportError:
        read_state = None  # type: ignore[assignment]
    status: str = "unknown"
    stale = False
    if read_state is not None:
        try:
            state = read_state(run_dir)
            status = str(state.status)
            try:
                mtime = (run_dir / "state.json").stat().st_mtime
            except OSError:
                mtime = 0.0
            import time as _time

            stale = is_stale_running(status, mtime, _time.time())
        except Exception:  # noqa: BLE001 - read-only status, torn state reads as unknown
            status = "unknown"
    try:
        committed = sum(
            1
            for entry in (run_dir / paths.SEGMENTS_DIRNAME).iterdir()
            if entry.is_dir() and (entry / paths.DONE_MARKER).exists()
        )
    except OSError:
        committed = 0
    _, counted_torn = read_all_metric_events_counted(run_dir)
    return {
        "status": cast(JsonValue, status),
        "stale_running": cast(JsonValue, stale),
        "committed": cast(JsonValue, committed),
        "partial": cast(JsonValue, partial_segment_ids(run_dir)),
        "torn": cast(JsonValue, counted_torn),
    }


def format_reconcile_output(
    deleted: Sequence[str], adopted: Sequence[str], partial: Sequence[str]
) -> str:
    """One-line reconcile summary surfacing partials (Track E, DESIGN §59).

    Pure formatter for the generate reconcile path: Track A/C prints this
    next to the `run_reconciled` metric event (`logrotate.reconcile_event`)
    so the screen and the stream agree. `partial` lists still-stalled
    non-DONE dirs after the heal — the reconcile is only done when this
    reads empty. Empty inputs render as explicit `none` tokens, never
    blank.
    """
    deleted_text = ", ".join(deleted) if deleted else "none"
    adopted_text = ", ".join(adopted) if adopted else "none"
    partial_text = ", ".join(partial) if partial else "none"
    return f"reconcile: deleted [{deleted_text}] adopted [{adopted_text}] partial [{partial_text}]"


def validate_scoreboard(run_dir: Path) -> list[str]:
    """Read-only scoreboard health hook for Track C (DESIGN §59).

    Returns error strings (empty = clean): torn metric lines are reported
    (`torn metric lines: N (...)` — crash tails the loaders skip), because
    a growing torn count means the log stream is losing history the table
    silently omits. Presence/shape checks stay in `validate_run` (Track C
    owns that); this hook is torn-only by design, never a second
    validator.
    """
    torn = torn_metric_lines(run_dir)
    if torn:
        return [f"torn metric lines: {torn} (loaders skip them; history may be short)"]
    return []


def prompt_enhanced_segment_id(event: dict[str, JsonValue] | dict[str, object]) -> str | None:
    """Segment id of a `prompt_enhanced` event, old or new key (Track E).

    Thin re-export of `logrotate.prompt_enhanced_segment_id`: new writers
    emit `segment_id`, the supervisor still emits `segment`. Scoreboard
    readers call this so the rename never breaks history scans.
    """
    return _prompt_enhanced_segment_id(event)


def partial_segment_ids(run_dir: Path) -> list[str]:
    """Sorted ids of segment dirs without a DONE marker (062).

    Why a helper, not a silent skip: non-DONE dirs are stalled partial
    commits, invisible in the very table meant for iteration. The row
    builder keeps skipping them (DONE-gating invariant,
    docs/STATE_AND_RECOVERY.md); the `python -m voyage.scoreboard`
    entry renders this list as a trailing `partial: [...]` line so
    stalls stay visible.
    """
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    if not segments_root.is_dir():
        return []
    return sorted(
        segment.name
        for segment in segments_root.iterdir()
        if segment.is_dir() and not (segment / paths.DONE_MARKER).exists()
    )


def scoreboard_rows(run_dir: Path) -> list[dict[str, JsonValue]]:
    """One scoreboard row per committed segment (DESIGN fast-iteration §140)."""
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    stages = _stages_by_segment(run_dir)
    adopted = adopted_segment_ids(run_dir)
    _, metric_torn = read_all_metric_events_counted(run_dir)
    rows: list[dict[str, JsonValue]] = []
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
        if metric_torn:
            errors.append(
                f"torn metric lines: {metric_torn} (loaders skip them; history may be short)"
            )
        deltas: dict[str, float | None] | None = None
        baseline_segment_id: str | None = None
        if current is not None:
            if previous is None:
                deltas = dict.fromkeys(current, 0.0)
            else:
                baseline_segment_id = previous_id
                deltas = {}
                for key in current:
                    baseline = previous.get(key)
                    if baseline is None:
                        deltas[key] = None
                    else:
                        deltas[key] = round(current[key] - baseline, _DELTA_ROUND_DIGITS)
        destination = transition.get("destination")
        video_path = segment / "video.mp4"
        row: dict[str, JsonValue] = {
            "segment_id": segment.name,
            "done": True,
            "adopted": segment.name in adopted,
            "frames": frames,
            "stages": cast(JsonValue, stages.get(segment.name, {})),
            "metrics": cast(JsonValue, current),
            "deltas": cast(JsonValue, deltas),
            "baseline_segment_id": baseline_segment_id,
            "errors": cast(JsonValue, errors),
            "destination": destination.get("canonical_name")
            if isinstance(destination, dict)
            else None,
            "phase": transition.get("phase"),
            "take_ids": audio_state.get("take_ids"),
            "video_path": str(video_path),
            "video_exists": video_path.exists(),
        }
        rows.append(row)
        if current is not None:
            previous = current
            previous_id = segment.name
    return rows


def format_scoreboard_table(
    rows: list[dict[str, JsonValue]],
    partial: list[str],
    status: dict[str, JsonValue] | None = None,
) -> str:
    """Compact per-segment table, one line per committed row (pure; no I/O)."""
    lines = [f"segments: {len(rows)}"]
    for row in rows:
        stages = row.get("stages")
        if isinstance(stages, dict):
            stage_cells = ",".join(
                f"{key}={value:.1f}"
                for key, value in sorted(stages.items())
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
        else:
            stage_cells = "-"
        errors = row.get("errors")
        error_cell = (
            ";".join(str(item) for item in errors) if isinstance(errors, list) and errors else "-"
        )
        adopted_flag = " adopted" if row.get("adopted") is True else ""
        lines.append(
            f"  {row.get('segment_id')} frames={row.get('frames')}{adopted_flag} "
            f"video={'ok' if row.get('video_exists') else 'missing'} "
            f"dest={row.get('destination')} phase={row.get('phase')} "
            f"stages={stage_cells or '-'} errors={error_cell}"
        )
    lines.append(f"partial: {partial}")
    if status is not None:
        lines.append(
            f"status: {status.get('status')} stale_running={status.get('stale_running')} "
            f"torn={status.get('torn')}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """CLI surface for the read-only scoreboard view."""
    parser = argparse.ArgumentParser(
        prog="voyage.scoreboard",
        description="Per-segment scoreboard table over a committed run (read-only).",
    )
    parser.add_argument("--run", required=True, help="Run directory (e.g. Voyage/output/crabz).")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw rows JSON instead of the text table.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: render one run's scoreboard, never writing into it."""
    args = _build_parser().parse_args(argv)
    run_path = Path(args.run)
    rows = scoreboard_rows(run_path)
    partial = partial_segment_ids(run_path)
    torn = torn_metric_lines(run_path)
    status = run_status_summary(run_path)
    if args.json:
        sys.stdout.write(
            json.dumps({"rows": rows, "partial": partial, "torn": torn, "status": status}, indent=2)
        )
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_scoreboard_table(rows, partial, status))
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin runner over main()
    raise SystemExit(main())
