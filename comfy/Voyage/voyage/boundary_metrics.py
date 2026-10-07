"""Seam-continuity metrics + prompt audit over committed runs (DESIGN §§22.5, 137A, 18).

Why this module exists: Track A Phase-0 needs CPU-only numbers for seam
continuity (within-segment drift vs cross-boundary jump) and prompt
adherence eyeball, without GPU runs or generation-path changes. This tool
reads committed `segments/*/video.mp4` + `segments/*/manifest.json` from a
run directory (e.g. `Voyage/output/<name>`) and reports the same
within/boundary ratio the §137A qualification harness
(`tests/test_qualification.py::summarize_run`) uses, plus the effective
block prompts (`prompt_plan.stages[].prompt` + transition) per segment.

CPU-only contract: stdlib + numpy + the ffmpeg CLI (via
`voyage.vision.metrics.sample_frames`). This module never imports
torch/transformers/diffusers, at module scope or lazily. Read-only: it
never writes into the run directory, so running it over
`Voyage/output/<name>` cannot disturb committed state.

Invoke: `python -m voyage.boundary_metrics --run Voyage/output/<name>`
(`--prompts` appends the prompt audit, `--json` prints the raw document).
Exit 0 on PASS/N/A, 1 on FAIL verdict, 3 on tool error, 2 on CLI misuse
(issue 242: FAIL and tool error used to share exit 1, so gate drivers
could not tell a continuity cut from broken inputs).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from voyage import paths
from voyage.errors import MediaError
from voyage.segment_manifest import load_segment_manifest
from voyage.vision.metrics import sample_frames

Frame: TypeAlias = NDArray[np.uint8]
"""One decoded RGB frame (height x width x 3, uint8)."""

BOUNDARY_RATIO_LIMIT = 3.0
"""Provisional continuity gate, mirroring the §137A harness.

The continuity investigation (DESIGN §22.5) measured ~6x frame-diff jumps
for fresh scenes, so a cross-boundary jump below 3x the in-segment drift
counts as a continuation (PASS), at/above as a cut (FAIL).
"""

DEFAULT_SAMPLES_PER_SEGMENT = 5
"""Frames sampled per segment clip (endpoints included, evenly spaced)."""

DEFAULT_DECODE_WIDTH = 160
"""Decode width in pixels (aspect kept); regulation signal, never pixels."""

_SINGLE_SEGMENT_VERDICT = "N/A (single segment)"
"""Verdict when the run holds one segment, so no boundary exists to score."""


def mean_abs_diff(first_frame: Frame, second_frame: Frame) -> float:
    """Mean absolute gray-level diff in [0, 1] between two RGB frames."""
    gray_first = first_frame.astype(np.float64) / 255.0
    gray_second = second_frame.astype(np.float64) / 255.0
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    delta = abs(gray_first.dot(weights) - gray_second.dot(weights))
    return float(delta.mean())


def boundary_verdict(
    within_mean: float, boundary_mean: float, *, limit: float = BOUNDARY_RATIO_LIMIT
) -> dict[str, Any]:
    """Score a cross-boundary jump against the in-segment drift (pure).

    PASS when the boundary jump stays below `limit` x the within drift.
    A zero within drift means `inf` (any visible jump off a frozen frame
    is a cut), matching the §137A harness so the two can never disagree.
    """
    ratio = boundary_mean / within_mean if within_mean > 0 else float("inf")
    return {
        "within_mean": within_mean,
        "boundary_mean": boundary_mean,
        "boundary_ratio": ratio,
        "limit": limit,
        "verdict": "PASS" if ratio < limit else "FAIL",
    }


def committed_segment_videos(run_dir: Path) -> list[tuple[str, Path]]:
    """Committed (segment_id, video.mp4) pairs in segment order.

    A directory counts as committed when it carries the DONE marker; its
    video must exist (fail loud like the §137A harness — a DONE segment
    without pixels is corruption, never a silent skip).
    """
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    if not segments_root.is_dir():
        raise ValueError(f"no committed segments in {run_dir}")
    committed = sorted(
        entry.name for entry in segments_root.iterdir() if (entry / paths.DONE_MARKER).is_file()
    )
    if not committed:
        raise ValueError(f"no committed segments in {run_dir}")
    pairs: list[tuple[str, Path]] = []
    for segment_id in committed:
        video_path = paths.segment_dir(run_dir, segment_id) / "video.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"missing segment video {video_path}")
        pairs.append((segment_id, video_path))
    return pairs


def summarize_boundaries(
    run_dir: Path | str,
    *,
    samples_per_segment: int = DEFAULT_SAMPLES_PER_SEGMENT,
    decode_width: int = DEFAULT_DECODE_WIDTH,
) -> dict[str, Any]:
    """Within-vs-boundary continuity summary for a committed run (CPU-only).

    Samples each committed segment video (endpoints included), takes mean
    absolute gray diffs between consecutive samples within segments vs
    tail(previous)->head(current) across boundaries. A single-segment run
    reports boundary fields as None with the N/A verdict instead of
    dividing by zero.
    """
    if samples_per_segment < 1:
        raise ValueError(f"samples_per_segment must be >= 1 (got {samples_per_segment})")
    if decode_width < 1:
        raise ValueError(f"decode_width must be >= 1 (got {decode_width})")
    run_path = Path(run_dir)
    pairs = committed_segment_videos(run_path)
    tails: dict[str, Frame] = {}
    heads: dict[str, Frame] = {}
    per_segment_within: dict[str, float] = {}
    within_diffs: list[float] = []
    for segment_id, video_path in pairs:
        frames = sample_frames(video_path, count=samples_per_segment, width=decode_width)
        heads[segment_id] = frames[0]
        tails[segment_id] = frames[-1]
        own_diffs = [mean_abs_diff(first, second) for first, second in pairwise(frames)]
        if own_diffs:
            per_segment_within[segment_id] = sum(own_diffs) / len(own_diffs)
            within_diffs.extend(own_diffs)
    boundaries: list[dict[str, Any]] = []
    for (previous_id, _), (current_id, _) in pairwise(pairs):
        diff = mean_abs_diff(tails[previous_id], heads[current_id])
        boundaries.append({"previous": previous_id, "current": current_id, "diff": diff})
    boundary_diffs = [float(entry["diff"]) for entry in boundaries]
    within_mean: float | None = sum(within_diffs) / len(within_diffs) if within_diffs else None
    boundary_mean: float | None = (
        sum(boundary_diffs) / len(boundary_diffs) if boundary_diffs else None
    )
    continuity: dict[str, Any]
    if within_mean is not None and boundary_mean is not None:
        continuity = boundary_verdict(within_mean, boundary_mean)
    else:
        continuity = {
            "within_mean": within_mean,
            "boundary_mean": boundary_mean,
            "boundary_ratio": None,
            "limit": BOUNDARY_RATIO_LIMIT,
            "verdict": (_SINGLE_SEGMENT_VERDICT if len(pairs) < 2 else "N/A (no frame pairs)"),
        }
    return {
        "run": str(run_path),
        "segments": [segment_id for segment_id, _ in pairs],
        "samples_per_segment": samples_per_segment,
        "decode_width": decode_width,
        "per_segment_within": per_segment_within,
        "boundaries": boundaries,
        **continuity,
    }


def _as_text(value: Any) -> str:
    """JSON value as text; non-strings degrade to empty (never raises)."""
    return value if isinstance(value, str) else ""


def _as_text_list(value: Any) -> list[str]:
    """JSON value as a text list; keeps only the string members."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _as_int(value: Any, default: int) -> int:
    """JSON value as int; bools and non-ints degrade to `default`."""
    if isinstance(value, bool):
        return default
    return value if isinstance(value, int) else default


def _stage_entries(prompt_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Effective block prompts (`stages[].prompt`) as plain dicts."""
    raw_stages = prompt_plan.get("stages")
    if not isinstance(raw_stages, list):
        return []
    entries: list[dict[str, Any]] = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict):
            continue
        entries.append(
            {
                "stage": _as_int(raw_stage.get("stage"), -1),
                "block_start": _as_int(raw_stage.get("block_start"), -1),
                "block_end": _as_int(raw_stage.get("block_end"), -1),
                "prompt": _as_text(raw_stage.get("prompt")),
            }
        )
    return entries


def audit_prompts(run_dir: Path | str) -> list[dict[str, Any]]:
    """Effective block prompts per committed segment, for adherence eyeball.

    Reads each committed segment's manifest (`prompt_plan.stages[]` plus
    the transition destination/mechanism). A torn manifest records its
    error on the entry instead of aborting the whole audit — the eyeball
    should show which segment needs attention, not hide the rest.
    """
    run_path = Path(run_dir)
    segments_root = run_path / paths.SEGMENTS_DIRNAME
    if not segments_root.is_dir():
        raise ValueError(f"no committed segments in {run_dir}")
    committed = sorted(
        entry.name for entry in segments_root.iterdir() if (entry / paths.DONE_MARKER).is_file()
    )
    if not committed:
        raise ValueError(f"no committed segments in {run_dir}")
    entries: list[dict[str, Any]] = []
    for segment_id in committed:
        segment_path = paths.segment_dir(run_path, segment_id)
        manifest_present = (segment_path / paths.SEGMENT_MANIFEST_FILENAME).is_file()
        try:
            manifest = load_segment_manifest(segment_path)
        except MediaError as exc:
            entries.append(
                {
                    "segment_id": segment_id,
                    "manifest_present": manifest_present,
                    "stages": [],
                    "destination": "",
                    "destination_summary": "",
                    "mechanism": "",
                    "intermediate_stages": [],
                    "error": str(exc),
                }
            )
            continue
        prompt_plan = manifest.get("prompt_plan")
        stages = _stage_entries(prompt_plan) if isinstance(prompt_plan, dict) else []
        transition = manifest.get("transition")
        transition_section = transition if isinstance(transition, dict) else {}
        destination_raw = transition_section.get("destination")
        destination = destination_raw if isinstance(destination_raw, dict) else {}
        inner_raw = transition_section.get("transition")
        inner = inner_raw if isinstance(inner_raw, dict) else {}
        entries.append(
            {
                "segment_id": segment_id,
                "manifest_present": manifest_present,
                "stages": stages,
                "destination": _as_text(destination.get("canonical_name")),
                "destination_summary": _as_text(destination.get("summary")),
                "mechanism": _as_text(inner.get("mechanism")),
                "intermediate_stages": _as_text_list(inner.get("intermediate_stages")),
                "error": None,
            }
        )
    return entries


def format_boundary_report(summary: dict[str, Any]) -> str:
    """Human-readable seam-continuity report (pure; no I/O)."""
    lines = [
        f"run: {summary.get('run')}",
        f"segments: {summary.get('segments')}",
        f"samples_per_segment: {summary.get('samples_per_segment')}",
    ]
    per_segment = summary.get("per_segment_within")
    if isinstance(per_segment, dict):
        for segment_id in sorted(per_segment):
            lines.append(f"  within {segment_id}: {per_segment[segment_id]:.4f}")
    boundaries = summary.get("boundaries")
    if isinstance(boundaries, list):
        for entry in boundaries:
            if isinstance(entry, dict):
                lines.append(
                    f"  boundary {entry.get('previous')} -> {entry.get('current')}: "
                    f"{entry.get('diff'):.4f}"
                )
    lines.append(f"within_mean: {summary.get('within_mean')}")
    lines.append(f"boundary_mean: {summary.get('boundary_mean')}")
    lines.append(f"boundary_ratio: {summary.get('boundary_ratio')} (limit {summary.get('limit')})")
    lines.append(f"verdict: {summary.get('verdict')}")
    return "\n".join(lines)


def format_prompt_audit(entries: list[dict[str, Any]]) -> str:
    """Human-readable prompt audit, one block per segment (pure; no I/O)."""
    lines: list[str] = []
    for entry in entries:
        lines.append(f"== segment {entry.get('segment_id')} ==")
        error = entry.get("error")
        if isinstance(error, str) and error:
            lines.append(f"  manifest error: {error}")
            continue
        destination = entry.get("destination")
        if isinstance(destination, str) and destination:
            lines.append(f"  destination: {destination}")
        summary_text = entry.get("destination_summary")
        if isinstance(summary_text, str) and summary_text:
            lines.append(f"  summary: {summary_text}")
        mechanism = entry.get("mechanism")
        if isinstance(mechanism, str) and mechanism:
            lines.append(f"  mechanism: {mechanism}")
        intermediates = entry.get("intermediate_stages")
        if isinstance(intermediates, list):
            for intermediate in intermediates:
                lines.append(f"  stage-text: {intermediate}")
        stages = entry.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                lines.append(
                    f"  [stage {stage.get('stage')} "
                    f"blocks {stage.get('block_start')}..{stage.get('block_end')}]"
                )
                lines.append(f"    {stage.get('prompt')}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """CLI surface for the read-only run inspection tool."""
    parser = argparse.ArgumentParser(
        prog="voyage.boundary_metrics",
        description="CPU-only seam continuity + prompt audit over a committed run.",
    )
    parser.add_argument("--run", required=True, help="Run directory (e.g. Voyage/output/crabz).")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_SEGMENT,
        help=f"Frames sampled per segment (default {DEFAULT_SAMPLES_PER_SEGMENT}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_DECODE_WIDTH,
        help=f"Decode width in pixels (default {DEFAULT_DECODE_WIDTH}).",
    )
    parser.add_argument(
        "--prompts",
        action="store_true",
        help="Append the per-segment prompt audit (adherence eyeball).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON document instead of the text report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: inspect one run directory, never writing into it."""
    args = _build_parser().parse_args(argv)
    run_path = Path(args.run)
    try:
        summary = summarize_boundaries(
            run_path, samples_per_segment=args.samples, decode_width=args.width
        )
        prompt_entries = audit_prompts(run_path) if args.prompts or args.json else []
    except (ValueError, FileNotFoundError, MediaError, OSError) as exc:
        sys.stderr.write(f"boundary_metrics: error: {exc}\n")
        return 3
    if args.json:
        sys.stdout.write(json.dumps({"boundaries": summary, "prompts": prompt_entries}, indent=2))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_boundary_report(summary))
        sys.stdout.write("\n")
        if args.prompts:
            sys.stdout.write("\n")
            sys.stdout.write(format_prompt_audit(prompt_entries))
            sys.stdout.write("\n")
    return 1 if summary.get("verdict") == "FAIL" else 0


if __name__ == "__main__":  # pragma: no cover - thin runner over main()
    raise SystemExit(main())
