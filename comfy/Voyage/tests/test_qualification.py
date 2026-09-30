"""Stream B §137A qualification harness for the `longlive2` backend.

CPU-runnable in the slim image (no torch/GPU): pure metric helpers with
unit tests, plus a fake-backend protocol dry-run proving the GPU-run
procedure (smoke → resolution/FPS → VRAM/RAM → steady state → crash
recovery → 3-segment visual review) works end to end.

The GPU leg itself (real `longlive2` numbers) is driven by
`scripts/qualify.sh` on an idle 4060 Ti and recorded in
`reports/video-backends.md`. Nothing here invents GPU numbers: every
GPU-gated field stays PENDING until measured.

Continuity method: sample evenly spaced frames from each committed
`segments/<id>/video.mp4` (endpoints included), take mean absolute gray
diffs within segments vs across segment boundaries. The continuity
investigation (DESIGN §22.5/§140) measured ~6x frame-diff jumps for fresh
scenes, so the provisional gate is boundary_mean < 3x within_mean.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.cli import validate_run
from voyage.config import load_config
from voyage.supervisor import Supervisor
from voyage.vision.metrics import sample_frames

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "video-backends.md"

QUALIFICATION_STAGES: tuple[str, ...] = (
    "smoke",
    "resolution_fps",
    "vram_ram",
    "steady_state",
    "crash_recovery",
    "visual_review",
)
"""§137A legs in run order: one new frame, geometry check, resource peaks,
warm-up-excluded throughput, kill recovery, 3-segment continuity review."""

REQUIRED_GPU_FIELDS: tuple[str, ...] = (
    "time_to_first_output_s",
    "seconds_generated",
    "wall_seconds",
    "steady_state_ratio",
    "peak_vram_bytes",
    "peak_cpu_ram_bytes",
    "within_mean",
    "boundary_mean",
    "boundary_ratio",
    "continuity_verdict",
    "kill_test_outcome",
)
"""Measured-only fields for a completed GPU leg (task §3: no invented numbers)."""

BOUNDARY_RATIO_LIMIT = 3.0
"""Provisional continuity gate: boundary jump must stay well below the ~6x
fresh-scene failure signature measured in the continuity investigation."""

Frame = NDArray[np.uint8]


def time_to_first_output(elapsed_seconds: list[float]) -> float:
    """Wall time of the first segment (includes model load + warm-up).

    Raises ValueError on empty input (a fresh/failed run with zero commits
    reports instead of IndexError — issue 132).
    """
    if not elapsed_seconds:
        raise ValueError("no segments: time_to_first_output needs at least one elapsed time")
    return elapsed_seconds[0]


def steady_state_mean(elapsed_seconds: list[float]) -> float:
    """Mean per-segment wall with the first (warm-up) segment excluded.

    A single-segment (smoke) run has no post-warm-up sample, so the lone
    value doubles as the mean (making steady_state_ratio 1.0) instead of
    dividing by zero; empty input is a ValueError like above (issue 132).
    """
    if not elapsed_seconds:
        raise ValueError("no segments: steady_state_mean needs at least one elapsed time")
    measured = elapsed_seconds[1:]
    if not measured:
        return elapsed_seconds[0]
    return sum(measured) / len(measured)


def steady_state_ratio(elapsed_seconds: list[float]) -> float:
    """First-segment wall over steady-state mean (>= 1; ~1 means no warm-up cost).

    Empty input is a ValueError (not IndexError): the `[0]` read below
    would crash before the mean's own guard runs (issue 132).
    """
    if not elapsed_seconds:
        raise ValueError("no segments: steady_state_ratio needs at least one elapsed time")
    return elapsed_seconds[0] / steady_state_mean(elapsed_seconds)


def seconds_per_wall_second(frames: int, fps: int, wall_seconds: float) -> float:
    """Novel generated seconds per wall second (throughput, higher is better).

    Raises ValueError on non-positive wall_seconds — a zero coarse-timer
    reading divided by zero before (issue 132).
    """
    if wall_seconds <= 0:
        raise ValueError(f"wall_seconds must be positive (got {wall_seconds})")
    return (frames / fps) / wall_seconds


def mean_abs_diff(first: Frame, second: Frame) -> float:
    """Mean absolute gray-level diff in [0, 1] between two RGB frames."""
    gray_first = first.astype(np.float64) / 255.0
    gray_second = second.astype(np.float64) / 255.0
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    delta = abs(gray_first.dot(weights) - gray_second.dot(weights))
    return float(delta.mean())


def boundary_verdict(
    within_mean: float, boundary_mean: float, *, limit: float = BOUNDARY_RATIO_LIMIT
) -> dict[str, Any]:
    """PASS when the cross-boundary jump stays below `limit` x the in-segment drift."""
    ratio = boundary_mean / within_mean if within_mean > 0 else float("inf")
    return {
        "within_mean": within_mean,
        "boundary_mean": boundary_mean,
        "boundary_ratio": ratio,
        "limit": limit,
        "verdict": "PASS" if ratio < limit else "FAIL",
    }


def missing_fields(record: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    """Required keys absent from a qualification record (honesty gate)."""
    return [key for key in required if key not in record]


def summarize_run(run_dir: Path | str, samples_per_segment: int = 5) -> dict[str, Any]:
    """Build a §137A summary from a committed run (backend-agnostic).

    Reads `segment_committed` metric events for stage timing, then samples
    each committed segment video for within/boundary continuity stats.
    Raises FileNotFoundError when a segment video is missing. A
    single-segment (smoke) run — or any run whose frame sampling yields no
    pairs — reports `within_mean`/`boundary_mean`/`boundary_ratio` as None
    with verdict "N/A (single segment)" instead of ZeroDivisionError
    (issue 132).
    """
    run_path = Path(run_dir)
    metrics_path = run_path / paths.LOGS_DIRNAME / "metrics.jsonl"
    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if '"event": "segment_committed"' in line
    ]
    segment_ids = [str(event["segment_id"]) for event in events]
    elapsed = [float(event["elapsed_seconds"]) for event in events]
    frames_per_segment = [int(event["frames"]) for event in events]
    tails: dict[str, Frame] = {}
    heads: dict[str, Frame] = {}
    within_diffs: list[float] = []
    for segment_id in segment_ids:
        video_path = paths.segment_dir(run_path, segment_id) / "video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"missing segment video {video_path}")
        frames = sample_frames(video_path, count=samples_per_segment)
        heads[segment_id] = frames[0]
        tails[segment_id] = frames[-1]
        within_diffs.extend(mean_abs_diff(first, second) for first, second in pairwise(frames))
    boundary_diffs = [
        mean_abs_diff(tails[previous], heads[current])
        for previous, current in pairwise(segment_ids)
    ]
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
            "verdict": ("N/A (single segment)" if len(segment_ids) < 2 else "N/A (no frame pairs)"),
        }
    return {
        "segments": segment_ids,
        "segment_elapsed_s": elapsed,
        "frames_per_segment": frames_per_segment,
        "time_to_first_output_s": time_to_first_output(elapsed),
        "steady_state_mean_s": steady_state_mean(elapsed),
        "steady_state_ratio": steady_state_ratio(elapsed),
        "continuity": continuity,
    }


def _init_run(run_dir: Path, run_id: str = "qualification") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def test_qualification_stages_cover_137a() -> None:
    assert QUALIFICATION_STAGES == (
        "smoke",
        "resolution_fps",
        "vram_ram",
        "steady_state",
        "crash_recovery",
        "visual_review",
    )


def test_required_fields_cover_task() -> None:
    for field in (
        "time_to_first_output_s",
        "seconds_generated",
        "wall_seconds",
        "steady_state_ratio",
        "peak_vram_bytes",
        "peak_cpu_ram_bytes",
        "kill_test_outcome",
    ):
        assert field in REQUIRED_GPU_FIELDS


def test_missing_fields_gate() -> None:
    assert missing_fields({"a": 1}, ("a", "b")) == ["b"]
    assert missing_fields({"a": 1}, ("a",)) == []


def test_steady_state_math() -> None:
    elapsed = [60.0, 20.0, 20.0]
    assert time_to_first_output(elapsed) == 60.0
    assert steady_state_mean(elapsed) == 20.0
    assert steady_state_ratio(elapsed) == 3.0


def test_steady_state_mean_single_segment_returns_the_sample() -> None:
    """One-segment (smoke) runs have no post-warm-up sample (issue 132)."""
    assert steady_state_mean([20.0]) == 20.0
    assert steady_state_ratio([20.0]) == 1.0


def test_empty_elapsed_raises_value_error_not_index_or_zero_division() -> None:
    """Zero-commit runs report, never IndexError/ZeroDivisionError (issue 132)."""
    with pytest.raises(ValueError, match="no segments"):
        time_to_first_output([])
    with pytest.raises(ValueError, match="no segments"):
        steady_state_mean([])
    with pytest.raises(ValueError, match="no segments"):
        steady_state_ratio([])


def test_seconds_per_wall_second_math() -> None:
    assert seconds_per_wall_second(48, 24, 120.0) == 2.0 / 120.0


def test_seconds_per_wall_second_zero_wall_raises_value_error() -> None:
    """A zero coarse-timer reading reports, never ZeroDivisionError (issue 132)."""
    with pytest.raises(ValueError, match="wall_seconds"):
        seconds_per_wall_second(48, 24, 0.0)


def test_mean_abs_diff_properties() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert mean_abs_diff(frame, frame) == 0.0
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert mean_abs_diff(frame, white) == pytest.approx(1.0)


def test_boundary_verdict_pass_and_fail() -> None:
    passing = boundary_verdict(0.01, 0.02)
    assert passing["verdict"] == "PASS"
    failing = boundary_verdict(0.01, 0.10)
    assert failing["verdict"] == "FAIL"
    assert failing["boundary_ratio"] == 10.0


def test_fake_single_segment_summarize_reports_no_boundary(tmp_path: Path) -> None:
    """Smoke-leg runs summarize with boundary_* None, never ZeroDivisionError (issue 132)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="qualification")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    assert validate_run(run_dir) == []
    summary = summarize_run(run_dir)
    assert summary["segments"] == ["000000"]
    assert summary["steady_state_mean_s"] == summary["time_to_first_output_s"]
    assert summary["steady_state_ratio"] == 1.0
    continuity = summary["continuity"]
    assert continuity["within_mean"] is not None and continuity["within_mean"] >= 0
    assert continuity["boundary_mean"] is None
    assert continuity["boundary_ratio"] is None
    assert continuity["verdict"] == "N/A (single segment)"


def test_fake_three_segment_dry_run(tmp_path: Path) -> None:
    """Protocol dry-run on the fake backend: commit 3, summarize, stay honest."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert Supervisor(run_dir, config).run_segments(3) == ["000000", "000001", "000002"]
    assert validate_run(run_dir) == []
    summary = summarize_run(run_dir)
    assert summary["segments"] == ["000000", "000001", "000002"]
    assert summary["time_to_first_output_s"] > 0
    assert summary["steady_state_mean_s"] > 0
    continuity = summary["continuity"]
    assert continuity["within_mean"] >= 0
    assert continuity["boundary_mean"] >= 0
    assert continuity["verdict"] in ("PASS", "FAIL")
    record = {**summary, "seconds_generated": 1.0, "wall_seconds": 1.0}
    assert "peak_vram_bytes" in missing_fields(record, REQUIRED_GPU_FIELDS)


def test_fake_kill_recovery_dry_run(tmp_path: Path) -> None:
    """Protocol dry-run for crash recovery: killed video worker still commits."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.inject_worker_crash("video")
        segment_id = supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert segment_id == "000000"
    assert (paths.segment_dir(run_dir, segment_id) / paths.DONE_MARKER).exists()
    metrics = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "worker_restart"' in metrics and '"worker": "video"' in metrics
    assert validate_run(run_dir) == []


def test_report_skeleton() -> None:
    """The report exists with a longlive2 leg and an explicit empty LTXV leg."""
    assert REPORT_PATH.exists(), f"missing {REPORT_PATH}"
    text = REPORT_PATH.read_text(encoding="utf-8").lower()
    assert "longlive2" in text
    assert "137a" in text
    assert "ltxv leg" in text
    assert "pending" in text
