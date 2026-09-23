"""Unit tests for the deterministic visual metrics (DESIGN §§43-44, 100).

Pure-numpy assertions on synthetic frames plus one real-ffmpeg sampling
test (testsrc-generated clip, same pattern as test_audio_planner.py).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from voyage.vision.metrics import (
    frame_histogram,
    histogram_distance,
    histogram_intersection,
    motion_energy,
    palette_distance,
    sample_frames,
    scene_boundary_strength,
    semantic_change_rate,
    style_similarity,
    summarize_segment,
    visual_complexity,
)


def _solid(value: int, width: int = 16, height: int = 16) -> NDArray[np.uint8]:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_motion_energy_zero_on_identical_frames() -> None:
    frame = _solid(128)
    assert motion_energy([frame, frame.copy(), frame.copy()]) == 0.0


def test_motion_energy_one_on_black_white_flip() -> None:
    assert motion_energy([_solid(0), _solid(255)]) == 1.0


def test_motion_energy_partial_change_is_fraction() -> None:
    frame = _solid(0)
    changed = frame.copy()
    changed[:, :8] = 255  # left half flips: 8 of 16 columns
    assert motion_energy([frame, changed]) == 0.5


def test_visual_complexity_zero_on_uniform_frame() -> None:
    assert visual_complexity([_solid(200)]) == 0.0


def test_visual_complexity_marks_the_edge_column() -> None:
    frame = _solid(0)
    frame[:, 8:] = 255  # one sharp vertical edge in a 16-wide frame
    complexity = visual_complexity([frame])
    assert 0.05 < complexity < 0.30


def test_semantic_change_rate_zero_on_identical_frames() -> None:
    frame = _solid(100)
    assert semantic_change_rate([frame, frame.copy()]) == 0.0


def test_semantic_change_rate_one_on_palette_swap() -> None:
    assert semantic_change_rate([_solid(0), _solid(255)]) == 1.0


def test_palette_distance_zero_on_identical_means() -> None:
    assert palette_distance([_solid(77), _solid(77)]) == 0.0


def test_palette_distance_one_on_black_white() -> None:
    assert palette_distance([_solid(0), _solid(255)]) == 1.0


def test_style_similarity_one_against_own_histogram() -> None:
    frame = _solid(150)
    assert style_similarity([frame], frame_histogram(frame)) == 1.0


def test_style_similarity_zero_on_disjoint_palettes() -> None:
    assert style_similarity([_solid(0)], frame_histogram(_solid(255))) == 0.0


def test_style_similarity_none_reference_is_identity() -> None:
    assert style_similarity([_solid(90)], None) == 1.0


def test_scene_boundary_strength_catches_the_cut() -> None:
    frames = [_solid(50), _solid(50), _solid(200)]
    assert scene_boundary_strength(frames) == semantic_change_rate([_solid(50), _solid(200)])


def test_scene_boundary_strength_zero_on_static_sequence() -> None:
    assert scene_boundary_strength([_solid(60), _solid(60)]) == 0.0


def test_histogram_intersection_is_symmetric() -> None:
    first = frame_histogram(_solid(40))
    second = frame_histogram(_solid(41))
    assert histogram_intersection(first, second) == histogram_intersection(second, first)


def test_histogram_distance_zero_on_identical() -> None:
    hist = frame_histogram(_solid(123))
    assert histogram_distance(hist, hist.copy()) == 0.0


def test_summarize_segment_reports_the_six_spec_metrics_in_order() -> None:
    summary = summarize_segment([_solid(100), _solid(110)], frame_histogram(_solid(100)))
    assert list(summary) == [
        "motion_energy",
        "visual_complexity",
        "semantic_change_rate",
        "palette_distance",
        "style_similarity",
        "scene_boundary_strength",
    ]
    assert all(0.0 <= value <= 1.0 for value in summary.values())


def test_summarize_single_frame_degrades_gracefully() -> None:
    summary = summarize_segment([_solid(100)], None)
    assert summary["motion_energy"] == 0.0
    assert summary["scene_boundary_strength"] == 0.0
    assert summary["semantic_change_rate"] == 0.0
    assert summary["style_similarity"] == 1.0


def test_sample_frames_decodes_testsrc_clip(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=8:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    frames = sample_frames(clip, count=3, width=160)
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8


def test_sample_frames_single_count_returns_middle_frame(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=8:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    (frame,) = sample_frames(clip, count=1, width=160)
    assert frame.shape == (120, 160, 3)
