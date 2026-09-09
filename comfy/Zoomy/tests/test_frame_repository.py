"""Tests for the frame repository."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from zoomy.frame_repository import FrameRepository

if TYPE_CHECKING:
    from pathlib import Path


def _write_file(path: Path, *, modification_time: float | None = None) -> None:
    """Create a stub file (and its parents), optionally setting the mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")
    if modification_time is not None:
        os.utime(path, (modification_time, modification_time))


def test_frame_count_and_latest_frame(tmp_path: Path) -> None:
    """Counts and newest frame follow zero-padded filename order."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "Zoomy" / "z_image" / "frame_00001_.png")
    _write_file(tmp_path / "Zoomy" / "z_image" / "frame_00002_.png")
    assert repository.frame_count("z_image") == 2
    assert repository.latest_frame_path("z_image") == (
        tmp_path / "Zoomy" / "z_image" / "frame_00002_.png"
    )


def test_empty_sequence_reports_zero_and_none(tmp_path: Path) -> None:
    """A missing sequence directory behaves like an empty sequence."""
    repository = FrameRepository(tmp_path)
    assert repository.frame_count("z_image") == 0
    assert repository.latest_frame_path("z_image") is None
    assert repository.latest_video_path("z_image") is None


def test_clear_frames_removes_the_sequence_directory(tmp_path: Path) -> None:
    """Clearing deletes the whole sequence folder and resets the count."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "Zoomy" / "z_image" / "frame_00001_.png")
    repository.clear_frames("z_image")
    assert repository.frame_count("z_image") == 0
    assert not (tmp_path / "Zoomy" / "z_image").exists()


def test_latest_video_prefers_the_audio_muxed_twin(tmp_path: Path) -> None:
    """The twin carries the soundtrack, so it wins over a newer silent file."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "Zoomy_z_image_00001.mp4", modification_time=100.0)
    _write_file(tmp_path / "Zoomy_z_image_00001-audio.mp4", modification_time=150.0)
    _write_file(tmp_path / "Zoomy_z_image_00002.mp4", modification_time=200.0)
    assert repository.latest_video_path("z_image") == (tmp_path / "Zoomy_z_image_00001-audio.mp4")


def test_latest_video_falls_back_to_the_silent_file(tmp_path: Path) -> None:
    """Without any twin the newest main file is returned."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "Zoomy_z_image_00001.mp4", modification_time=100.0)
    _write_file(tmp_path / "Zoomy_z_image_00002.mp4", modification_time=200.0)
    assert repository.latest_video_path("z_image") == tmp_path / "Zoomy_z_image_00002.mp4"


def test_sequences_are_isolated_by_key(tmp_path: Path) -> None:
    """Frames of one sequence never count toward another."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "Zoomy" / "ernie_turbo" / "frame_00001_.png")
    _write_file(tmp_path / "Zoomy" / "z_image" / "frame_00001_.png")
    _write_file(tmp_path / "Zoomy" / "z_image" / "frame_00002_.png")
    assert repository.frame_count("ernie_turbo") == 1
    assert repository.frame_count("z_image") == 2
