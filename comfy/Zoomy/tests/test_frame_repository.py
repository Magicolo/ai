"""Tests for the frame repository."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from hypothesis import given
from hypothesis import strategies as st

from zoomy.frame_repository import RECENT_FRAME_LIMIT, FrameRepository

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def temporary_repository() -> Iterator[tuple[FrameRepository, Path]]:
    """Yield a repository in a fresh temporary directory per Hypothesis example.

    Sharing pytest's function-scoped tmp_path across examples leaks files
    from one example into the next; a dedicated directory per example keeps
    every case hermetic. Everything lives under the container's /tmp and is
    removed automatically.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_directory = Path(temporary_directory)
        yield FrameRepository(output_directory), output_directory


def _write_file(path: Path, *, modification_time: float | None = None) -> None:
    """Create a stub file (and its parents), optionally setting the mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")
    if modification_time is not None:
        os.utime(path, (modification_time, modification_time))


def test_frame_count_and_latest_frame(tmp_path: Path) -> None:
    """Counts and newest frame follow zero-padded filename order."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image" / "frame_00001_.png")
    _write_file(tmp_path / "z_image" / "frame_00002_.png")
    assert repository.frame_count("z_image") == 2
    assert repository.latest_frame_path("z_image") == (tmp_path / "z_image" / "frame_00002_.png")


def test_empty_sequence_reports_zero_and_none(tmp_path: Path) -> None:
    """A missing sequence directory behaves like an empty sequence."""
    repository = FrameRepository(tmp_path)
    assert repository.frame_count("z_image") == 0
    assert repository.latest_frame_path("z_image") is None
    assert repository.latest_video_path("z_image") is None


def test_save_next_frame_numbers_sequentially(tmp_path: Path) -> None:
    """Saved frames land under incrementing counters in render order."""
    from PIL import Image as PillowImage  # noqa: PLC0415

    repository = FrameRepository(tmp_path)
    first = repository.save_next_frame("z_image", PillowImage.new("RGB", (4, 4)))
    second = repository.save_next_frame("z_image", PillowImage.new("RGB", (4, 4)))
    assert first.name == "frame_00001_.png"
    assert second.name == "frame_00002_.png"
    assert repository.frame_count("z_image") == 2
    assert repository.latest_frame_path("z_image") == second


def test_clear_frames_removes_the_sequence_directory(tmp_path: Path) -> None:
    """Clearing deletes the whole sequence folder and resets the count."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image" / "frame_00001_.png")
    repository.clear_frames("z_image")
    assert repository.frame_count("z_image") == 0
    assert not (tmp_path / "z_image").exists()


def test_latest_video_prefers_the_audio_muxed_twin(tmp_path: Path) -> None:
    """The twin carries the soundtrack, so it wins over a newer silent file."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image_00001.mp4", modification_time=100.0)
    _write_file(tmp_path / "z_image_00001-audio.mp4", modification_time=150.0)
    _write_file(tmp_path / "z_image_00002.mp4", modification_time=200.0)
    assert repository.latest_video_path("z_image") == (tmp_path / "z_image_00001-audio.mp4")


def test_latest_video_falls_back_to_the_silent_file(tmp_path: Path) -> None:
    """Without any twin the newest main file is returned."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image_00001.mp4", modification_time=100.0)
    _write_file(tmp_path / "z_image_00002.mp4", modification_time=200.0)
    assert repository.latest_video_path("z_image") == tmp_path / "z_image_00002.mp4"


def test_latest_video_ignores_segment_twins(tmp_path: Path) -> None:
    """Segment window twins never pose as the finished video."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image_00001.mp4", modification_time=100.0)
    _write_file(tmp_path / "z_image_00001-audio.mp4", modification_time=150.0)
    _write_file(tmp_path / "z_image_seg000_music_00001-audio.mp4", modification_time=300.0)
    assert repository.latest_video_path("z_image") == (tmp_path / "z_image_00001-audio.mp4")


def test_sequences_are_isolated_by_key(tmp_path: Path) -> None:
    """Frames of one sequence never count toward another."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "ernie_turbo" / "frame_00001_.png")
    _write_file(tmp_path / "z_image" / "frame_00001_.png")
    _write_file(tmp_path / "z_image" / "frame_00002_.png")
    assert repository.frame_count("ernie_turbo") == 1
    assert repository.frame_count("z_image") == 2


@given(frame_count=st.integers(min_value=0, max_value=30))
def test_frame_inventory_scales(frame_count: int) -> None:
    """Any number of zero-padded frames counts and reports consistently."""
    with temporary_repository() as (repository, output_directory):
        for number in range(1, frame_count + 1):
            _write_file(output_directory / "z_image" / f"frame_{number:05d}_.png")
        assert repository.frame_count("z_image") == frame_count
        statistics = repository.sequence_statistics("z_image")
        assert statistics.frame_count == frame_count
        assert statistics.frames_bytes == frame_count * len(b"stub")
        assert len(statistics.recent_frame_paths) == min(frame_count, RECENT_FRAME_LIMIT)
        if frame_count:
            assert statistics.recent_frame_paths[-1].name == f"frame_{frame_count:05d}_.png"


@given(
    frame_count=st.integers(min_value=0, max_value=20),
    limit=st.integers(min_value=0, max_value=10),
)
def test_recent_frame_limit_is_honored(frame_count: int, limit: int) -> None:
    """The recent strip holds at most the requested limit, newest last."""
    with temporary_repository() as (repository, output_directory):
        for number in range(1, frame_count + 1):
            _write_file(output_directory / "z_image" / f"frame_{number:05d}_.png")
        statistics = repository.sequence_statistics("z_image", recent_frame_limit=limit)
        assert len(statistics.recent_frame_paths) == min(frame_count, limit)
        names = [path.name for path in statistics.recent_frame_paths]
        assert names == sorted(names)


@given(
    # Whole-second mtimes: distinct by construction and exact through any
    # filesystem timestamp granularity, so ties are impossible by design.
    modification_times=st.lists(
        st.integers(min_value=1_000_000_000, max_value=2_000_000_000),
        min_size=1,
        max_size=8,
        unique=True,
    )
)
def test_newest_video_wins(modification_times: list[int]) -> None:
    """Distinct modification times always resolve to the newest video."""
    with temporary_repository() as (repository, output_directory):
        for index, modified in enumerate(modification_times):
            _write_file(
                output_directory / f"z_image_{index:05d}.mp4",
                modification_time=modified,
            )
        expected = max(range(len(modification_times)), key=modification_times.__getitem__)
        assert repository.latest_video_path("z_image") == (
            output_directory / f"z_image_{expected:05d}.mp4"
        )


def test_sequence_statistics_summarizes_frames_and_video(tmp_path: Path) -> None:
    """One call reports counts, bytes, recent paths, and video details."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image" / "frame_00001_.png")
    _write_file(tmp_path / "z_image" / "frame_00002_.png")
    _write_file(tmp_path / "z_image" / "frame_00003_.png")
    _write_file(tmp_path / "z_image_00001-audio.mp4", modification_time=200.0)
    statistics = repository.sequence_statistics("z_image", recent_frame_limit=2)
    assert statistics.sequence_key == "z_image"
    assert statistics.frame_count == 3
    assert statistics.frames_bytes == 3 * len(b"stub")
    assert [path.name for path in statistics.recent_frame_paths] == [
        "frame_00002_.png",
        "frame_00003_.png",
    ]
    assert statistics.video_path == tmp_path / "z_image_00001-audio.mp4"
    assert statistics.video_bytes == len(b"stub")
    assert statistics.video_modified_timestamp == 200.0


def test_sequence_statistics_of_an_empty_sequence(tmp_path: Path) -> None:
    """Missing directories yield zeros and Nones instead of errors."""
    repository = FrameRepository(tmp_path)
    statistics = repository.sequence_statistics("z_image")
    assert statistics.frame_count == 0
    assert statistics.frames_bytes == 0
    assert statistics.recent_frame_paths == ()
    assert statistics.video_path is None
    assert statistics.video_bytes is None
    assert statistics.video_modified_timestamp is None


def test_segment_twin_paths_resolves_the_newest_music_and_effects_twins(
    tmp_path: Path,
) -> None:
    """Each window's two soundtrack twins resolve independently by recency."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image_seg002_music_00001-audio.mp4", modification_time=100.0)
    _write_file(tmp_path / "z_image_seg002_music_00002-audio.mp4", modification_time=200.0)
    _write_file(tmp_path / "z_image_seg002_sfx_00001-audio.mp4", modification_time=150.0)
    twins = repository.segment_twin_paths("z_image", 2)
    assert twins == (
        tmp_path / "z_image_seg002_music_00002-audio.mp4",
        tmp_path / "z_image_seg002_sfx_00001-audio.mp4",
    )


def test_segment_twin_paths_needs_both_stems(tmp_path: Path) -> None:
    """A window with only one rendered twin is not ready for assembly."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image_seg002_music_00001-audio.mp4")
    assert repository.segment_twin_paths("z_image", 2) is None
    assert repository.segment_twin_paths("z_image", 3) is None


def test_segment_stem_paths_resolves_the_newest_flac_stems(tmp_path: Path) -> None:
    """Each window's two overlap stems resolve independently by recency."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image_seg002_music_stem.flac", modification_time=100.0)
    _write_file(tmp_path / "z_image_seg002_music_stem_00001.flac", modification_time=200.0)
    _write_file(tmp_path / "z_image_seg002_sfx_stem.flac", modification_time=150.0)
    stems = repository.segment_stem_paths("z_image", 2)
    assert stems == (
        tmp_path / "z_image_seg002_music_stem_00001.flac",
        tmp_path / "z_image_seg002_sfx_stem.flac",
    )
    assert repository.segment_stem_paths("z_image", 3) is None


def test_next_video_stem_continues_the_main_sequence(tmp_path: Path) -> None:
    """Assembly names follow the VHS counter, skipping twins and segments."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image_00001.mp4")
    _write_file(tmp_path / "z_image_00001-audio.mp4")
    _write_file(tmp_path / "z_image_00002.mp4")
    _write_file(tmp_path / "z_image_seg000_music_00001-audio.mp4")
    assert repository.next_video_stem("z_image") == "z_image_00003"


def test_next_video_stem_starts_a_fresh_sequence(tmp_path: Path) -> None:
    """No previous videos means the first counter, matching VHS numbering."""
    repository = FrameRepository(tmp_path)
    assert repository.next_video_stem("z_image") == "z_image_00001"


def test_remove_segment_files_deletes_only_segments(tmp_path: Path) -> None:
    """Post-mux cleanup removes every intermediate kind and reports the count."""
    repository = FrameRepository(tmp_path)
    _write_file(tmp_path / "z_image_seg000_music_00001-audio.mp4")
    _write_file(tmp_path / "z_image_seg000_sfx_00001.mp4")
    _write_file(tmp_path / "z_image_seg000_music_stem.flac")
    _write_file(tmp_path / "z_image_seg000_music_00001.png")
    _write_file(tmp_path / "z_image_00001.mp4")
    _write_file(tmp_path / "z_image_00001-audio.mp4")
    assert repository.remove_segment_files("z_image") == 4
    assert (tmp_path / "z_image_00001.mp4").exists()
    assert (tmp_path / "z_image_00001-audio.mp4").exists()


def test_assembly_directory_lives_beside_the_sequence(tmp_path: Path) -> None:
    """Scratch waves and concat lists stay inside the zoomy output tree."""
    repository = FrameRepository(tmp_path)
    assert repository.assembly_directory("z_image") == (tmp_path / "z_image_assembly")
