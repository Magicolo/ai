"""Tests for the shared frame/video helpers above the heavy imports.

Frame buffers, artifact checks, and the global-music slice run without
torch or ffmpeg installed: the binary path and command runner inject.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zoomy import video_io
from zoomy.errors import AssemblyError, EngineConfigurationError

if TYPE_CHECKING:
    from pathlib import Path


def test_slice_audio_seeks_and_trims_with_flac_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window stem slices out of the global take at its timeline offset."""
    recorded: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...]) -> None:
        recorded.append(command)
        (tmp_path / "stem.flac").write_bytes(b"slice")

    monkeypatch.setattr(video_io, "ffmpeg_binary_cached", lambda: "/usr/bin/ffmpeg")
    video_io.slice_audio(
        tmp_path / "global.flac",
        tmp_path / "stem.flac",
        start_seconds=5.90625,
        duration_seconds=6.90625,
        run=fake_run,
    )
    (command,) = recorded
    assert command[command.index("-ss") + 1] == "5.906"
    assert command[command.index("-t") + 1] == "6.906"
    assert command[command.index("-c:a") + 1] == "flac"


def test_slice_audio_rejects_a_failed_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dying slice fails naming the destination, not silently empty."""
    import subprocess  # noqa: PLC0415

    def failing_run(command: tuple[str, ...]) -> None:
        raise subprocess.CalledProcessError(1, list(command))

    monkeypatch.setattr(video_io, "ffmpeg_binary_cached", lambda: "/usr/bin/ffmpeg")
    with pytest.raises(AssemblyError, match=r"stem\.flac"):
        video_io.slice_audio(
            tmp_path / "global.flac",
            tmp_path / "stem.flac",
            start_seconds=0.0,
            duration_seconds=1.0,
            run=failing_run,
        )


def test_verify_paths_rejects_missing_artifacts(tmp_path: Path) -> None:
    """A window that wrote nothing fails at its own index."""
    with pytest.raises(AssemblyError, match="Segment 3"):
        video_io.verify_paths(3, (tmp_path / "missing.mp4",))


def test_discard_partial_artifacts_tolerates_absent_files(tmp_path: Path) -> None:
    """Cleanup on the failure path never masks the error that caused it."""
    video_io.discard_partial_artifacts((tmp_path / "never-rendered.mp4",))


def test_pad_frames_rejects_empty_batches() -> None:
    """Tiling from zero frames would spin forever; fail fast instead."""
    with pytest.raises(EngineConfigurationError, match="empty frame batch"):
        video_io.pad_frames_to_minimum([], 17)
