"""Access to the rendered frame sequence and finalized videos on disk.

Frames land in ``<output_directory>/Zoomy/<sequence_key>/frame_*.png`` because
the frame workflow saves with the prefix ``Zoomy/<sequence_key>/frame``. Videos
land directly in ``<output_directory>`` as ``Zoomy_<sequence_key>_*.mp4`` — the
VideoHelperSuite combine node writes a silent main file plus a ``-audio`` twin
carrying the muxed soundtrack, and the twin is what previews should show.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

ZOOMY_DIRECTORY_NAME = "Zoomy"
FRAME_FILE_PATTERN = "frame_*.png"
VIDEO_FILE_TEMPLATE = "Zoomy_{sequence_key}*.mp4"
AUDIO_FILE_MARKER = "-audio"


class FrameRepository:
    """Reads and clears zoomy artifacts under one ComfyUI output directory."""

    def __init__(self, output_directory: Path) -> None:
        """Remember the ComfyUI output directory that holds zoomy artifacts."""
        self.output_directory = output_directory

    def frame_directory(self, sequence_key: str) -> Path:
        """Return the directory holding the sequence's rendered frames."""
        return self.output_directory / ZOOMY_DIRECTORY_NAME / sequence_key

    def frame_paths(self, sequence_key: str) -> list[Path]:
        """Return the sequence's frames in render order.

        ComfyUI zero-pads frame counters, so a plain name sort equals
        chronological order.
        """
        directory = self.frame_directory(sequence_key)
        if not directory.is_dir():
            return []
        return sorted(directory.glob(FRAME_FILE_PATTERN))

    def frame_count(self, sequence_key: str) -> int:
        """Return how many frames the sequence currently holds."""
        return len(self.frame_paths(sequence_key))

    def latest_frame_path(self, sequence_key: str) -> Path | None:
        """Return the most recently rendered frame, or ``None`` when empty."""
        frame_paths = self.frame_paths(sequence_key)
        return frame_paths[-1] if frame_paths else None

    def clear_frames(self, sequence_key: str) -> None:
        """Delete the sequence's entire frame directory."""
        shutil.rmtree(self.frame_directory(sequence_key), ignore_errors=True)

    def latest_video_path(self, sequence_key: str) -> Path | None:
        """Return the newest finalized video, preferring the audio-muxed twin.

        The video node always writes a silent main file plus, when an audio
        track is connected, a ``-audio`` twin carrying the muxed soundtrack;
        the twin is the artifact users want to preview. When several videos
        exist the newest by modification time wins, which is the one the
        latest finalize just wrote.
        """
        pattern = VIDEO_FILE_TEMPLATE.format(sequence_key=sequence_key)
        candidates = list(self.output_directory.glob(pattern))
        if not candidates:
            return None
        twins = [path for path in candidates if AUDIO_FILE_MARKER in path.stem]
        pool = twins or candidates
        return max(pool, key=lambda path: path.stat().st_mtime)
