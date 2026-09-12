"""Access to the rendered frame sequence and finalized videos on disk.

Frames land in ``<output_directory>/Zoomy/<sequence_key>/frame_*.png`` because
the frame workflow saves with the prefix ``Zoomy/<sequence_key>/frame``. Videos
land directly in ``<output_directory>`` as ``Zoomy_<sequence_key>_*.mp4`` — the
VideoHelperSuite combine node writes a silent main file plus a ``-audio`` twin
carrying the muxed soundtrack, and the twin is what previews should show.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

ZOOMY_DIRECTORY_NAME = "Zoomy"
FRAME_FILE_PATTERN = "frame_*.png"
VIDEO_FILE_TEMPLATE = "Zoomy_{sequence_key}*.mp4"
AUDIO_FILE_MARKER = "-audio"
SEGMENT_FILE_MARKER = "_seg"
ASSEMBLY_DIRECTORY_SUFFIX = "_assembly"
VIDEO_COUNTER_DIGITS = 5
RECENT_FRAME_LIMIT = 8


@dataclass(frozen=True, slots=True)
class SequenceStatistics:
    """One-glance summary of a frame sequence and its finalized video.

    Attributes:
        sequence_key: The sequence these figures describe.
        frame_count: Frames currently in the sequence directory.
        frames_bytes: Combined size of those frames in bytes.
        recent_frame_paths: The newest frames (at most the requested limit),
            oldest first, backing the interface gallery strip.
        video_path: Newest finalized video (audio twin preferred), if any.
        video_bytes: Size of that video in bytes, if any.
        video_modified_timestamp: Its modification time as epoch seconds, for
            display formatting, if any.
    """

    sequence_key: str
    frame_count: int
    frames_bytes: int
    recent_frame_paths: tuple[Path, ...]
    video_path: Path | None
    video_bytes: int | None
    video_modified_timestamp: float | None


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

    def assembly_directory(self, sequence_key: str) -> Path:
        """Return the scratch directory for assembly waves and concat lists."""
        leaf = f"{sequence_key}{ASSEMBLY_DIRECTORY_SUFFIX}"
        return self.output_directory / ZOOMY_DIRECTORY_NAME / leaf

    def segment_twin_paths(self, sequence_key: str, index: int) -> tuple[Path, Path] | None:
        """Return a segment window's music and effects twins, newest each.

        Returns ``None`` unless both twins landed, so the finalize loop never
        assembles half a window. The twins carry the concatenated pixels;
        the full-length soundtrack stems resolve separately.
        """
        music_twin = self._newest_match(f"Zoomy_{sequence_key}_seg{index:03d}_music*-audio.mp4")
        effects_twin = self._newest_match(f"Zoomy_{sequence_key}_seg{index:03d}_sfx*-audio.mp4")
        if music_twin is None or effects_twin is None:
            return None
        return (music_twin, effects_twin)

    def segment_stem_paths(self, sequence_key: str, index: int) -> tuple[Path, Path] | None:
        """Return a segment window's music and effects stems, newest each.

        Stems keep the over-generated overlap tails the twins trim away, so
        both must be present before the window joins the assembly.
        """
        music_stem = self._newest_match(f"Zoomy_{sequence_key}_seg{index:03d}_music_stem*.flac")
        sound_stem = self._newest_match(f"Zoomy_{sequence_key}_seg{index:03d}_sfx_stem*.flac")
        if music_stem is None or sound_stem is None:
            return None
        return (music_stem, sound_stem)

    def next_video_stem(self, sequence_key: str) -> str:
        """Return the next ``Zoomy_<sequence>_NNNNN`` stem, VHS-style.

        Segment intermediates and audio twins never advance the counter, so
        assembly names continue the main video sequence seamlessly.
        """
        prefix = f"Zoomy_{sequence_key}_"
        counters = []
        for candidate in self.output_directory.glob(f"{prefix}*.mp4"):
            stem = candidate.stem
            if AUDIO_FILE_MARKER in stem or SEGMENT_FILE_MARKER in stem:
                continue
            counter_text = stem[len(prefix) :]
            if len(counter_text) == VIDEO_COUNTER_DIGITS and counter_text.isdigit():
                counters.append(int(counter_text))
        return f"{prefix}{max(counters, default=0) + 1:05d}"

    def remove_segment_files(self, sequence_key: str) -> int:
        """Delete a sequence's segment intermediates, returning the count.

        Matches every segment artifact regardless of extension: videos,
        twins, stems, and the metadata preview images VHS writes alongside.
        """
        removed = 0
        for candidate in self.output_directory.glob(f"Zoomy_{sequence_key}_seg*"):
            if candidate.is_dir():
                continue
            try:
                candidate.unlink()
            except OSError:
                continue
            removed += 1
        return removed

    def _newest_match(self, pattern: str) -> Path | None:
        """Return the newest file matching a glob, or ``None`` when empty."""
        candidates = list(self.output_directory.glob(pattern))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

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

    def sequence_statistics(
        self, sequence_key: str, *, recent_frame_limit: int = RECENT_FRAME_LIMIT
    ) -> SequenceStatistics:
        """Summarize one sequence in a single pass for the stats panel.

        File sizes tolerate files vanishing mid-read (ComfyUI may be writing
        while the interface refreshes); such files simply count as empty.
        """
        frame_paths = self.frame_paths(sequence_key)
        recent_frame_paths = (
            tuple(frame_paths[-recent_frame_limit:]) if recent_frame_limit > 0 else ()
        )
        video_path = self.latest_video_path(sequence_key)
        video_bytes: int | None = None
        video_modified_timestamp: float | None = None
        if video_path is not None:
            try:
                video_status = video_path.stat()
            except OSError:
                video_path = None
            else:
                video_bytes = video_status.st_size
                video_modified_timestamp = video_status.st_mtime
        return SequenceStatistics(
            sequence_key=sequence_key,
            frame_count=len(frame_paths),
            frames_bytes=sum(_file_size_bytes(path) for path in frame_paths),
            recent_frame_paths=recent_frame_paths,
            video_path=video_path,
            video_bytes=video_bytes,
            video_modified_timestamp=video_modified_timestamp,
        )


def _file_size_bytes(path: Path) -> int:
    """Return the file size, treating a file that vanished mid-read as empty."""
    try:
        return path.stat().st_size
    except OSError:
        return 0
