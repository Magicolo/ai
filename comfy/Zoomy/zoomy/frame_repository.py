"""Access to the rendered frame sequence and finalized videos on disk.

Frames land directly in ``<output_directory>/<sequence_key>/frame_*.png``.
Videos land directly in ``<output_directory>`` as ``<sequence_key>_*.mp4`` —
the finalize writes a silent main file plus a ``-audio`` twin carrying the
muxed soundtrack, and the twin is what previews should show.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zoomy.errors import ZoomyError

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image

FRAME_FILE_PATTERN = "frame_*.png"
VIDEO_FILE_TEMPLATE = "{sequence_key}_*.mp4"
AUDIO_FILE_MARKER = "-audio"
SEGMENT_FILE_MARKER = "_seg"
ASSEMBLY_DIRECTORY_SUFFIX = "_assembly"
VIDEO_COUNTER_DIGITS = 5
RECENT_FRAME_LIMIT = 8
_SAFE_SEQUENCE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _check_sequence_key(sequence_key: str) -> None:
    """Reject keys that could escape the output directory or inject globs.

    Every repository path derives from the sequence key, and two methods
    delete (``clear_frames``, ``remove_segment_files``), so every keyed
    entry point validates: only letters, digits, underscore, and hyphen —
    the catalog's key shape — which admits no separators, ``..``, empties,
    absolute paths, or glob metacharacters.
    """
    if not _SAFE_SEQUENCE_KEY.fullmatch(sequence_key):
        message = f"Invalid sequence key: {sequence_key!r}"
        raise ZoomyError(message)


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
    """Reads and clears zoomy artifacts under one output directory."""

    def __init__(self, output_directory: Path) -> None:
        """Remember the output directory that holds zoomy artifacts."""
        self.output_directory = output_directory

    def frame_directory(self, sequence_key: str) -> Path:
        """Return the directory holding the sequence's rendered frames."""
        _check_sequence_key(sequence_key)
        return self.output_directory / sequence_key

    def frame_paths(self, sequence_key: str) -> list[Path]:
        """Return the sequence's frames in render order.

        Frame counters are zero-padded, so a plain name sort equals
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

    def save_next_frame(self, sequence_key: str, frame_image: Image) -> Path:
        """Save one rendered frame under the next counter name and return it.

        Names keep the ``frame_00001_.png`` counter shape, so the lexical
        sort in :meth:`frame_paths` stays chronological.
        """
        directory = self.frame_directory(sequence_key)
        directory.mkdir(parents=True, exist_ok=True)
        frame_number = self.frame_count(sequence_key) + 1
        frame_path = directory / f"frame_{frame_number:05d}_.png"
        frame_image.save(frame_path)
        return frame_path

    def clear_frames(self, sequence_key: str) -> None:
        """Delete the sequence's entire frame directory.

        A missing directory is a no-op; a failed delete raises ZoomyError
        so callers never report a clear they did not perform.
        """
        directory = self.frame_directory(sequence_key)
        if not directory.exists() and not directory.is_symlink():
            return
        try:
            shutil.rmtree(directory)
        except OSError as failure:
            message = f"Could not clear frames of sequence {sequence_key!r}: {failure}"
            raise ZoomyError(message) from failure

    def assembly_directory(self, sequence_key: str) -> Path:
        """Return the scratch directory for assembly waves and concat lists."""
        _check_sequence_key(sequence_key)
        leaf = f"{sequence_key}{ASSEMBLY_DIRECTORY_SUFFIX}"
        return self.output_directory / leaf

    def segment_twin_paths(self, sequence_key: str, index: int) -> tuple[Path, Path] | None:
        """Return a segment window's music and effects twins, newest each.

        Returns ``None`` unless both twins landed, so the finalize loop never
        assembles half a window. The twins carry the concatenated pixels;
        the full-length soundtrack stems resolve separately.
        """
        _check_sequence_key(sequence_key)
        music_twin = self._newest_match(f"{sequence_key}_seg{index:03d}_music*-audio.mp4")
        effects_twin = self._newest_match(f"{sequence_key}_seg{index:03d}_sfx*-audio.mp4")
        if music_twin is None or effects_twin is None:
            return None
        return (music_twin, effects_twin)

    def segment_stem_paths(self, sequence_key: str, index: int) -> tuple[Path, Path] | None:
        """Return a segment window's music and effects stems, newest each.

        Stems keep the over-generated overlap tails the twins trim away, so
        both must be present before the window joins the assembly.
        """
        _check_sequence_key(sequence_key)
        music_stem = self._newest_match(f"{sequence_key}_seg{index:03d}_music_stem*.flac")
        sound_stem = self._newest_match(f"{sequence_key}_seg{index:03d}_sfx_stem*.flac")
        if music_stem is None or sound_stem is None:
            return None
        return (music_stem, sound_stem)

    def next_video_stem(self, sequence_key: str) -> str:
        """Return the next ``<sequence>_NNNNN`` stem.

        Segment intermediates and audio twins never advance the counter, so
        assembly names continue the main video sequence seamlessly.
        """
        _check_sequence_key(sequence_key)
        prefix = f"{sequence_key}_"
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
        twins, stems, and the metadata preview images written alongside.
        """
        _check_sequence_key(sequence_key)
        removed = 0
        for candidate in self.output_directory.glob(f"{sequence_key}_seg*"):
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

        The finalize always writes a silent main file plus, when an audio
        track is connected, a ``-audio`` twin carrying the muxed soundtrack;
        the twin is the artifact users want to preview. Segment windows write
        their own twins, which never count as finished videos. When several
        videos exist the newest by modification time wins, which is the one
        the latest finalize just wrote.
        """
        _check_sequence_key(sequence_key)
        pattern = VIDEO_FILE_TEMPLATE.format(sequence_key=sequence_key)
        candidates = [
            path
            for path in self.output_directory.glob(pattern)
            if SEGMENT_FILE_MARKER not in path.stem
        ]
        if not candidates:
            return None
        twins = [path for path in candidates if AUDIO_FILE_MARKER in path.stem]
        pool = twins or candidates
        return max(pool, key=lambda path: path.stat().st_mtime)

    def sequence_statistics(
        self, sequence_key: str, *, recent_frame_limit: int = RECENT_FRAME_LIMIT
    ) -> SequenceStatistics:
        """Summarize one sequence in a single pass for the stats panel.

        File sizes tolerate files vanishing mid-read (a finalize may be
        writing while the interface refreshes); such files count as empty.
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
