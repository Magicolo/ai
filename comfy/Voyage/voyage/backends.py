"""Backend interface + fake backends (DESIGN §§78, 83; task groups E/H/K).

The supervisor talks to *workers* over RPC; workers own backend objects.
Fake backends generate real tiny media with ffmpeg so the whole
persistence/validation/finalize path is exercised without GPUs.
Real LongLive / ACE-Step adapters implement the same interface in
their own worker environments later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class VideoBackend(Protocol):
    name: str

    def generate_segment(
        self,
        output_path: Path,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        fps: int,
        frames: int,
    ) -> dict[str, object]: ...


class AudioBackend(Protocol):
    name: str

    def generate_segment(
        self,
        output_path: Path,
        style: str,
        energy: float,
        seed: int,
        sample_rate: int,
        channels: int,
        duration_seconds: float,
    ) -> dict[str, object]: ...
