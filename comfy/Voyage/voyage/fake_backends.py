"""Fake media backends: tiny deterministic ffmpeg renders.

Video: `testsrc` pattern (deterministic, no model). Audio: `sine`
tone. Both are real codecs in real containers so ffprobe validation
and the finalizer concat path run exactly as in production.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from voyage.errors import MediaError


def _run(argv: list[str]) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise MediaError(f"{argv[0]} failed: {proc.stderr[-2000:]}")


class FakeVideoBackend:
    name = "fake"

    def generate_segment(
        self,
        output_path: Path,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        fps: int,
        frames: int,
    ) -> dict[str, object]:
        del prompt, seed
        duration = frames / fps
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                str(output_path),
            ]
        )
        return {"frames": frames, "fps": fps, "width": width, "height": height}


class FakeAudioBackend:
    name = "fake"

    def generate_segment(
        self,
        output_path: Path,
        style: str,
        energy: float,
        seed: int,
        sample_rate: int,
        channels: int,
        duration_seconds: float,
    ) -> dict[str, object]:
        del style, seed
        frequency = 220.0 + 220.0 * energy
        # Take files are FLAC (ACE-Step's native container); segment slices
        # are WAV. Match the encoder to the output extension so the fake
        # backend stays a drop-in for either (§38: WAV intermediates, FLAC
        # takes).
        codec = "flac" if output_path.suffix.lower() == ".flac" else "pcm_s16le"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={duration_seconds}",
                "-c:a",
                codec,
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate),
                str(output_path),
            ]
        )
        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_seconds": duration_seconds,
        }


class LongLiveBackend:
    """Real LongLive 2.0 adapter — Phase 1/2 work (task group E).

    Pinned commit, loader, stream session, per-block causal generation,
    relative RoPE, recovery replay. Not implemented in the Phase 0
    skeleton; the worker refuses `generate_blocks` until this lands.
    """

    name = "longlive2"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("LongLiveBackend lands in Phase 1/2 (task group E)")


class AceStepBackend:
    """Real ACE-Step 1.5 adapter — Phase 4 work (task group H)."""

    name = "acestep"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("AceStepBackend lands in Phase 4 (task group H)")
