"""Fake media backends: tiny deterministic ffmpeg renders (DESIGN §§38, 56, I).

Video: `testsrc` pattern (deterministic, no model). Audio: sine tone at
the energy-derived frequency. Both are real codecs in real containers so
ffprobe validation and the finalizer concat path run exactly as in
production.

Why these exist instead of mocks: the commit/validate/finalize contract
is exercised with genuine ffmpeg media (real durations, frames, codecs),
so a regression in probing or assembly fails here on CPU instead of in a
long GPU run. Seeds reach the bytes (issue 040): the video hue angle and
the audio start phase both derive from `seed`, so different seeds render
different media while the same seed stays bit-stable.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

from voyage.errors import MediaError


def video_hue_angle(seed: int) -> float:
    """Hue rotation in degrees derived from `seed` (issue 040).

    Pure helper so tests pin the mapping without rendering: the angle
    wraps at 360 (seeds 360 apart share a hue — documented, acceptable
    for a fake), and the same seed always yields the same angle.
    """
    return float(seed % 360)


def audio_start_phase(seed: int) -> float:
    """Sine start phase in radians derived from `seed` (issue 040).

    Pure helper so tests pin the mapping without rendering: 1000 distinct
    phases across 0..2π, same seed → same phase.
    """
    return (seed % 1000) * 2.0 * math.pi / 1000.0


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
                "-vf",
                f"hue=h={video_hue_angle(seed)}",
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
        del style
        frequency = 220.0 + 220.0 * energy
        # Seed reaches the bytes via the start phase (issue 040): same
        # tone, different phase per seed — `sine` has no phase knob, so the
        # equivalent `aevalsrc` expression carries it explicitly.
        phase = audio_start_phase(seed)
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
                f"aevalsrc=sin(2*PI*{frequency}*t+{phase}):s={sample_rate}:d={duration_seconds}",
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


class FakeSfxBackend:
    """Fake SFX backend behind the same `generate_sfx` shape as MMAudio.

    Deterministic seeded pink noise through ffmpeg so the finalize-time
    windowing/sharding/mix path is genuine with no GPU, no weights, no
    network. The seed reaches the bytes via anoisesrc's seed knob, so
    the same window re-renders byte-identical bytes (ledger-friendly).
    """

    name = "fake"

    def generate_window(
        self,
        output_path: Path,
        caption: str,
        seed: int,
        sample_rate: int,
        channels: int,
        duration_seconds: float,
    ) -> dict[str, object]:
        del caption
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anoisesrc=color=pink:duration={duration_seconds}:seed={seed}:"
                f"sample_rate={sample_rate}",
                "-c:a",
                "pcm_s16le",
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
