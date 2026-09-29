"""Direct tests for the fake media backends (issue 038).

`fake_backends` previously had zero direct imports — everything exercised
it incidentally through the supervisor — so seed/shape regressions
surfaced only in long runs. These pin the contract: real codecs in real
containers, honest durations/shapes, and seeds reaching the bytes
(issue 040: hue angle for video, start phase for audio).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from voyage.fake_backends import (
    FakeAudioBackend,
    FakeVideoBackend,
    audio_start_phase,
    video_hue_angle,
)
from voyage.media import validate_audio, validate_video

NARROW_WIDTH = 64
NARROW_HEIGHT = 64
NARROW_FPS = 8
NARROW_FRAMES = 8
PROBE_SAMPLE_RATE = 48000
PROBE_CHANNELS = 2
PROBE_DURATION_SECONDS = 0.5


def _sha256hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fake_video_renders_valid_container(tmp_path: Path) -> None:
    output_path = tmp_path / "segment.mp4"
    result = FakeVideoBackend().generate_segment(
        output_path, "a meadow", 11, NARROW_WIDTH, NARROW_HEIGHT, NARROW_FPS, NARROW_FRAMES
    )
    assert result == {
        "frames": NARROW_FRAMES,
        "fps": NARROW_FPS,
        "width": NARROW_WIDTH,
        "height": NARROW_HEIGHT,
    }
    info = validate_video(output_path, NARROW_WIDTH, NARROW_HEIGHT, NARROW_FPS)
    assert info["duration"] > 0


def test_fake_audio_renders_valid_wav_and_flac(tmp_path: Path) -> None:
    backend = FakeAudioBackend()
    for suffix in (".wav", ".flac"):
        output_path = tmp_path / f"take{suffix}"
        result = backend.generate_segment(
            output_path,
            "ambient",
            0.5,
            7,
            PROBE_SAMPLE_RATE,
            PROBE_CHANNELS,
            PROBE_DURATION_SECONDS,
        )
        assert result == {
            "sample_rate": PROBE_SAMPLE_RATE,
            "channels": PROBE_CHANNELS,
            "duration_seconds": PROBE_DURATION_SECONDS,
        }
        info = validate_audio(output_path, PROBE_SAMPLE_RATE, PROBE_CHANNELS)
        assert info["duration"] > 0


def test_fake_video_seed_reaches_bytes(tmp_path: Path) -> None:
    """Same seed → identical bytes; different seeds → different bytes (issue 040)."""
    backend = FakeVideoBackend()
    first = tmp_path / "first.mp4"
    repeat = tmp_path / "repeat.mp4"
    other = tmp_path / "other.mp4"
    for output_path, seed in ((first, 11), (repeat, 11), (other, 12)):
        backend.generate_segment(
            output_path, "a meadow", seed, NARROW_WIDTH, NARROW_HEIGHT, NARROW_FPS, NARROW_FRAMES
        )
    assert _sha256hex(first) == _sha256hex(repeat)
    assert _sha256hex(first) != _sha256hex(other)


def test_fake_audio_seed_reaches_bytes(tmp_path: Path) -> None:
    """Same seed → identical bytes; different seeds → different bytes (issue 040)."""
    backend = FakeAudioBackend()
    first = tmp_path / "first.wav"
    repeat = tmp_path / "repeat.wav"
    other = tmp_path / "other.wav"
    for output_path, seed in ((first, 11), (repeat, 11), (other, 12)):
        backend.generate_segment(
            output_path,
            "ambient",
            0.5,
            seed,
            PROBE_SAMPLE_RATE,
            PROBE_CHANNELS,
            PROBE_DURATION_SECONDS,
        )
    assert _sha256hex(first) == _sha256hex(repeat)
    assert _sha256hex(first) != _sha256hex(other)


def test_video_hue_angle_mapping() -> None:
    assert video_hue_angle(0) == 0.0
    assert video_hue_angle(90) == 90.0
    assert video_hue_angle(360) == 0.0
    assert video_hue_angle(11) == video_hue_angle(11)


def test_audio_start_phase_mapping() -> None:
    assert audio_start_phase(0) == 0.0
    assert audio_start_phase(1000) == 0.0
    assert 0.0 < audio_start_phase(11) < math.pi * 2
    assert audio_start_phase(11) == audio_start_phase(11)
    assert audio_start_phase(11) != audio_start_phase(12)
