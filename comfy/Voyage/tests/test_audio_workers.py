"""Direct tests for the fake audio worker ops (issue 038).

`workers.audio` had zero direct tests: validators, the `generate_audio`
contract, and the benchmark shape were exercised only incidentally
through full supervisor commits. Rendering here is a sub-second sine
tone via real ffmpeg — no GPU, no weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.media import validate_audio
from voyage.workers import audio as audio_worker


def test_health_reports_fake_backend() -> None:
    assert audio_worker.handle_health({}) == {"status": "READY", "backend": "fake"}


def test_generate_audio_renders_valid_tone(tmp_path: Path) -> None:
    output_path = tmp_path / "segment.wav"
    result = audio_worker.handle_generate_audio(
        {
            "segment_id": "000000",
            "style": "ambient",
            "energy": 0.5,
            "seed": 7,
            "output_path": str(output_path),
            "sample_rate": 48000,
            "channels": 2,
            "duration_seconds": 0.5,
        }
    )
    assert result["artifacts"] == [str(output_path)]
    assert output_path.exists()
    info = validate_audio(output_path, 48000, 2)
    assert info["duration"] > 0


def test_generate_audio_rejects_bad_energy(tmp_path: Path) -> None:
    base_payload: dict[str, Any] = {
        "segment_id": "000000",
        "style": "ambient",
        "energy": 0.5,
        "seed": 7,
        "output_path": str(tmp_path / "segment.wav"),
        "sample_rate": 48000,
        "channels": 2,
        "duration_seconds": 0.5,
    }
    for bad_energy in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="energy"):
            audio_worker.handle_generate_audio({**base_payload, "energy": bad_energy})


def test_generate_audio_rejects_missing_field(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="energy"):
        audio_worker.handle_generate_audio(
            {
                "segment_id": "000000",
                "style": "ambient",
                "seed": 7,
                "output_path": str(tmp_path / "segment.wav"),
                "sample_rate": 48000,
                "channels": 2,
                "duration_seconds": 0.5,
            }
        )


def test_generate_audio_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="duration_seconds"):
        audio_worker.validate_duration_seconds(0.0)
    with pytest.raises(ValueError, match="sample_rate"):
        audio_worker.validate_sample_rate(0)
    with pytest.raises(ValueError, match="channels"):
        audio_worker.validate_channels(3)
    with pytest.raises(ValueError, match="output_path"):
        audio_worker.validate_output_path("   ")
    audio_worker.validate_energy(0.0)
    audio_worker.validate_energy(1.0)


def test_benchmark_reports_take_shape() -> None:
    result = audio_worker.handle_benchmark({"warmup": 0, "measured": 1, "duration_seconds": 0.5})
    assert result["backend"] == "fake"
    assert result["warmup_takes"] == 0
    assert result["measured_takes"] == 1
    assert len(result["take_wall_seconds"]) == 1
    assert result["takes_per_second"] > 0


def test_benchmark_rejects_empty_counts() -> None:
    with pytest.raises(ValueError, match="benchmark"):
        audio_worker.handle_benchmark({"warmup": 0, "measured": 0})


def test_lifecycle_ops_echo_without_state() -> None:
    assert audio_worker.handle_checkpoint({"segment_id": "000000"}) == {
        "checkpoint_id": "audio-000000"
    }
    assert audio_worker.handle_evict_gpu({}) == {"evicted": True}
    assert audio_worker.handle_resume({"checkpoint_id": "audio-000000"}) == {
        "resumed": True,
        "checkpoint_id": "audio-000000",
    }
    assert audio_worker.handle_shutdown({}) == {"stopped": True}
