"""Audio RPC/library parameter validation (issue 063).

CPU-only: every rejection happens before the ACE stack loads or ffmpeg
runs, so no GPU, weights, or audio rendering is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.audio.acestep import (
    validate_bpm,
    validate_duration_seconds,
    validate_reference_audio,
    validate_task_type,
)
from voyage.workers import audio_acestep


def test_bpm_rejects_zero_and_out_of_range() -> None:
    for bad_bpm in (0, -12, 301, 1000000):
        with pytest.raises(ValueError, match="bpm"):
            validate_bpm(bad_bpm)


def test_bpm_accepts_none_and_musical_range() -> None:
    validate_bpm(None)
    for good_bpm in (1, 60, 110, 140, 300):
        validate_bpm(good_bpm)


def test_duration_rejects_non_positive_and_non_finite() -> None:
    for bad_duration in (0.0, -3.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="duration_seconds"):
            validate_duration_seconds(bad_duration)


def test_duration_accepts_positive_values() -> None:
    validate_duration_seconds(0.5)
    validate_duration_seconds(15.0)


def test_task_type_rejects_unknown_modes() -> None:
    for bad_task in ("not-a-task", "", "variation"):
        with pytest.raises(ValueError, match="task_type"):
            validate_task_type(bad_task)


def test_task_type_accepts_known_modes() -> None:
    validate_task_type("text2music")
    validate_task_type("repaint")


def test_reference_audio_rejects_non_paths_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reference_audio"):
        validate_reference_audio(12345)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reference_audio"):
        validate_reference_audio(str(tmp_path / "missing.wav"))


def test_reference_audio_accepts_none_and_existing_file(tmp_path: Path) -> None:
    validate_reference_audio(None)
    existing = tmp_path / "take.wav"
    existing.write_bytes(b"fake-wav")
    validate_reference_audio(existing)
    validate_reference_audio(str(existing))


def test_sample_rate_and_channels_bounds() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        audio_acestep.validate_sample_rate(0)
    with pytest.raises(ValueError, match="channels"):
        audio_acestep.validate_channels(0)
    with pytest.raises(ValueError, match="channels"):
        audio_acestep.validate_channels(3)
    audio_acestep.validate_sample_rate(48000)
    audio_acestep.validate_channels(1)
    audio_acestep.validate_channels(2)


def _generate_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "segment_id": "000001",
        "style": "pastel neon line-art, peaceful",
        "energy": 0.5,
        "seed": 7,
        "output_path": "/tmp/voyage-audio-probe/take.wav",
        "sample_rate": 48000,
        "channels": 2,
        "duration_seconds": 15.0,
    }
    payload.update(overrides)
    return payload


def test_handle_generate_audio_rejects_bad_bpm_before_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> Any:
        raise AssertionError("stack must not load for invalid params")

    monkeypatch.setattr(audio_acestep, "_require_stack", _boom)
    with pytest.raises(ValueError, match="bpm"):
        audio_acestep.handle_generate_audio(_generate_payload(bpm=0))


def test_handle_generate_audio_rejects_bad_audio_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> Any:
        raise AssertionError("stack must not load for invalid params")

    monkeypatch.setattr(audio_acestep, "_require_stack", _boom)
    with pytest.raises(ValueError, match="sample_rate"):
        audio_acestep.handle_generate_audio(_generate_payload(sample_rate=0))
    with pytest.raises(ValueError, match="channels"):
        audio_acestep.handle_generate_audio(_generate_payload(channels=0))
    with pytest.raises(ValueError, match="duration_seconds"):
        audio_acestep.handle_generate_audio(_generate_payload(duration_seconds=-3.0))
    with pytest.raises(ValueError, match="task_type"):
        audio_acestep.handle_generate_audio(_generate_payload(task_type="not-a-task"))
    with pytest.raises(ValueError, match="reference_audio"):
        audio_acestep.handle_generate_audio(_generate_payload(reference_audio=12345))
