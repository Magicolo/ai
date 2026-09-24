"""Adapter over generate_blocks (Stream C, DESIGN §§5.1/14/45-46).

CPU-only: every test injects a fake transport — no workers, no ffmpeg,
no GPU. Covers frame/segment accounting, payload mapping, state_mode
passthrough, result normalization, and error propagation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from voyage.backends import (
    BACKEND_STATE_MODES,
    Transport,
    VideoBackendAdapter,
    VideoSegmentRequest,
    frames_for_segment_seconds,
    segment_seconds_for_frames,
)
from voyage.config import VideoConfig
from voyage.errors import ConfigurationError, FatalWorkerError, RecoverableWorkerError


def _video_config(**overrides: Any) -> VideoConfig:
    fields: dict[str, Any] = {"backend": "fake", "blocks_per_segment": 1}
    fields.update(overrides)
    return VideoConfig(**fields)


def _request(**overrides: Any) -> VideoSegmentRequest:
    fields: dict[str, Any] = {
        "segment_id": "000007",
        "prompt": "pastel neon line-art, peaceful",
        "seed": 11,
        "width": 768,
        "height": 432,
        "fps": 24,
        "segment_seconds": 2.0,
        "state_mode": "independent_clip",
    }
    fields.update(overrides)
    return VideoSegmentRequest(**fields)


def _stub_transport(
    result: dict[str, Any],
) -> tuple[Transport, list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def transport(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((operation, payload))
        return result

    return transport, calls


def test_frames_for_segment_seconds_exact() -> None:
    assert frames_for_segment_seconds(2.0, 24) == 48
    assert frames_for_segment_seconds(16.0, 24) == 384


def test_frames_for_segment_seconds_rounds_up() -> None:
    assert frames_for_segment_seconds(5.04, 24) == 121
    assert frames_for_segment_seconds(0.001, 24) == 1


def test_frames_for_segment_seconds_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        frames_for_segment_seconds(0.0, 24)
    with pytest.raises(ValueError):
        frames_for_segment_seconds(2.0, 0)


def test_segment_seconds_for_frames() -> None:
    assert segment_seconds_for_frames(48, 24) == 2.0
    assert segment_seconds_for_frames(121, 24) == pytest.approx(121 / 24)
    with pytest.raises(ValueError):
        segment_seconds_for_frames(0, 24)
    with pytest.raises(ValueError):
        segment_seconds_for_frames(48, 0)


def test_seconds_frames_roundtrip_never_runs_short() -> None:
    for seconds, fps in [(0.5, 24), (2.0, 24), (5.04, 24), (16.0, 24), (1.0, 16)]:
        frames = frames_for_segment_seconds(seconds, fps)
        assert segment_seconds_for_frames(frames, fps) >= seconds


def test_fake_payload_is_single_prompt_form() -> None:
    transport, calls = _stub_transport({"video": {"frames": 48, "fps": 24}})
    adapter = VideoBackendAdapter(transport, "fake", _video_config(blocks_per_segment=3))
    result = adapter.generate_segment(_request(segment_seconds=2.0), Path("seg/video.mp4"))
    assert calls[0][0] == "generate_blocks"
    payload = calls[0][1]
    assert payload["prompt"] == "pastel neon line-art, peaceful"
    assert payload["seed"] == 11
    assert payload["frames"] == 48
    assert "prompts" not in payload
    assert result.returned_frames == 48
    assert result.novel_frames == 48
    assert result.conditioning_frames == 0


def test_streaming_payload_expands_per_block() -> None:
    transport, calls = _stub_transport({"video": {"frames": 73, "fps": 24}})
    config = _video_config(backend="ltxv", blocks_per_segment=3)
    adapter = VideoBackendAdapter(transport, "ltxv", config)
    request = _request(state_mode="reconstructable_prefix", scene_cut=True)
    adapter.generate_segment(request, Path("seg/video.mp4"))
    payload = calls[0][1]
    assert payload["prompts"] == [request.prompt] * 3
    assert payload["seeds"] == [11, 12, 13]
    assert payload["scene_cuts"] == [True, False, False]
    assert payload["segment_id"] == "000007"


def test_streaming_payload_scene_cut_defaults_false() -> None:
    transport, calls = _stub_transport({"video": {"frames": 49, "fps": 24}})
    config = _video_config(backend="longlive2", blocks_per_segment=2)
    adapter = VideoBackendAdapter(transport, "longlive2", config)
    request = _request(state_mode="persistent_kv")
    adapter.generate_segment(request, Path("seg/video.mp4"))
    assert calls[0][1]["scene_cuts"] == [False, False]


def test_backend_state_modes_cover_spec_trio() -> None:
    assert BACKEND_STATE_MODES["longlive2"] == "persistent_kv"
    assert BACKEND_STATE_MODES["ltxv"] == "reconstructable_prefix"
    assert BACKEND_STATE_MODES["causvid"] == "reconstructable_prefix"
    assert BACKEND_STATE_MODES["fake"] == "independent_clip"


def test_capabilities_report_backend_mode_and_streaming() -> None:
    transport, _ = _stub_transport({})
    fake_caps = VideoBackendAdapter(transport, "fake", _video_config()).capabilities()
    assert (fake_caps.backend, fake_caps.state_mode, fake_caps.streaming) == (
        "fake",
        "independent_clip",
        False,
    )
    live_caps = VideoBackendAdapter(transport, "longlive2", _video_config()).capabilities()
    assert (live_caps.backend, live_caps.state_mode, live_caps.streaming) == (
        "longlive2",
        "persistent_kv",
        True,
    )


def test_state_mode_mismatch_raises_before_transport() -> None:
    transport, calls = _stub_transport({})
    adapter = VideoBackendAdapter(transport, "fake", _video_config())
    bad = _request(state_mode="persistent_kv")
    with pytest.raises(ConfigurationError):
        adapter.build_payload(bad, Path("seg/video.mp4"))
    with pytest.raises(ConfigurationError):
        adapter.generate_segment(bad, Path("seg/video.mp4"))
    assert calls == []


def test_matching_state_mode_override_accepted() -> None:
    transport, _ = _stub_transport({})
    adapter = VideoBackendAdapter(transport, "fake", _video_config(), state_mode="independent_clip")
    assert adapter.state_mode == "independent_clip"


def test_mismatched_state_mode_override_rejected() -> None:
    transport, _ = _stub_transport({})
    with pytest.raises(ConfigurationError):
        VideoBackendAdapter(transport, "fake", _video_config(), state_mode="persistent_kv")


def test_unknown_backend_rejected() -> None:
    transport, _ = _stub_transport({})
    with pytest.raises(ConfigurationError):
        VideoBackendAdapter(transport, "imaginary", _video_config())


def test_reported_frames_win_over_requested() -> None:
    transport, _ = _stub_transport({"video": {"frames": 96, "fps": 24}})
    adapter = VideoBackendAdapter(transport, "fake", _video_config())
    result = adapter.generate_segment(_request(segment_seconds=2.0), Path("seg/video.mp4"))
    assert result.requested_frames == 48
    assert result.returned_frames == 96
    assert result.novel_frames == 96
    assert result.backend == "fake"
    assert result.state_mode == "independent_clip"


def test_missing_report_falls_back_to_requested() -> None:
    for worker_result in ({}, {"video": {}}, {"video": {"frames": 0}}, {"video": None}):
        transport, _ = _stub_transport(worker_result)
        adapter = VideoBackendAdapter(transport, "fake", _video_config())
        result = adapter.generate_segment(_request(segment_seconds=2.0), Path("seg/video.mp4"))
        assert result.returned_frames == 48


def test_reported_fps_wins_so_backends_are_never_relabeled() -> None:
    transport, _ = _stub_transport({"video": {"frames": 32, "fps": 16}})
    adapter = VideoBackendAdapter(transport, "fake", _video_config())
    result = adapter.generate_segment(_request(fps=24, segment_seconds=2.0), Path("seg/v.mp4"))
    assert result.native_fps == 16


def test_recoverable_error_propagates_untouched() -> None:
    def failing(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise RecoverableWorkerError("VIDEO_OOM: out of memory")

    adapter = VideoBackendAdapter(failing, "fake", _video_config())
    with pytest.raises(RecoverableWorkerError, match="VIDEO_OOM"):
        adapter.generate_segment(_request(), Path("seg/video.mp4"))


def test_fatal_error_propagates_untouched() -> None:
    def failing(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise FatalWorkerError("WORKER_ERROR: boom")

    adapter = VideoBackendAdapter(failing, "fake", _video_config())
    with pytest.raises(FatalWorkerError, match="WORKER_ERROR"):
        adapter.generate_segment(_request(), Path("seg/video.mp4"))


def test_request_rejects_empty_id_and_nonpositive_geometry() -> None:
    with pytest.raises(ValidationError):
        _request(segment_id="  ")
    with pytest.raises(ValidationError):
        _request(fps=0)
    with pytest.raises(ValidationError):
        _request(segment_seconds=0.0)


def test_config_segment_seconds_bridges_stored_schema() -> None:
    transport, _ = _stub_transport({})
    adapter = VideoBackendAdapter(transport, "fake", _video_config())
    assert adapter.config_segment_seconds() == 2.0
