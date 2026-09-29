"""Adapter contract: request + generate_segment over injected transport (issue 023).

CPU-only: every test injects a fake transport — no workers, no ffmpeg, no
GPU. Pins that VideoSegmentRequest carries everything the supervisor's
inline payload build used to (per-block prompts/seeds included), that the
streaming-vs-fake fork lives in build_payload, and that
transport_from_restarting_call forwards the supervisor's restarting call
shape untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.backends import (
    Transport,
    VideoBackendAdapter,
    VideoSegmentRequest,
    transport_from_restarting_call,
)
from voyage.config import BACKEND_REGISTRY, ProjectConfig, VideoConfig, with_video_backend
from voyage.errors import ConfigurationError, RecoverableWorkerError


def _base_config() -> ProjectConfig:
    return ProjectConfig(style="contract-probe")


def _configured_video(backend: str, blocks: int) -> VideoConfig:
    configured = with_video_backend(_base_config(), backend)  # type: ignore[arg-type]
    return VideoConfig(**{**configured.video.model_dump(), "blocks_per_segment": blocks})


def _stub_transport(result: dict[str, Any]) -> tuple[Transport, list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def transport(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((operation, payload))
        return result

    return transport, calls


def test_request_from_config_uses_stored_geometry_and_mode() -> None:
    for name, row in BACKEND_REGISTRY.items():
        video = _configured_video(name, 1)
        request = VideoBackendAdapter.request_from_config(
            video, segment_id="000007", prompt="pastel neon line-art, peaceful", seed=11
        )
        assert (request.width, request.height) == (row.width, row.height)
        assert request.fps == row.fps
        assert request.state_mode == row.state_mode
        assert request.segment_seconds == pytest.approx(video.segment_frames / video.fps)
        assert request.scene_cut is False
        assert request.block_prompts is None
        assert request.block_seeds is None


def test_request_from_config_explicit_seconds_scene_cut_and_blocks() -> None:
    video = _configured_video("ltxv", 3)
    request = VideoBackendAdapter.request_from_config(
        video,
        segment_id="000007",
        prompt="stage prompt",
        seed=11,
        scene_cut=True,
        segment_seconds=2.0,
        block_prompts=["one", "two", "three"],
        block_seeds=[101, 102, 103],
    )
    assert request.segment_seconds == 2.0
    assert request.scene_cut is True
    assert request.block_prompts == ["one", "two", "three"]
    assert request.block_seeds == [101, 102, 103]


def test_request_from_config_unknown_backend_rejected() -> None:
    video = VideoConfig.model_construct(backend="nope", segment_frames=48, fps=24)
    with pytest.raises(ConfigurationError, match="unknown video backend"):
        VideoBackendAdapter.request_from_config(video, segment_id="000007", prompt="pastel", seed=1)


def test_streaming_payload_matches_supervisor_inline_shape() -> None:
    transport, calls = _stub_transport({"video": {"frames": 73, "fps": 24}})
    video = _configured_video("ltxv", 3)
    adapter = VideoBackendAdapter(transport, "ltxv", video)
    request = VideoBackendAdapter.request_from_config(
        video, segment_id="000007", prompt="pastel neon line-art, peaceful", seed=11
    )
    adapter.generate_segment(request, Path("seg/video.mp4"))
    assert calls[0][0] == "generate_blocks"
    payload = calls[0][1]
    assert set(payload) == {
        "segment_id",
        "output_path",
        "width",
        "height",
        "fps",
        "frames",
        "prompts",
        "seeds",
        "scene_cuts",
    }
    assert payload["prompts"] == ["pastel neon line-art, peaceful"] * 3
    assert payload["seeds"] == [11, 12, 13]
    assert payload["scene_cuts"] == [False, False, False]
    # ltxv registry row: 96 novel frames/segment (was 48 before the
    # BackendRecord.segment_frames column removed the fake-row leak).
    assert payload["frames"] == 96


def test_staged_block_sequences_ride_through_verbatim() -> None:
    transport, calls = _stub_transport({"video": {"frames": 73, "fps": 24}})
    video = _configured_video("ltxv", 3)
    adapter = VideoBackendAdapter(transport, "ltxv", video)
    request = VideoBackendAdapter.request_from_config(
        video,
        segment_id="000007",
        prompt="fallback",
        seed=11,
        scene_cut=True,
        block_prompts=["stage-one", "stage-two", "stage-three"],
        block_seeds=[101, 102, 103],
    )
    adapter.generate_segment(request, Path("seg/video.mp4"))
    payload = calls[0][1]
    assert payload["prompts"] == ["stage-one", "stage-two", "stage-three"]
    assert payload["seeds"] == [101, 102, 103]
    assert payload["scene_cuts"] == [True, False, False]


def test_block_sequence_length_mismatch_rejected() -> None:
    transport, _ = _stub_transport({})
    video = _configured_video("ltxv", 3)
    adapter = VideoBackendAdapter(transport, "ltxv", video)
    short_prompts = VideoBackendAdapter.request_from_config(
        video, segment_id="s", prompt="p", seed=1, block_prompts=["one", "two"]
    )
    with pytest.raises(ConfigurationError, match="block prompts"):
        adapter.build_payload(short_prompts, Path("seg/video.mp4"))
    long_seeds = VideoBackendAdapter.request_from_config(
        video, segment_id="s", prompt="p", seed=1, block_seeds=[1, 2, 3, 4]
    )
    with pytest.raises(ConfigurationError, match="block seeds"):
        adapter.build_payload(long_seeds, Path("seg/video.mp4"))


def test_fake_payload_shape_and_block_rejection() -> None:
    transport, calls = _stub_transport({"video": {"frames": 48, "fps": 24}})
    video = _configured_video("fake", 1)
    adapter = VideoBackendAdapter(transport, "fake", video)
    request = VideoBackendAdapter.request_from_config(
        video, segment_id="000007", prompt="pastel neon line-art, peaceful", seed=11
    )
    adapter.generate_segment(request, Path("seg/video.mp4"))
    payload = calls[0][1]
    assert set(payload) == {
        "segment_id",
        "output_path",
        "width",
        "height",
        "fps",
        "frames",
        "prompt",
        "seed",
    }
    assert payload["prompt"] == "pastel neon line-art, peaceful"
    assert payload["seed"] == 11
    staged = VideoBackendAdapter.request_from_config(
        video, segment_id="s", prompt="p", seed=1, block_prompts=["one"]
    )
    with pytest.raises(ConfigurationError, match="not streaming"):
        adapter.build_payload(staged, Path("seg/video.mp4"))


def test_transport_from_restarting_call_forwards_shape() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    worker = object()

    def restarting_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"video": {"frames": 48}}

    hook_calls: list[str] = []

    def resume_hook(segment_id: str) -> None:
        hook_calls.append(segment_id)

    transport = transport_from_restarting_call(
        restarting_call, worker, "video", "000007", restart_hook=resume_hook
    )
    result = transport("generate_blocks", {"frames": 48})
    assert result == {"video": {"frames": 48}}
    assert len(calls) == 1
    call_args, call_kwargs = calls[0]
    assert call_args == (worker, "video", "000007", "generate_blocks", {"frames": 48})
    assert call_kwargs == {"restart_hook": resume_hook}


def test_transport_from_restarting_call_defaults_to_no_hook() -> None:
    seen: list[dict[str, Any]] = []

    def restarting_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(dict(kwargs))
        return {}

    transport = transport_from_restarting_call(restarting_call, object(), "video", "000007")
    transport("generate_blocks", {})
    assert seen == [{"restart_hook": None}]


def test_adapter_end_to_end_over_injected_transport() -> None:
    def restarting_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert args[3] == "generate_blocks"
        return {"video": {"frames": 72, "fps": 16, "novel_frames": 72}}

    video = _configured_video("causvid", 1)
    adapter = VideoBackendAdapter(
        transport_from_restarting_call(restarting_call, object(), "video", "000007"),
        "causvid",
        video,
    )
    request = VideoBackendAdapter.request_from_config(
        video, segment_id="000007", prompt="pastel", seed=5
    )
    result = adapter.generate_segment(request, Path("seg/video.mp4"))
    assert result.returned_frames == 72
    assert result.novel_frames == 72
    assert result.native_fps == 16
    assert result.backend == "causvid"


def test_adapter_never_retries_transport_errors() -> None:
    attempts: list[str] = []

    def failing(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        attempts.append(operation)
        raise RecoverableWorkerError("VIDEO_OOM: out of memory")

    adapter = VideoBackendAdapter(failing, "fake", _configured_video("fake", 1))
    request = VideoSegmentRequest(
        segment_id="000007",
        prompt="pastel",
        seed=1,
        width=768,
        height=432,
        fps=24,
        segment_seconds=2.0,
        state_mode="independent_clip",
    )
    with pytest.raises(RecoverableWorkerError, match="VIDEO_OOM"):
        adapter.generate_segment(request, Path("seg/video.mp4"))
    assert attempts == ["generate_blocks"]
