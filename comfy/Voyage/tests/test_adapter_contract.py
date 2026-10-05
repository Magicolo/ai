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
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.backends import (
    BACKEND_STATE_MODES,
    Transport,
    VideoBackendAdapter,
    VideoSegmentRequest,
    frames_for_segment_seconds,
    segment_seconds_for_frames,
    transport_from_restarting_call,
)
from voyage.cli_validate import validate_run
from voyage.config import (
    BACKEND_REGISTRY,
    ProjectConfig,
    VideoConfig,
    with_video_backend,
)
from voyage.errors import (
    ConfigurationError,
    FatalWorkerError,
    MediaError,
    RecoverableWorkerError,
)
from voyage.hashing import sha256_file as hashing_sha256_file
from voyage.models import StyleSpec
from voyage.persistence import read_effective_config, read_state
from voyage.seeds import video_seed
from voyage.supervisor import CoveredAudio, ProposedSegment, RenderedVideo, Supervisor
from voyage.supervisor import VideoBackendAdapter as SupervisorAdapter  # type: ignore[attr-defined]


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


# --- 088 fold: tests/test_backends_adapter.py (22 tests) ---
# """Adapter over generate_blocks (Stream C, DESIGN §§5.1/14/45-46).
#
# CPU-only: every test injects a fake transport — no workers, no ffmpeg,
# no GPU. Covers frame/segment accounting, payload mapping, state_mode
# passthrough, result normalization, and error propagation.
# """
# NOTE: source `_stub_transport` is semantically identical to this file's
# `_stub_transport` (modulo signature wrapping) — reused, not duplicated.
# Drift adaptations (assertions byte-identical, see 124 precedent):
# `_video_config` ignore `[literal-required]`→`[index]` (022 Literals);
# `test_unknown_backend_rejected` gains `[arg-type]` (intentional invalid);
# `test_missing_report_falls_back_to_requested` pre-loop declaration
# (var-annotated).


def _video_config(**overrides: Any) -> VideoConfig:
    from voyage.config import BACKEND_REGISTRY

    backend = str(overrides.get("backend", "fake"))
    row = BACKEND_REGISTRY[backend]  # type: ignore[index]
    fields: dict[str, Any] = {
        "backend": backend,
        "blocks_per_segment": 1,
        # Registry row owns the per-backend shape (2026-09-29 ltxv
        # decision: VideoConfig field defaults are the ltxv row, so a
        # bare backend= override would otherwise leak them).
        "segment_frames": row.segment_frames,
        "fps": row.fps,
    }
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
    config = _video_config(backend="ltxv", blocks_per_segment=2)
    adapter = VideoBackendAdapter(transport, "ltxv", config)
    request = _request(state_mode="reconstructable_prefix")
    adapter.generate_segment(request, Path("seg/video.mp4"))
    assert calls[0][1]["scene_cuts"] == [False, False]


def test_backend_state_modes_cover_spec_trio() -> None:
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
    live_caps = VideoBackendAdapter(transport, "ltxv", _video_config()).capabilities()
    assert (live_caps.backend, live_caps.state_mode, live_caps.streaming) == (
        "ltxv",
        "reconstructable_prefix",
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
        VideoBackendAdapter(transport, "imaginary", _video_config())  # type: ignore[arg-type]


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
    worker_result: dict[str, Any]
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


def test_causvid_reported_novel_and_conditioning_win() -> None:
    """CausVid's 72-novel/9-conditioning accounting must survive the adapter."""
    transport, _ = _stub_transport(
        {"video": {"frames": 81, "fps": 16, "novel_frames": 72, "conditioning_frames": 9}}
    )
    adapter = VideoBackendAdapter(
        transport, "causvid", _video_config(backend="causvid", blocks_per_segment=1)
    )
    result = adapter.generate_segment(
        _request(fps=16, segment_seconds=4.5, state_mode="reconstructable_prefix"),
        Path("seg/video.mp4"),
    )
    assert result.requested_frames == 72
    assert result.returned_frames == 81
    assert result.novel_frames == 72
    assert result.conditioning_frames == 9
    assert result.native_fps == 16
    assert result.backend == "causvid"
    assert result.state_mode == "reconstructable_prefix"


# --- 088 fold: tests/test_commit_split.py (7 tests) ---
# """Commit-split tests (issues 020, 023-supervisor-half).
#
# Helper-level coverage for the `_propose/_render_video/_cover_audio/_commit`
# split of `commit_one_segment` (020) plus the adapter-through-supervisor
# wiring (023): the commit path must call `VideoBackendAdapter.generate_segment`
# while keeping the staged per-block prompts, the issue-006 frame ceiling,
# and the recovery-tape confinement that the adapter alone cannot express.
#
# All against fake backends or stubbed transports (real media, no GPU).
# """
# Drift adaptations (assertions byte-identical, see 124 precedent):
# ignore codes are per-line as merged-solo mypy demands (incompatible-value
# assigns carry assignment, compatible restores carry method-assign);
# state-line arg-type ignores kept on the three streaming `_render_video`
# calls; config-line ignores + the backend-assignment ignore dropped (stale
# after pruning); supervisor re-export keeps an attr-defined ignore
# (import source unchanged).


def _started_supervisor(run_dir: Path) -> Supervisor:
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    return supervisor


def test_supervisor_sha256_delegates_to_hashing(tmp_path: Path) -> None:
    """`supervisor.sha256_file` is the shared helper, not a sixth copy (021)."""
    import voyage.supervisor as supervisor_module

    assert supervisor_module.sha256_file is hashing_sha256_file
    probe = tmp_path / "probe.bin"
    probe.write_bytes(b"voyage" * 4096)
    assert supervisor_module.sha256_file(probe) == hashing_sha256_file(probe)


def test_propose_segment_returns_staged_plan(tmp_path: Path) -> None:
    """`_propose_segment` accepts a decision and maps stages to blocks (020)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="commit-split")
    supervisor = _started_supervisor(run_dir)
    try:
        config = read_effective_config(run_dir)
        state = read_state(run_dir)
        stage_seconds: dict[str, float] = {}
        proposed = supervisor._propose_segment(
            config, state, 0, "000000", StyleSpec(prompt=config.style), stage_seconds
        )
        assert isinstance(proposed, ProposedSegment)
        assert proposed.decision.destination_concept.strip()
        assert proposed.prompt_plan.stages
        assert len(proposed.block_prompts) == proposed.num_blocks == 1
        assert set(stage_seconds) == {"director"}
        assert proposed.prefetch_hit is False
        assert proposed.drift_hold == (0 % max(1, config.voyage.drift_every_n_segments) != 0)
    finally:
        supervisor.stop_workers()


def test_render_video_goes_through_adapter(tmp_path: Path) -> None:
    """The commit video path calls the adapter with the fake payload (023)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="commit-split")
    supervisor = _started_supervisor(run_dir)
    try:
        config = read_effective_config(run_dir)
        state = read_state(run_dir)
        stage_seconds: dict[str, float] = {}
        proposed = supervisor._propose_segment(
            config, state, 0, "000000", StyleSpec(prompt=config.style), stage_seconds
        )
        segment = paths.segment_dir(run_dir, "000000")
        segment.mkdir(parents=True, exist_ok=True)

        adapter_calls: list[tuple[Any, Path]] = []
        video_calls: list[dict[str, Any]] = []
        original_call = supervisor._call_with_restart

        def _recording_call(
            worker: Any,
            worker_name: str,
            segment_id: str,
            operation: str,
            payload: dict[str, object],
            restart_hook: Any = None,
        ) -> dict[str, object]:
            if operation == "generate_blocks":
                video_calls.append(dict(payload))
            return original_call(worker, worker_name, segment_id, operation, payload, restart_hook)

        supervisor._call_with_restart = _recording_call  # type: ignore[assignment]
        try:
            original_method = SupervisorAdapter.generate_segment

            def _spy_generate(adapter: Any, request: Any, output_path: Path) -> Any:
                adapter_calls.append((request, output_path))
                return original_method(adapter, request, output_path)

            SupervisorAdapter.generate_segment = _spy_generate  # type: ignore[assignment]
            try:
                rendered = supervisor._render_video(
                    config, state, 0, "000000", segment, proposed, stage_seconds
                )
            finally:
                SupervisorAdapter.generate_segment = original_method  # type: ignore[method-assign]
        finally:
            supervisor._call_with_restart = original_call  # type: ignore[method-assign]

        assert len(adapter_calls) == 1
        request, output_path = adapter_calls[0]
        assert request.prompt == proposed.prompt_plan.stages[0].prompt
        assert request.seed == video_seed(config.seed, 0, 0)
        assert request.state_mode == "independent_clip"
        assert output_path == segment / "video.mp4"
        assert len(video_calls) == 1
        assert video_calls[0]["prompt"] == proposed.prompt_plan.stages[0].prompt
        assert "prompts" not in video_calls[0]
        assert isinstance(rendered, RenderedVideo)
        assert rendered.frames > 0
        assert rendered.duration == rendered.frames / config.video.fps
        assert rendered.video_time == 0.0
        assert rendered.recovery_tape is None
        assert "video" in stage_seconds
    finally:
        supervisor.stop_workers()


def test_cover_audio_and_commit_advance_state(tmp_path: Path) -> None:
    """`_cover_audio` + `_commit_segment` commit exactly one segment (020).

    All-deferred pin: cover is always deferred (take_action "deferred",
    empty take_ids — finalize takes live under run/audio/takes.jsonl),
    commit is video-only with no audio.wav ever written.
    """
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="commit-split")
    supervisor = _started_supervisor(run_dir)
    try:
        config = read_effective_config(run_dir)
        state = read_state(run_dir)
        stage_seconds: dict[str, float] = {}
        proposed = supervisor._propose_segment(
            config, state, 0, "000000", StyleSpec(prompt=config.style), stage_seconds
        )
        segment = paths.segment_dir(run_dir, "000000")
        segment.mkdir(parents=True, exist_ok=True)
        rendered = supervisor._render_video(
            config, state, 0, "000000", segment, proposed, stage_seconds
        )
        covered = supervisor._cover_audio(
            config,
            0,
            "000000",
            segment,
            rendered.video_time,
            rendered.duration,
            proposed.decision,
            rendered.recovery_tape,
            stage_seconds,
        )
        assert isinstance(covered, CoveredAudio)
        assert covered.take_action == "deferred"
        assert covered.audio_plan.take_ids == []
        committed = supervisor._commit_segment(
            config, state, 0, "000000", segment, proposed, rendered, covered, stage_seconds, 0.0
        )
        assert committed == "000000"
        assert (segment / paths.DONE_MARKER).exists()
        assert (segment / paths.SEGMENT_MANIFEST_FILENAME).exists()
        assert not (segment / "sha256.json").exists()
        assert not (segment / "metrics.json").exists()
        assert not (segment / "audio.wav").exists()
        fresh = read_state(run_dir)
        assert fresh.committed_segments == 1
        assert fresh.timeline_frames == rendered.frames
        assert set(stage_seconds) == {"director", "video", "audio", "validate", "commit"}
        assert validate_run(run_dir) == []
    finally:
        supervisor.stop_workers()


def _streaming_supervisor(run_dir: Path) -> Supervisor:
    """Unstarted supervisor with a causvid (streaming) video config.

    Workers never start: `_call_with_restart` is stubbed per test, so no
    GPU, no subprocess, no ffmpeg — only the adapter + overlay wiring.
    """
    config = read_effective_config(run_dir)
    config.video.backend = "causvid"
    config.video.blocks_per_segment = 3
    return Supervisor(run_dir, config)


def _canned_proposal() -> ProposedSegment:
    decision = SimpleNamespace(
        destination_concept="harbor at dawn",
        phase="drift",
        novelty_accepted=True,
        transition=SimpleNamespace(mechanism="cut", intermediate_stages=["mist", "gulls"]),
        audio=SimpleNamespace(
            music_caption="soft pulse", energy=0.5, texture="haze", environment=["dock"]
        ),
        notes="canned",
    )
    prompt_plan = SimpleNamespace(stages=[SimpleNamespace(prompt="stage zero")])
    return ProposedSegment(
        decision=decision,  # type: ignore[arg-type]
        prompt_plan=prompt_plan,  # type: ignore[arg-type]
        block_prompts=["mist over water", "gulls aloft", "harbor lights"],
        num_blocks=3,
        prefetch_hit=False,
        drift_hold=False,
        director_tokens={"prompt_tokens": 0, "completion_tokens": 0},
    )


def _stub_video_call(
    supervisor: Supervisor,
    result: dict[str, object],
    seen: list[dict[str, Any]],
    hooks: list[Any],
) -> Any:
    original = supervisor._call_with_restart

    def _stub(
        worker: Any,
        worker_name: str,
        segment_id: str,
        operation: str,
        payload: dict[str, object],
        restart_hook: Any = None,
    ) -> dict[str, object]:
        seen.append({"op": operation, "payload": dict(payload)})
        hooks.append(restart_hook)
        return dict(result)

    supervisor._call_with_restart = _stub  # type: ignore[assignment]
    return original


def test_render_video_streaming_overlay_uses_staged_prompts(tmp_path: Path) -> None:
    """Streaming payloads carry the staged plan, not one prompt tripled (023).

    Always-continue (ltx25-compare fix): drift never cuts — scene_cuts
    stays all-False even when destination != current. Fresh happens only
    when the worker has no tail.
    """
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="commit-split")
    supervisor = _streaming_supervisor(run_dir)
    seen: list[dict[str, Any]] = []
    hooks: list[Any] = []
    original = _stub_video_call(
        supervisor,
        {"video": {"frames": 81, "novel_frames": 72, "conditioning_frames": 9, "fps": 16}},
        seen,
        hooks,
    )
    try:
        config = read_effective_config(run_dir)
        config.video.backend = "causvid"
        config.video.blocks_per_segment = 3
        state = SimpleNamespace(
            destination_concept="harbor at dawn", current_concept="open sea", timeline_frames=0
        )
        segment = run_dir / "segments" / "000000"
        segment.mkdir(parents=True, exist_ok=True)
        rendered = supervisor._render_video(
            config,
            state,  # type: ignore[arg-type]
            0,
            "000000",
            segment,
            _canned_proposal(),
            {},
        )
        assert len(seen) == 1
        assert seen[0]["op"] == "generate_blocks"
        payload = seen[0]["payload"]
        assert payload["prompts"] == ["mist over water", "gulls aloft", "harbor lights"]
        assert payload["seeds"] == [video_seed(11, 0, block) for block in range(3)]
        assert payload["scene_cuts"] == [False, False, False]
        assert hooks[0] is not None
        assert rendered.frames == 81
        assert rendered.recovery_tape is None
    finally:
        supervisor._call_with_restart = original  # type: ignore[method-assign]


def test_render_video_rejects_implausible_report(tmp_path: Path) -> None:
    """The 006 ceiling survives the adapter move (the adapter alone accepts)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="commit-split")
    supervisor = _streaming_supervisor(run_dir)
    seen: list[dict[str, Any]] = []
    hooks: list[Any] = []
    original = _stub_video_call(supervisor, {"video": {"frames": 10**9}}, seen, hooks)
    try:
        config = read_effective_config(run_dir)
        state = SimpleNamespace(
            destination_concept="harbor at dawn", current_concept="open sea", timeline_frames=0
        )
        segment = run_dir / "segments" / "000000"
        segment.mkdir(parents=True, exist_ok=True)
        with pytest.raises(MediaError, match="implausible"):
            supervisor._render_video(
                config,
                state,  # type: ignore[arg-type]
                0,
                "000000",
                segment,
                _canned_proposal(),
                {},
            )
    finally:
        supervisor._call_with_restart = original  # type: ignore[method-assign]


def test_render_video_rejects_foreign_tape(tmp_path: Path) -> None:
    """Tape confinement survives the adapter move (the result drops tapes)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="commit-split")
    supervisor = _streaming_supervisor(run_dir)
    seen: list[dict[str, Any]] = []
    hooks: list[Any] = []
    original = _stub_video_call(
        supervisor, {"video": {"frames": 48, "recovery_path": "/etc/passwd"}}, seen, hooks
    )
    try:
        config = read_effective_config(run_dir)
        state = SimpleNamespace(
            destination_concept="harbor at dawn", current_concept="open sea", timeline_frames=0
        )
        segment = run_dir / "segments" / "000000"
        segment.mkdir(parents=True, exist_ok=True)
        with pytest.raises(MediaError, match="escapes the run dir"):
            supervisor._render_video(
                config,
                state,  # type: ignore[arg-type]
                0,
                "000000",
                segment,
                _canned_proposal(),
                {},
            )
    finally:
        supervisor._call_with_restart = original  # type: ignore[method-assign]
