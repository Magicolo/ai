"""Commit-split tests (issues 020, 023-supervisor-half).

Helper-level coverage for the `_propose/_render_video/_cover_audio/_commit`
split of `commit_one_segment` (020) plus the adapter-through-supervisor
wiring (023): the commit path must call `VideoBackendAdapter.generate_segment`
while keeping the staged per-block prompts, the issue-006 frame ceiling,
and the recovery-tape confinement that the adapter alone cannot express.

All against fake backends or stubbed transports (real media, no GPU).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voyage import paths
from voyage.cli import validate_run
from voyage.config import default_config_toml, load_config
from voyage.errors import MediaError
from voyage.hashing import sha256_file as hashing_sha256_file
from voyage.models import StyleSpec
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_state,
    write_manifest,
    write_state,
)
from voyage.seeds import video_seed
from voyage.supervisor import CoveredAudio, ProposedSegment, RenderedVideo, Supervisor
from voyage.supervisor import VideoBackendAdapter as SupervisorAdapter


def _init_run(run_dir: Path, run_id: str = "commit-split") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.CONFIG_FILENAME).write_text(
        default_config_toml(run_id, "pastel neon line-art, peaceful", 11),
        encoding="utf-8",
    )
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


def _started_supervisor(run_dir: Path) -> Supervisor:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
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
    _init_run(run_dir)
    supervisor = _started_supervisor(run_dir)
    try:
        config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
        state = read_state(run_dir)
        stage_seconds: dict[str, float] = {}
        proposed = supervisor._propose_segment(
            config, state, 0, "000000", StyleSpec(prompt=config.style), stage_seconds
        )
        assert isinstance(proposed, ProposedSegment)
        assert proposed.decision.destination_concept.strip()
        assert proposed.prompt_plan.stages
        assert len(proposed.block_prompts) == proposed.num_blocks == 1
        assert set(stage_seconds) == {"inspect", "director"}
        assert proposed.prefetch_hit is False
        assert proposed.drift_hold == (0 % max(1, config.voyage.drift_every_n_segments) != 0)
    finally:
        supervisor.stop_workers()


def test_render_video_goes_through_adapter(tmp_path: Path) -> None:
    """The commit video path calls the adapter with the fake payload (023)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _started_supervisor(run_dir)
    try:
        config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
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

        supervisor._call_with_restart = _recording_call  # type: ignore[method-assign]
        try:
            original_method = SupervisorAdapter.generate_segment

            def _spy_generate(adapter: Any, request: Any, output_path: Path) -> Any:
                adapter_calls.append((request, output_path))
                return original_method(adapter, request, output_path)

            SupervisorAdapter.generate_segment = _spy_generate  # type: ignore[method-assign]
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
    """`_cover_audio` + `_commit_segment` commit exactly one segment (020)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _started_supervisor(run_dir)
    try:
        config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
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
        assert covered.audio_plan.take_ids
        committed = supervisor._commit_segment(
            config, state, 0, "000000", segment, proposed, rendered, covered, stage_seconds, 0.0
        )
        assert committed == "000000"
        assert (segment / paths.DONE_MARKER).exists()
        assert (segment / "sha256.json").exists()
        assert (segment / "metrics.json").exists()
        fresh = read_state(run_dir)
        assert fresh.committed_segments == 1
        assert fresh.timeline_frames == rendered.frames
        assert set(stage_seconds) == {"inspect", "director", "video", "audio", "validate", "commit"}
        assert validate_run(run_dir) == []
    finally:
        supervisor.stop_workers()


def _streaming_supervisor(run_dir: Path) -> Supervisor:
    """Unstarted supervisor with a causvid (streaming) video config.

    Workers never start: `_call_with_restart` is stubbed per test, so no
    GPU, no subprocess, no ffmpeg — only the adapter + overlay wiring.
    """
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config.video.backend = "causvid"  # type: ignore[assignment]
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

    supervisor._call_with_restart = _stub  # type: ignore[method-assign]
    return original


def test_render_video_streaming_overlay_uses_staged_prompts(tmp_path: Path) -> None:
    """Streaming payloads carry the staged plan, not one prompt tripled (023)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
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
        config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
        config.video.backend = "causvid"  # type: ignore[assignment]
        config.video.blocks_per_segment = 3
        state = SimpleNamespace(
            destination_concept="harbor at dawn", current_concept="open sea", timeline_frames=0
        )
        segment = run_dir / "segments" / "000000"
        segment.mkdir(parents=True, exist_ok=True)
        rendered = supervisor._render_video(
            config,  # type: ignore[arg-type]
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
        assert payload["scene_cuts"] == [True, False, False]
        assert hooks[0] is not None
        assert rendered.frames == 81
        assert rendered.recovery_tape is None
    finally:
        supervisor._call_with_restart = original  # type: ignore[method-assign]


def test_render_video_rejects_implausible_report(tmp_path: Path) -> None:
    """The 006 ceiling survives the adapter move (the adapter alone accepts)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _streaming_supervisor(run_dir)
    seen: list[dict[str, Any]] = []
    hooks: list[Any] = []
    original = _stub_video_call(supervisor, {"video": {"frames": 10**9}}, seen, hooks)
    try:
        config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
        state = SimpleNamespace(
            destination_concept="harbor at dawn", current_concept="open sea", timeline_frames=0
        )
        segment = run_dir / "segments" / "000000"
        segment.mkdir(parents=True, exist_ok=True)
        with pytest.raises(MediaError, match="implausible"):
            supervisor._render_video(
                config,  # type: ignore[arg-type]
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
    _init_run(run_dir)
    supervisor = _streaming_supervisor(run_dir)
    seen: list[dict[str, Any]] = []
    hooks: list[Any] = []
    original = _stub_video_call(
        supervisor, {"video": {"frames": 48, "recovery_path": "/etc/passwd"}}, seen, hooks
    )
    try:
        config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
        state = SimpleNamespace(
            destination_concept="harbor at dawn", current_concept="open sea", timeline_frames=0
        )
        segment = run_dir / "segments" / "000000"
        segment.mkdir(parents=True, exist_ok=True)
        with pytest.raises(MediaError, match="escapes the run dir"):
            supervisor._render_video(
                config,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                0,
                "000000",
                segment,
                _canned_proposal(),
                {},
            )
    finally:
        supervisor._call_with_restart = original  # type: ignore[method-assign]
