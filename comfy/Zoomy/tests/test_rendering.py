"""Tests for the render orchestration against a scripted engine."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from PIL import Image

import zoomy.rendering as rendering_module
from zoomy.engine_protocol import FinalizeRequest, FrameRenderRequest, ProgressUpdate
from zoomy.errors import (
    EmptyFrameSequenceError,
    EngineConfigurationError,
    EngineExecutionError,
    RenderInterruptedError,
    ZoomyError,
)
from zoomy.family_catalog import FAMILY_CATALOG, find_family
from zoomy.frame_repository import FrameRepository
from zoomy.rendering import (
    LoopOptions,
    RenderEnvironment,
    VideoGenerationOptions,
    clear_loop_stop,
    finalize_video,
    generate_video,
    render_loop,
    render_next_frame,
    request_loop_stop,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path
    from typing import Any


class ScriptedEngine:
    """Test double that replays scripted frame and finalize outcomes."""

    def __init__(
        self,
        *,
        frame_results: Sequence[Image.Image | ZoomyError] | None = None,
        finalize_updates: Sequence[Sequence[ProgressUpdate] | ZoomyError] | None = None,
    ) -> None:
        """Store the scripted outcomes and start with no calls."""
        self.frame_results = list(frame_results or [])
        self.finalize_updates = list(finalize_updates or [])
        self.render_requests: list[FrameRenderRequest] = []
        self.finalize_requests: list[FinalizeRequest] = []
        self.interrupt_calls = 0

    def render_frame(self, request: FrameRenderRequest) -> Image.Image:
        """Record the request, then return the next image or raise it."""
        self.render_requests.append(request)
        if not self.frame_results:
            return Image.new("RGB", (8, 8))
        outcome = self.frame_results.pop(0)
        if isinstance(outcome, ZoomyError):
            raise outcome
        return outcome

    def finalize_sequence(self, request: FinalizeRequest) -> Iterator[ProgressUpdate]:
        """Record the request, then replay the next updates or raise."""
        self.finalize_requests.append(request)
        if not self.finalize_updates:
            return
            yield  # Make this a generator even when nothing is scripted.
        outcome = self.finalize_updates.pop(0)
        if isinstance(outcome, ZoomyError):
            raise outcome
        yield from outcome

    def request_interrupt(self) -> None:
        """Count interrupt calls; no test drives engine aborts here."""
        self.interrupt_calls += 1

    def is_ready(self) -> bool:
        """Report a ready engine."""
        return True

    def engine_statistics(self) -> Any:
        """Report nothing; statistics never flow through this double."""
        return None


def _repository_with_frames(tmp_path: Path, frame_count: int) -> FrameRepository:
    """Create a repository whose sequence already holds stub frames."""
    sequence_directory = tmp_path / "z_image"
    sequence_directory.mkdir(parents=True, exist_ok=True)
    for frame_number in range(1, frame_count + 1):
        (sequence_directory / f"frame_{frame_number:05d}_.png").write_bytes(b"stub")
    return FrameRepository(tmp_path)


def _frame_request(repository: FrameRepository) -> FrameRenderRequest:
    """Build a frame request matching the repository's z_image sequence."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    return FrameRenderRequest(
        family=family,
        prompt="a prompt",
        negative_prompt="a negative",
        frame_count=repository.frame_count(family.sequence_key),
        lora_selections=(),
        seed=42,
    )


def _test_environment(engine: ScriptedEngine, repository: FrameRepository) -> RenderEnvironment:
    """Build the engine plus repository bundle one test needs."""
    return RenderEnvironment(engine=engine, repository=repository)


def test_render_next_frame_success_saves_new_frame(tmp_path: Path) -> None:
    """A rendered image lands under the next counter name with its count."""
    repository = _repository_with_frames(tmp_path, frame_count=2)
    engine = ScriptedEngine()
    updates = list(
        render_next_frame(
            _frame_request(repository),
            _test_environment(engine, repository),
        )
    )
    final_update = updates[-1]
    assert final_update.frame_count == 3
    assert final_update.frame_path is not None
    assert final_update.frame_path.name == "frame_00003_.png"
    assert final_update.frame_path.is_file()
    assert len(engine.render_requests) == 1


def test_render_next_frame_raises_engine_error(tmp_path: Path) -> None:
    """An engine stage failure propagates with its stage detail."""
    repository = _repository_with_frames(tmp_path, frame_count=1)
    engine = ScriptedEngine(frame_results=[EngineExecutionError("z-frame", "out of memory")])
    with pytest.raises(EngineExecutionError, match="z-frame"):
        list(
            render_next_frame(
                _frame_request(repository),
                _test_environment(engine, repository),
            )
        )


def test_render_next_frame_raises_interrupted(tmp_path: Path) -> None:
    """An engine interrupt propagates instead of saving anything."""
    repository = _repository_with_frames(tmp_path, frame_count=1)
    engine = ScriptedEngine(frame_results=[RenderInterruptedError("stopped")])
    with pytest.raises(RenderInterruptedError):
        list(
            render_next_frame(
                _frame_request(repository),
                _test_environment(engine, repository),
            )
        )
    assert repository.frame_count("z_image") == 1


def test_finalize_video_requires_frames(tmp_path: Path) -> None:
    """Finalizing an empty sequence raises before the engine runs."""
    repository = FrameRepository(tmp_path)
    engine = ScriptedEngine()
    family = find_family(FAMILY_CATALOG, "z_fast")
    request = FinalizeRequest(family=family, frame_count=0)
    with pytest.raises(EmptyFrameSequenceError, match="no frames"):
        list(finalize_video(request, _test_environment(engine, repository)))
    assert not engine.finalize_requests


def test_finalize_video_replays_engine_updates(tmp_path: Path) -> None:
    """Finalize streams the engine's updates, ending with its video."""
    repository = _repository_with_frames(tmp_path, frame_count=3)
    video_path = tmp_path / "z_image_00001-audio.mp4"
    engine = ScriptedEngine(
        finalize_updates=[
            [
                ProgressUpdate(message="Segment 1/1 complete.", frame_count=3),
                ProgressUpdate(
                    message="Video complete.",
                    video_path=video_path,
                    frame_count=3,
                    elapsed_seconds=4.0,
                ),
            ]
        ]
    )
    family = find_family(FAMILY_CATALOG, "z_fast")
    request = FinalizeRequest(family=family, frame_count=3)
    updates = list(finalize_video(request, _test_environment(engine, repository)))
    assert next(update.message for update in updates if "Submitting finalize" in update.message)
    final_update = updates[-1]
    assert final_update.video_path == video_path
    assert final_update.frame_count == 3
    assert len(engine.finalize_requests) == 1


def _counted_request_factory() -> Callable[[int], FrameRenderRequest]:
    """Build frame requests honoring the frame count the loop supplies."""
    family = find_family(FAMILY_CATALOG, "z_fast")

    def build_request(frame_count: int) -> FrameRenderRequest:
        """Assemble one frame request for the given count."""
        return FrameRenderRequest(
            family=family,
            prompt="a prompt",
            negative_prompt="a negative",
            frame_count=frame_count,
            lora_selections=(),
            seed=frame_count,
        )

    return build_request


def _drain_loop(
    engine: ScriptedEngine,
    repository: FrameRepository,
    *,
    frame_target: int | None,
    max_attempts_per_frame: int = 3,
) -> list[str]:
    """Run a loop to completion and return every yielded message."""
    environment = RenderEnvironment(engine=engine, repository=repository)
    updates = render_loop(
        _counted_request_factory(),
        "z_image",
        environment,
        LoopOptions(
            frame_target=frame_target,
            max_attempts_per_frame=max_attempts_per_frame,
        ),
    )
    return [update.message for update in updates]


def test_render_loop_rejects_zero_attempts(tmp_path: Path) -> None:
    """Zero per-frame attempts would spin forever, so the loop refuses them."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    engine = ScriptedEngine()
    environment = RenderEnvironment(engine=engine, repository=repository)
    with pytest.raises(ValueError, match="max_attempts_per_frame"):
        list(
            render_loop(
                _counted_request_factory(),
                "z_image",
                environment,
                LoopOptions(frame_target=1, max_attempts_per_frame=0),
            )
        )


def test_render_next_frame_reports_elapsed_seconds(tmp_path: Path) -> None:
    """Terminal updates carry the measured render time."""
    repository = _repository_with_frames(tmp_path, frame_count=1)
    engine = ScriptedEngine()
    updates = list(
        render_next_frame(
            _frame_request(repository),
            _test_environment(engine, repository),
        )
    )
    assert updates[-1].elapsed_seconds is not None
    assert updates[-1].elapsed_seconds >= 0.0


def test_render_loop_stops_at_the_frame_target(tmp_path: Path) -> None:
    """A numeric target ends the loop after exactly that many frames."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    engine = ScriptedEngine()
    messages = _drain_loop(engine, repository, frame_target=2)
    assert len(engine.render_requests) == 2
    assert messages[-1] == "Loop target reached after 2 frames."


def test_concurrent_loops_stop_independently(tmp_path: Path) -> None:
    """Stopping one loop's flag leaves a loop on another flag running."""
    first_environment = _test_environment(ScriptedEngine(), FrameRepository(tmp_path / "first"))
    second_environment = _test_environment(ScriptedEngine(), FrameRepository(tmp_path / "second"))
    factory = _counted_request_factory()
    first_stop = threading.Event()
    second_stop = threading.Event()
    first_loop = render_loop(
        factory, "z_image", first_environment, LoopOptions(frame_target=None, stop_event=first_stop)
    )
    second_loop = render_loop(
        factory,
        "z_image",
        second_environment,
        LoopOptions(frame_target=None, stop_event=second_stop),
    )
    assert next(first_loop).message == "Loop started — rendering until stopped…"
    assert next(second_loop).message == "Loop started — rendering until stopped…"
    for _ in range(3):
        next(first_loop)
        next(second_loop)
    first_stop.set()
    first_messages = [update.message for update in first_loop]
    assert first_messages[-1] == "Loop stopped after 1 frame."
    second_messages = [next(second_loop).message for _ in range(3)]
    assert not any("Loop stopped" in message for message in second_messages)
    assert second_environment.repository.frame_count("z_image") == 2


def test_render_loop_stops_gracefully_on_request(tmp_path: Path) -> None:
    """A stop requested mid-loop finishes the running frame, then exits."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    engine = ScriptedEngine()
    factory_calls = 0

    def stopping_factory(frame_count: int) -> FrameRenderRequest:
        """Request the stop while the second frame is being built."""
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 2:
            request_loop_stop()
        return _counted_request_factory()(frame_count)

    environment = RenderEnvironment(engine=engine, repository=repository)
    messages = list(
        render_loop(
            stopping_factory,
            "z_image",
            environment,
            LoopOptions(frame_target=None),
        )
    )
    clear_loop_stop()
    assert len(engine.render_requests) == 2
    assert messages[-1].message == "Loop stopped after 2 frames."


def test_render_loop_retries_then_gives_up(tmp_path: Path) -> None:
    """Transient failures retry per frame; exhaustion ends the loop."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    failure = EngineExecutionError("z-frame", "boom")
    engine = ScriptedEngine(frame_results=[failure, failure, failure])
    messages = _drain_loop(engine, repository, frame_target=None, max_attempts_per_frame=3)
    assert len(engine.render_requests) == 3
    assert "failed after 3 attempts" in messages[-1]
    assert "Ending loop" in messages[-1]


def test_render_loop_succeeds_on_retry(tmp_path: Path) -> None:
    """A frame that fails once still completes the loop on its retry."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    engine = ScriptedEngine(frame_results=[EngineExecutionError("z-frame", "boom")])
    messages = _drain_loop(engine, repository, frame_target=1, max_attempts_per_frame=3)
    assert len(engine.render_requests) == 2
    assert messages[-1] == "Loop target reached after 1 frame."


def test_render_loop_propagates_interrupts_without_retry(tmp_path: Path) -> None:
    """A user interrupt ends the loop immediately on the first attempt."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    engine = ScriptedEngine(frame_results=[RenderInterruptedError("stopped")])
    with pytest.raises(RenderInterruptedError):
        _drain_loop(engine, repository, frame_target=None)
    assert len(engine.render_requests) == 1


def test_render_loop_propagates_configuration_errors_without_retry(tmp_path: Path) -> None:
    """A configuration error is a spec bug, so the loop never retries it."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    engine = ScriptedEngine(frame_results=[EngineConfigurationError("bad request")])
    with pytest.raises(EngineConfigurationError, match="bad request"):
        _drain_loop(engine, repository, frame_target=None)
    assert len(engine.render_requests) == 1


def test_interrupt_module_flag_is_independent_of_engine() -> None:
    """The loop stop flag still toggles without any engine involved."""
    clear_loop_stop()
    assert rendering_module.is_loop_stop_requested() is False
    request_loop_stop()
    assert rendering_module.is_loop_stop_requested() is True
    clear_loop_stop()


def test_generate_video_renders_until_duration_then_finalizes(tmp_path: Path) -> None:
    """Ten seconds need 81 frames; the video lands after the last one."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    video_path = tmp_path / "z_image_00001-audio.mp4"
    engine = ScriptedEngine(
        finalize_updates=[
            [
                ProgressUpdate(
                    message="Video complete.",
                    video_path=video_path,
                    frame_count=81,
                    elapsed_seconds=4.0,
                ),
            ]
        ]
    )
    family = find_family(FAMILY_CATALOG, "z_fast")
    updates = list(
        generate_video(
            family,
            _counted_request_factory(),
            _test_environment(engine, repository),
            options=VideoGenerationOptions(target_seconds=10.0),
        )
    )
    assert len(engine.render_requests) == 81
    assert len(engine.finalize_requests) == 1
    assert engine.finalize_requests[0].frame_count == 81
    assert updates[-1].video_path == video_path


def test_generate_video_skips_the_loop_when_long_enough(tmp_path: Path) -> None:
    """A sequence that already covers the target goes straight to finalize."""
    repository = _repository_with_frames(tmp_path, frame_count=81)
    engine = ScriptedEngine(finalize_updates=[[]])
    family = find_family(FAMILY_CATALOG, "z_fast")
    updates = list(
        generate_video(
            family,
            _counted_request_factory(),
            _test_environment(engine, repository),
            options=VideoGenerationOptions(target_seconds=10.0),
        )
    )
    assert not engine.render_requests
    assert len(engine.finalize_requests) == 1
    assert any("already" in update.message for update in updates)


def test_generate_video_manual_stop_finalizes_when_enabled(tmp_path: Path) -> None:
    """An unlimited run that is stopped still gets its video by default."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    engine = ScriptedEngine(finalize_updates=[[]])
    factory_calls = 0

    def stopping_factory(frame_count: int) -> FrameRenderRequest:
        """Stop while the second frame is being built."""
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 2:
            request_loop_stop()
        return _counted_request_factory()(frame_count)

    family = find_family(FAMILY_CATALOG, "z_fast")
    list(
        generate_video(
            family,
            stopping_factory,
            _test_environment(engine, repository),
            options=VideoGenerationOptions(target_seconds=None),
        )
    )
    clear_loop_stop()
    assert len(engine.render_requests) == 2
    assert len(engine.finalize_requests) == 1


def test_generate_video_manual_stop_skips_finalize_when_disabled(tmp_path: Path) -> None:
    """With auto-finalize off, a stopped run keeps its frames only."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    engine = ScriptedEngine()
    factory_calls = 0

    def stopping_factory(frame_count: int) -> FrameRenderRequest:
        """Stop while the second frame is being built."""
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 2:
            request_loop_stop()
        return _counted_request_factory()(frame_count)

    family = find_family(FAMILY_CATALOG, "z_fast")
    messages = [
        update.message
        for update in generate_video(
            family,
            stopping_factory,
            _test_environment(engine, repository),
            options=VideoGenerationOptions(target_seconds=None, finalize_on_stop=False),
        )
    ]
    clear_loop_stop()
    assert len(engine.render_requests) == 2
    assert not engine.finalize_requests
    assert any("not finalized" in message for message in messages)


def test_generate_video_reports_without_finalizing_when_empty(tmp_path: Path) -> None:
    """A loop that renders nothing never reaches the finalize stage."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    failure = EngineExecutionError("z-frame", "boom")
    engine = ScriptedEngine(frame_results=[failure, failure, failure])
    family = find_family(FAMILY_CATALOG, "z_fast")
    messages = [
        update.message
        for update in generate_video(
            family,
            _counted_request_factory(),
            _test_environment(engine, repository),
            options=VideoGenerationOptions(target_seconds=10.0, max_attempts_per_frame=3),
        )
    ]
    assert not engine.finalize_requests
    assert any("nothing to finalize" in message for message in messages)


def test_generate_video_rejects_non_positive_durations(tmp_path: Path) -> None:
    """Zero or negative durations cannot size the frame loop."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    engine = ScriptedEngine()
    family = find_family(FAMILY_CATALOG, "z_fast")
    with pytest.raises(ValueError, match="positive"):
        list(
            generate_video(
                family,
                _counted_request_factory(),
                _test_environment(engine, repository),
                options=VideoGenerationOptions(target_seconds=0.0),
            )
        )
    assert not engine.render_requests
