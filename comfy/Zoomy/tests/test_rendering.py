"""Tests for the render orchestration against a scripted connection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zoomy.comfy_connection import HistoryEntry, SystemStatistics
from zoomy.errors import (
    ComfyExecutionError,
    ComfyRejectedWorkflowError,
    EmptyFrameSequenceError,
    OperationTimeoutError,
    RenderInterruptedError,
    ZoomyError,
)
from zoomy.family_catalog import FAMILY_CATALOG, find_family
from zoomy.finalize_workflow import FinalizeRequest
from zoomy.frame_repository import FrameRepository
from zoomy.frame_workflow import FrameRenderRequest
from zoomy.rendering import (
    RenderEnvironment,
    _raise_if_execution_failed,
    clear_loop_stop,
    finalize_video,
    render_loop,
    render_next_frame,
    request_loop_stop,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path
    from typing import Any

SCRIPTED_PROMPT_IDENTIFIER = "scripted-prompt"


class ScriptedConnection:
    """Test double that replays scripted history transitions in order."""

    def __init__(
        self,
        entries: Sequence[HistoryEntry | None],
        *,
        queue_failure: ZoomyError | None = None,
    ) -> None:
        """Store the scripted history transitions and start with no calls."""
        self.entries = list(entries)
        self.queue_failure = queue_failure
        self.queued_workflows: list[Mapping[str, Any]] = []

    def is_reachable(self) -> bool:
        """Report a healthy connection."""
        return True

    def system_statistics(self) -> SystemStatistics:
        """Report canned memory figures."""
        return SystemStatistics(
            system_memory_free_bytes=12000000000,
            system_memory_total_bytes=32000000000,
            video_memory_free_bytes=9000000000,
            video_memory_total_bytes=16000000000,
        )

    def queue_workflow(self, workflow: Mapping[str, Any]) -> str:
        """Record the submitted workflow and return the scripted id."""
        if self.queue_failure is not None:
            raise self.queue_failure
        self.queued_workflows.append(workflow)
        return SCRIPTED_PROMPT_IDENTIFIER

    def fetch_history(self, prompt_identifier: str) -> HistoryEntry | None:
        """Hand back the next scripted entry, then the last one forever."""
        assert prompt_identifier == SCRIPTED_PROMPT_IDENTIFIER
        if self.entries:
            return self.entries.pop(0)
        return None

    def interrupt(self) -> None:
        """Do nothing; no test drives interrupts through this double."""


def _completed_entry(
    status_messages: list[list[Any]] | None = None,
) -> HistoryEntry:
    """Build a completed history entry with optional status messages."""
    return HistoryEntry(outputs={}, status_messages=status_messages or [], is_completed=True)


def _repository_with_frames(tmp_path: Path, frame_count: int) -> FrameRepository:
    """Create a repository whose sequence already holds stub frames."""
    sequence_directory = tmp_path / "Zoomy" / "z_image"
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
        output_directory="/unused/by/fakes",
    )


def _test_environment(
    connection: ScriptedConnection,
    repository: FrameRepository,
    *,
    operation_timeout_seconds: float = 5.0,
) -> RenderEnvironment:
    """Build standard fast-polling execution budgets for one test."""
    return RenderEnvironment(
        connection=connection,
        repository=repository,
        operation_timeout_seconds=operation_timeout_seconds,
        poll_interval_seconds=0.0,
    )


def test_render_next_frame_success_yields_final_frame(tmp_path: Path) -> None:
    """A completed job yields a final update with the newest frame."""
    repository = _repository_with_frames(tmp_path, frame_count=2)
    connection = ScriptedConnection([None, _completed_entry()])
    updates = list(
        render_next_frame(
            _frame_request(repository),
            _test_environment(connection, repository),
        )
    )
    final_update = updates[-1]
    # The scripted connection produces no new file, so the implementation
    # re-reads the repository and still sees the two stub frames.
    assert final_update.frame_count == 2
    assert final_update.frame_path is not None
    assert final_update.frame_path.name == "frame_00002_.png"
    assert len(connection.queued_workflows) == 1


def test_render_next_frame_raises_execution_error(tmp_path: Path) -> None:
    """An execution_error status becomes a ComfyExecutionError with detail."""
    repository = _repository_with_frames(tmp_path, frame_count=1)
    failure_entry = _completed_entry(
        [
            [
                "execution_error",
                {"node_type": "KSampler", "exception_message": "out of memory"},
            ]
        ]
    )
    connection = ScriptedConnection([failure_entry])
    with pytest.raises(ComfyExecutionError, match="KSampler"):
        list(
            render_next_frame(
                _frame_request(repository),
                _test_environment(connection, repository),
            )
        )


def test_render_next_frame_raises_interrupted(tmp_path: Path) -> None:
    """An execution_interrupted status becomes a RenderInterruptedError."""
    repository = _repository_with_frames(tmp_path, frame_count=1)
    interrupted_entry = _completed_entry([["execution_interrupted", None]])
    connection = ScriptedConnection([interrupted_entry])
    with pytest.raises(RenderInterruptedError):
        list(
            render_next_frame(
                _frame_request(repository),
                _test_environment(connection, repository),
            )
        )


def test_render_next_frame_times_out(tmp_path: Path) -> None:
    """A job that never completes raises OperationTimeoutError."""
    repository = _repository_with_frames(tmp_path, frame_count=1)
    connection = ScriptedConnection([])
    with pytest.raises(OperationTimeoutError):
        list(
            render_next_frame(
                _frame_request(repository),
                _test_environment(connection, repository, operation_timeout_seconds=0.0),
            )
        )


def test_finalize_video_requires_frames(tmp_path: Path) -> None:
    """Finalizing an empty sequence raises before anything is queued."""
    repository = FrameRepository(tmp_path)
    connection = ScriptedConnection([])
    family = find_family(FAMILY_CATALOG, "z_fast")
    request = FinalizeRequest(family=family, frame_count=0, output_directory="/output")
    with pytest.raises(EmptyFrameSequenceError, match="no frames"):
        list(
            finalize_video(
                request,
                _test_environment(connection, repository),
            )
        )
    assert not connection.queued_workflows


def test_finalize_video_success_yields_video(tmp_path: Path) -> None:
    """A completed finalize yields the newest sequence video."""
    repository = _repository_with_frames(tmp_path, frame_count=3)
    video_path = tmp_path / "Zoomy_z_image_00001.mp4"
    video_path.write_bytes(b"stub")
    connection = ScriptedConnection([_completed_entry()])
    family = find_family(FAMILY_CATALOG, "z_fast")
    request = FinalizeRequest(family=family, frame_count=3, output_directory="/output")
    updates = list(
        finalize_video(
            request,
            _test_environment(connection, repository),
        )
    )
    final_update = updates[-1]
    assert final_update.video_path == video_path
    assert final_update.frame_count == 3


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
            output_directory="/unused/by/fakes",
        )

    return build_request


def _drain_loop(
    connection: ScriptedConnection,
    repository: FrameRepository,
    *,
    frame_target: int | None,
    max_attempts_per_frame: int = 3,
) -> list[str]:
    """Run a loop to completion and return every yielded message."""
    environment = RenderEnvironment(
        connection=connection,
        repository=repository,
        operation_timeout_seconds=5.0,
        poll_interval_seconds=0.0,
    )
    updates = render_loop(
        _counted_request_factory(),
        "z_image",
        environment,
        frame_target=frame_target,
        max_attempts_per_frame=max_attempts_per_frame,
    )
    return [update.message for update in updates]


def test_render_loop_rejects_zero_attempts(tmp_path: Path) -> None:
    """Zero per-frame attempts would spin forever, so the loop refuses them."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    connection = ScriptedConnection([])
    environment = RenderEnvironment(
        connection=connection,
        repository=repository,
        operation_timeout_seconds=5.0,
        poll_interval_seconds=0.0,
    )
    with pytest.raises(ValueError, match="max_attempts_per_frame"):
        list(
            render_loop(
                _counted_request_factory(),
                "z_image",
                environment,
                frame_target=1,
                max_attempts_per_frame=0,
            )
        )


def test_render_next_frame_reports_elapsed_seconds(tmp_path: Path) -> None:
    """Terminal updates carry the measured queue-to-completion time."""
    repository = _repository_with_frames(tmp_path, frame_count=1)
    connection = ScriptedConnection([_completed_entry()])
    updates = list(
        render_next_frame(
            _frame_request(repository),
            _test_environment(connection, repository),
        )
    )
    assert updates[-1].elapsed_seconds is not None
    assert updates[-1].elapsed_seconds >= 0.0


def test_render_loop_stops_at_the_frame_target(tmp_path: Path) -> None:
    """A numeric target ends the loop after exactly that many frames."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    connection = ScriptedConnection([None, _completed_entry()] * 2)
    messages = _drain_loop(connection, repository, frame_target=2)
    assert len(connection.queued_workflows) == 2
    assert messages[-1] == "Loop target reached after 2 frames."


def test_render_loop_stops_gracefully_on_request(tmp_path: Path) -> None:
    """A stop requested mid-loop finishes the running frame, then exits."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    connection = ScriptedConnection([None, _completed_entry()] * 5)
    factory_calls = 0

    def stopping_factory(frame_count: int) -> FrameRenderRequest:
        """Request the stop while the second frame is being built."""
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 2:
            request_loop_stop()
        return _counted_request_factory()(frame_count)

    environment = RenderEnvironment(
        connection=connection,
        repository=repository,
        operation_timeout_seconds=5.0,
        poll_interval_seconds=0.0,
    )
    messages = list(
        render_loop(
            stopping_factory,
            "z_image",
            environment,
            frame_target=None,
        )
    )
    clear_loop_stop()
    assert len(connection.queued_workflows) == 2
    assert messages[-1].message == "Loop stopped after 2 frames."


def test_render_loop_retries_then_gives_up(tmp_path: Path) -> None:
    """Transient failures retry per frame; exhaustion ends the loop."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    failure = _completed_entry(
        [["execution_error", {"node_type": "KSampler", "exception_message": "boom"}]]
    )
    connection = ScriptedConnection([failure, failure, failure])
    messages = _drain_loop(connection, repository, frame_target=None, max_attempts_per_frame=3)
    assert len(connection.queued_workflows) == 3
    assert "failed after 3 attempts" in messages[-1]
    assert "Ending loop" in messages[-1]


def test_render_loop_succeeds_on_retry(tmp_path: Path) -> None:
    """A frame that fails once still completes the loop on its retry."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    failure = _completed_entry(
        [["execution_error", {"node_type": "KSampler", "exception_message": "boom"}]]
    )
    connection = ScriptedConnection([failure, None, _completed_entry()])
    messages = _drain_loop(connection, repository, frame_target=1, max_attempts_per_frame=3)
    assert len(connection.queued_workflows) == 2
    assert messages[-1] == "Loop target reached after 1 frame."


def test_render_loop_propagates_interrupts_without_retry(tmp_path: Path) -> None:
    """A user interrupt ends the loop immediately on the first attempt."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    interrupted = _completed_entry([["execution_interrupted", None]])
    connection = ScriptedConnection([interrupted, None, _completed_entry()])
    with pytest.raises(RenderInterruptedError):
        _drain_loop(connection, repository, frame_target=None)
    assert len(connection.queued_workflows) == 1


def test_render_loop_propagates_rejections_without_retry(tmp_path: Path) -> None:
    """A validation rejection is a spec bug, so the loop never retries it."""
    repository = _repository_with_frames(tmp_path, frame_count=0)
    rejection = ComfyRejectedWorkflowError(summary="bad node", node_errors={})
    connection = ScriptedConnection([], queue_failure=rejection)
    with pytest.raises(ComfyRejectedWorkflowError, match="bad node"):
        _drain_loop(connection, repository, frame_target=None)
    assert not connection.queued_workflows


def _open_entry(
    status_messages: list[list[Any]] | None = None,
) -> HistoryEntry:
    """Build a not-completed entry, the shape live interrupts use."""
    return HistoryEntry(outputs={}, status_messages=status_messages or [], is_completed=False)


# Arbitrary JSON for status messages: NUL-free text (matching the settings
# domain) with bounded nesting so cases stay small and fast.
json_atom = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.text(alphabet=st.characters(blacklist_characters="\x00"), max_size=20)
)
json_value = st.recursive(
    json_atom | st.floats(allow_nan=False, allow_infinity=False),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=3)
    ),
    max_leaves=8,
)


def _scan_outcome(entry: HistoryEntry) -> str:
    """Run the status scanner, translating its result to a testable label."""
    try:
        _raise_if_execution_failed(entry)
    except ComfyExecutionError:
        return "execution_error"
    except RenderInterruptedError:
        return "interrupted"
    return "silent"


@given(messages=st.lists(json_value, max_size=4))
def test_status_message_scan_only_raises_typed_errors(messages: list[Any]) -> None:
    """Arbitrary message shapes surface as typed errors or silence, never a crash."""
    entry = HistoryEntry(outputs={}, status_messages=messages, is_completed=False)
    assert _scan_outcome(entry) in ("silent", "execution_error", "interrupted")


def test_interrupted_entry_without_completed_flag_raises_immediately(
    tmp_path: Path,
) -> None:
    """Live interrupts stay completed=False; the loop must not poll forever."""
    repository = _repository_with_frames(tmp_path, frame_count=1)
    interrupted = _open_entry([["execution_interrupted", {"node_id": "54"}]])
    connection = ScriptedConnection([interrupted])
    with pytest.raises(RenderInterruptedError):
        list(
            render_next_frame(
                _frame_request(repository),
                _test_environment(connection, repository),
            )
        )


def test_error_entry_without_completed_flag_raises_immediately(tmp_path: Path) -> None:
    """Live errors stay completed=False; they surface instead of timing out."""
    repository = _repository_with_frames(tmp_path, frame_count=1)
    failure = _open_entry(
        [["execution_error", {"node_type": "KSampler", "exception_message": "boom"}]]
    )
    connection = ScriptedConnection([failure])
    with pytest.raises(ComfyExecutionError, match="KSampler"):
        list(
            render_next_frame(
                _frame_request(repository),
                _test_environment(connection, repository),
            )
        )
