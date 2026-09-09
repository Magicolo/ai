"""Tests for the render orchestration against a scripted connection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zoomy.comfy_connection import HistoryEntry
from zoomy.errors import (
    ComfyExecutionError,
    EmptyFrameSequenceError,
    OperationTimeoutError,
    RenderInterruptedError,
)
from zoomy.family_catalog import FAMILY_CATALOG, find_family
from zoomy.finalize_workflow import FinalizeRequest
from zoomy.frame_repository import FrameRepository
from zoomy.frame_workflow import FrameRenderRequest
from zoomy.rendering import finalize_video, render_next_frame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Any

SCRIPTED_PROMPT_IDENTIFIER = "scripted-prompt"


class ScriptedConnection:
    """Test double that replays scripted history transitions in order."""

    def __init__(self, entries: Sequence[HistoryEntry | None]) -> None:
        """Store the scripted history transitions and start with no calls."""
        self.entries = list(entries)
        self.queued_workflows: list[Mapping[str, Any]] = []

    def is_reachable(self) -> bool:
        """Report a healthy connection."""
        return True

    def queue_workflow(self, workflow: Mapping[str, Any]) -> str:
        """Record the submitted workflow and return the scripted id."""
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


def test_render_next_frame_success_yields_final_frame(tmp_path: Path) -> None:
    """A completed job yields a final update with the newest frame."""
    repository = _repository_with_frames(tmp_path, frame_count=2)
    connection = ScriptedConnection([None, _completed_entry()])
    updates = list(
        render_next_frame(
            _frame_request(repository),
            connection,
            repository,
            operation_timeout_seconds=5.0,
            poll_interval_seconds=0.0,
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
                connection,
                repository,
                operation_timeout_seconds=5.0,
                poll_interval_seconds=0.0,
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
                connection,
                repository,
                operation_timeout_seconds=5.0,
                poll_interval_seconds=0.0,
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
                connection,
                repository,
                operation_timeout_seconds=0.0,
                poll_interval_seconds=0.0,
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
                connection,
                repository,
                operation_timeout_seconds=5.0,
                poll_interval_seconds=0.0,
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
            connection,
            repository,
            operation_timeout_seconds=5.0,
            poll_interval_seconds=0.0,
        )
    )
    final_update = updates[-1]
    assert final_update.video_path == video_path
    assert final_update.frame_count == 3
