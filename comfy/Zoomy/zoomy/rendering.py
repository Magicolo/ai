"""Render orchestration: queue a workflow, poll history, report progress.

The two entry points are generators so the Gradio interface can stream status
updates while ComfyUI works:

    render_next_frame(...)  -> yields ProgressUpdate with the new frame
    finalize_video(...)     -> yields ProgressUpdate with the finished video

Both poll ``fetch_history`` until ComfyUI marks the prompt completed, then
translate the recorded status messages into typed errors (execution failure,
interrupt) and resolve the produced artifact through the frame repository.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zoomy.errors import (
    ComfyExecutionError,
    EmptyFrameSequenceError,
    OperationTimeoutError,
    RenderInterruptedError,
)
from zoomy.finalize_workflow import build_finalize_workflow
from zoomy.frame_workflow import build_frame_workflow

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from zoomy.comfy_connection import ConnectionProtocol, HistoryEntry
    from zoomy.finalize_workflow import FinalizeRequest
    from zoomy.frame_repository import FrameRepository
    from zoomy.frame_workflow import FrameRenderRequest

EXECUTION_ERROR_EVENT = "execution_error"
EXECUTION_INTERRUPTED_EVENT = "execution_interrupted"


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """One status update yielded to the interface.

    Attributes:
        message: Human-readable progress line.
        frame_path: Newest frame when one was produced this operation.
        video_path: Finished video when one was produced this operation.
        frame_count: Frames in the sequence after this operation.
    """

    message: str
    frame_path: Path | None = None
    video_path: Path | None = None
    frame_count: int = 0


def render_next_frame(
    request: FrameRenderRequest,
    connection: ConnectionProtocol,
    repository: FrameRepository,
    *,
    operation_timeout_seconds: float,
    poll_interval_seconds: float,
) -> Iterator[ProgressUpdate]:
    """Render one frame and yield progress until it lands on disk.

    Yields:
        ProgressUpdate: Submission and wait messages, ending with one that
        carries the new frame path and updated frame count.

    Raises:
        ComfyExecutionError: ComfyUI recorded an execution_error status.
        RenderInterruptedError: The job was interrupted.
        OperationTimeoutError: The job exceeded ``operation_timeout_seconds``.
    """
    next_frame_number = request.frame_count + 1
    yield ProgressUpdate(
        message=f"Submitting frame {next_frame_number} to ComfyUI…",
        frame_count=request.frame_count,
    )
    prompt_identifier = connection.queue_workflow(build_frame_workflow(request))
    yield ProgressUpdate(
        message=f"ComfyUI is rendering frame {next_frame_number}…",
        frame_count=request.frame_count,
    )
    yield from _await_completion(
        connection,
        prompt_identifier,
        operation_timeout_seconds=operation_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        activity_message=f"Rendering frame {next_frame_number}",
    )
    updated_frame_count = repository.frame_count(request.family.sequence_key)
    latest_frame_path = repository.latest_frame_path(request.family.sequence_key)
    yield ProgressUpdate(
        message=f"Frame {next_frame_number} complete ({updated_frame_count} in sequence).",
        frame_path=latest_frame_path,
        frame_count=updated_frame_count,
    )


def finalize_video(
    request: FinalizeRequest,
    connection: ConnectionProtocol,
    repository: FrameRepository,
    *,
    operation_timeout_seconds: float,
    poll_interval_seconds: float,
) -> Iterator[ProgressUpdate]:
    """Turn the sequence into a video and yield progress until it lands.

    Raises:
        EmptyFrameSequenceError: The sequence has no frames to finalize.
        ComfyExecutionError: ComfyUI recorded an execution_error status.
        RenderInterruptedError: The job was interrupted.
        OperationTimeoutError: The job exceeded ``operation_timeout_seconds``.
    """
    if request.frame_count < 1:
        raise EmptyFrameSequenceError(
            "Cannot finalize a video: the sequence has no frames yet. "
            "Render at least one frame first."
        )
    sequence_key = request.family.sequence_key
    yield ProgressUpdate(message="Submitting finalize workflow to ComfyUI…")
    prompt_identifier = connection.queue_workflow(build_finalize_workflow(request))
    yield ProgressUpdate(
        message=(
            f"ComfyUI is finalizing {request.frame_count} frames "
            f"into the {sequence_key} video (interpolation, music, sound "
            "effects, and video encode all run here)…"
        )
    )
    yield from _await_completion(
        connection,
        prompt_identifier,
        operation_timeout_seconds=operation_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        activity_message="Finalizing video",
    )
    video_path = repository.latest_video_path(sequence_key)
    yield ProgressUpdate(
        message="Video complete.",
        video_path=video_path,
        frame_count=request.frame_count,
    )


def _await_completion(
    connection: ConnectionProtocol,
    prompt_identifier: str,
    *,
    operation_timeout_seconds: float,
    poll_interval_seconds: float,
    activity_message: str,
) -> Iterator[ProgressUpdate]:
    """Poll history until the prompt completes, yielding progress updates.

    The timeout check happens *after* the first history fetch so a fast job
    never trips a zero-second budget, and the error scan runs only once, on
    the completed entry.
    """
    operation_started = time.monotonic()
    while True:
        history_entry = connection.fetch_history(prompt_identifier)
        if history_entry is not None and history_entry.is_completed:
            _raise_if_execution_failed(history_entry)
            return
        if time.monotonic() - operation_started >= operation_timeout_seconds:
            message = (
                f"{activity_message} did not finish within {operation_timeout_seconds:.0f} seconds"
            )
            raise OperationTimeoutError(message)
        elapsed_seconds = int(time.monotonic() - operation_started)
        yield ProgressUpdate(message=f"{activity_message}… {elapsed_seconds}s elapsed")
        time.sleep(poll_interval_seconds)


def _raise_if_execution_failed(history_entry: HistoryEntry) -> None:
    """Translate terminal status messages into typed zoomy errors.

    Raises:
        ComfyExecutionError: An execution_error message is present.
        RenderInterruptedError: An execution_interrupted message is present.
    """
    for message in history_entry.status_messages:
        event_name = message[0] if message else None
        if event_name == EXECUTION_ERROR_EVENT:
            details = message[1] if len(message) > 1 and isinstance(message[1], dict) else {}
            node_type = str(details.get("node_type") or "unknown node")
            exception_message = str(details.get("exception_message") or "unknown failure")
            raise ComfyExecutionError(node_type=node_type, exception_message=exception_message)
        if event_name == EXECUTION_INTERRUPTED_EVENT:
            raise RenderInterruptedError("The ComfyUI job was interrupted.")
