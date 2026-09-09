"""Render orchestration: queue workflows, poll history, report progress.

The entry points are generators so the Gradio interface can stream status
updates while ComfyUI works:

    render_next_frame(...)  -> yields ProgressUpdate with the new frame
    finalize_video(...)     -> yields ProgressUpdate with the finished video
    render_loop(...)        -> yields ProgressUpdate per frame until the target
        count, a stop request, or exhausted per-frame retries

Single renders poll ``fetch_history`` until ComfyUI marks the prompt
completed, then translate the recorded status messages into typed errors
(execution failure, interrupt) and resolve the produced artifact through the
frame repository.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zoomy.errors import (
    ComfyExecutionError,
    ComfyRejectedWorkflowError,
    EmptyFrameSequenceError,
    OperationTimeoutError,
    RenderInterruptedError,
    ZoomyError,
)
from zoomy.finalize_workflow import build_finalize_workflow
from zoomy.frame_workflow import build_frame_workflow

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from pathlib import Path

    from zoomy.comfy_connection import ConnectionProtocol, HistoryEntry
    from zoomy.finalize_workflow import FinalizeRequest
    from zoomy.frame_repository import FrameRepository
    from zoomy.frame_workflow import FrameRenderRequest

EXECUTION_ERROR_EVENT = "execution_error"
EXECUTION_INTERRUPTED_EVENT = "execution_interrupted"
DEFAULT_MAX_ATTEMPTS_PER_FRAME = 3

_loop_stop_requested = threading.Event()


@dataclass(frozen=True, slots=True)
class RenderEnvironment:
    """Everything a render operation needs besides its workflow request.

    Bundles the connection, the artifact repository, and the polling budgets
    so multi-parameter operations (like the frame loop) stay under the
    argument-count budget. Attributes are documented at the call sites.
    """

    connection: ConnectionProtocol
    repository: FrameRepository
    operation_timeout_seconds: float
    poll_interval_seconds: float


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """One status update yielded to the interface.

    Attributes:
        message: Human-readable progress line.
        frame_path: Newest frame when one was produced this operation.
        video_path: Finished video when one was produced this operation.
        frame_count: Frames in the sequence after this operation.
        elapsed_seconds: Queue-to-completion seconds on terminal updates;
            ``None`` on interim progress updates.
    """

    message: str
    frame_path: Path | None = None
    video_path: Path | None = None
    frame_count: int = 0
    elapsed_seconds: float | None = None


def request_loop_stop() -> None:
    """Signal a running frame loop to stop after the current frame.

    Safe to call from any thread: the Stop button handler runs outside the
    queued loop generator. One global flag is sufficient because zoomy serves
    a single local user, so two loops can never run concurrently.
    """
    _loop_stop_requested.set()


def clear_loop_stop() -> None:
    """Clear a previous stop request, call it when starting a new loop."""
    _loop_stop_requested.clear()


def is_loop_stop_requested() -> bool:
    """Return True when a graceful loop stop was requested."""
    return _loop_stop_requested.is_set()


def render_next_frame(
    request: FrameRenderRequest,
    environment: RenderEnvironment,
) -> Iterator[ProgressUpdate]:
    """Render one frame and yield progress until it lands on disk.

    Yields:
        ProgressUpdate: Submission and wait messages, ending with one that
        carries the new frame path, the updated frame count, and the measured
        queue-to-completion time.

    Raises:
        ComfyExecutionError: ComfyUI recorded an execution_error status.
        RenderInterruptedError: The job was interrupted.
        OperationTimeoutError: The job exceeded the environment timeout.
    """
    next_frame_number = request.frame_count + 1
    yield ProgressUpdate(
        message=f"Submitting frame {next_frame_number} to ComfyUI…",
        frame_count=request.frame_count,
    )
    operation_started = time.monotonic()
    prompt_identifier = environment.connection.queue_workflow(build_frame_workflow(request))
    yield ProgressUpdate(
        message=f"ComfyUI is rendering frame {next_frame_number}…",
        frame_count=request.frame_count,
    )
    yield from _await_completion(
        environment,
        prompt_identifier,
        activity_message=f"Rendering frame {next_frame_number}",
    )
    updated_frame_count = environment.repository.frame_count(request.family.sequence_key)
    latest_frame_path = environment.repository.latest_frame_path(request.family.sequence_key)
    elapsed_seconds = time.monotonic() - operation_started
    yield ProgressUpdate(
        message=(
            f"Frame {next_frame_number} complete in {elapsed_seconds:.0f} s "
            f"({updated_frame_count} in sequence)."
        ),
        frame_path=latest_frame_path,
        frame_count=updated_frame_count,
        elapsed_seconds=elapsed_seconds,
    )


def finalize_video(
    request: FinalizeRequest,
    environment: RenderEnvironment,
) -> Iterator[ProgressUpdate]:
    """Turn the sequence into a video and yield progress until it lands.

    Raises:
        EmptyFrameSequenceError: The sequence has no frames to finalize.
        ComfyExecutionError: ComfyUI recorded an execution_error status.
        RenderInterruptedError: The job was interrupted.
        OperationTimeoutError: The job exceeded the environment timeout.
    """
    if request.frame_count < 1:
        raise EmptyFrameSequenceError(
            "Cannot finalize a video: the sequence has no frames yet. "
            "Render at least one frame first."
        )
    sequence_key = request.family.sequence_key
    yield ProgressUpdate(message="Submitting finalize workflow to ComfyUI…")
    operation_started = time.monotonic()
    prompt_identifier = environment.connection.queue_workflow(build_finalize_workflow(request))
    yield ProgressUpdate(
        message=(
            f"ComfyUI is finalizing {request.frame_count} frames "
            f"into the {sequence_key} video (interpolation, music, sound "
            "effects, and video encode all run here)…"
        )
    )
    yield from _await_completion(
        environment,
        prompt_identifier,
        activity_message="Finalizing video",
    )
    video_path = environment.repository.latest_video_path(sequence_key)
    elapsed_seconds = time.monotonic() - operation_started
    yield ProgressUpdate(
        message=f"Video complete in {elapsed_seconds:.0f} s.",
        video_path=video_path,
        frame_count=request.frame_count,
        elapsed_seconds=elapsed_seconds,
    )


def _await_completion(
    environment: RenderEnvironment,
    prompt_identifier: str,
    *,
    activity_message: str,
) -> Iterator[ProgressUpdate]:
    """Poll history until the prompt completes, yielding progress updates.

    The timeout check happens *after* the first history fetch so a fast job
    never trips a zero-second budget. Terminal failure messages are scanned
    *before* the completion flag because interrupted and errored prompts keep
    ``completed: False`` forever (verified live: an interrupted prompt stays
    ``{status_str: error, completed: False}`` with its ``execution_interrupted``
    message); the error scan runs on every poll so such entries can never
    wedge the loop until timeout.
    """
    operation_started = time.monotonic()
    operation_timeout_seconds = environment.operation_timeout_seconds
    poll_interval_seconds = environment.poll_interval_seconds
    while True:
        history_entry = environment.connection.fetch_history(prompt_identifier)
        if history_entry is not None:
            _raise_if_execution_failed(history_entry)
            if history_entry.is_completed:
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
        if not isinstance(message, list) or not message:
            continue
        event_name = message[0]
        if event_name == EXECUTION_ERROR_EVENT:
            details = message[1] if len(message) > 1 and isinstance(message[1], dict) else {}
            node_type = str(details.get("node_type") or "unknown node")
            exception_message = str(details.get("exception_message") or "unknown failure")
            raise ComfyExecutionError(node_type=node_type, exception_message=exception_message)
        if event_name == EXECUTION_INTERRUPTED_EVENT:
            raise RenderInterruptedError("The ComfyUI job was interrupted.")


def render_loop(
    request_factory: Callable[[int], FrameRenderRequest],
    sequence_key: str,
    environment: RenderEnvironment,
    *,
    frame_target: int | None,
    max_attempts_per_frame: int = DEFAULT_MAX_ATTEMPTS_PER_FRAME,
) -> Iterator[ProgressUpdate]:
    """Render frames until the target, a stop request, or exhausted retries.

    Each iteration builds a fresh request through ``request_factory`` (called
    with the current frame count, so sequences grow across iterations) with a
    fresh random seed. A frame that fails with a transient error (execution
    failure, timeout, connection loss) is retried up to
    ``max_attempts_per_frame`` times; user interrupts and workflow rejections
    end the loop immediately because retrying them is pointless.

    Yields:
        ProgressUpdate: Loop lifecycle lines plus every nested frame update.

    Raises:
        RenderInterruptedError: The running job was interrupted.
        ComfyRejectedWorkflowError: ComfyUI rejected a frame workflow.
        ValueError: ``max_attempts_per_frame`` is below 1 (zero attempts
            would spin the outer loop forever without rendering).
    """
    if max_attempts_per_frame < 1:
        message = f"max_attempts_per_frame must be at least 1, received {max_attempts_per_frame}"
        raise ValueError(message)
    clear_loop_stop()
    rendered_frame_count = 0
    if frame_target is None:
        yield ProgressUpdate(message="Loop started — rendering until stopped…")
    else:
        yield ProgressUpdate(message=f"Loop started — rendering {frame_target} frames…")
    while True:
        if is_loop_stop_requested():
            yield ProgressUpdate(
                message=f"Loop stopped after {_pluralize(rendered_frame_count, 'frame')}."
            )
            return
        if frame_target is not None and rendered_frame_count >= frame_target:
            yield ProgressUpdate(
                message=f"Loop target reached after {_pluralize(rendered_frame_count, 'frame')}."
            )
            return
        request = request_factory(environment.repository.frame_count(sequence_key))
        if not (
            yield from _attempt_frame_with_retries(request, environment, max_attempts_per_frame)
        ):
            return
        rendered_frame_count += 1


def _attempt_frame_with_retries(
    request: FrameRenderRequest,
    environment: RenderEnvironment,
    max_attempts_per_frame: int,
) -> Generator[ProgressUpdate, None, bool]:
    """Render one frame, retrying transient failures; True when rendered.

    User interrupts and workflow rejections propagate immediately; other
    failures retry until the attempt budget is spent, ending with False.
    """
    for attempt in range(1, max_attempts_per_frame + 1):
        try:
            yield from render_next_frame(request, environment)
        except (RenderInterruptedError, ComfyRejectedWorkflowError):
            raise
        except ZoomyError as failure:
            if attempt >= max_attempts_per_frame:
                yield ProgressUpdate(
                    message=(
                        f"Frame {request.frame_count + 1} failed after "
                        f"{max_attempts_per_frame} attempts: {failure} "
                        "Ending loop."
                    )
                )
                return False
            yield ProgressUpdate(
                message=(
                    f"Frame {request.frame_count + 1} attempt {attempt} failed: {failure} Retrying…"
                )
            )
        else:
            return True
    return True


def _pluralize(count: int, singular: str) -> str:
    """Format a count with a correctly pluralized noun ("1 frame", "2 frames")."""
    if count == 1:
        return f"1 {singular}"
    return f"{count} {singular}s"
