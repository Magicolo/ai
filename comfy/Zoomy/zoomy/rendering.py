"""Render orchestration: run engine jobs and report progress.

The entry points are generators so the Gradio interface can stream status
updates while the in-process engine works:

    render_next_frame(...)  -> yields ProgressUpdate with the new frame
    finalize_video(...)     -> yields ProgressUpdate with the finished video
    render_loop(...)        -> yields ProgressUpdate per frame until the target
        count, a stop request, or exhausted per-frame retries

Frame renders call the engine once (a blocking diffusion pass), save the
returned image through the frame repository, and report the measured time.
Finalize delegates progress streaming to the engine, which yields one update
per segment window plus the terminal update carrying the finished video.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zoomy.engine_protocol import ProgressUpdate
from zoomy.errors import (
    EmptyFrameSequenceError,
    EngineConfigurationError,
    RenderInterruptedError,
    ZoomyError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from zoomy.engine_protocol import EngineProtocol, FinalizeRequest, FrameRenderRequest
    from zoomy.frame_repository import FrameRepository

DEFAULT_MAX_ATTEMPTS_PER_FRAME = 3

_loop_stop_requested = threading.Event()


@dataclass(frozen=True, slots=True)
class RenderEnvironment:
    """Everything a render operation needs besides its workflow request.

    Bundles the generation engine with the artifact repository so
    multi-parameter operations (like the frame loop) stay under the
    argument-count budget.
    """

    engine: EngineProtocol
    repository: FrameRepository


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
        ProgressUpdate: Submission and rendering messages, ending with one
        that carries the new frame path, the updated frame count, and the
        measured render time.

    Raises:
        EngineExecutionError: The engine stage failed while rendering.
        RenderInterruptedError: The job was interrupted.
    """
    next_frame_number = request.frame_count + 1
    yield ProgressUpdate(
        message=f"Submitting frame {next_frame_number} to the engine…",
        frame_count=request.frame_count,
    )
    operation_started = time.monotonic()
    yield ProgressUpdate(
        message=f"The engine is rendering frame {next_frame_number}…",
        frame_count=request.frame_count,
    )
    frame_image = environment.engine.render_frame(request)
    frame_path = environment.repository.save_next_frame(request.family.sequence_key, frame_image)
    updated_frame_count = environment.repository.frame_count(request.family.sequence_key)
    elapsed_seconds = time.monotonic() - operation_started
    yield ProgressUpdate(
        message=(
            f"Frame {next_frame_number} complete in {elapsed_seconds:.0f} s "
            f"({updated_frame_count} in sequence)."
        ),
        frame_path=frame_path,
        frame_count=updated_frame_count,
        elapsed_seconds=elapsed_seconds,
    )


def finalize_video(
    request: FinalizeRequest,
    environment: RenderEnvironment,
) -> Iterator[ProgressUpdate]:
    """Turn the sequence into a video and yield progress until it lands.

    Progress streams from the engine (one update per segment window); short
    sequences finalize in one pass while long ones render window by window
    with bounded VRAM per job and assemble in Python.

    Raises:
        EmptyFrameSequenceError: The sequence has no frames to finalize.
        EngineExecutionError: The engine stage failed while finalizing.
        RenderInterruptedError: The job was interrupted.
        AssemblyError: Segment outputs are missing or the ffmpeg assembly
            failed.
    """
    if request.frame_count < 1:
        raise EmptyFrameSequenceError(
            "Cannot finalize a video: the sequence has no frames yet. "
            "Render at least one frame first."
        )
    yield ProgressUpdate(message="Submitting finalize to the engine…")
    yield from environment.engine.finalize_sequence(request)


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
    fresh random seed. A frame that fails with a transient error is retried
    up to ``max_attempts_per_frame`` times; user interrupts and engine
    configuration errors end the loop immediately because retrying them is
    pointless.

    Yields:
        ProgressUpdate: Loop lifecycle lines plus every nested frame update.

    Raises:
        RenderInterruptedError: The running job was interrupted.
        EngineConfigurationError: The engine rejected a frame request.
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

    User interrupts and engine configuration errors propagate immediately;
    other failures retry until the attempt budget is spent, ending with False.
    """
    for attempt in range(1, max_attempts_per_frame + 1):
        try:
            yield from render_next_frame(request, environment)
        except (RenderInterruptedError, EngineConfigurationError):
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
