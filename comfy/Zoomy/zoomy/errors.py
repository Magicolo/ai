"""Exception hierarchy for zoomy.

Every error zoomy raises inherits from :class:`ZoomyError`, so the interface
layer can catch one type and present any failure to the user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class ZoomyError(Exception):
    """Base class for every error raised by zoomy."""


class ComfyConnectionError(ZoomyError):
    """ComfyUI could not be reached or returned an unexpected response."""


class ComfyRejectedWorkflowError(ZoomyError):
    """ComfyUI rejected a workflow before queueing it (HTTP 400).

    This is a specification error: a workflow ComfyUI refuses to validate is a
    zoomy bug, not a transient failure, so retrying is pointless.
    """

    def __init__(self, summary: str, node_errors: dict[str, Any]) -> None:
        """Store the server-provided rejection details."""
        self.summary = summary
        self.node_errors = node_errors
        super().__init__(f"ComfyUI rejected the workflow: {summary}")

    def __str__(self) -> str:
        """Format the rejection, appending per-node details when present."""
        if not self.node_errors:
            return f"ComfyUI rejected the workflow: {self.summary}"
        details = "; ".join(f"{key}: {value}" for key, value in self.node_errors.items())
        return f"ComfyUI rejected the workflow: {self.summary} ({details})"


class ComfyExecutionError(ZoomyError):
    """A queued workflow failed while executing on ComfyUI.

    The node class and the raw exception message come from the execution_error
    status message ComfyUI records in the prompt's history entry.
    """

    def __init__(self, node_type: str, exception_message: str) -> None:
        """Store which node class failed and why."""
        self.node_type = node_type
        self.exception_message = exception_message
        super().__init__(f"ComfyUI execution failed in {node_type}: {exception_message}")


class RenderInterruptedError(ZoomyError):
    """The user interrupted the running ComfyUI job."""


class OperationTimeoutError(ZoomyError):
    """ComfyUI did not finish the job within the configured timeout."""


class EmptyFrameSequenceError(ZoomyError):
    """A video finalize was requested without any rendered frames."""
