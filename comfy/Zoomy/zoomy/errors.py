"""Exception hierarchy for zoomy.

Every error zoomy raises inherits from :class:`ZoomyError`, so the interface
layer can catch one type and present any failure to the user.
"""

from __future__ import annotations


class ZoomyError(Exception):
    """Base class for every error raised by zoomy."""


class EngineConfigurationError(ZoomyError):
    """The engine was misconfigured or received an invalid request.

    This is a specification error: bad directories, missing model files, or
    an invalid request are zoomy bugs, not transient failures, so retrying
    is pointless.
    """


class EngineExecutionError(ZoomyError):
    """A generation stage failed while the engine was running it.

    The stage names the pipeline step (for example ``ernie-frame`` or
    ``music``) and the raw exception message carries the underlying cause.
    """

    def __init__(self, stage: str, exception_message: str) -> None:
        """Store which engine stage failed and why."""
        self.stage = stage
        self.exception_message = exception_message
        super().__init__(f"Engine stage {stage} failed: {exception_message}")


class RenderInterruptedError(ZoomyError):
    """The user interrupted the running engine job."""


class EmptyFrameSequenceError(ZoomyError):
    """A video finalize was requested without any rendered frames."""


class AssemblyError(ZoomyError):
    """The Python-side assembly of segmented finalize outputs failed."""
