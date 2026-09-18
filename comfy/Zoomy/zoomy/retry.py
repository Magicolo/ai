"""Shared retry policy for generation stages and render loops.

Both the in-process engine and the render orchestration retry work, so the
out-of-memory detector and the attempt budget live here instead of in two
places. The engine retries OOMs after evicting resident stacks; the loop
retries transient stage failures but never OOMs (the engine already spent
its budget on those) and never interrupts or configuration errors.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from zoomy.errors import EngineConfigurationError, EngineExecutionError, RenderInterruptedError

if TYPE_CHECKING:
    from collections.abc import Callable

# Attempts per generation stage before giving up: each retry first evicts
# resident stacks, so transient fragmentation clears while a genuinely
# oversized job still fails fast enough to read about.
MAXIMUM_STAGE_ATTEMPTS = 3

# Substrings marking a video-memory failure across backends and wrappers.
# Torch raises ``OutOfMemoryError`` with "out of memory" on every backend,
# but allocator fragmentation, cuDNN workspace, NCCL, and HIP failures
# surface with different text that eviction can still fix.
_OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "out-of-memory",
    "outofmemory",
    "cuda error",
    "cudnn",
    "cublas",
    "nccl",
    "hip",
    "memory allocation",
    "allocator",
    "fragmentation",
    "allow_growth",
)

# Bare "oom" only counts as its own word ("CUDA OOM"): a plain substring
# would also match ordinary words like "boom" and misroute generic stage
# failures into the evict-and-retry path.
_OUT_OF_MEMORY_WORD_PATTERN = re.compile(r"\boom\b")


def is_out_of_memory(failure: BaseException) -> bool:
    """Detect a video-memory failure without importing torch.

    The heavy stack stays behind lazy imports (the slim test image has no
    torch), so detection matches the exception shape instead: the class
    name or a case-insensitive substring of the message. ``EngineExecutionError``
    wraps stage failures, so its embedded message is checked too.
    """
    if type(failure).__name__ == "OutOfMemoryError":
        return True
    message = str(failure).lower()
    if isinstance(failure, EngineExecutionError):
        message = f"{failure.stage} {failure.exception_message}".lower()
    return any(marker in message for marker in _OUT_OF_MEMORY_MARKERS) or bool(
        _OUT_OF_MEMORY_WORD_PATTERN.search(message)
    )


def is_retryable_transient(failure: BaseException) -> bool:
    """Return True when the render loop should retry a stage failure.

    Interrupts and configuration errors never retry (a stop request or a
    bad request repeats deterministically), and OOMs never retry here
    because the engine already spent its evict-and-retry budget inside
    :func:`run_stage_with_retries`.
    """
    if isinstance(failure, (RenderInterruptedError, EngineConfigurationError)):
        return False
    return not is_out_of_memory(failure)


def run_stage_with_retries[StageResultT](
    stage_name: str,
    operation: Callable[[], StageResultT],
    *,
    max_attempts: int = MAXIMUM_STAGE_ATTEMPTS,
    evict_resident_stacks: Callable[[], None],
) -> StageResultT:
    """Run one generation stage, retrying video-memory failures after eviction.

    Every attempt runs ``operation``; when it dies with an OOM the resident
    stacks are evicted (returning their VRAM) and the stage runs again.
    Interrupts and configuration errors propagate immediately — retrying a
    stop request or a bad request is pointless — as does any non-memory
    failure, which eviction cannot fix. An exhausted OOM budget raises an
    :class:`EngineExecutionError` naming the stage so the outer loop can
    recognize it and avoid a second retry layer (previously up to 9
    diffusion passes per frame: 3 inner x 3 outer).

    Raises:
        ValueError: ``max_attempts`` is below 1.
    """
    if max_attempts < 1:
        message = f"max_attempts must be at least 1, received {max_attempts}"
        raise ValueError(message)
    last_failure: BaseException | None = None
    for _attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except (RenderInterruptedError, EngineConfigurationError):
            raise
        except Exception as failure:
            last_failure = failure
            if not is_out_of_memory(failure):
                raise
            evict_resident_stacks()
    message = f"Stage {stage_name!r} ran out of video memory after {max_attempts} attempts"
    if last_failure is not None:
        message = f"{message}: {last_failure}"
        raise EngineExecutionError(stage_name, message) from last_failure
    raise EngineExecutionError(stage_name, message)
