"""RIFE frame interpolation shared by every finalize window.

The x4 multiplier comes from two bisection levels: each level doubles the
frames, so depth 2 turns ``[a, b]`` into ``[a, m1, m, m2, b]``. The model
stays resident across windows; fewer than two frames have no pairs to
bisect and pass through unchanged.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from zoomy.engine_protocol import INTERPOLATION_MULTIPLIER
from zoomy.errors import EngineExecutionError

if TYPE_CHECKING:
    from collections.abc import Callable

_RIFE_MODEL_NAME = "RIFE_IFNet_v426_heavy"
# Bisection levels deriving the x4 multiplier: each level doubles the frames.
INTERPOLATION_BISECTION_DEPTH = int(math.log2(INTERPOLATION_MULTIPLIER))


def bisect(model: Any, first: Any, second: Any, depth: int) -> list[Any]:
    """Bisect one frame pair: depth 2 yields ``[a, m1, m, m2, b]``."""
    middle = model.inference_image_list([first, second])[0]
    if depth <= 1:
        return [first, middle, second]
    left = bisect(model, first, middle, depth - 1)
    right = bisect(model, middle, second, depth - 1)
    return [*left, *right[1:]]


def interpolate_arrays(
    model: Any,
    arrays: list[Any],
    raise_if_interrupted: Callable[[], None],
) -> list[Any]:
    """Expand numpy frame arrays 4x, batching pairs to share Python overhead.

    Pairs still bisect sequentially (one RIFE forward per bisection, three
    per pair for x4) because ``inference_image_list`` takes a single pair,
    but the loop checks interrupts once per pair and extends the output in
    place instead of copying prefix lists per pair.
    """
    interpolated: list[Any] = []
    for first, second in pairwise(arrays):
        raise_if_interrupted()
        sequence = bisect(model, first, second, INTERPOLATION_BISECTION_DEPTH)
        if not interpolated:
            interpolated.extend(sequence)
        else:
            interpolated.extend(sequence[1:])
    return interpolated


def load_rife_model() -> Any:
    """Load the heavy RIFE model, naming it in load failures."""
    try:
        from ccvfi import AutoModel, ConfigType  # noqa: PLC0415

        return AutoModel.from_pretrained(pretrained_model_name=ConfigType[_RIFE_MODEL_NAME])
    except Exception as failure:
        message = f"RIFE model load failed: {failure}"
        raise EngineExecutionError("interpolate-load", message) from failure
