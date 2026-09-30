"""Shared resident-stack helpers (issue 084).

Single home for `BYTES_PER_GIB` plus the `find_spec` optional-dependency
guards. The ACE-Step and MMAudio workers each carried their own
`BYTES_PER_GIB = 1024**3` plus a bespoke `_require_torch` with a
hardcoded needing-stack message; the director carries `_require_module`
as the generic template. Everything here is stdlib-only
(`importlib.util`) so slim images import it freely; `torch` itself only
appears inside caller functions behind these guards (§12 GPU ban).

Full stack-lifecycle collapse (`_require_stack` / `_convert` / evict /
shutdown across ACE-Step + MMAudio) is recorded as residual in the issue:
the two stacks differ in type (`AceStepStack` vs `SfxStack`), init
signature (model_size), and convert flags, so this pass unifies the
constant + guards only and leaves per-worker lifecycle in place.
"""

from __future__ import annotations

import importlib.util

BYTES_PER_GIB = 1024**3
"""Byte-to-GiB divisor for VRAM peak reporting (benchmark only)."""


def require_module(module_name: str, kind: str) -> None:
    """Fail fast with ImportError when an optional model stack is absent.

    Generic form of the director's `_require_module` template: a
    `find_spec` guard (never a bare import) so the worker loop maps the
    failure to retryable WORKER_ERROR with the needing stack named,
    and — because it runs before any cache mutation — a failed load never
    clobbers a resident entry.
    """
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"{kind} needs optional dependency {module_name!r} "
            "(slim image carries the fake worker only)"
        )


def require_torch(kind: str) -> None:
    """Fail fast with ImportError when `torch` is absent (slim image).

    Thin wrapper over :func:`require_module` for the common case: the
    benchmark's peak-memory accounting needs `torch.cuda`; without the
    guard the bare import raises ImportError anyway, but naming the
    needing op keeps the failure attributable.
    """
    require_module("torch", kind)
