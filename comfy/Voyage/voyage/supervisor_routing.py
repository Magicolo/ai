"""Backend-routing tables for the supervisor (DESIGN §§73, 5).

Split from `voyage.supervisor` (issue 081): the commit path's
backend-routing block — worker-module maps, the streaming-backend set,
and the unknown-backend resolvers — as importable module-level names
with no supervisor state. `voyage.supervisor` re-exports every name
below so existing importers keep working; new code imports from here
directly.

- `VIDEO_WORKER_MODULES` / `AUDIO_WORKER_MODULES`: backend → worker module.
- `STREAMING_VIDEO_BACKENDS`: backends holding a resident session
  (multi-block payload, resume-hook restart path, acestep audio GPU swap).
- `audio_worker_module` / `video_worker_module`: resolvers raising
  `ConfigurationError` on unknown backends.
"""

from __future__ import annotations

from voyage.config import BACKEND_REGISTRY
from voyage.errors import ConfigurationError

VIDEO_WORKER_MODULES = {
    "fake": "voyage.workers.video",
    "ltxv": "voyage.workers.video_ltxv",
    "causvid": "voyage.workers.video_causvid",
    "ltx25": "voyage.workers.video_ltx25",
    "ltx23": "voyage.workers.video_ltx23",
}
"""Backend name → worker module. ltxv/causvid only exist in the CUDA image;
ltx25/ltx23 only exist in the voyage-ltx image."""

STREAMING_VIDEO_BACKENDS: tuple[str, ...] = tuple(
    name for name, record in BACKEND_REGISTRY.items() if record.streaming
)
"""Backends whose worker holds a resident session across blocks/segments.

Derived from `config.BACKEND_REGISTRY` (issues 023/083) — the single
source, so the supervisor tuple, `backends._STREAMING_BACKENDS`, and the
registry projection can never fork (see `test_streaming_sets_agree`).

These get the multi-block prompts/seeds payload, the resume-hook restart
path, and the acestep audio GPU swap (their DiT is GPU-resident, so audio
must evict + rebuild around takes). Fake renders statelessly per segment.
"""

AUDIO_WORKER_MODULES = {
    "fake": "voyage.workers.audio",
    "acestep": "voyage.workers.audio_acestep",
}
"""Backend name → worker module. acestep only exists in the GPU image."""


def audio_worker_module(backend: str) -> str:
    try:
        return AUDIO_WORKER_MODULES[backend]
    except KeyError:
        raise ConfigurationError(
            f"unknown audio backend {backend!r} (known: {sorted(AUDIO_WORKER_MODULES)})"
        ) from None


def video_worker_module(backend: str) -> str:
    try:
        return VIDEO_WORKER_MODULES[backend]
    except KeyError:
        raise ConfigurationError(
            f"unknown video backend {backend!r} (known: {sorted(VIDEO_WORKER_MODULES)})"
        ) from None
