"""Routing agreement (issue 081 extraction from `supervisor`).

`VIDEO/AUDIO_WORKER_MODULES`, `STREAMING_VIDEO_BACKENDS`,
`audio_worker_module` and `video_worker_module` are the verbatim
backend-routing block moved to `voyage.supervisor_routing` so
`supervisor.py` shrinks toward the §12 split signal. Behavior contract:
identical to the pre-split module-level names — the facade re-export is
the same object (single source, not a copy), the streaming tuple and the
worker-module keys agree with the `BACKEND_REGISTRY` projection
(issues 023/025), and unknown backends raise `ConfigurationError`
(the video path keeps the 079 longlive2 migration hint).
"""

from __future__ import annotations

import pytest

import voyage.supervisor as supervisor
import voyage.supervisor_routing as supervisor_routing
from voyage.supervisor_routing import audio_worker_module, video_worker_module


def test_facade_reexport_is_single_sourced() -> None:
    """The facade names are the new home objects, not copies (issue 081)."""
    assert supervisor.VIDEO_WORKER_MODULES is supervisor_routing.VIDEO_WORKER_MODULES
    assert supervisor.STREAMING_VIDEO_BACKENDS is supervisor_routing.STREAMING_VIDEO_BACKENDS
    assert supervisor.AUDIO_WORKER_MODULES is supervisor_routing.AUDIO_WORKER_MODULES
    assert supervisor.audio_worker_module is supervisor_routing.audio_worker_module
    assert supervisor.video_worker_module is supervisor_routing.video_worker_module


def test_streaming_sets_agree_with_registry() -> None:
    """Supervisor tuple, routing tuple, and registry projection match (023/025)."""
    from voyage import backends
    from voyage.config import BACKEND_REGISTRY

    expected = {name for name, record in BACKEND_REGISTRY.items() if record.streaming}
    assert set(supervisor_routing.STREAMING_VIDEO_BACKENDS) == expected
    assert set(supervisor.STREAMING_VIDEO_BACKENDS) == expected
    assert set(backends._STREAMING_BACKENDS) == expected


def test_worker_modules_cover_registry() -> None:
    """Every registry backend has a worker module (no silent gap)."""
    from voyage.config import BACKEND_REGISTRY

    assert set(supervisor_routing.VIDEO_WORKER_MODULES) == set(BACKEND_REGISTRY)
    assert set(supervisor.VIDEO_WORKER_MODULES) == set(BACKEND_REGISTRY)


def test_known_backends_resolve() -> None:
    """Known backends resolve to their worker modules (verbatim behavior)."""
    assert video_worker_module("fake") == "voyage.workers.video"
    assert video_worker_module("ltxv") == "voyage.workers.video_ltxv"
    assert audio_worker_module("fake") == "voyage.workers.audio"


def test_unknown_audio_backend_raises() -> None:
    """Unknown audio backends fail fast with the known set."""
    from voyage.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="unknown audio backend"):
        audio_worker_module("nope")


def test_unknown_video_backend_keeps_longlive2_hint() -> None:
    """Unknown video backends fail fast; removed longlive2 keeps the 079 hint."""
    from voyage.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="unknown video backend"):
        video_worker_module("nope")
    with pytest.raises(ConfigurationError) as excinfo:
        video_worker_module("longlive2")
    message = str(excinfo.value).lower()
    assert "longlive2" in message
    assert "ltxv" in message
    assert "tape" in message
