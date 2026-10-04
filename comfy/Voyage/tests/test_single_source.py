"""Single-source agreement tests (issue 085).

Every hand-duplicated set that cannot take a runtime import (stdlib-only
leaves, light media module, concurrently-owned supervisor) is pinned by
an import-time agreement test instead: the test fails the gate the moment
two copies diverge, without adding import weight or merge contention.
Sets that already derive (presets, state modes, escapers) are pinned too,
so a future hand copy fails loudly.
"""

from __future__ import annotations

import inspect

from voyage import backends, supervisor
from voyage.config import (
    BACKEND_REGISTRY,
    AugmentConfig,
    _audio_preset,
    _sfx_preset,
    _video_preset,
)
from voyage.media import (
    AUGMENT_DEFAULT_MIN_FPS,
    AUGMENT_DEFAULT_MIN_HEIGHT,
    AUGMENT_DEFAULT_MIN_WIDTH,
    FinalizeOptions,
)


def test_duration_epsilon_is_single_sourced() -> None:
    """Planning math uses the backends epsilon, not a restated literal."""
    import voyage.cli_planning as cli_planning

    source = inspect.getsource(cli_planning.segments_for_duration)
    assert "FLOAT_DUST_EPSILON" in source
    assert "1e-9" not in source
    assert backends.FLOAT_DUST_EPSILON == 1e-9


def test_streaming_sets_agree() -> None:
    """Supervisor tuple, backends frozenset, and registry projection match."""
    from_registry = {name for name, record in BACKEND_REGISTRY.items() if record.streaming}
    assert set(supervisor.STREAMING_VIDEO_BACKENDS) == from_registry
    assert set(backends._STREAMING_BACKENDS) == from_registry
    assert dict(backends.BACKEND_STATE_MODES) == {
        name: record.state_mode for name, record in BACKEND_REGISTRY.items()
    }


def test_worker_modules_cover_registry() -> None:
    """Every registry backend has a worker module (no silent gap)."""
    assert set(supervisor.VIDEO_WORKER_MODULES) == set(BACKEND_REGISTRY)


def test_augment_floors_agree() -> None:
    """Config, media, and finalize defaults ship the same floors."""
    config_defaults = AugmentConfig()
    finalize_defaults = FinalizeOptions()
    assert (
        config_defaults.min_fps,
        config_defaults.min_width,
        config_defaults.min_height,
    ) == (AUGMENT_DEFAULT_MIN_FPS, AUGMENT_DEFAULT_MIN_WIDTH, AUGMENT_DEFAULT_MIN_HEIGHT)
    assert (
        finalize_defaults.min_fps,
        finalize_defaults.min_width,
        finalize_defaults.min_height,
    ) == (
        config_defaults.min_fps,
        config_defaults.min_width,
        config_defaults.min_height,
    )


def test_presets_derive_from_registry() -> None:
    """Preset views project the registry rows (no second source)."""
    for name, record in BACKEND_REGISTRY.items():
        video = _video_preset(name)
        assert video["segment_frames"] == record.segment_frames
        assert video["fps"] == record.fps
        assert video["width"] == record.width
        assert video["height"] == record.height
        assert _audio_preset(name)["backend"] == record.audio_backend
        assert _sfx_preset(name)["backend"] == record.sfx_backend
