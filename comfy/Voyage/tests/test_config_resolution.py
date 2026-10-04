"""Single config resolver: preset → overrides (issue 025).

CPU-only: pure config transforms — no workers, no ffmpeg, no GPU. Pins
the resolution order (backend preset, then targeted overrides), the
wrapper parity (resolve_config vs with_video_backend), and the rejection
behavior.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voyage.config import (
    BACKEND_REGISTRY,
    DEV_MIN_FREE_SPACE_GIB,
    SPEC_MIN_FREE_SPACE_GIB,
    AudioConfig,
    ProjectConfig,
    VideoConfig,
    preset_config,
    resolve_config,
    with_video_backend,
)


def _base_config() -> ProjectConfig:
    return ProjectConfig(
        style="resolution-probe", audio=AudioConfig(take_seconds=30.0, ahead_seconds=20.0)
    )


def test_resolve_without_options_is_pure_noop() -> None:
    base = _base_config()
    before = base.model_dump()
    resolved = resolve_config(base)
    assert resolved.model_dump() == before
    assert base.model_dump() == before
    assert resolved is not base


def test_backend_branch_matches_with_video_backend() -> None:
    for name in BACKEND_REGISTRY:
        assert resolve_config(_base_config(), backend=name).model_dump() == (
            with_video_backend(_base_config(), name).model_dump()
        )


def test_resolution_order_preset_then_overrides() -> None:
    base = _base_config()
    # Explicit overrides win over the backend row …
    assert resolve_config(base, backend="ltxv", blocks=3).video.blocks_per_segment == 3
    assert resolve_config(base, take_seconds=31.0).audio.take_seconds == 31.0
    # … and without overrides the backend row rules the geometry.
    plain = resolve_config(base, backend="ltxv")
    row = BACKEND_REGISTRY["ltxv"]
    assert (plain.video.width, plain.video.height) == (row.width, row.height)


def test_full_matrix_backend_geometry() -> None:
    for name, row in BACKEND_REGISTRY.items():
        plain = resolve_config(_base_config(), backend=name)
        assert (plain.video.width, plain.video.height) == (row.width, row.height)
        assert plain.video.fps == row.fps
        assert plain.video.latent_shape == list(row.latent_shape)
        assert plain.audio.backend == row.audio_backend
        assert plain.audio.device == row.audio_device
        # The 30 s base take (not the 45 s default) survives resolution …
        assert plain.audio.take_seconds == 30.0
        assert plain.audio.backend == row.audio_backend


def test_targeted_overrides() -> None:
    resolved = resolve_config(
        _base_config(),
        director="qwen",
        blocks=2,
        take_seconds=31.0,
        quantization="bf16",
        beats_per_segment=8,
        drift_every_n_segments=3,
    )
    assert resolved.director.backend == "qwen"
    assert resolved.video.blocks_per_segment == 2
    assert resolved.audio.take_seconds == 31.0
    assert resolved.video.quantization == "bf16"
    assert resolved.audio.beats_per_segment == 8
    assert resolved.voyage.drift_every_n_segments == 3


def test_invalid_overrides_rejected() -> None:
    with pytest.raises(ValidationError):
        resolve_config(_base_config(), blocks=0)
    with pytest.raises(ValidationError):
        resolve_config(_base_config(), take_seconds=-1.0)
    with pytest.raises(ValidationError):
        resolve_config(_base_config(), quantization="ultra")


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="unknown video backend"):
        resolve_config(_base_config(), backend="framepack")  # type: ignore[arg-type]


def test_free_space_reserve_defaults_are_named_constants() -> None:
    """Triple default is explicit: spec 20.0 vs dev 5.0 (052)."""
    assert SPEC_MIN_FREE_SPACE_GIB == 20.0
    assert DEV_MIN_FREE_SPACE_GIB == 5.0
    assert ProjectConfig(style="reserve-probe").min_free_space_gib == SPEC_MIN_FREE_SPACE_GIB
    generated = preset_config("reserve-probe", "line art", 7)
    assert generated.min_free_space_gib == DEV_MIN_FREE_SPACE_GIB


def test_local_attn_size_defaults_to_continuity_capacity() -> None:
    # 16 = sink 8 + one 8-frame block: the minimum KV capacity at which
    # every chunk attends to real history (smaller caches evict the sink
    # and each chunk denoises from noise+text alone — a fresh scene per
    # chunk, measured ~6x boundary jumps). Rescued from the deleted
    # draft-mode tests: the default is backend-independent.
    assert VideoConfig().local_attn_size == 16


def test_local_attn_size_survives_resolution() -> None:
    out = resolve_config(_base_config(), backend="ltxv", blocks=3)
    assert out.video.local_attn_size == 16
    with pytest.raises(ValidationError):
        VideoConfig(local_attn_size=0)
