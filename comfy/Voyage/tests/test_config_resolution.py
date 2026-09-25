"""Single config resolver: preset → draft overlay → overrides (issue 025).

CPU-only: pure config transforms — no workers, no ffmpeg, no GPU. Pins
the resolution order (backend preset, then the stored [draft] overlay,
then targeted overrides), the wrapper parities (resolve_config vs
apply_draft_overrides/with_video_backend), and the rejection behavior.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voyage.config import (
    BACKEND_REGISTRY,
    AudioConfig,
    ProjectConfig,
    apply_draft_overrides,
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


def test_draft_and_override_branches_match_apply_draft_overrides() -> None:
    base = _base_config()
    assert resolve_config(base, draft=True).model_dump() == (
        apply_draft_overrides(base, draft=True).model_dump()
    )
    assert resolve_config(
        base, director="qwen", blocks=2, take_seconds=30.0, quantization="bf16"
    ).model_dump() == (
        apply_draft_overrides(
            base, director="qwen", blocks=2, take_seconds=30.0, quantization="bf16"
        ).model_dump()
    )


def test_resolution_order_preset_then_draft_then_overrides() -> None:
    base = _base_config()
    overlay = base.draft
    # Draft overlay wins over the backend row …
    drafted = resolve_config(base, backend="ltxv", draft=True)
    assert (drafted.video.width, drafted.video.height) == (overlay.width, overlay.height)
    assert drafted.video.latent_shape == list(overlay.latent_shape)
    assert drafted.audio.take_seconds == overlay.take_seconds
    # … and explicit overrides win over the overlay.
    assert resolve_config(base, backend="ltxv", draft=True, blocks=3).video.blocks_per_segment == 3
    assert resolve_config(base, draft=True, take_seconds=31.0).audio.take_seconds == 31.0
    # Without draft the backend row rules the geometry.
    plain = resolve_config(base, backend="ltxv")
    row = BACKEND_REGISTRY["ltxv"]
    assert (plain.video.width, plain.video.height) == (row.width, row.height)


def test_full_matrix_backend_by_draft() -> None:
    for name, row in BACKEND_REGISTRY.items():
        plain = resolve_config(_base_config(), backend=name)
        assert (plain.video.width, plain.video.height) == (row.width, row.height)
        assert plain.video.fps == row.fps
        assert plain.video.latent_shape == list(row.latent_shape)
        assert plain.audio.backend == row.audio_backend
        assert plain.audio.device == row.audio_device
        # The 30 s base take (not the 45 s default) survives without draft …
        assert plain.audio.take_seconds == 30.0
        drafted = resolve_config(_base_config(), backend=name, draft=True)
        # … and the overlay replaces geometry + take while the audio
        # pairing still rides the backend row.
        assert (drafted.video.width, drafted.video.height) == (640, 352)
        assert drafted.video.latent_shape == [1, 8, 48, 22, 40]
        assert drafted.audio.take_seconds == 45.0
        assert drafted.audio.take_seconds > drafted.audio.ahead_seconds
        assert drafted.audio.backend == row.audio_backend


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
