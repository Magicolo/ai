"""Draft mode: cheap iteration profile + targeted CLI overrides (fast loop)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voyage.config import (
    DraftConfig,
    apply_draft_overrides,
    default_config_toml,
    load_config,
)


def _base_config(tmp_path):  # type: ignore[no-untyped-def]
    toml_path = tmp_path / "voyage.toml"
    toml_path.write_text(default_config_toml("draft-test", "line art", 7), encoding="utf-8")
    config, _ = load_config(toml_path)
    return config


def test_draft_profile_defaults() -> None:
    draft = DraftConfig()
    assert (draft.width, draft.height) == (640, 352)
    assert draft.latent_shape == [1, 8, 48, 22, 40]
    assert draft.blocks_per_segment == 1
    assert draft.take_seconds == 45.0


def test_no_flags_leaves_config_unchanged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _base_config(tmp_path)
    assert apply_draft_overrides(config).model_dump() == config.model_dump()


def test_draft_flag_applies_profile(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = apply_draft_overrides(_base_config(tmp_path), draft=True)
    assert (out.video.width, out.video.height) == (640, 352)
    assert out.video.latent_shape == [1, 8, 48, 22, 40]
    assert out.video.blocks_per_segment == 1
    assert out.audio.take_seconds == 45.0
    # Invariant: a draft take must outlast the audio-ahead window, or every
    # segment triggers a render + full GPU swap (measured: swap every segment
    # at take 15 < ahead 20).
    assert out.audio.take_seconds > out.audio.ahead_seconds


def test_targeted_overrides_without_draft(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = apply_draft_overrides(
        _base_config(tmp_path), director="deterministic", blocks=2, take_seconds=30.0
    )
    assert out.director.backend == "deterministic"
    assert out.video.blocks_per_segment == 2
    assert out.audio.take_seconds == 30.0


def test_overrides_compose_on_top_of_draft(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = apply_draft_overrides(_base_config(tmp_path), draft=True, blocks=3)
    assert (out.video.width, out.video.height) == (640, 352)
    assert out.video.blocks_per_segment == 3


def test_invalid_overrides_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        apply_draft_overrides(_base_config(tmp_path), blocks=0)
    with pytest.raises(ValidationError):
        apply_draft_overrides(_base_config(tmp_path), take_seconds=-1.0)
