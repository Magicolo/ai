"""Precision option: fp8 (default) vs bf16 DiT weights (slice 4).

fp8 W8A8 dynamic activation quantization proved to be the highlight-blowout
amplifier (bf16 probe renders clean). The config still carries the
quantization knob (a removed video backend derived its recovery
profile from it so tapes never resumed across numerics — issue 079).
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from voyage.config import (
    ProjectConfig,
    VideoConfig,
    apply_draft_overrides,
    default_config_toml,
    load_config,
)


def _base_config(tmp_path: Path) -> ProjectConfig:
    toml_path = tmp_path / "voyage.toml"
    toml_path.write_text(default_config_toml("precision-test", "line art", 7), encoding="utf-8")
    config, _ = load_config(toml_path)
    return config


def test_default_is_fp8() -> None:
    assert VideoConfig().quantization == "fp8"


def test_accepts_bf16() -> None:
    assert VideoConfig(quantization="bf16").quantization == "bf16"


def test_rejects_unknown_quantization() -> None:
    with pytest.raises(ValidationError):
        VideoConfig(**{"quantization": "int4"})  # type: ignore[arg-type]


def test_toml_carries_fp8_default() -> None:
    assert 'quantization = "fp8"' in default_config_toml("x", "pastel", 1)


def test_draft_preserves_quantization(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.video = VideoConfig(**{**config.video.model_dump(), "quantization": "bf16"})
    out = apply_draft_overrides(config, draft=True)
    assert out.video.quantization == "bf16"


def test_quantization_override(tmp_path: Path) -> None:
    out = apply_draft_overrides(_base_config(tmp_path), quantization="bf16")
    assert out.video.quantization == "bf16"
    with pytest.raises(ValidationError):
        apply_draft_overrides(_base_config(tmp_path), quantization="int4")
