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
    preset_config,
    resolve_config,
)


def _base_config(tmp_path: Path) -> ProjectConfig:
    return preset_config("precision-test", "line art", 7)


def test_default_is_fp8() -> None:
    assert VideoConfig().quantization == "fp8"


def test_accepts_bf16() -> None:
    assert VideoConfig(quantization="bf16").quantization == "bf16"


def test_rejects_unknown_quantization() -> None:
    with pytest.raises(ValidationError):
        VideoConfig(**{"quantization": "int4"})  # type: ignore[arg-type]


def test_toml_carries_fp8_default() -> None:
    assert preset_config("x", "pastel", 1).video.quantization == "fp8"


def test_resolve_preserves_quantization(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.video = VideoConfig(**{**config.video.model_dump(), "quantization": "bf16"})
    out = resolve_config(config)
    assert out.video.quantization == "bf16"


def test_quantization_override(tmp_path: Path) -> None:
    out = resolve_config(_base_config(tmp_path), quantization="bf16")
    assert out.video.quantization == "bf16"
    with pytest.raises(ValidationError):
        resolve_config(_base_config(tmp_path), quantization="int4")
