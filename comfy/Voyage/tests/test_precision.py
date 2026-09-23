"""Precision option: fp8 (default) vs bf16 DiT weights (slice 4).

fp8 W8A8 dynamic activation quantization proved to be the highlight-blowout
amplifier (bf16 probe renders clean); the worker therefore offers a bf16
mode with its own recovery profile so tapes never resume across numerics.
"""

import pytest
from pydantic import ValidationError

from voyage.config import (
    VideoConfig,
    apply_draft_overrides,
    default_config_toml,
    load_config,
)
from voyage.workers.video_longlive import profile_for_quantization


def _base_config(tmp_path):  # type: ignore[no-untyped-def]
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
        VideoConfig(**{"quantization": "int4"})


def test_profile_mapping() -> None:
    assert profile_for_quantization("fp8") == "longlive2-bf16-fp8"
    assert profile_for_quantization("bf16") == "longlive2-bf16"
    with pytest.raises(ValueError, match="unknown quantization"):
        profile_for_quantization("int4")


def test_toml_carries_fp8_default() -> None:
    assert 'quantization = "fp8"' in default_config_toml("x", "pastel", 1)


def test_draft_preserves_quantization(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _base_config(tmp_path)
    config.video = VideoConfig(**{**config.video.model_dump(), "quantization": "bf16"})
    out = apply_draft_overrides(config, draft=True)
    assert out.video.quantization == "bf16"


def test_quantization_override(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = apply_draft_overrides(_base_config(tmp_path), quantization="bf16")
    assert out.video.quantization == "bf16"
    with pytest.raises(ValidationError):
        apply_draft_overrides(_base_config(tmp_path), quantization="int4")
