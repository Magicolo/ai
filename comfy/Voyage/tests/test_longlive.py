"""Unit tests: model registry pins + video backend selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import model_registry
from voyage.errors import ConfigurationError
from voyage.supervisor import video_worker_module


def test_longlive_pins_are_set() -> None:
    assert len(model_registry.LONGLIVE_COMMIT) == 40
    assert len(model_registry.LONGLIVE_HF_REVISION) == 40
    assert model_registry.LONGLIVE_HF_FILE == "model_bf16.pt"
    assert model_registry.WAN_HF_REPO == "Wan-AI/Wan2.2-TI2V-5B"
    assert any(p.endswith(".pth") for p in model_registry.WAN_ALLOW)


def test_models_dir_layout_keys(tmp_path: Path) -> None:
    layout = model_registry.models_dir_layout(tmp_path)
    assert set(layout) == {"wan_dir", "generator_ckpt", "qwen_dir", "minilm_dir", "manifest"}
    assert layout["generator_ckpt"].endswith("model_bf16.pt")


def test_director_pins(tmp_path: Path) -> None:
    assert model_registry.QWEN_HF_REPO == "Qwen/Qwen3-8B"
    assert len(model_registry.QWEN_HF_REVISION) == 40
    assert model_registry.MINILM_HF_REPO == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(model_registry.MINILM_HF_REVISION) == 40
    ok, message = model_registry.verify_director_models(tmp_path)
    assert not ok
    assert "missing" in message


def test_verify_reports_missing_on_empty_dir(tmp_path: Path) -> None:
    ok, message = model_registry.verify_longlive2_bf16(tmp_path)
    assert not ok
    assert "missing" in message


def test_video_worker_module_map() -> None:
    assert video_worker_module("fake") == "voyage.workers.video"
    assert video_worker_module("longlive2") == "voyage.workers.video_longlive"
    with pytest.raises(ConfigurationError):
        video_worker_module("kling")
