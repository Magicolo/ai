"""GPU director placement: device plumbing + AWQ substitution (DESIGN §140).

The director serves Qwen3-4B-AWQ on `cuda:1` by default and keeps the bf16
Qwen3-8B only for the explicit `--director-device cpu` opt-out. These tests
pin the contract without a GPU: config defaults, TOML round-trip, resolve
override, `required_specs` device pick, registry pins, and the worker's
placement validator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.config import DirectorConfig, ProjectConfig, with_video_backend


def test_normalize_device_accepts_cpu_and_cuda() -> None:
    from voyage.workers.director import _normalize_device

    assert _normalize_device("cpu") == "cpu"
    assert _normalize_device("cuda:1") == "cuda:1"
    assert _normalize_device("cuda:0") == "cuda:0"


def test_normalize_device_rejects_garbage() -> None:
    from voyage.workers.director import _normalize_device

    with pytest.raises(ValueError):
        _normalize_device("tpu:0")
    with pytest.raises(ValueError):
        _normalize_device("gpu")
    # Falsy input means "no preference" → the cuda:1 default, never a guess.
    assert _normalize_device("") == "cuda:1"
    assert _normalize_device(None) == "cuda:1"


def test_worker_default_model_ids_are_distinct_pins() -> None:
    from voyage.workers.director import DEFAULT_CPU_MODEL_ID, DEFAULT_CUDA_MODEL_ID

    assert DEFAULT_CPU_MODEL_ID == "Qwen/Qwen3-8B"
    assert DEFAULT_CUDA_MODEL_ID == "Qwen/Qwen3-4B-AWQ"
    assert DEFAULT_CPU_MODEL_ID != DEFAULT_CUDA_MODEL_ID


def test_director_config_defaults_to_second_gpu() -> None:
    assert DirectorConfig().device == "cuda:1"


def test_director_config_rejects_garbage_device() -> None:
    # Fail fast at the config layer: without the validator, garbage
    # survives into TOML and dies deep in the worker (after ensuring
    # weights). Mirrors _normalize_device across the import boundary.
    with pytest.raises(ValueError):
        DirectorConfig(device="tpu:0")
    assert DirectorConfig(device="cpu").device == "cpu"
    assert DirectorConfig(device="cuda:0").device == "cuda:0"


def test_default_toml_round_trips_director_device(tmp_path: Path) -> None:
    from voyage.config import preset_config

    config = preset_config("device-probe", "pastel neon line-art", 7)
    assert config.director.device == "cuda:1"
    assert config.director.model_id == "Qwen/Qwen3-8B"

    cpu_config = preset_config("device-probe", "pastel neon line-art", 7, director_device="cpu")
    assert cpu_config.director.device == "cpu"


def test_resolve_config_director_device_override() -> None:
    from voyage.config import resolve_config

    config = ProjectConfig(style="pastel neon line-art, peaceful")
    assert resolve_config(config, director_device="cpu").director.device == "cpu"
    assert resolve_config(config, director_device="cuda:0").director.device == "cuda:0"
    # Unset leaves the stored value alone.
    assert resolve_config(config).director.device == config.director.device


def _ltxv_config() -> ProjectConfig:
    return with_video_backend(ProjectConfig(style="pastel neon line-art, peaceful"), "ltxv")


def test_required_specs_default_to_awq_decider() -> None:
    from voyage.models_ensure import required_specs

    specs = {item.spec for item in required_specs(_ltxv_config(), sfx_enabled=False)}
    assert "director-qwen35-gguf" in specs
    assert "director-qwen4b-awq" not in specs
    assert "director-qwen8b" not in specs


def test_required_specs_cpu_opt_out_keeps_8b() -> None:
    from voyage.config import resolve_config
    from voyage.models_ensure import required_specs

    config = resolve_config(_ltxv_config(), director="qwen", director_device="cpu")
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert "director-qwen8b" in specs
    assert "director-qwen4b-awq" not in specs


def test_effective_qwen_id_substitutes_8b_on_cuda_only() -> None:
    from voyage.workers.director import (
        DEFAULT_CPU_MODEL_ID,
        DEFAULT_CUDA_MODEL_ID,
        _effective_qwen_id,
    )

    assert _effective_qwen_id(DEFAULT_CPU_MODEL_ID, "cuda:1") == DEFAULT_CUDA_MODEL_ID
    assert _effective_qwen_id(DEFAULT_CPU_MODEL_ID, "cuda:0") == DEFAULT_CUDA_MODEL_ID
    # CPU keeps the bf16 default; explicit ids pass through on any device.
    assert _effective_qwen_id(DEFAULT_CPU_MODEL_ID, "cpu") == DEFAULT_CPU_MODEL_ID
    assert _effective_qwen_id(DEFAULT_CUDA_MODEL_ID, "cuda:1") == DEFAULT_CUDA_MODEL_ID
    assert _effective_qwen_id("Qwen/Qwen3-4B", "cuda:1") == "Qwen/Qwen3-4B"


def test_registry_pins_awq_revision() -> None:
    from voyage import model_registry

    assert model_registry.QWEN4B_AWQ_HF_REPO == "Qwen/Qwen3-4B-AWQ"
    assert model_registry.QWEN4B_AWQ_HF_REVISION == "74d4bd2bd4bff9cafc9345221320bffb08b406a3"
    assert "director-qwen4b-awq" in model_registry.MODEL_SPECS
    spec = model_registry.MODEL_SPECS["director-qwen4b-awq"]
    assert spec.manifest_key == "director-awq"
