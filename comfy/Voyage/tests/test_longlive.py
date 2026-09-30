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
    assert set(layout) == {
        "wan_dir",
        "generator_ckpt",
        "wan21_dir",
        "causvid_dir",
        "ltxv_dir",
        "ltxv_text_encoder_dir",
        "qwen_dir",
        "inspector_dir",
        "minilm_dir",
        "acestep_dir",
        "sfx_dir",
        "film_dir",
        "realesrgan_dir",
        "manifest",
    }
    assert layout["generator_ckpt"].endswith("model_bf16.pt")
    assert layout["causvid_dir"].endswith(model_registry.CAUSVID_SUBDIR)
    assert layout["wan21_dir"].endswith(model_registry.WAN21_SUBDIR)
    assert layout["inspector_dir"].endswith(model_registry.QWEN35_SUBDIR)
    assert layout["ltxv_text_encoder_dir"].endswith(model_registry.LTXV_TE_SUBDIR)
    assert layout["sfx_dir"].endswith(model_registry.MMAUDIO_SUBDIR)
    assert layout["film_dir"].endswith(model_registry.FILM_SUBDIR)
    assert layout["realesrgan_dir"].endswith(model_registry.REALESRGAN_SUBDIR)


def test_director_pins(tmp_path: Path) -> None:
    assert model_registry.QWEN_HF_REPO == "Qwen/Qwen3-8B"
    assert len(model_registry.QWEN_HF_REVISION) == 40
    assert model_registry.MINILM_HF_REPO == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(model_registry.MINILM_HF_REVISION) == 40
    ok, message = model_registry.verify_director_models(tmp_path)
    assert not ok
    assert "missing" in message


def test_audio_pins(tmp_path: Path) -> None:
    assert model_registry.ACE_MAIN_REPO == "ACE-Step/Ace-Step1.5"
    assert len(model_registry.ACE_MAIN_REVISION) == 40
    assert model_registry.ACE_LM_REPO == "ACE-Step/acestep-5Hz-lm-0.6B"
    assert len(model_registry.ACE_LM_REVISION) == 40
    ok, message = model_registry.verify_audio_models(tmp_path)
    assert not ok
    assert "missing" in message


def test_inspector_pins(tmp_path: Path) -> None:
    assert model_registry.QWEN35_HF_REPO == "Qwen/Qwen3.5-9B"
    assert len(model_registry.QWEN35_HF_REVISION) == 40
    assert "chat_template.jinja" in model_registry.QWEN35_ALLOW
    ok, message = model_registry.verify_inspector_models(tmp_path)
    assert not ok
    assert "missing" in message


def test_verify_reports_missing_on_empty_dir(tmp_path: Path) -> None:
    ok, message = model_registry.verify_longlive2_bf16(tmp_path)
    assert not ok
    assert "missing" in message


def test_registry_carries_augment_specs() -> None:
    film = model_registry.MODEL_SPECS["film"]
    assert film.manifest_key == "film"
    assert [file.repo_id for file in film.files] == ["Comfy-Org/frame_interpolation"]
    esrgan = model_registry.MODEL_SPECS["realesrgan-anime"]
    assert esrgan.manifest_key == "realesrgan"
    assert [file.repo_id for file in esrgan.files] == ["amd/realesrgan-x4plus-anime-6b"]


def test_film_pins(tmp_path: Path) -> None:
    assert model_registry.FILM_HF_REPO == "Comfy-Org/frame_interpolation"
    assert len(model_registry.FILM_HF_REVISION) == 40
    assert model_registry.FILM_FILE == "film_net_fp16.safetensors"
    assert model_registry.FILM_MIN_BYTES == 60_000_000
    ok, message = model_registry.verify_film_models(tmp_path)
    assert not ok
    assert "missing" in message
    assert "film_net_fp16.safetensors" in message


def test_realesrgan_pins(tmp_path: Path) -> None:
    assert model_registry.REALESRGAN_HF_REPO == "amd/realesrgan-x4plus-anime-6b"
    assert len(model_registry.REALESRGAN_HF_REVISION) == 40
    assert model_registry.REALESRGAN_ANIME_FILE == "RealESRGAN_x4plus_anime_6B.pth"
    assert model_registry.REALESRGAN_ANIME_MIN_BYTES == 15_000_000
    ok, message = model_registry.verify_realesrgan_models(tmp_path)
    assert not ok
    assert "missing" in message
    assert "RealESRGAN_x4plus_anime_6B.pth" in message


def test_augment_specs_verify_ok_above_floor(tmp_path: Path) -> None:
    film_path = tmp_path / model_registry.FILM_REPO_PATH
    film_path.parent.mkdir(parents=True)
    with film_path.open("wb") as handle:
        handle.truncate(model_registry.FILM_MIN_BYTES)
    ok, message = model_registry.verify_film_models(tmp_path)
    assert ok
    assert message.startswith("film OK")
    esrgan_path = tmp_path / model_registry.REALESRGAN_SUBDIR / model_registry.REALESRGAN_ANIME_FILE
    esrgan_path.parent.mkdir(parents=True)
    with esrgan_path.open("wb") as handle:
        handle.truncate(model_registry.REALESRGAN_ANIME_MIN_BYTES)
    ok, message = model_registry.verify_realesrgan_models(tmp_path)
    assert ok
    assert message.startswith("realesrgan-anime OK")


def test_video_worker_module_map() -> None:
    assert video_worker_module("fake") == "voyage.workers.video"
    assert video_worker_module("longlive2") == "voyage.workers.video_longlive"
    with pytest.raises(ConfigurationError):
        video_worker_module("kling")
