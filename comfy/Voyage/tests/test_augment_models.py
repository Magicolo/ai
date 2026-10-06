"""Track C: FILM + Real-ESRGAN registry specs and ensure wiring (DESIGN §§84-85).

The finalize-stage augmentation weights (FILM interpolation + Real-ESRGAN
anime upscaler) ship as two `model_registry` rows fetched at runtime into
`/models` and ensured by `generate` on CUDA backends only. These tests pin
the registry shape (repo/revision/file/floor/license, mirroring the
LTXV/MMAudio rows) and the ensure gate (CUDA includes, fake stays empty,
`augment_enabled=False` restores the pre-augmentation set). CPU-only: file
presence/size checks on `tmp_path` plus pure `required_specs` transforms —
no hub, no torch, no ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import model_registry
from voyage.config import ProjectConfig, VideoBackendName, with_video_backend


def _base_config() -> ProjectConfig:
    return ProjectConfig(style="pastel neon line-art, peaceful")


def _write_sized_file(candidate: Path, size_bytes: int) -> None:
    """Create a sparse file of exactly `size_bytes` (instant, no 60 MiB write)."""
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open("wb") as handle:
        handle.truncate(size_bytes)


def test_film_spec_pins_repo_file_and_license() -> None:
    spec = model_registry.MODEL_SPECS["film"]
    assert spec.manifest_key == "film"
    assert spec.snapshots == ()
    assert len(spec.files) == 1
    filereq = spec.files[0]
    assert filereq.repo_id == model_registry.FILM_HF_REPO == "Comfy-Org/frame_interpolation"
    assert filereq.revision == model_registry.FILM_HF_REVISION != ""
    assert filereq.filename.endswith("film_net_fp16.safetensors")
    assert model_registry.FILM_MIN_BYTES >= 60_000_000
    assert "Apache" in model_registry.FILM_LICENSE
    assert "MIT" in model_registry.FILM_LICENSE


def test_realesrgan_spec_pins_mirror_upstream_and_license() -> None:
    spec = model_registry.MODEL_SPECS["realesrgan-anime"]
    assert spec.manifest_key == "realesrgan"
    assert spec.snapshots == ()
    assert len(spec.files) == 1
    filereq = spec.files[0]
    assert filereq.repo_id == model_registry.REALESRGAN_HF_REPO != ""
    assert filereq.revision == model_registry.REALESRGAN_HF_REVISION != ""
    assert filereq.filename == model_registry.REALESRGAN_ANIME_FILE
    assert filereq.filename == "realesr-animevideov3.pth"
    assert model_registry.REALESRGAN_ANIME_MIN_BYTES >= 2_100_000
    assert "xinntao/Real-ESRGAN" in model_registry.REALESRGAN_UPSTREAM_URL
    assert "BSD" in model_registry.REALESRGAN_LICENSE


def test_film_verify_missing_message_names_weights(tmp_path: Path) -> None:
    ok, message = model_registry.verify_model(tmp_path, "film")
    assert ok is False
    assert message.startswith("missing")
    assert "film_net_fp16.safetensors" in message


def test_realesrgan_verify_missing_message_names_weights(tmp_path: Path) -> None:
    ok, message = model_registry.verify_model(tmp_path, "realesrgan-anime")
    assert ok is False
    assert message.startswith("missing")
    assert "realesr-animevideov3.pth" in message


def test_film_verify_passes_with_sized_file(tmp_path: Path) -> None:
    candidate = tmp_path / model_registry.FILM_REPO_PATH
    _write_sized_file(candidate, model_registry.FILM_MIN_BYTES)
    ok, message = model_registry.verify_model(tmp_path, "film")
    assert ok is True
    assert message.startswith("film OK")


def test_realesrgan_verify_passes_with_sized_file(tmp_path: Path) -> None:
    candidate = tmp_path / model_registry.REALESRGAN_SUBDIR / model_registry.REALESRGAN_ANIME_FILE
    _write_sized_file(candidate, model_registry.REALESRGAN_ANIME_MIN_BYTES)
    ok, message = model_registry.verify_model(tmp_path, "realesrgan-anime")
    assert ok is True
    assert message.startswith("realesrgan-anime OK")


@pytest.mark.parametrize("backend", ["ltxv", "causvid"])
def test_cuda_backends_include_augmentation_by_default(backend: VideoBackendName) -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_base_config(), backend)
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert {"rife", "realesrgan-anime"} <= specs


def test_fake_backend_stays_empty_with_augment_enabled() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_base_config(), "fake")
    assert required_specs(config, sfx_enabled=False, augment_enabled=True) == []
    assert required_specs(config, sfx_enabled=False, augment_enabled=False) == []


def test_augment_disabled_excludes_film_and_realesrgan() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_base_config(), "ltxv")
    specs = {item.spec for item in required_specs(config, sfx_enabled=False, augment_enabled=False)}
    assert "film" not in specs
    assert "rife" not in specs
    assert "realesrgan-anime" not in specs
    assert "ltxv-2b" in specs


def test_augment_specs_share_video_models_dir() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_base_config(), "ltxv")
    entries = {item.spec: item.models_dir for item in required_specs(config, sfx_enabled=False)}
    assert entries["rife"] == entries["realesrgan-anime"] == Path(config.video.models_dir)
