"""CUDA preflight vocabulary separation (issue 021).

The hand-maintained ``_CUDA_BACKENDS`` mixed the audio ``acestep`` into a
set consumed for video-backend checks and omitted the SFX ``mmaudio``
backend entirely, so ``video=fake, audio=fake, sfx=mmaudio`` on a
torch-less image passed the fast-fail and died late inside the SFX
worker. These tests pin the registry-derived sets plus the SFX branch of
the offenders/require helpers. CPU-only: torch availability is stubbed.
"""

from __future__ import annotations

import argparse

import pytest

from voyage import cli
from voyage.config import (
    BACKEND_REGISTRY,
    AudioConfig,
    ProjectConfig,
    SfxConfig,
    VideoConfig,
)
from voyage.tui_state import gpu_warning


def _all_fake_config() -> ProjectConfig:
    """CPU-only config (product default video is ltxv/CUDA, so pin fake)."""
    base = ProjectConfig(style="probe")
    return base.model_copy(
        update={
            "video": VideoConfig(backend="fake"),
            "audio": AudioConfig(backend="fake"),
            "sfx": SfxConfig(backend="fake"),
        }
    )


def _sfx_only_cuda_config() -> ProjectConfig:
    """Fake video/audio with a CUDA SFX backend (the 021 false negative)."""
    base = ProjectConfig(style="cuda-probe")
    return base.model_copy(
        update={
            "video": VideoConfig(backend="fake"),
            "audio": AudioConfig(backend="fake"),
            "sfx": SfxConfig(backend="mmaudio", device="cuda:0"),
        }
    )


def test_video_cuda_set_derives_from_registry_devices() -> None:
    """Every cuda-device video row is flagged, every cpu row is not."""
    for name, record in BACKEND_REGISTRY.items():
        assert (name in cli._CUDA_VIDEO_BACKENDS) == record.device.startswith("cuda")
    assert "fake" not in cli._CUDA_VIDEO_BACKENDS
    assert "ltxv" in cli._CUDA_VIDEO_BACKENDS


def test_audio_and_sfx_cuda_sets_use_own_vocabularies() -> None:
    """Audio/SFX CUDA sets derive from their own device columns."""
    expected_audio = {
        record.audio_backend
        for record in BACKEND_REGISTRY.values()
        if record.audio_device.startswith("cuda")
    }
    expected_sfx = {
        record.sfx_backend
        for record in BACKEND_REGISTRY.values()
        if record.sfx_device.startswith("cuda")
    }
    assert frozenset(expected_audio) == cli._CUDA_AUDIO_BACKENDS
    assert frozenset(expected_sfx) == cli._CUDA_SFX_BACKENDS
    assert frozenset({"acestep"}) == cli._CUDA_AUDIO_BACKENDS
    assert frozenset({"mmaudio"}) == cli._CUDA_SFX_BACKENDS
    assert "acestep" not in cli._CUDA_VIDEO_BACKENDS
    assert "mmaudio" not in cli._CUDA_VIDEO_BACKENDS


def test_cuda_offenders_names_sfx_branch() -> None:
    """fake/fake/mmaudio reports the SFX backend, all-fake reports none."""
    offenders = cli._cuda_offenders(_sfx_only_cuda_config())
    assert offenders == ["sfx 'mmaudio'"]
    assert cli._cuda_offenders(_all_fake_config()) == []


def test_require_cuda_stack_fails_for_sfx_only_without_torch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The SFX-only CUDA run fast-fails naming sfx (was: silent pass)."""
    monkeypatch.setattr(cli, "_torch_available", lambda: False)
    assert cli._require_cuda_stack(_sfx_only_cuda_config()) is False
    assert "sfx 'mmaudio'" in capsys.readouterr().err


def test_require_cuda_stack_passes_all_fake_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-only runs never need the worker stack, torch or not."""
    monkeypatch.setattr(cli, "_torch_available", lambda: False)
    assert cli._require_cuda_stack(_all_fake_config()) is True


def test_gpu_warning_names_mmaudio_and_stays_silent_for_fake() -> None:
    """The TUI warning covers the real SFX CUDA need (was: silent)."""
    assert gpu_warning("mmaudio")
    assert "voyage-video" in gpu_warning("mmaudio")
    assert gpu_warning("fake") == ""


def test_generate_video_check_uses_video_vocabulary() -> None:
    """An audio-vocabulary name is never treated as a video backend."""
    assert "acestep" not in cli._CUDA_VIDEO_BACKENDS
    namespace = argparse.Namespace(backend="acestep")
    assert namespace.backend not in cli._CUDA_VIDEO_BACKENDS
