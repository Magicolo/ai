"""Worker `init` rejects unknown fields instead of silently ignoring (127).

CPU-only: both SFX and ACE `handle_init` paths record config without
touching the GPU. A typo'd key (`model_is`, `backemd`, `model_dir`)
must fail as INVALID_PAYLOAD at startup, never boot defaults.
Module globals mutated here are restored so sibling tests keep
their defaults.
"""

from __future__ import annotations

import os

import pytest

from voyage.workers import audio_acestep, sfx_mmaudio


@pytest.fixture(autouse=True)
def _restore_worker_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sfx_mmaudio, "_models_dir", sfx_mmaudio._models_dir)
    monkeypatch.setattr(sfx_mmaudio, "_device", sfx_mmaudio._device)
    monkeypatch.setattr(sfx_mmaudio, "_model_size", sfx_mmaudio._model_size)
    monkeypatch.setattr(audio_acestep, "_models_dir", audio_acestep._models_dir)
    monkeypatch.setattr(audio_acestep, "_device", audio_acestep._device)


def test_sfx_init_rejects_unknown_key() -> None:
    with pytest.raises(TypeError, match="model_is"):
        sfx_mmaudio.handle_init({"models_dir": "/models", "model_is": "/models/custom"})


def test_sfx_init_accepts_all_documented_keys() -> None:
    response = sfx_mmaudio.handle_init(
        {"models_dir": "/models", "device": "cuda:1", "model_size": "small_44k"}
    )
    assert response["status"] == "READY"
    assert response["device"] == "cuda:1"
    assert response["model_size"] == "small_44k"


def test_sfx_init_rejects_mistyped_known_key() -> None:
    with pytest.raises(TypeError, match="device"):
        sfx_mmaudio.handle_init({"device": 1})


def test_ace_init_rejects_unknown_key_before_any_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """Unknown keys fail before the CWD redirect — a typo never moves state."""
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError, match="backemd"):
        audio_acestep.handle_init({"models_dir": "/models", "backemd": "qwen"})
    assert os.getcwd() == str(tmp_path)


def test_ace_init_accepts_all_documented_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    monkeypatch.chdir(tmp_path)
    response = audio_acestep.handle_init({"models_dir": "/models", "device": "cuda:0"})
    assert response["status"] == "READY"
    assert response["device"] == "cuda:0"
