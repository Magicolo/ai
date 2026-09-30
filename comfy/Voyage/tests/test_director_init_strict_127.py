"""Director `init` rejects unknown fields instead of silently ignoring (127).

CPU-only: `handle_init` records config without touching the GPU. A typo'd
key (`model_is`, `backemd`, `model_dir`) must fail as INVALID_PAYLOAD at
startup, never boot defaults. `_CONFIG` is restored so sibling tests keep
their defaults (candidate 3 pins).
"""

from __future__ import annotations

import pytest

from voyage.workers import director


@pytest.fixture(autouse=True)
def _restore_director_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(director, "_CONFIG", dict(director._CONFIG))


def test_director_init_rejects_unknown_key() -> None:
    with pytest.raises(TypeError, match="model_is"):
        director.handle_init({"model_is": "/models/custom"})


def test_director_init_rejects_unknown_among_known() -> None:
    with pytest.raises(TypeError, match="backemd"):
        director.handle_init({"backend": "qwen", "backemd": "qwen"})


def test_director_init_rejects_models_dir_typo() -> None:
    with pytest.raises(TypeError, match="model_dir"):
        director.handle_init({"model_dir": "/models"})


def test_director_init_accepts_all_documented_keys() -> None:
    response = director.handle_init(
        {
            "backend": "qwen",
            "model_id": "Qwen/Qwen3-4B-AWQ",
            "device": "cuda:1",
            "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "inspector_model_id": "Qwen/Qwen3.5-9B",
            "models_dir": "/models",
        }
    )
    assert response == {"status": "READY", "backend": "qwen"}
    assert director._CONFIG["model_id"] == "Qwen/Qwen3-4B-AWQ"
    assert director._CONFIG["models_dir"] == "/models"


def test_director_init_rejects_mistyped_known_key() -> None:
    with pytest.raises(TypeError, match="device"):
        director.handle_init({"device": 1})


def test_director_init_empty_payload_keeps_defaults() -> None:
    response = director.handle_init({})
    assert response["status"] == "READY"
