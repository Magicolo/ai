"""Step 2 VLM inspector op tests (DESIGN §44, retry→skip).

`handle_inspect` must never raise: two attempts, then `inspected: False`.
Model loading is monkeypatched — no weights needed (live VLM runs in Step 6).
"""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from voyage.workers import director as director_worker


def test_normalize_inspect_strips_summary() -> None:
    result = director_worker._normalize_inspect('{"scene_summary": "  neon dunes  "}')
    assert result["scene_summary"] == "neon dunes"


def test_normalize_inspect_accepts_fences() -> None:
    result = director_worker._normalize_inspect('```json\n{"scene_summary": "quiet harbor"}\n```')
    assert result["scene_summary"] == "quiet harbor"


def test_normalize_inspect_rejects_missing_summary() -> None:
    with pytest.raises(ValueError, match="scene_summary"):
        director_worker._normalize_inspect('{"other": "field"}')


def test_handle_inspect_skips_when_generation_fails(monkeypatch: MonkeyPatch) -> None:
    def _boom(model_id: str, frame_path: str, prompt: str, max_new_tokens: int) -> str:
        raise RuntimeError("no weights here")

    monkeypatch.setattr(director_worker, "_inspector_generate", _boom)
    result = director_worker.handle_inspect({"frame_path": "/tmp/frame.png"})
    assert result["inspected"] is False
    assert "no weights here" in str(result["error"])
    assert result["model_id"] == "Qwen/Qwen3.5-9B"


def test_handle_inspect_success(monkeypatch: MonkeyPatch) -> None:
    def _fake_generate(model_id: str, frame_path: str, prompt: str, max_new_tokens: int) -> str:
        assert frame_path == "/tmp/frame.png"
        return '{"scene_summary": "amber canyon"}'

    monkeypatch.setattr(director_worker, "_inspector_generate", _fake_generate)
    result = director_worker.handle_inspect({"frame_path": "/tmp/frame.png"})
    assert result == {
        "inspected": True,
        "scene_summary": "amber canyon",
        "model_id": "Qwen/Qwen3.5-9B",
    }


def test_handle_inspect_requires_frame_path() -> None:
    with pytest.raises(KeyError, match="frame_path"):
        director_worker.handle_inspect({})
