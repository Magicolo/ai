"""Step 2 VLM inspector op tests (DESIGN §44, retry→skip).

`handle_inspect` must never raise: two attempts, then `inspected: False`.
Model loading is monkeypatched — no weights needed (live VLM runs in Step 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def test_handle_inspect_skips_when_generation_fails(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(model_id: str, frame_path: str, prompt: str, max_new_tokens: int) -> str:
        raise RuntimeError("no weights here")

    monkeypatch.setattr(director_worker, "_inspector_generate", _boom)
    frame_file = tmp_path / "frame.png"
    frame_file.write_bytes(b"fake-frame")
    result = director_worker.handle_inspect({"frame_path": str(frame_file)})
    assert result["inspected"] is False
    assert "no weights here" in str(result["error"])
    assert result["model_id"] == "Qwen/Qwen3.5-9B"


def test_handle_inspect_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    frame_file = tmp_path / "frame.png"
    frame_file.write_bytes(b"fake-frame")

    def _fake_generate(model_id: str, frame_path: str, prompt: str, max_new_tokens: int) -> str:
        assert frame_path == str(frame_file)
        return '{"scene_summary": "amber canyon"}'

    monkeypatch.setattr(director_worker, "_inspector_generate", _fake_generate)
    result = director_worker.handle_inspect({"frame_path": str(frame_file)})
    assert result == {
        "inspected": True,
        "scene_summary": "amber canyon",
        "model_id": "Qwen/Qwen3.5-9B",
    }


def test_handle_inspect_forwards_missing_path_unchecked(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Hook for the director track (issue 090): no existence check yet.

    The payload is `checked_request`ed then forwarded straight to the
    loader — the mocks bypass the real loader's missing-file error path
    entirely. This pins the current forwarding behavior; adding the
    boundary existence check is the director track's follow-up.
    """
    missing = tmp_path / "missing.png"
    seen: dict[str, Any] = {}

    def _record(model_id: str, frame_path: str, prompt: str, max_new_tokens: int) -> str:
        seen["frame_path"] = frame_path
        return '{"scene_summary": "empty room"}'

    monkeypatch.setattr(director_worker, "_inspector_generate", _record)
    result = director_worker.handle_inspect({"frame_path": str(missing)})
    assert result["inspected"] is True
    assert seen["frame_path"] == str(missing)


def test_handle_inspect_requires_frame_path() -> None:
    with pytest.raises(KeyError, match="frame_path"):
        director_worker.handle_inspect({})
