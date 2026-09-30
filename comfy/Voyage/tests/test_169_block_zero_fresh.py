"""Issue 169: LTXV block-0 fresh restart must be counted + visible (CPU-only).

Worker-side pin mirroring causvid's ``fresh_rollouts``: block 0 rendered
with ``conditioning_source is None`` increments ``fresh_blocks`` and sets
``resume_fallback`` ``{reason, tail_path}`` (reasons: ``fresh_session``,
``scene_cut``, ``missing_tail``). Only the degraded ``missing_tail``
branch logs to stderr. The supervisor ``video_resume_fallback`` metric
stays a documented residual (supervisor.py is out of scope).
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voyage.workers import video_ltxv


class _FakeBlockTensor:
    def __init__(self, frames: int) -> None:
        self._array = np.zeros((1, 3, frames, 8, 8), dtype=np.float32)
        self.shape = self._array.shape

    @classmethod
    def _wrap(cls, array: Any) -> _FakeBlockTensor:
        instance = cls.__new__(cls)
        instance._array = np.asarray(array)
        instance.shape = instance._array.shape
        return instance

    def __getitem__(self, key: Any) -> _FakeBlockTensor:
        return _FakeBlockTensor._wrap(self._array[key])


class _FakeLtxvTorch:
    def cat(self, tensors: list[Any], dim: int = 0) -> _FakeBlockTensor:
        return _FakeBlockTensor._wrap(
            np.concatenate([tensor._array for tensor in tensors], axis=dim)
        )


def _fake_session(frame_counts: list[int], resident_tail: str | None = None) -> Any:
    def _generate_block(
        prompt: str,
        seed: int,
        width: int,
        height: int,
        frames: int,
        fps: int,
        conditioning: Any,
    ) -> _FakeBlockTensor:
        del prompt, seed, width, height, frames, fps, conditioning
        return _FakeBlockTensor(frame_counts[0])

    return types.SimpleNamespace(
        _torch=_FakeLtxvTorch(),
        _generate_block=_generate_block,
        _conditioning_tail_path=resident_tail,
        _last_prompt=None,
        _fp8_fallback=False,
    )


def _stub_save_mp4(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(images: Any, path: Path, fps: int) -> None:
        del images, fps
        path.write_bytes(b"stub-video")

    monkeypatch.setattr(video_ltxv, "_save_mp4", _stub)


def _render(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resident_tail: str | None,
    scene_cut: bool,
) -> dict[str, Any]:
    _stub_save_mp4(monkeypatch)
    session = _fake_session([121], resident_tail=resident_tail)
    return video_ltxv.LTXVSession.generate_blocks(
        session,
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[scene_cut],
        output_path=tmp_path / "seg.mp4",
        width=768,
        height=512,
        fps=24,
    )


def test_missing_tail_counts_fresh_block_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = str(tmp_path / "gone.mp4")
    result = _render(monkeypatch, tmp_path, missing, False)
    assert result["frames"] == 121
    assert result["fresh_blocks"] == 1
    fallback = result["resume_fallback"]
    assert fallback["reason"] == "missing_tail"
    assert fallback["tail_path"] == missing
    assert "missing" in capsys.readouterr().err


def test_fresh_session_counts_without_stderr_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _render(monkeypatch, tmp_path, None, False)
    assert result["fresh_blocks"] == 1
    assert result["resume_fallback"]["reason"] == "fresh_session"
    assert result["resume_fallback"]["tail_path"] is None
    assert "resume tail" not in capsys.readouterr().err


def test_scene_cut_counts_as_intended_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    resident = tmp_path / "video_tail.mp4"
    resident.write_bytes(b"resident-tail")
    result = _render(monkeypatch, tmp_path, str(resident), True)
    assert result["fresh_blocks"] == 1
    assert result["resume_fallback"]["reason"] == "scene_cut"
    assert "resume tail" not in capsys.readouterr().err


def test_conditioned_block_carries_no_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resident = tmp_path / "video_tail.mp4"
    resident.write_bytes(b"resident-tail")
    result = _render(monkeypatch, tmp_path, str(resident), False)
    assert result["novel_frames"] == 96
    assert result["fresh_blocks"] == 0
    assert result["resume_fallback"] is None
