"""LTXV failure hygiene, fps validation, mp4 clipping, encode device.

Covers issues 064 (chain-tail cleanup + short-tail assert + fps>0),
065 (`_save_mp4` clips before uint8), and the 074 LTXV `_encode` device.
CPU-only: block rendering and media I/O are faked/stubbed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voyage.workers import video_ltxv


class _FakeBlockTensor:
    """Numpy-backed (B,C,T,H,W) stand-in: shape reads and slicing are real."""

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


def _ltxv_session_self(
    frame_counts: list[int],
    fail_at: int | None = None,
    resident_tail: str | None = None,
) -> Any:
    """Fake LTXV session: scripted block lengths, optional mid-chain failure."""
    calls: list[dict[str, Any]] = []

    def _generate_block(
        prompt: str,
        seed: int,
        width: int,
        height: int,
        frames: int,
        fps: int,
        conditioning: str | None,
    ) -> _FakeBlockTensor:
        del prompt, seed, width, height, frames, fps
        calls.append({"conditioning": conditioning})
        if fail_at is not None and len(calls) - 1 == fail_at:
            raise RuntimeError("simulated block OOM")
        return _FakeBlockTensor(frame_counts[len(calls) - 1])

    return types.SimpleNamespace(
        _torch=_FakeLtxvTorch(),
        _generate_block=_generate_block,
        _conditioning_tail_path=resident_tail,
        _last_prompt=None,
        _fp8_fallback=False,
        _calls=calls,
    )


def _stub_save_mp4(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, int]]:
    """Replace `_save_mp4` with a stub that writes marker bytes."""
    saved: list[tuple[Path, int]] = []

    def _stub(images: Any, path: Path, fps: int) -> None:
        del images
        saved.append((path, fps))
        path.write_bytes(b"stub-video")

    monkeypatch.setattr(video_ltxv, "_save_mp4", _stub)
    return saved


def _chain_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*_chain*.mp4"))


def test_block_failure_cleans_up_chain_tails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_save_mp4(monkeypatch)
    session_self = _ltxv_session_self([121, 121], fail_at=1)
    output = tmp_path / "seg.mp4"
    with pytest.raises(RuntimeError, match="simulated block OOM"):
        video_ltxv.LTXVSession.generate_blocks(
            session_self,
            prompts=["amber dunes", "teal spires"],
            seeds=[7, 8],
            scene_cuts=[True, False],
            output_path=output,
            width=768,
            height=512,
            fps=24,
        )
    assert _chain_files(tmp_path) == []
    assert not output.exists()


def test_short_tail_anchor_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_save_mp4(monkeypatch)
    resident = tmp_path / "video_tail.mp4"
    resident.write_bytes(b"resident-tail")
    session_self = _ltxv_session_self([30], resident_tail=str(resident))
    with pytest.raises(ValueError, match="tail has 5 frames"):
        video_ltxv.LTXVSession.generate_blocks(
            session_self,
            prompts=["amber dunes"],
            seeds=[7],
            scene_cuts=[False],
            output_path=tmp_path / "seg.mp4",
            width=768,
            height=512,
            fps=24,
        )
    assert _chain_files(tmp_path) == []


def test_fresh_success_commits_without_orphans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _stub_save_mp4(monkeypatch)
    session_self = _ltxv_session_self([121])
    output = tmp_path / "seg.mp4"
    result = video_ltxv.LTXVSession.generate_blocks(
        session_self,
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[True],
        output_path=output,
        width=768,
        height=512,
        fps=24,
    )
    assert result["frames"] == 121
    assert output.exists()
    assert Path(str(result["conditioning_tail_path"])).exists()
    assert _chain_files(tmp_path) == []
    assert [fps for _, fps in saved] == [24, 24]


def test_validate_fps_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        video_ltxv.validate_fps(0)
    with pytest.raises(ValueError, match="fps must be positive"):
        video_ltxv.validate_fps(-24)
    video_ltxv.validate_fps(24)


def test_generate_blocks_rejects_bad_fps_before_render(tmp_path: Path) -> None:
    session_self = _ltxv_session_self([121])
    with pytest.raises(ValueError, match="fps must be positive"):
        video_ltxv.LTXVSession.generate_blocks(
            session_self,
            prompts=["amber dunes"],
            seeds=[7],
            scene_cuts=[True],
            output_path=tmp_path / "seg.mp4",
            width=768,
            height=512,
            fps=0,
        )
    assert session_self._calls == []


def test_handle_generate_blocks_rejects_bad_fps_without_session() -> None:
    assert video_ltxv._SESSION is None
    with pytest.raises(ValueError, match="fps must be positive"):
        video_ltxv.handle_generate_blocks(
            {
                "segment_id": "000000",
                "prompt": "amber dunes",
                "seed": 7,
                "output_path": "/tmp/voyage-ltxv-probe/seg.mp4",
                "fps": 0,
            }
        )


class _FakeVideoTensor:
    """(T,H,W,C) float frames with torch-style chained accessors."""

    def __init__(self, array: Any) -> None:
        self._array = np.asarray(array, dtype=np.float32)
        self.shape = self._array.shape

    def dim(self) -> int:
        return len(self.shape)

    def permute(self, *order: int) -> _FakeVideoTensor:
        return _FakeVideoTensor(np.transpose(self._array, order))

    def float(self) -> _FakeVideoTensor:
        return self

    def cpu(self) -> _FakeVideoTensor:
        return self

    def numpy(self) -> Any:
        return self._array

    def __getitem__(self, key: Any) -> _FakeVideoTensor:
        return _FakeVideoTensor(self._array[key])


def _stub_mimsave(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub `imageio.v2.mimsave` (slim has no imageio)."""
    captured: dict[str, Any] = {}

    def _mimsave(path: str, frames: Any, fps: Any = None, codec: Any = None) -> None:
        captured["path"] = path
        captured["frames"] = np.stack(list(frames))
        captured["fps"] = fps
        captured["codec"] = codec

    imageio_mod = types.ModuleType("imageio")
    v2_mod = types.ModuleType("imageio.v2")
    v2_mod.mimsave = _mimsave  # type: ignore[attr-defined]
    imageio_mod.v2 = v2_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", imageio_mod)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2_mod)
    return captured


def test_save_mp4_clips_overshoot_instead_of_wrapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _stub_mimsave(monkeypatch)
    values = np.full((3, 2, 2, 3), 0.5, dtype=np.float32)
    values[0, 0, 0, 0] = 1.01  # VAE overshoot: must become 255, not 1
    values[1, 0, 0, 1] = -0.01  # undershoot: must become 0, not 254
    values[2, 1, 1, 2] = 1.0
    video_ltxv._save_mp4([_FakeVideoTensor(values)], tmp_path / "tail.mp4", 24)
    frames = captured["frames"]
    assert frames.dtype == np.uint8
    assert captured["codec"] == "libx264"
    # permute(1,2,3,0) maps old[l,i,j,k] -> new[i,j,k,l].
    assert int(frames[0, 0, 0, 0]) == 255
    assert int(frames[0, 0, 1, 1]) == 0
    assert int(frames[1, 1, 2, 2]) == 255
    assert int(frames[0, 1, 1, 1]) == 127  # 0.5 * 255 truncated
    assert int(frames.min()) == 0
    assert int(frames.max()) == 255


class _RecordedMask:
    def __init__(self) -> None:
        self.destinations: list[Any] = []

    def to(self, device: Any) -> _RecordedMask:
        self.destinations.append(device)
        return self


class _FakeEmbeds:
    def to(self, dtype: Any) -> _FakeEmbeds:
        del dtype
        return self


class _FakeTextEncoder:
    def __init__(self, embeds: _FakeEmbeds) -> None:
        self._embeds = embeds

    def __call__(self, input_ids: Any) -> tuple[_FakeEmbeds, None]:
        del input_ids
        return (self._embeds, None)


def test_encode_moves_mask_to_session_device() -> None:
    """Issue 074: the mask must follow `self._device` (e.g. cuda:1)."""
    import contextlib

    mask = _RecordedMask()
    session_self = types.SimpleNamespace(
        _device="cuda:1",
        _tokenizer=lambda *args, **kwargs: types.SimpleNamespace(
            input_ids=object(), attention_mask=mask
        ),
        _text_encoder=_FakeTextEncoder(_FakeEmbeds()),
        _torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext, bfloat16="bfloat16"),
        _embed_cache={},
    )
    video_ltxv.LTXVSession._encode(session_self, "probe prompt")  # type: ignore[arg-type]
    assert mask.destinations == ["cuda:1"]
