"""Issue 134: causvid resume-anchor fallback must be loud + structured (CPU-only).

Worker-side pin: a missing/unreadable/shape-mismatched tail still renders
fresh (crash-free), but the segment result must carry ``resume_fallback``
``{reason, tail_path}`` and the anchor must only clear after the fallback
is recorded (never before validation). The supervisor-read half stays a
documented residual (supervisor.py is out of scope).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from voyage.workers import video_causvid
from voyage.workers.video_causvid import CausvidSession


class _FakeTensor:
    def __init__(self, array: Any) -> None:
        self._array = np.asarray(array)
        self.shape = self._array.shape

    def __getitem__(self, key: Any) -> _FakeTensor:
        return _FakeTensor(self._array[key])

    def permute(self, *order: int) -> _FakeTensor:
        return _FakeTensor(np.transpose(self._array, order))

    def transpose(self, dim0: int, dim1: int) -> _FakeTensor:
        return _FakeTensor(np.swapaxes(self._array, dim0, dim1))

    def unsqueeze(self, dim: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self._array, dim))

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> Any:
        return self._array

    @property
    def device(self) -> str:
        return "cpu"

    def to(self, *args: Any, **kwargs: Any) -> _FakeTensor:
        del args, kwargs
        return self

    def astype(self, dtype: Any) -> _FakeTensor:
        return _FakeTensor(self._array.astype(dtype))

    def __mul__(self, other: Any) -> _FakeTensor:
        return _FakeTensor(self._array * other)

    def __sub__(self, other: Any) -> _FakeTensor:
        return _FakeTensor(self._array - other)

    def __truediv__(self, other: Any) -> _FakeTensor:
        return _FakeTensor(self._array / other)

    def __rtruediv__(self, other: Any) -> _FakeTensor:
        return _FakeTensor(other / self._array)


class _FakeGenerator:
    def __init__(self, torch: _FakeTorch) -> None:
        self._torch = torch

    def manual_seed(self, seed: int) -> _FakeGenerator:
        self._torch.generator_seeds.append(seed)
        return self


class _FakeCuda:
    def is_available(self) -> bool:
        return False

    def empty_cache(self) -> None:
        return None


class _FakeModule:
    """Minimal ``torch.nn.Module``: attribute assignment + ``forward`` dispatch."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError

    def to(self, *args: Any, **kwargs: Any) -> _FakeModule:
        del args, kwargs
        return self


class _FakeNn:
    Module = _FakeModule


class _FakeTorch:
    nn = _FakeNn()

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.bfloat16 = "bfloat16"
        self.generator_seeds: list[int] = []
        self.randn_calls: list[dict[str, Any]] = []

    def manual_seed(self, seed: int) -> None:
        self.generator_seeds.append(seed)

    def Generator(self, device: str = "cpu") -> _FakeGenerator:
        del device
        return _FakeGenerator(self)

    def randn(self, *args: Any, **kwargs: Any) -> _FakeTensor:
        shape = list(args[0]) if args else [1, 1]
        self.randn_calls.append({"shape": shape, "kwargs": kwargs})
        return _FakeTensor(np.zeros(shape, dtype=np.float32))

    def cat(self, tensors: list[Any], dim: int = 0) -> _FakeTensor:
        arrays = [t._array if isinstance(t, _FakeTensor) else np.asarray(t) for t in tensors]
        return _FakeTensor(np.concatenate(arrays, axis=dim))

    def from_numpy(self, array: Any) -> _FakeTensor:
        return _FakeTensor(np.asarray(array))

    def set_grad_enabled(self, enabled: bool) -> None:
        del enabled


class _FakeVaeModel:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, tensor: Any, scale: Any) -> _FakeTensor:
        del scale
        self.encode_calls += 1
        array = tensor._array if isinstance(tensor, _FakeTensor) else np.asarray(tensor)
        latent_frames = (int(array.shape[2]) - 1) // 4 + 1
        return _FakeTensor(np.zeros((1, 16, latent_frames, 4, 4), dtype=np.float32))


class _FakeVae:
    def __init__(self) -> None:
        self.mean = _FakeTensor(np.zeros(16, dtype=np.float32))
        self.std = _FakeTensor(np.ones(16, dtype=np.float32))
        self.model = _FakeVaeModel()


class _FakeTextEncoder:
    def to(self, device: str) -> _FakeTextEncoder:
        del device
        return self

    def __call__(self, prompts: Any) -> dict[str, Any]:
        del prompts
        return {"prompt_embeds": _FakeTensor(np.zeros((1, 16, 4, 4), dtype=np.float32))}


class _FakePipeline:
    DECODED = 81

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.vae = _FakeVae()
        self.text_encoder = _FakeTextEncoder()

    def inference(
        self,
        noise: Any,
        text_prompts: list[str],
        return_latents: bool = False,
        start_latents: Any = None,
    ) -> tuple[_FakeTensor, _FakeTensor]:
        del noise, return_latents
        self.calls.append({"prompts": list(text_prompts), "start": start_latents})
        video = np.zeros((1, self.DECODED, 3, 4, 6), dtype=np.float32)
        for index in range(self.DECODED):
            video[0, index] = index / self.DECODED
        latents = np.zeros((1, 21, 16, 4, 4), dtype=np.float32)
        return _FakeTensor(video), _FakeTensor(latents)


def _install_imageio_stub(monkeypatch: pytest.MonkeyPatch, reads: dict[str, Any]) -> None:
    recorded_reads: list[str] = []

    def _mimsave(path: str, frames: Any, fps: int = 16, codec: str = "libx264") -> None:
        del codec, frames, fps
        Path(path).write_bytes(b"fake-mp4")

    def _mimread(path: str) -> Any:
        recorded_reads.append(path)
        if path not in reads:
            raise OSError(f"no media stub for {path}")
        value = reads[path]
        if isinstance(value, Exception):
            raise value
        return value

    imageio_mod = types.ModuleType("imageio")
    v2_mod = types.ModuleType("imageio.v2")
    v2_mod.mimsave = _mimsave  # type: ignore[attr-defined]
    v2_mod.mimread = _mimread  # type: ignore[attr-defined]
    imageio_mod.v2 = v2_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", imageio_mod)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2_mod)


def _test_session(monkeypatch: pytest.MonkeyPatch, reads: dict[str, Any]) -> CausvidSession:
    _install_imageio_stub(monkeypatch, reads)
    session = CausvidSession.__new__(CausvidSession)
    session._torch = _FakeTorch()
    session._device = "cuda:0"
    session._latent_shape = [1, 21, 16, 60, 104]
    session._overlap_frames = 3
    session._num_frame_per_block = 3
    session._config_sha256 = "0" * 64
    session._pipeline = _FakePipeline()
    session._start_latents = None
    session._pending_tail_path = None
    session._pending_overlap = 3
    session._last_prompt = None
    return session


def _generate(session: CausvidSession, tmp_path: Path, pending: str | None) -> dict[str, Any]:
    session._pending_tail_path = pending
    return session.generate_blocks(
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[False],
        output_path=tmp_path / "seg.mp4",
        segment_id="000007",
    )


def test_missing_tail_reports_structured_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session = _test_session(monkeypatch, {})
    missing = str(tmp_path / "gone.mp4")
    result = _generate(session, tmp_path, missing)
    assert result["fresh_rollouts"] == 1
    fallback = result["resume_fallback"]
    assert fallback["reason"] == "missing"
    assert fallback["tail_path"] == missing
    assert "missing" in capsys.readouterr().err
    # Anchor clears only after the fallback is recorded, never before validation.
    assert session._pending_tail_path is None


def test_unreadable_tail_reports_structured_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tail = tmp_path / "video_tail.mp4"
    tail.write_bytes(b"corrupt-bytes")
    session = _test_session(monkeypatch, {str(tail): OSError("undecodable")})
    result = _generate(session, tmp_path, str(tail))
    assert result["fresh_rollouts"] == 1
    fallback = result["resume_fallback"]
    assert fallback["reason"] == "unreadable"
    assert fallback["tail_path"] == str(tail)
    assert "unreadable" in capsys.readouterr().err


def test_shape_mismatch_tail_reports_structured_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tail = tmp_path / "video_tail.mp4"
    tail.write_bytes(b"tail-bytes")
    frames = np.zeros((9, 4, 6, 3), dtype=np.uint8)
    session = _test_session(monkeypatch, {str(tail): [frames[index] for index in range(9)]})
    real_encode = video_causvid._vae_encode_window
    encode_calls = 0

    def _short_encode_once(wrapper: Any, scaled: Any, dtype: Any) -> _FakeTensor:
        nonlocal encode_calls
        encode_calls += 1
        if encode_calls == 1:
            # Resume window only: re-encode short so the shape gate fires.
            # Later calls (the fresh chain's advance math) use the real path.
            return _FakeTensor(np.zeros((1, 2, 16, 4, 4), dtype=np.float32))
        return cast(_FakeTensor, real_encode(wrapper, scaled, dtype))

    monkeypatch.setattr(video_causvid, "_vae_encode_window", _short_encode_once)
    result = _generate(session, tmp_path, str(tail))
    assert result["fresh_rollouts"] == 1
    fallback = result["resume_fallback"]
    assert fallback["reason"] == "shape_mismatch"
    assert fallback["tail_path"] == str(tail)
    assert "expects 3" in capsys.readouterr().err


def test_happy_path_carries_no_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _test_session(monkeypatch, {})
    result = _generate(session, tmp_path, None)
    assert result["resume_fallback"] is None
    assert result["fresh_rollouts"] == 1
