"""LTXV stage telemetry: `perf_counter` timings in `generate_blocks` (DESIGN §22.5).

Stage A (additive-only): every `generate_blocks` result carries `stage_ms`
with `encode_ms` (TE `_encode`), `denoise_ms` (per-block denoise loop),
`save_ms` (`_save_mp4`), `tape_ms` (recovery-tape write) — floats >= 0,
always present (no gating flag; `perf_counter` only, negligible overhead).

Key-path note: `VideoBackendAdapter.generate_segment` (voyage/backends.py)
normalizes the worker result into `VideoSegmentResult`, which has no
`stage_ms` field — extras are stripped there (another track owns that
file, so it is intentionally untouched). The supervisor-visible path is
the raw worker return: `handle_generate_blocks(...)["video"]["stage_ms"]`.

CPU-only: sessions are SimpleNamespace doubles, media I/O is stubbed —
no GPU, no model downloads.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voyage.workers import video_ltxv

EXPECTED_STAGE_KEYS = frozenset({"encode_ms", "denoise_ms", "save_ms", "tape_ms"})
"""Stage A key contract: every `generate_blocks` result carries these."""


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


def _fake_session(frame_counts: list[int], resident_tail: str | None = None) -> Any:
    """Fake LTXV session: scripted block lengths, no timing split reported."""

    def _generate_block(
        prompt: str,
        seed: int,
        width: int,
        height: int,
        frames: int,
        fps: int,
        conditioning: str | None,
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


def _stub_save_mp4(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, int]]:
    """Replace `_save_mp4` with a stub that writes marker bytes."""
    saved: list[tuple[Path, int]] = []

    def _stub(images: Any, path: Path, fps: int) -> None:
        del images
        saved.append((path, fps))
        path.write_bytes(b"stub-video")

    monkeypatch.setattr(video_ltxv, "_save_mp4", _stub)
    return saved


def _assert_stage_ms_shape(stage_ms: Any) -> None:
    """Pin the Stage A contract: four keys, floats, non-negative."""
    assert set(stage_ms) == EXPECTED_STAGE_KEYS
    for value in stage_ms.values():
        assert isinstance(value, float)
        assert value >= 0.0


def test_generate_blocks_reports_stage_ms_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh single block: result carries the four stage timings."""
    _stub_save_mp4(monkeypatch)
    session = _fake_session([121])
    result = video_ltxv.LTXVSession.generate_blocks(
        session,
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[True],
        output_path=tmp_path / "seg.mp4",
        width=768,
        height=512,
        fps=24,
    )
    assert result["frames"] == 121
    _assert_stage_ms_shape(result["stage_ms"])


def test_generate_blocks_reports_stage_ms_conditioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conditioned block (prefix drop): accounting intact, timings present."""
    _stub_save_mp4(monkeypatch)
    resident = tmp_path / "video_tail.mp4"
    resident.write_bytes(b"resident-tail")
    session = _fake_session([121], resident_tail=str(resident))
    result = video_ltxv.LTXVSession.generate_blocks(
        session,
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[False],
        output_path=tmp_path / "seg.mp4",
        width=768,
        height=512,
        fps=24,
    )
    assert result["novel_frames"] == 96
    assert result["conditioning_frames"] == 25
    _assert_stage_ms_shape(result["stage_ms"])


def test_save_and_tape_stages_use_live_perf_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The save/tape timings bracket real calls (not hardcoded zeros).

    Counts `perf_counter` invocations through a delegating wrapper (global
    patch, auto-reverted): a fresh block performs two `_save_mp4` writes
    (chain tail + segment video) plus one tape write — six brackets
    minimum. Proves the instrumentation is live without timing flakiness.
    """
    _stub_save_mp4(monkeypatch)
    call_count = 0
    real_perf_counter = time.perf_counter

    def _counting_perf_counter() -> float:
        nonlocal call_count
        call_count += 1
        return real_perf_counter()

    monkeypatch.setattr(time, "perf_counter", _counting_perf_counter)
    session = _fake_session([121])
    result = video_ltxv.LTXVSession.generate_blocks(
        session,
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[True],
        output_path=tmp_path / "seg.mp4",
        width=768,
        height=512,
        fps=24,
    )
    _assert_stage_ms_shape(result["stage_ms"])
    assert call_count >= 6


def test_handle_generate_blocks_exposes_stage_ms_under_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surviving key path: `response["video"]["stage_ms"]` (adapter strips extras)."""
    output = tmp_path / "seg.mp4"
    expected_stage_ms = {
        "encode_ms": 25.0,
        "denoise_ms": 1200.0,
        "save_ms": 40.0,
        "tape_ms": 3.0,
    }

    class _StubSession:
        def generate_blocks(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return {
                "frames": 121,
                "conditioning_tail_path": str(tmp_path / "video_tail.mp4"),
                "recovery_path": str(tmp_path / "recovery.pt"),
                "stage_ms": dict(expected_stage_ms),
            }

    monkeypatch.setattr(video_ltxv, "_SESSION", _StubSession())
    response = video_ltxv.handle_generate_blocks(
        {
            "segment_id": "000007",
            "prompt": "amber dunes",
            "seed": 7,
            "output_path": str(output),
            "fps": 24,
        }
    )
    assert response["video"]["stage_ms"] == expected_stage_ms


def _stub_ltx_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `ltx_video.inference` (fresh path needs padding + conditioning)."""
    parent = types.ModuleType("ltx_video")
    monkeypatch.setitem(sys.modules, "ltx_video", parent)
    inference = types.ModuleType("ltx_video.inference")
    inference.calculate_padding = lambda *args: (0, 0, 0, 0)  # type: ignore[attr-defined]
    inference.prepare_conditioning = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ltx_video.inference", inference)


def test_generate_block_reports_encode_denoise_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real `_generate_block`: encode split covers both TE calls (negative + prompt)."""
    _stub_ltx_inference(monkeypatch)
    sentinel = object()
    encode_calls: list[str] = []

    class _Generator:
        def manual_seed(self, seed: int) -> _Generator:
            del seed
            return self

    fake_torch = types.SimpleNamespace(Generator=lambda device: _Generator())

    def _encode(text: str) -> tuple[str, str]:
        encode_calls.append(text)
        return (f"embeds:{text}", f"mask:{text}")

    session: Any = types.SimpleNamespace(
        _torch=fake_torch,
        _device="cuda:0",
        _negative=None,
        _encode=_encode,
        _run_multiscale=lambda *args, **kwargs: sentinel,
        _fp8_fallback=False,
    )
    assert (
        video_ltxv.LTXVSession._generate_block(session, "amber dunes", 7, 768, 512, 121, 24, None)
        is sentinel
    )
    assert encode_calls == [video_ltxv.NEGATIVE_PROMPT, "amber dunes"]
    assert isinstance(session._last_block_encode_ms, float)
    assert session._last_block_encode_ms >= 0.0
    assert isinstance(session._last_block_denoise_ms, float)
    assert session._last_block_denoise_ms >= 0.0
