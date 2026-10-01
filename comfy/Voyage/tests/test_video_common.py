"""Shared video-worker scaffolding (issue 019).

CPU-only: imageio is stubbed (the slim gates image has none), torch/GPU
are never touched, and the benchmark harness runs on fake clocks.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voyage.workers import video_causvid, video_common, video_ltxv
from voyage.workers.video_common import (
    BenchmarkHarnessOutcome,
    clip_array_to_uint8,
    run_benchmark_harness,
    save_mp4,
    standard_serve_map,
    write_tape_atomic,
)


def test_write_tape_atomic_roundtrip(tmp_path: Path) -> None:
    tape_path = tmp_path / "recovery.pt"
    tape = {"backend": "ltxv", "width": 768, "nested": {"alpha": 1}}
    assert write_tape_atomic(tape_path, tape) == tape_path
    assert tape_path.read_text(encoding="utf-8") == (
        json.dumps(tape, indent=2, sort_keys=True) + "\n"
    )
    assert json.loads(tape_path.read_text(encoding="utf-8")) == tape
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_tape_atomic_overwrites(tmp_path: Path) -> None:
    tape_path = tmp_path / "recovery.pt"
    tape_path.write_text("stale", encoding="utf-8")
    write_tape_atomic(tape_path, {"backend": "causvid"})
    assert json.loads(tape_path.read_text(encoding="utf-8")) == {"backend": "causvid"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_clip_array_to_uint8_holds_black_and_white() -> None:
    clipped = clip_array_to_uint8(np.array([[[0.0, 0.5, 1.0, 1.01, 2.0, -0.25]]]))
    assert clipped.dtype == np.uint8
    assert list(clipped[0, 0]) == [0, 127, 255, 255, 255, 0]


def _install_imageio_stub(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub imageio.v2.mimsave with a recording fake (slim image has none)."""
    calls: list[dict[str, Any]] = []

    def fake_mimsave(path: str, frames: Any, **kwargs: Any) -> None:
        calls.append({"path": path, "frames": frames, **kwargs})

    package = types.ModuleType("imageio")
    submodule = types.ModuleType("imageio.v2")
    submodule.mimsave = fake_mimsave  # type: ignore[attr-defined]
    package.v2 = submodule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", package)
    monkeypatch.setitem(sys.modules, "imageio.v2", submodule)
    return calls


def test_save_mp4_delegates_to_mimsave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_imageio_stub(monkeypatch)
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    dest = tmp_path / "segment.mp4"
    save_mp4(frames, dest, 24)
    assert len(calls) == 1
    assert calls[0]["path"] == str(dest)
    assert calls[0]["frames"] == frames
    assert calls[0]["fps"] == 24
    assert calls[0]["codec"] == "libx264"


def test_save_mp4_rejects_nonpositive_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_imageio_stub(monkeypatch)
    with pytest.raises(ValueError, match="must be positive"):
        save_mp4([np.zeros((2, 2, 3), dtype=np.uint8)], tmp_path / "x.mp4", 0)
    assert calls == []


def test_run_benchmark_harness_measures_only_measured() -> None:
    seen: list[tuple[str, bool]] = []
    parents: list[str] = []
    clock_ticks = iter([0.0, 1.0, 10.0, 12.0, 20.0, 25.0])
    peak_reads = iter([5.0, 6.0, 7.0])
    resets: list[str] = []

    def probe(output_path: Path, measured: bool) -> None:
        seen.append((output_path.name, measured))
        parents.append(output_path.parent.name)
        output_path.write_bytes(b"probe")

    outcome = run_benchmark_harness(
        1,
        2,
        "voyage-test-bench-",
        probe,
        clock=lambda: next(clock_ticks),
        reset_peak_memory=lambda: resets.append("reset"),
        read_peak_gib=lambda: next(peak_reads),
    )
    assert isinstance(outcome, BenchmarkHarnessOutcome)
    assert [name for name, _ in seen] == ["b0.mp4", "b1.mp4", "b2.mp4"]
    assert [measured for _, measured in seen] == [False, True, True]
    assert all(parent.startswith("voyage-test-bench-") for parent in parents)
    assert outcome.wall_seconds == [2.0, 5.0]
    assert outcome.peak_gib == [6.0, 7.0]
    assert resets == ["reset", "reset", "reset"]


def test_run_benchmark_harness_rejects_bad_counts() -> None:
    def probe(output_path: Path, measured: bool) -> None:
        del output_path, measured

    with pytest.raises(ValueError, match="warmup"):
        run_benchmark_harness(-1, 3, "voyage-test-bench-", probe)
    with pytest.raises(ValueError, match="warmup"):
        run_benchmark_harness(1, 0, "voyage-test-bench-", probe)


def test_standard_serve_map_keys_and_checkpoint() -> None:
    def stub(payload: dict[str, Any]) -> dict[str, Any]:
        return dict(payload)

    handlers = standard_serve_map(
        "ltxv",
        handle_init=stub,
        handle_health=stub,
        handle_generate_blocks=stub,
        handle_benchmark=stub,
        handle_evict_gpu=stub,
        handle_rebuild=stub,
        handle_resume=stub,
    )
    assert sorted(handlers) == [
        "benchmark",
        "checkpoint",
        "evict_gpu",
        "generate_blocks",
        "health",
        "init",
        "rebuild",
        "resume",
        "shutdown",
    ]
    assert handlers["checkpoint"]({"segment_id": "000007"}) == {"checkpoint_id": "ltxv-000007"}
    assert handlers["checkpoint"]({}) == {"checkpoint_id": "ltxv-none"}
    assert handlers["shutdown"]({}) == {"stopped": True}
    assert handlers["init"] is stub


def test_worker_tail_tape_constants_match_common() -> None:
    assert video_ltxv.TAIL_FILENAME == video_common.TAIL_FILENAME == "video_tail.mp4"
    assert video_ltxv.TAPE_FILENAME == video_common.TAPE_FILENAME == "recovery.pt"
    assert video_causvid.TAIL_FILENAME == video_common.TAIL_FILENAME
    assert video_causvid.TAPE_FILENAME == video_common.TAPE_FILENAME


class _FakeTorchTensor:
    """Minimal torch-tensor stand-in: permute/float/cpu/numpy chain."""

    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        self._array = array

    def dim(self) -> int:
        return int(self._array.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dimension) for dimension in self._array.shape)

    def permute(self, *axes: int) -> _FakeTorchTensor:
        return _FakeTorchTensor(np.transpose(self._array, axes))

    def float(self) -> _FakeTorchTensor:
        return self

    def cpu(self) -> _FakeTorchTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self._array


def test_ltxv_save_mp4_bridge_clips_and_drops_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ltxv tensor bridge converts layout, then shares the writer."""
    saved: list[dict[str, Any]] = []

    def fake_save(frames: list[Any], output_path: Path, frames_per_second: int) -> None:
        saved.append({"frames": frames, "output_path": output_path, "fps": frames_per_second})

    monkeypatch.setattr(video_common, "save_mp4", fake_save)
    tensor = _FakeTorchTensor(np.full((4, 2, 2, 3), 1.01, dtype=np.float64))
    video_ltxv._save_mp4([tensor], Path("seg.mp4"), 24)
    assert len(saved) == 1
    frames = saved[0]["frames"]
    assert len(frames) == 2
    # (C=4, T=2, H=2, W=3) → permute to (T, H, W, C); the 4-channel tail
    # reads as alpha and is dropped, leaving (H, W, C=3) per frame.
    assert frames[0].shape == (2, 3, 3)
    assert frames[0].dtype == np.uint8
    assert int(frames[0].max()) == 255
    assert int(frames[0].min()) == 255
    assert saved[0]["fps"] == 24
