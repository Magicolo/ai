"""Issue 212: direct upscale path finishes inline (no chunk-sized device hold).

`upscale_frames` used to accumulate every frame's x4-native device tensor
in a `natives` list outside the batch-halving loop, then post-process
afterwards — a 32-frame chunk holds ~2.6 GiB of natives at 1216x704
outside any OOM relief. The direct path now finishes each frame inline
like the tiled path (peak: one live native per frame).

Needs torch (run in voyage-video, not the slim gates image); weights are
throwaway tmp files because `_get_prepared_model` is stubbed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.workers import augment_worker


def _stub_prepared(monkeypatch: pytest.MonkeyPatch, model: Any, torch: Any) -> None:
    """Route `_get_prepared_model` to a caller-owned model on CPU."""
    monkeypatch.setattr(
        augment_worker,
        "_get_prepared_model",
        lambda cache, key, device, loader: (model, torch.device("cpu"), torch.float32),
    )


def _torch() -> Any:
    """Real torch (run in voyage-video, not the slim gates image)."""
    return pytest.importorskip("torch", reason="upscale path needs torch")


def test_direct_path_finishes_inline_per_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run/finish events strictly alternate — one live native at a time."""
    torch = _torch()
    events: list[str] = []
    real_run = augment_worker._run_frame_batches
    real_finish = augment_worker._finish_upscaled

    def spy_run(*args: Any, **kwargs: Any) -> list[Any]:
        events.append("run")
        return real_run(*args, **kwargs)

    def spy_finish(*args: Any, **kwargs: Any) -> Any:
        events.append("finish")
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(augment_worker, "_run_frame_batches", spy_run)
    monkeypatch.setattr(augment_worker, "_finish_upscaled", spy_finish)
    _stub_prepared(monkeypatch, lambda batch: batch, torch)
    weights = tmp_path / "realesr-animevideov3.pth"
    weights.write_bytes(b"\x00" * 64)
    frames = [torch.zeros(3, 8, 8) for _ in range(6)]
    outputs = augment_worker.upscale_frames(frames, weights, scale=2, device="cpu")
    assert events == ["run", "finish"] * 6
    assert len(outputs) == 6
    assert all(tuple(frame.shape) == (3, 4, 4) for frame in outputs)


def test_direct_path_outputs_match_finish_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inline finishing yields clamped [0, 1] CPU float frames at target scale."""
    torch = _torch()
    _stub_prepared(monkeypatch, lambda batch: batch, torch)
    weights = tmp_path / "realesr-animevideov3.pth"
    weights.write_bytes(b"\x00" * 64)
    frames = [torch.full((3, 8, 8), 2.0), torch.full((3, 8, 8), -1.0)]
    outputs = augment_worker.upscale_frames(frames, weights, scale=1, device="cpu")
    assert len(outputs) == 2
    for output in outputs:
        assert tuple(output.shape) == (3, 2, 2)
        assert bool(((output >= 0.0) & (output <= 1.0)).all())
