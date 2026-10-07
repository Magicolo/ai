"""Issue 257: RIFE pair batching with OOM-halving (mirrors FILM).

`interpolate_rife_mids` ran strictly serially (one pair per encode, one
`.cpu()` per mid, no `pair_batch`, OOM propagating). It now stacks
`pair_batch` pairs per encode+forward set with `_run_rife_window`
halving on OOM — default 1, bit-exact with the old loop.

Validation tests run torch-free (slim gates image); structure tests need
torch (run in voyage-video) but no weights (`_get_prepared_model` is
stubbed with a recording lerp model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.workers import augment_worker


def _torch() -> Any:
    """Real torch (run in voyage-video, not the slim gates image)."""
    return pytest.importorskip("torch", reason="RIFE batching needs torch")


def test_rife_rejects_bad_pair_batch() -> None:
    """Pair batch must be a positive int — validated before any torch import."""
    with pytest.raises(TypeError, match="must be an int"):
        augment_worker.interpolate_rife_mids(
            ["frame-a", "frame-b"], "/models/rife.safetensors", moments=[0.5], pair_batch=True
        )
    with pytest.raises(ValueError, match="must be positive"):
        augment_worker.interpolate_rife_mids(
            ["frame-a", "frame-b"], "/models/rife.safetensors", moments=[0.5], pair_batch=0
        )


class _LerpRife:
    """Recording RIFE double: elementwise lerp (batch-size independent)."""

    def __init__(self) -> None:
        self.encode_batches: list[int] = []
        self.forward_batches: list[int] = []
        self.fail_above: int | None = None

    def encode(self, stacked: Any) -> Any:
        self.encode_batches.append(int(stacked.shape[0]))
        return stacked

    def __call__(self, first: Any, second: Any, timestep: Any = 0.5, cache: Any = None) -> Any:
        del cache
        self.forward_batches.append(int(first.shape[0]))
        if self.fail_above is not None and int(first.shape[0]) > self.fail_above:
            raise RuntimeError("CUDA out of memory: simulated pressure")
        moment = float(timestep)
        return first * moment + second * (1.0 - moment)


def _ramp_frames(torch: Any, count: int) -> list[Any]:
    """Constant-value frames (pair/moment identity readable from values)."""
    return [torch.full((3, 64, 64), index / 4.0) for index in range(count)]


def _run_rife(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    torch: Any,
    model: _LerpRife,
    frames: list[Any],
    **kwargs: Any,
) -> list[Any]:
    weights = tmp_path / "rife_v4.25_heavy.safetensors"
    weights.write_bytes(b"\x00" * 64)
    monkeypatch.setattr(
        augment_worker,
        "_get_prepared_model",
        lambda cache, key, device, loader: (model, torch.device("cpu"), torch.float32),
    )
    return augment_worker.interpolate_rife_mids(frames, weights, device="cpu", **kwargs)


def test_rife_pair_batch_stacks_encodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pair_batch=2 encodes two pairs per window; default stays one at a time.

    Each window encodes twice (img0 + img1), so two windows record four
    batch-2 encodes and the serial default eight batch-1 encodes.
    """
    torch = _torch()
    frames = _ramp_frames(torch, 5)
    batched_model = _LerpRife()
    _run_rife(monkeypatch, tmp_path, torch, batched_model, frames, moments=[0.5], pair_batch=2)
    assert batched_model.encode_batches == [2, 2, 2, 2]
    assert batched_model.forward_batches == [2, 2]
    serial_model = _LerpRife()
    _run_rife(monkeypatch, tmp_path, torch, serial_model, frames, moments=[0.5])
    assert serial_model.encode_batches == [1] * 8
    assert serial_model.forward_batches == [1] * 4


def test_rife_batches_match_serial_exactly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Batched output equals the serial loop bit-exactly (pair-major)."""
    torch = _torch()
    frames = _ramp_frames(torch, 5)
    moments = [0.25, 0.5, 0.75]
    serial = _run_rife(monkeypatch, tmp_path, torch, _LerpRife(), frames, moments=moments)
    batched = _run_rife(
        monkeypatch, tmp_path, torch, _LerpRife(), frames, moments=moments, pair_batch=4
    )
    assert len(serial) == len(batched) == 12
    for got, want in zip(batched, serial, strict=True):
        assert torch.equal(got, want)
    for pair_index in range(4):
        for moment_index, moment in enumerate(moments):
            expected = pair_index / 4.0 * moment + (pair_index + 1) / 4.0 * (1.0 - moment)
            assert float(batched[pair_index * 3 + moment_index].mean()) == pytest.approx(expected)


def test_rife_oom_halves_to_singles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A window that OOMs above one pair still completes via halving.

    The batch-4 attempt (plus its halved retries) records larger batches
    first; the completed tail is all singles, and the healed output
    equals the serial run exactly.
    """
    torch = _torch()
    frames = _ramp_frames(torch, 5)
    pressured = _LerpRife()
    pressured.fail_above = 1
    healed = _run_rife(monkeypatch, tmp_path, torch, pressured, frames, moments=[0.5], pair_batch=4)
    # Halving multisets: the batch-4 attempt, three batch-2 retries, then
    # four batch-1 leaf completions (encodes come in img0 + img1 pairs).
    assert sorted(pressured.encode_batches) == [1] * 8 + [2] * 4 + [4] * 2
    assert pressured.encode_batches[-4:] == [1] * 4
    assert sorted(pressured.forward_batches) == [1] * 4 + [2] * 2 + [4]
    assert pressured.forward_batches[-2:] == [1] * 2
    serial = _run_rife(monkeypatch, tmp_path, torch, _LerpRife(), frames, moments=[0.5])
    assert len(healed) == len(serial) == 4
    for got, want in zip(healed, serial, strict=True):
        assert torch.equal(got, want)


def test_rife_on_pair_fires_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """on_pair fires per finished pair in index order across windows."""
    torch = _torch()
    frames = _ramp_frames(torch, 5)
    fired: list[tuple[int, int]] = []
    _run_rife(
        monkeypatch,
        tmp_path,
        torch,
        _LerpRife(),
        frames,
        moments=[0.5],
        pair_batch=2,
        on_pair=lambda index, total: fired.append((index, total)),
    )
    assert fired == [(0, 4), (1, 4), (2, 4), (3, 4)]
