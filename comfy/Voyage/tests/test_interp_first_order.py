"""Interp-first order on small GPUs (DESIGN §140 GPU defaults).

Live incident (this change): with the upscale leg tiled, the pass
reaches FILM at 2432x1408 per pair — ~5.9 GiB, OOM on the 6 GB 2060
where batch-halving again bottoms out at one pair. Chunks whose device
reports under 8 GiB free therefore interpolate at 1x first and upscale
after (the established interp-then-upscale pipeline); capable devices
keep the validated upscale-first recipe byte-for-byte.

`interp_first_for_small_device` answers False torch-free (non-CUDA,
unknown, probe failure); the order switch itself is covered with stubbed
legs so gates prove it without torch or weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _dummy_legs(tmp_path: Path) -> Any:
    from voyage.augment import AugmentWeights

    film = tmp_path / "film.safetensors"
    esrgan = tmp_path / "esrgan.pth"
    film.write_bytes(b"\x00" * 64)
    esrgan.write_bytes(b"\x00" * 64)
    return AugmentWeights(film=film, realesrgan=esrgan)


def test_small_device_probe_fails_safe_without_cuda() -> None:
    """Non-CUDA and unparsable devices keep the validated order (False)."""
    from voyage.workers.augment_worker import interp_first_for_small_device

    assert interp_first_for_small_device("cpu") is False
    assert interp_first_for_small_device("cuda:x") is False


def test_enhance_order_switches_on_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stubbed legs record upscale/interp call order under both probe answers."""
    import voyage.workers.augment_worker as worker
    from voyage.augment import enhance_frames

    calls: list[str] = []

    def _fake_upscale(frames: list[Any], weights: object, **kwargs: Any) -> list[Any]:
        calls.append("upscale")
        return list(frames)

    def _fake_mids(frames: list[Any], weights: object, **kwargs: Any) -> list[Any]:
        moments = kwargs.get("moments", [0.5])
        calls.extend(["interp"] * len(list(moments)))
        pairs = max(len(frames) - 1, 0)
        return [frames[0]] * (pairs * len(list(moments)))

    monkeypatch.setattr(worker, "upscale_frames", _fake_upscale)
    monkeypatch.setattr(worker, "interpolate_mids", _fake_mids)
    legs = _dummy_legs(tmp_path)
    frames = ["frame-a", "frame-b"]

    monkeypatch.setattr(worker, "interp_first_for_small_device", lambda _device: False)
    enhance_frames(frames, legs, device="cpu", interp_backend="film")
    # Multiplier 4 → 3 mids per pair, all after the upscale.
    assert calls == ["upscale", "interp", "interp", "interp"]

    calls.clear()
    monkeypatch.setattr(worker, "interp_first_for_small_device", lambda _device: True)
    out = enhance_frames(frames, legs, device="cpu", interp_backend="film")
    assert calls == ["interp", "interp", "interp", "upscale"]
    # Interp-first blends before upscaling: 2 frames x4 in, 5 out.
    assert len(out) == 5
