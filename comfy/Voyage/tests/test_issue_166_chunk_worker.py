"""Issue 166 remainder: chunk-scale model-pass wiring (torch-free fallback + GPU legs).

`resolve_augment_weights` maps a models dir to loader-ready paths; the
chunk worker passes non-None legs to `augment_worker.upscale_frames` /
`interpolate_pair` with `device=chunk.device` and keeps the ffmpeg encode
for None legs. `use_model_pass=False` (default) returns the input frames
unchanged — ffmpeg stays the default, so the knob is opt-in.

Fallback legs run torch-free in the slim gates image; provisioned legs
skip loudly without torch+weights (run them in `voyage-video` with the
models volume mounted, `--gpus all` for the CUDA proof).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from voyage.augment import (
    AugmentChunk,
    AugmentWeights,
    augment_plan,
    run_augment_chunks,
)


def _torch_available() -> bool:
    """Whether real torch imports (slim gates image: no — provisioned legs skip)."""
    return importlib.util.find_spec("torch") is not None


def test_enhance_frames_import_fails_red() -> None:
    """TDD red anchor: the chunk-scale seam does not exist yet."""
    from voyage import augment as augment_module

    assert hasattr(augment_module, "enhance_frames")
    assert hasattr(augment_module, "make_enhance_chunk_worker")
    assert hasattr(augment_module, "run_model_augment_chunks")


def test_opt_out_returns_input_unchanged_without_torch() -> None:
    """Default knob keeps ffmpeg bytes: input frames pass through untouched."""
    from voyage.augment import enhance_frames

    source = ["frame-a", "frame-b", "frame-c"]
    weights = AugmentWeights(film=None, realesrgan=None)
    assert enhance_frames(source, weights, device="cuda:0") == source
    assert enhance_frames(source, weights, device="cuda:0") is not source


def test_opt_in_without_weights_skips_without_torch() -> None:
    """Opt-in with absent legs still skips (no torch import, same frames)."""
    from voyage.augment import enhance_frames

    source = ["frame-a", "frame-b"]
    weights = AugmentWeights(film=None, realesrgan=None)
    assert enhance_frames(source, weights, device="cuda:0", use_model_pass=True) == source


def test_enhance_frames_rejects_bad_inputs() -> None:
    """Bad knob/frame/factor inputs fail loud before any model work."""
    from voyage.augment import enhance_frames

    weights = AugmentWeights(film=None, realesrgan=None)
    with pytest.raises(ValueError, match="at least one frame"):
        enhance_frames([], weights, device="cuda:0")
    with pytest.raises(TypeError, match="device"):
        enhance_frames(["frame-a"], weights, device=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="use_model_pass"):
        enhance_frames(["frame-a"], weights, device="cpu", use_model_pass="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="upscale_factor"):
        enhance_frames(["frame-a"], weights, device="cpu", upscale_factor=3)
    with pytest.raises(ValueError, match="multiplier"):
        enhance_frames(["frame-a"], weights, device="cpu", multiplier=0)


def test_chunk_worker_forwards_chunk_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model legs receive `device=chunk.device` (the SFX pairing contract)."""
    from voyage import augment as augment_module
    from voyage.workers import augment_worker

    seen: dict[str, Any] = {}

    def _fake_upscale(
        frames: list[Any],
        weights: Path | str,
        *,
        scale: int = 2,
        device: str = "cpu",
        **kwargs: Any,
    ) -> list[Any]:
        seen["upscale_device"] = device
        seen["upscale_scale"] = scale
        return list(frames)

    def _fake_interpolate(
        before: Any,
        after: Any,
        weights: Path | str,
        *,
        moment: float = 0.5,
        device: str = "cpu",
        **kwargs: Any,
    ) -> Any:
        seen.setdefault("interp_devices", []).append(device)
        seen.setdefault("interp_moments", []).append(moment)
        return before

    monkeypatch.setattr(augment_worker, "upscale_frames", _fake_upscale)
    monkeypatch.setattr(augment_worker, "interpolate_pair", _fake_interpolate)
    weights = AugmentWeights(
        film=Path("/models/frame_interpolation/film_net_fp16.safetensors"),
        realesrgan=Path("/models/realesrgan/RealESRGAN_x4plus_anime_6B.pth"),
    )
    worker = augment_module.make_enhance_chunk_worker(
        {0: ["frame-a", "frame-b"]},
        weights,
        use_model_pass=True,
    )
    chunk = AugmentChunk(
        index=0, start_frame=0, source_frames=2, expected_frames=5, device="cuda:1"
    )
    outcome = worker(chunk, chunk.device)
    assert outcome == ["frame-a", "frame-a", "frame-a", "frame-a", "frame-b"]
    assert seen["upscale_device"] == "cuda:1"
    assert seen["interp_devices"] == ["cuda:1", "cuda:1", "cuda:1"]
    assert seen["interp_moments"] == [0.25, 0.5, 0.75]


def test_chunk_worker_missing_chunk_index_fails_loud() -> None:
    """A worker without source frames for its chunk index fails loud."""
    from voyage import augment as augment_module

    weights = AugmentWeights(film=None, realesrgan=None)
    worker = augment_module.make_enhance_chunk_worker({}, weights, use_model_pass=False)
    chunk = AugmentChunk(
        index=3, start_frame=96, source_frames=4, expected_frames=13, device="cuda:0"
    )
    with pytest.raises(KeyError, match="chunk 3"):
        worker(chunk, chunk.device)


def test_run_model_chunks_preserves_order_without_model() -> None:
    """Opt-out run threads through `run_augment_chunks` with identical order."""
    from voyage import augment as augment_module

    plan = augment_plan(5, chunk=2, multiplier=2, devices=("cuda:0",))
    source_frames = {
        chunk.index: [
            f"chunk-{chunk.index}-frame-{offset}" for offset in range(chunk.source_frames)
        ]
        for chunk in plan
    }
    weights = AugmentWeights(film=None, realesrgan=None)
    outcomes = augment_module.run_model_augment_chunks(
        plan, weights, source_frames, use_model_pass=False
    )
    assert outcomes == [source_frames[chunk.index] for chunk in plan]
    assert run_augment_chunks(plan, lambda chunk, device: chunk.index) == [0, 1, 2]


def _find_provisioned_weight(relative_path: str) -> Path | None:
    """First existing provisioned copy of `relative_path`, or None (skip signal)."""
    candidates = [
        os.environ.get("VOYAGE_MODELS", ""),
        "/models",
        str(Path.home() / ".cache" / "voyage-models"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        found = Path(candidate) / relative_path
        try:
            if found.is_file() and found.stat().st_size > 0:
                return found
        except OSError:
            continue
    return None


def test_provisioned_model_pass_upscales_and_interpolates_on_cpu() -> None:
    """Provisioned weights upscale then interpolate a synthetic pair (CPU fp32)."""
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    if importlib.util.find_spec("safetensors") is None:
        pytest.skip("needs safetensors (run in voyage-video)")
    from voyage.augment import enhance_frames, resolve_augment_weights
    from voyage.registry_film import FILM_REPO_PATH
    from voyage.registry_realesrgan import REALESRGAN_ANIME_FILE, REALESRGAN_SUBDIR

    film = _find_provisioned_weight(FILM_REPO_PATH)
    realesrgan = _find_provisioned_weight(f"{REALESRGAN_SUBDIR}/{REALESRGAN_ANIME_FILE}")
    if film is None or realesrgan is None:
        pytest.skip("needs provisioned film + realesrgan weights (models volume absent)")
    assert film is not None
    assert realesrgan is not None
    import torch

    from voyage.workers import augment_worker

    augment_worker.evict_augment_models()
    resolved = resolve_augment_weights(film.parent.parent)
    assert resolved.film == film
    assert resolved.realesrgan == realesrgan
    rows = torch.linspace(0.0, 1.0, 16).unsqueeze(1).expand(16, 16)
    before = torch.stack([rows, rows, rows])
    after = torch.stack([1.0 - rows, 1.0 - rows, 1.0 - rows])
    enhanced = enhance_frames(
        [before, after],
        resolved,
        device="cpu",
        upscale_factor=1,
        multiplier=2,
        use_model_pass=True,
    )
    assert len(enhanced) == 3
    assert tuple(enhanced[0].shape) == (3, 16, 16)
    assert tuple(enhanced[1].shape) == (3, 16, 16)
    assert tuple(enhanced[2].shape) == (3, 16, 16)
    for frame in enhanced:
        assert bool(torch.isfinite(frame).all())
