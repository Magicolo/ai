"""Issue 166: FileSpec filename round-trips through its worker loader.

Contract: every `FileSpec` filename under the `film` /
`realesrgan-anime` registry rows must resolve to a worker loader in
`voyage.workers.augment_worker`, and the provisioned weights must load
strict plus run synthetic tensors end to end — the ESRGAN anime-video-XS
`.pth` upscales, the FILM `.safetensors` strict-loads its 82-key
extract/fuse/predict_flow state and interpolates a synthetic pair.

Runs torch-free in the slim gates image (mapping + stub-rejection legs);
the provisioned legs skip loudly unless `torch` imports AND the weight
file exists under `$VOYAGE_MODELS`, `/models`, or
`~/.cache/voyage-models` — run them in `voyage-video` with the models
volume mounted. CPU-only throughout (device="cpu").
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from voyage import model_registry
from voyage.errors import ModelCompatibilityError
from voyage.model_registry import RequiredFile
from voyage.workers import augment_worker

_AUGMENT_SPECS = ("film", "realesrgan-anime")
"""Registry rows whose FileSpec filenames this contract covers."""


def _loader_for_filename(filename: str) -> Callable[[Path], Any]:
    """Worker loader owning `filename` (suffix decides, mirroring the registry)."""
    if filename.endswith(".pth"):
        return augment_worker._load_esrgan_net
    if filename.endswith(".safetensors"):
        return augment_worker._load_film_net
    raise AssertionError(f"no augment loader owns filename {filename!r}")


def _spec_filenames(spec_name: str) -> list[str]:
    """Filenames (not paths) of every FileSpec in the registry row."""
    return [file_spec.filename for file_spec in model_registry.MODEL_SPECS[spec_name].files]


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


def _torch_available() -> bool:
    """Whether real torch imports (slim gates image: no — provisioned legs skip)."""
    return importlib.util.find_spec("torch") is not None


def test_every_filespec_filename_resolves_to_a_loader() -> None:
    """Each FileSpec filename under film/realesrgan-anime maps to its loader."""
    resolved = {
        spec: [_loader_for_filename(name) for name in _spec_filenames(spec)]
        for spec in _AUGMENT_SPECS
    }
    assert resolved["film"] == [augment_worker._load_film_net]
    assert resolved["realesrgan-anime"] == [augment_worker._load_esrgan_net]


def test_loader_mapping_rejects_unknown_suffix() -> None:
    """Filenames outside the two shipped suffixes fail loud (no silent default)."""
    with pytest.raises(AssertionError, match="no augment loader"):
        _loader_for_filename("weights.onnx")


def test_esrgan_loader_rejects_wrong_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A floor-sized blob with non-anime keys stays ModelCompatibilityError (torch-free)."""
    from voyage import model_registry as registry

    weights_path = tmp_path / registry.REALESRGAN_ANIME_FILE
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with weights_path.open("wb") as handle:
        handle.truncate(registry.REALESRGAN_ANIME_MIN_BYTES)

    def _wrong_keys(_path: Path) -> dict[str, Any]:
        return {"not_a_net_key": "junk"}

    class _RejectsAll:
        def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
            raise RuntimeError(f"missing keys (got {len(state)})")

    monkeypatch.setattr(augment_worker, "_load_state_dict", _wrong_keys)
    monkeypatch.setattr(augment_worker, "_build_rrdb_net", lambda: _RejectsAll())
    monkeypatch.setattr(augment_worker, "_build_srvgg_net", lambda _depth: _RejectsAll())
    with pytest.raises(ModelCompatibilityError, match="Real-ESRGAN"):
        augment_worker._load_esrgan_net(weights_path)


def test_srvgg_compact_state_sniff_and_strict_load() -> None:
    """Provisioned anime-video-XS `.pth` sniffs SRVGG-compact and strict-loads.

    The pinned `realesr-animevideov3.pth` carries the SRVGGNetCompact
    PReLU layout (body.0..body.34, odd indices weight-only, last conv
    (48, 64, 3, 3) for num_feat 64 / upscale 4); the worker sniffs it
    (not the RRDB layouts) and `load_state_dict(strict=True)` covers
    every key. Skips (not fails) without torch+weights.
    """
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    check = model_registry.MODEL_SPECS["realesrgan-anime"].checks[0]
    assert isinstance(check, RequiredFile)
    found = _find_provisioned_weight(check.relative_path)
    if found is None:
        pytest.skip(f"needs provisioned weights ({check.relative_path})")
    weights_path = found

    decoded = augment_worker._load_state_dict(weights_path)
    # The file wraps params one level deep (`params`, ESRGAN-family
    # convention) — unwrap exactly like `_load_esrgan_net` before sniffing.
    state = augment_worker._unwrap_esrgan_state(decoded)
    assert len(state) == 53
    assert augment_worker._is_srvgg_compact_state(state) is True
    assert augment_worker._upstream_block_count(state) is None
    last_weight = state["body.34.weight"]
    assert tuple(last_weight.shape) == (48, 64, 3, 3)
    model = augment_worker._build_srvgg_net(16)
    model.eval()
    model.load_state_dict(state, strict=True)
    assert set(model.state_dict()) == set(state)


def test_esrgan_loads_provisioned_weights_and_upscales() -> None:
    """Provisioned anime-video-XS `.pth` loads strict and upscales a synthetic frame.

    Red pre-fix (issue 166): ModelCompatibilityError ("needs its own
    loader"). Green post-fix: strict load plus a (3, 16, 16) finite
    upscale at scale 2 on CPU. Skips (not fails) without torch+weights.
    """
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    check = model_registry.MODEL_SPECS["realesrgan-anime"].checks[0]
    assert isinstance(check, RequiredFile)
    found = _find_provisioned_weight(check.relative_path)
    if found is None:
        pytest.skip(f"needs provisioned weights ({check.relative_path})")
    weights_path = found
    import torch

    augment_worker.evict_augment_models()
    frame = torch.rand(3, 8, 8, dtype=torch.float32)
    outputs = augment_worker.upscale_frames([frame], weights_path, scale=2, device="cpu")
    assert len(outputs) == 1
    assert tuple(outputs[0].shape) == (3, 16, 16)
    assert bool(torch.isfinite(outputs[0]).all())
    # Bicubic downscale (scale 2 of the native x4 pass) can ring slightly past
    # the [0, 1] clamp applied before it — bound the overshoot, not exact range.
    assert float(outputs[0].min()) >= -0.1
    assert float(outputs[0].max()) <= 1.1


def test_film_loads_provisioned_weights_strict() -> None:
    """Provisioned FILM `.safetensors` strict-loads into the upstream port.

    The pinned `film_net_fp16.safetensors` carries the 82-key
    extract/fuse/predict_flow state (fp16); the worker builds the
    matching net and `load_state_dict(strict=True)` succeeds — no
    ModelCompatibilityError. Skips (not fails) without
    torch+safetensors+weights.
    """
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    if importlib.util.find_spec("safetensors") is None:
        pytest.skip("needs safetensors (run in voyage-video)")
    check = model_registry.MODEL_SPECS["film"].checks[0]
    assert isinstance(check, RequiredFile)
    found = _find_provisioned_weight(check.relative_path)
    if found is None:
        pytest.skip(f"needs provisioned weights ({check.relative_path})")
    weights_path = found
    augment_worker.evict_augment_models()
    decoded = augment_worker._load_state_dict(weights_path)
    assert len(decoded) == 82
    assert {key.split(".")[0] for key in decoded} == {"extract", "fuse", "predict_flow"}
    assert {str(value.dtype) for value in decoded.values()} == {"torch.float16"}
    model = augment_worker._load_film_net(weights_path)
    model.eval()
    assert set(model.state_dict()) == set(decoded)


def test_film_rejects_frames_below_min_side() -> None:
    """Sides below FILM_MIN_SIDE fail loud (the fusion decoder needs 4 levels).

    Random init, no weights: the floor lives in the forward, before any
    model work. Upstream always runs 7 levels (needs >=64px); the worker
    clamps depth down to 4 levels (8px) and rejects below that instead of
    crashing deep in the pyramid.
    """
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    import torch

    model = augment_worker._build_film_net()
    model.eval()
    with torch.no_grad(), pytest.raises(ValueError, match="frame sides"):
        model(torch.zeros(1, 2, 3, 4, 4), 0.5)


def test_film_interpolate_pair_synthetic_round_trip() -> None:
    """Provisioned FILM weights interpolate a synthetic pair on CPU.

    Two (3, 16, 16) gradient frames at moment 0.5 give one finite
    (3, 16, 16) mid frame. 16px exercises the worker's small-input
    pyramid clamp (upstream runs 7 levels, which needs >=64px).
    Skips (not fails) without torch+safetensors+weights.
    """
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    if importlib.util.find_spec("safetensors") is None:
        pytest.skip("needs safetensors (run in voyage-video)")
    check = model_registry.MODEL_SPECS["film"].checks[0]
    assert isinstance(check, RequiredFile)
    found = _find_provisioned_weight(check.relative_path)
    if found is None:
        pytest.skip(f"needs provisioned weights ({check.relative_path})")
    weights_path = found
    import torch

    augment_worker.evict_augment_models()
    rows = torch.linspace(0.0, 1.0, 16).unsqueeze(1).expand(16, 16)
    before = torch.stack([rows, rows, rows])
    after = torch.stack([1.0 - rows, 1.0 - rows, 1.0 - rows])
    mid = augment_worker.interpolate_pair(before, after, weights_path, moment=0.5, device="cpu")
    assert tuple(mid.shape) == (3, 16, 16)
    assert bool(torch.isfinite(mid).all())
    # The fuse head is unbounded (Comfy clamps at the pipe end); bound the
    # overshoot instead of the exact range.
    assert float(mid.min()) >= -0.5
    assert float(mid.max()) <= 1.5
