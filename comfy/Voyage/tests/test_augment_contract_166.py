"""Issue 166: FileSpec filename round-trips through its worker loader.

Contract: every `FileSpec` filename under the `film` /
`realesrgan-anime` registry rows must resolve to a worker loader in
`voyage.workers.augment_worker`, and the provisioned weights must behave
as documented — the ESRGAN anime-6B `.pth` loads and upscales synthetic
tensors, while the FILM `.safetensors` fails loud with the upstream-port
note until the full FILM port lands (explicit remainder, not silent).

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
        return augment_worker._load_rrdb_net
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
    assert resolved["realesrgan-anime"] == [augment_worker._load_rrdb_net]


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
    with pytest.raises(ModelCompatibilityError, match="Real-ESRGAN"):
        augment_worker._load_rrdb_net(weights_path)


def test_esrgan_loads_provisioned_weights_and_upscales() -> None:
    """Provisioned anime-6B `.pth` loads strict and upscales a synthetic frame.

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


def test_film_provisioned_weights_fail_loud_with_port_note() -> None:
    """Provisioned FILM `.safetensors` raises ModelCompatibilityError naming the port.

    The full upstream FILM port (82-key extract/fuse/predict_flow net) is
    the explicit remainder: this pins the fail-loud contract so the day
    the port lands this leg flips to a load test deliberately, not silently.
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
    with pytest.raises(ModelCompatibilityError, match="FILM port"):
        augment_worker._load_film_net(weights_path)
