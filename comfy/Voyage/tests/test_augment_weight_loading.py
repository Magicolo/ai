"""Issue 074: augment weight loads branch on format with pre-verify gates.

Both augment loaders called `torch.load` unconditionally, yet FILM ships a
`.safetensors` file (not pickle). The format branch (safetensors for
`.safetensors`, `torch.load(weights_only=True)` for `.pth`) plus the size and
manifest pre-checks and the `UnpicklingError` mapping are the standard
controls. NOTE (issue 166): the vendored RRDBNet/FilmNetMini architectures
structurally do NOT match the pinned Real-ESRGAN/FILM weights (`strict=True`
raises by construction) — these gates are testable on synthetic blobs, but
end-to-end loading of the pinned weights awaits the upstream FILM port +
SRVGG loader. CPU-only: stubbed torch/safetensors, synthetic blobs, tmp_path.
"""

from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from voyage.errors import ModelCompatibilityError
from voyage.workers import augment_worker


def test_unpickling_error_maps_to_compatibility() -> None:
    """`UnpicklingError` means weights-unusable-here (not a generic worker error)."""
    assert pickle.UnpicklingError in augment_worker._LOAD_ERRORS


def test_safetensors_suffix_uses_safe_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.safetensors` paths decode via safetensors (never the pickle machine)."""
    weights_path = tmp_path / "film_net_fp16.safetensors"
    weights_path.write_bytes(b"fake-safetensors-payload")
    seen: dict[str, Any] = {}

    def _fake_load_file(filename: str | Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        seen["safetensors_file"] = str(filename)
        return {"marker": "safetensors-state"}

    def _boom_torch_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("torch.load must not run for .safetensors files")

    safetensors_module = types.ModuleType("safetensors")
    torch_submodule = types.ModuleType("safetensors.torch")
    torch_submodule.load_file = _fake_load_file  # type: ignore[attr-defined]
    safetensors_module.torch = torch_submodule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_module)
    monkeypatch.setitem(sys.modules, "safetensors.torch", torch_submodule)
    torch_stub = types.ModuleType("torch")
    torch_stub.load = _boom_torch_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    state = augment_worker._load_state_dict(weights_path)
    assert state == {"marker": "safetensors-state"}
    assert seen["safetensors_file"] == str(weights_path)


def test_pth_suffix_uses_weights_only_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`.pth` paths load via `torch.load(weights_only=True)` (no safetensors)."""
    weights_path = tmp_path / "realesr-animevideov3.pth"
    weights_path.write_bytes(b"fake-pth-payload")
    seen: dict[str, Any] = {}

    def _fake_torch_load(filename: str | Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        seen["torch_file"] = str(filename)
        seen["weights_only"] = kwargs.get("weights_only")
        return {"marker": "torch-state"}

    def _boom_safe_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("safetensors must not run for .pth files")

    torch_stub = types.ModuleType("torch")
    torch_stub.load = _fake_torch_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    safetensors_module = types.ModuleType("safetensors")
    torch_submodule = types.ModuleType("safetensors.torch")
    torch_submodule.load_file = _boom_safe_load  # type: ignore[attr-defined]
    safetensors_module.torch = torch_submodule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_module)
    monkeypatch.setitem(sys.modules, "safetensors.torch", torch_submodule)

    state = augment_worker._load_state_dict(weights_path)
    assert state == {"marker": "torch-state"}
    assert seen["torch_file"] == str(weights_path)
    assert seen["weights_only"] is True


def test_corrupt_pickle_maps_to_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt `.pth` blob surfaces as ModelCompatibilityError (actionable)."""
    from voyage import model_registry

    weights_path = tmp_path / "realesr-animevideov3.pth"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with weights_path.open("wb") as handle:
        handle.truncate(model_registry.REALESRGAN_ANIME_MIN_BYTES)

    def _raise_unpickling(*args: Any, **kwargs: Any) -> Any:
        raise pickle.UnpicklingError("corrupt pickle payload")

    torch_stub = types.ModuleType("torch")
    torch_stub.load = _raise_unpickling  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    fake_model: Any = types.SimpleNamespace(
        load_state_dict=lambda state, strict: None,
    )
    monkeypatch.setattr(augment_worker, "_build_rrdb_net", lambda: fake_model)
    monkeypatch.setattr(augment_worker, "_build_srvgg_net", lambda _depth: fake_model)
    with pytest.raises(ModelCompatibilityError, match="Real-ESRGAN"):
        augment_worker._load_esrgan_net(weights_path)


def test_size_floor_rejects_tiny_file(tmp_path: Path) -> None:
    """Undersized blobs fail the size pre-check (not deep in the loader)."""
    tiny = tmp_path / "film_net_fp16.safetensors"
    tiny.write_bytes(b"x" * 16)
    with pytest.raises(ModelCompatibilityError, match="size"):
        augment_worker._load_film_net(tiny)


def test_manifest_mismatch_rejects_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest-recorded sha mismatch fails before any loader runs."""
    from voyage import model_registry

    weights_path = tmp_path / model_registry.FILM_SUBDIR / model_registry.FILM_FILE
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"y" * model_registry.FILM_MIN_BYTES)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "film": {
                    "checkpoint_sha256": "0" * 64,
                    "checkpoint_file": model_registry.FILM_REPO_PATH,
                }
            }
        ),
        encoding="utf-8",
    )

    def _boom_loader(path: Path) -> dict[str, Any]:
        raise AssertionError("loader must not run after a manifest mismatch")

    monkeypatch.setattr(augment_worker, "_load_state_dict", _boom_loader)
    with pytest.raises(ModelCompatibilityError, match="[Hh]ash|mismatch|manifest"):
        augment_worker._load_film_net(weights_path)
