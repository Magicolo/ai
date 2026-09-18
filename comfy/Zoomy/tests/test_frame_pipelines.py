"""Tests for frame pipeline LoRA bookkeeping above the heavy imports.

LoRA selection never touches torch or diffusers — the pipeline is
duck-typed — so activation, caching, and the dirty check run in the slim
test image with a recording stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zoomy.errors import EngineConfigurationError
from zoomy.family_catalog import LoraDefinition
from zoomy.frame_pipelines import LoadedFramePipeline, apply_lora_selection

if TYPE_CHECKING:
    from pathlib import Path


class _StubFramePipeline:
    """Record LoRA loads and activations without any model weights."""

    def __init__(self) -> None:
        """Start with no registered adapters and no activations."""
        self.loads: list[str] = []
        self.activations: list[tuple[list[str], list[float]]] = []

    def load_lora_weights(self, path: str) -> None:
        """Pretend one file registers one adapter named after the file."""
        self.loads.append(path)

    def get_list_adapters(self) -> dict[str, list[str]]:
        """Report one adapter per loaded file, named after its stem."""
        return {"default": [f"adapter-{len(self.loads)}"]}

    def set_adapters(self, names: list[str], adapter_weights: list[float]) -> None:
        """Record the activation the engine requested."""
        self.activations.append((list(names), list(adapter_weights)))


def _lora(file_name: str) -> LoraDefinition:
    """Build a catalog LoRA entry pointing at a test file name."""
    return LoraDefinition(
        file_name=file_name,
        display_name=file_name,
        default_strength=1.0,
        selected_by_default=False,
    )


def _loaded() -> LoadedFramePipeline:
    """Build a resident pipeline stub; LoRA files resolve under tmp_path."""
    return LoadedFramePipeline(family_key="z_fast", pipeline=_StubFramePipeline())


def test_empty_selection_activates_nothing(tmp_path: Path) -> None:
    """Deselecting every LoRA skips the weight swap without touching disk."""
    loaded = _loaded()
    apply_lora_selection(loaded, tmp_path, ())
    assert loaded.pipeline.activations == []
    assert loaded.active_adapter_key == ()


def test_unchanged_selection_skips_the_weight_swap(tmp_path: Path) -> None:
    """A repeated selection reuses the active adapters: no disk, no swap."""
    loras_directory = tmp_path / "loras"
    loras_directory.mkdir(parents=True)
    (loras_directory / "chalk.safetensors").write_bytes(b"stub")
    loaded = _loaded()
    selection = ((_lora("chalk.safetensors"), 0.85),)
    apply_lora_selection(loaded, tmp_path, selection)
    apply_lora_selection(loaded, tmp_path, selection)
    assert loaded.pipeline.loads == [str(loras_directory / "chalk.safetensors")]
    assert len(loaded.pipeline.activations) == 1
    assert loaded.pipeline.activations[0] == (["adapter-1"], [0.85])


def test_changed_strength_reactivates_without_reloading(tmp_path: Path) -> None:
    """A strength change swaps weights on the registered adapter, no reload."""
    loras_directory = tmp_path / "loras"
    loras_directory.mkdir(parents=True)
    (loras_directory / "chalk.safetensors").write_bytes(b"stub")
    loaded = _loaded()
    apply_lora_selection(loaded, tmp_path, ((_lora("chalk.safetensors"), 0.85),))
    apply_lora_selection(loaded, tmp_path, ((_lora("chalk.safetensors"), 1.0),))
    assert len(loaded.pipeline.loads) == 1
    assert loaded.pipeline.activations[-1] == (["adapter-1"], [1.0])


def test_missing_lora_file_names_itself(tmp_path: Path) -> None:
    """A selection pointing at no file fails before any adapter loads."""
    loaded = _loaded()
    with pytest.raises(EngineConfigurationError, match=r"chalk\.safetensors"):
        apply_lora_selection(loaded, tmp_path, ((_lora("chalk.safetensors"), 1.0),))


def test_activation_failure_names_the_stage(tmp_path: Path) -> None:
    """A pipeline refusing the swap surfaces as a stage error, not a crash."""

    class _RefusingPipeline(_StubFramePipeline):
        def set_adapters(self, names: list[str], adapter_weights: list[float]) -> None:
            del names, adapter_weights
            raise RuntimeError("adapter mismatch")

    loras_directory = tmp_path / "loras"
    loras_directory.mkdir(parents=True)
    (loras_directory / "chalk.safetensors").write_bytes(b"stub")
    loaded = LoadedFramePipeline(family_key="z_fast", pipeline=_RefusingPipeline())
    from zoomy.errors import EngineExecutionError  # noqa: PLC0415

    with pytest.raises(EngineExecutionError, match="LoRA activation failed"):
        apply_lora_selection(loaded, tmp_path, ((_lora("chalk.safetensors"), 1.0),))
