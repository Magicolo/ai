"""Tests for the API-format workflow builder."""

from __future__ import annotations

import pytest

from zoomy.graph import ComfyWorkflow, NodeReference


def test_add_converts_node_references_to_link_arrays() -> None:
    """NodeReference inputs serialize to [key, slot] arrays."""
    workflow = ComfyWorkflow()
    loader = workflow.add(
        "load_model", "UNETLoader", {"unet_name": "model.safetensors", "weight_dtype": "default"}
    )
    workflow.add(
        "encode_prompt",
        "CLIPTextEncode",
        {"clip": NodeReference(key=loader.key, output_slot=1), "text": "a cat"},
    )
    document = workflow.build()
    assert document["encode_prompt"]["inputs"]["clip"] == ["load_model", 1]


def test_add_converts_references_inside_lists() -> None:
    """References nested in lists serialize recursively."""
    workflow = ComfyWorkflow()
    first = workflow.add("first", "LoadImage", {"image": "a.png"})
    second = workflow.add("second", "LoadImage", {"image": "b.png"})
    workflow.add("combine", "BatchImages", {"images": [first, second]})
    document = workflow.build()
    assert document["combine"]["inputs"]["images"] == [["first", 0], ["second", 0]]


def test_add_records_class_types_and_plain_inputs() -> None:
    """Nodes keep their class type and pass plain values through unchanged."""
    workflow = ComfyWorkflow()
    workflow.add("load_model", "UNETLoader", {"unet_name": "model.safetensors"})
    document = workflow.build()
    assert document["load_model"]["class_type"] == "UNETLoader"
    assert document["load_model"]["inputs"] == {"unet_name": "model.safetensors"}


def test_add_rejects_duplicate_keys() -> None:
    """Node keys are unique because they double as link targets."""
    workflow = ComfyWorkflow()
    workflow.add("load_model", "UNETLoader", {"unet_name": "a.safetensors"})
    with pytest.raises(ValueError, match="load_model"):
        workflow.add("load_model", "UNETLoader", {"unet_name": "b.safetensors"})


def test_build_returns_an_independent_copy() -> None:
    """Mutating the built document leaves the builder reusable."""
    workflow = ComfyWorkflow()
    workflow.add("load_model", "UNETLoader", {"unet_name": "a.safetensors"})
    document = workflow.build()
    document["load_model"]["inputs"]["unet_name"] = "mutated.safetensors"
    fresh_document = workflow.build()
    assert fresh_document["load_model"]["inputs"]["unet_name"] == "a.safetensors"
