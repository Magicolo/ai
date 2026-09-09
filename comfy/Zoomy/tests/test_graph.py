"""Tests for the API-format workflow builder."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zoomy.graph import ComfyWorkflow, NodeReference

# Node keys are free-form non-empty strings in practice; link slots count
# outputs from zero.
node_key_text = st.text(
    alphabet=st.characters(blacklist_characters="\x00"), min_size=1, max_size=12
)
output_slot = st.integers(min_value=0, max_value=8)


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


@given(key=node_key_text, slot=output_slot)
def test_reference_conversion_roundtrip(key: str, slot: int) -> None:
    """Any key/slot reference serializes to exactly its link array."""
    workflow = ComfyWorkflow()
    workflow.add("consumer", "KSampler", {"model": NodeReference(key=key, output_slot=slot)})
    assert workflow.build()["consumer"]["inputs"]["model"] == [key, slot]


@given(
    keys=st.lists(node_key_text, min_size=1, max_size=6, unique=True),
    slot=output_slot,
)
def test_chained_workflow_links_all_resolve(keys: list[str], slot: int) -> None:
    """A chain where each node feeds the next has no dangling links."""
    workflow = ComfyWorkflow()
    for position, key in enumerate(keys):
        inputs: dict[str, object] = {"label": f"node-{position}"}
        if position > 0:
            inputs["model"] = NodeReference(key=keys[position - 1], output_slot=slot)
        workflow.add(key, "SomeNode", inputs)
    document = workflow.build()
    assert len(document) == len(keys)
    for node in document.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in document
