"""ComfyUI API-format workflow construction.

ComfyUI's REST API accepts a workflow as a flat dictionary:

    {"<node key>": {"class_type": "...", "inputs": {"name": value, ...}}}

A link between two nodes is an input whose value is ``[source_key, output_slot]``.
:class:`NodeReference` lets callers express links as typed values, and
:class:`ComfyWorkflow` converts them during ``add`` so the assembled document
is exactly what ComfyUI expects.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


@dataclass(frozen=True, slots=True)
class NodeReference:
    """A typed pointer to one output slot of a workflow node."""

    key: str
    output_slot: int = 0


class ComfyWorkflow:
    """Incremental builder for a ComfyUI API-format workflow document."""

    def __init__(self) -> None:
        """Start with an empty node dictionary."""
        self._nodes: dict[str, dict[str, Any]] = {}

    def add(self, key: str, class_type: str, inputs: Mapping[str, Any]) -> NodeReference:
        """Add one node and return a reference to its first output slot.

        Any :class:`NodeReference` in ``inputs`` (directly or inside a list) is
        converted to the ``[key, slot]`` array format ComfyUI expects.

        Raises:
            ValueError: If ``key`` was already added; node keys must be unique
                because they double as link targets.
        """
        if key in self._nodes:
            message = f"Workflow already contains a node with key {key!r}"
            raise ValueError(message)
        converted = {name: self._convert(value) for name, value in inputs.items()}
        self._nodes[key] = {"class_type": class_type, "inputs": converted}
        return NodeReference(key=key)

    def build(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of the assembled workflow document."""
        return deepcopy(self._nodes)

    def _convert(self, value: Any) -> Any:
        """Convert references to link arrays, recursing into plain lists."""
        if isinstance(value, NodeReference):
            return [value.key, value.output_slot]
        if isinstance(value, list):
            return [self._convert(item) for item in value]
        return value
