"""Inspector family split surface tests (issue 082).

Behavior-preservation contract for the `voyage/registry_inspector.py`
per-family extraction: the Qwen3.5 inspector pins + record/describe
builders live once in the new family module, and both
`voyage.registry_records` and `voyage.model_registry` re-export the
identical objects. Unlike the FILM/Real-ESRGAN families this row carries
no EXPECTED ingest hash (the manifest record carries no sha — same open
residual as the CausVid checkpoint, see `registry_records.py`).
"""

from __future__ import annotations

import voyage.model_registry as model_registry
import voyage.registry_inspector as registry_inspector
import voyage.registry_records as registry_records

INSPECTOR_PIN_NAMES = (
    "QWEN35_HF_REPO",
    "QWEN35_HF_REVISION",
    "QWEN35_SUBDIR",
    "QWEN35_ALLOW",
    "QWEN35_MIN_BYTES",
    "QWEN35_LICENSE",
    "QWEN35_LICENSE_URL",
)


def test_inspector_pins_are_single_sourced() -> None:
    """Inspector pins live once, in registry_inspector (issue 082)."""
    for name in INSPECTOR_PIN_NAMES:
        assert getattr(registry_records, name) is getattr(registry_inspector, name)
        assert getattr(model_registry, name) == getattr(registry_inspector, name)


def test_inspector_builders_are_single_sourced() -> None:
    """Inspector record/describe helpers live once, in registry_inspector (issue 082)."""
    assert registry_records._record_inspector is registry_inspector._record_inspector
    assert registry_records._describe_inspector is registry_inspector._describe_inspector
    assert model_registry._record_inspector is registry_inspector._record_inspector
    assert model_registry._describe_inspector is registry_inspector._describe_inspector


def test_inspector_spec_row_points_at_family_builders() -> None:
    """MODEL_SPECS inspector-qwen35 row calls the single-sourced builders (issue 082)."""
    spec = model_registry.MODEL_SPECS["inspector-qwen35"]
    assert spec.name == "inspector-qwen35"
    assert spec.record_builder is registry_inspector._record_inspector
    assert spec.success_message is registry_inspector._describe_inspector
