"""LTXV family split surface tests (issue 082).

Behavior-preservation contract for the `voyage/registry_ltxv.py`
per-family extraction: the LTXV pins + record/describe builders live
once in the new family module, and both `voyage.registry_records`
and `voyage.model_registry` re-export the identical objects.
"""

from __future__ import annotations

import voyage.model_registry as model_registry
import voyage.registry_ltxv as registry_ltxv
import voyage.registry_records as registry_records

LTXV_PIN_NAMES = (
    "LTXV_HF_REPO",
    "LTXV_HF_REVISION",
    "LTXV_SUBDIR",
    "LTXV_DIT_FILE",
    "LTXV_UPSC_FILE",
    "LTXV_DIT_MIN_BYTES",
    "LTXV_UPSC_MIN_BYTES",
    "LTXV_TE_REPO",
    "LTXV_TE_REVISION",
    "LTXV_TE_SUBDIR",
    "LTXV_TE_ALLOW",
    "LTXV_COMMIT",
    "LTXV_COMMIT_SHORT",
    "EXPECTED_LTXV_DIT_SHA256",
    "EXPECTED_LTXV_UPSC_SHA256",
)


def test_ltxv_pins_are_single_sourced() -> None:
    """LTXV pins live once, in registry_ltxv (issue 082)."""
    for name in LTXV_PIN_NAMES:
        assert getattr(registry_records, name) is getattr(registry_ltxv, name)
        assert getattr(model_registry, name) == getattr(registry_ltxv, name)


def test_ltxv_builders_are_single_sourced() -> None:
    """LTXV record/describe helpers live once, in registry_ltxv (issue 082)."""
    assert registry_records._record_ltxv is registry_ltxv._record_ltxv
    assert registry_records._describe_ltxv is registry_ltxv._describe_ltxv
    assert model_registry._record_ltxv is registry_ltxv._record_ltxv
    assert model_registry._describe_ltxv is registry_ltxv._describe_ltxv


def test_ltxv_spec_row_points_at_family_builders() -> None:
    """MODEL_SPECS ltxv-2b row calls the single-sourced builders (issue 082)."""
    spec = model_registry.MODEL_SPECS["ltxv-2b"]
    assert spec.name == "ltxv-2b"
    assert spec.record_builder is registry_ltxv._record_ltxv
    assert spec.success_message is registry_ltxv._describe_ltxv
