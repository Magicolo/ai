"""Causvid family split surface tests (issue 082).

Behavior-preservation contract for the `voyage/registry_causvid.py`
per-family extraction: the CausVid + Wan2.1 base pins + record/describe
builders live once in the new family module, and both
`voyage.registry_records` and `voyage.model_registry` re-export the
identical objects. Unlike the FILM/Real-ESRGAN/LTXV families this row
carries no EXPECTED ingest hash (the DMD checkpoint was pruned
2026-09-24 — same open residual as the inspector row, see
`registry_records.py`).
"""

from __future__ import annotations

import voyage.model_registry as model_registry
import voyage.registry_causvid as registry_causvid
import voyage.registry_records as registry_records

CAUSVID_PIN_NAMES = (
    "CAUSVID_COMMIT",
    "CAUSVID_COMMIT_SHORT",
    "CAUSVID_HF_REPO",
    "CAUSVID_HF_REVISION",
    "CAUSVID_SUBDIR",
    "CAUSVID_CHECKPOINT_SUBDIR",
    "CAUSVID_CHECKPOINT_NAME",
    "CAUSVID_CHECKPOINT_FILE",
    "CAUSVID_CKPT_MIN_BYTES",
    "CAUSVID_LICENSE",
    "CAUSVID_LICENSE_URL",
    "WAN21_HF_REPO",
    "WAN21_HF_REVISION",
    "WAN21_SUBDIR",
    "WAN21_ALLOW",
    "WAN21_DIT_MIN_BYTES",
    "WAN21_VAE_MIN_BYTES",
    "WAN21_T5_MIN_BYTES",
    "WAN21_LICENSE",
    "WAN21_LICENSE_URL",
)


def test_causvid_pins_are_single_sourced() -> None:
    """Causvid pins live once, in registry_causvid (issue 082)."""
    for name in CAUSVID_PIN_NAMES:
        assert getattr(registry_records, name) is getattr(registry_causvid, name)
        assert getattr(model_registry, name) == getattr(registry_causvid, name)


def test_causvid_builders_are_single_sourced() -> None:
    """Causvid record/describe helpers live once, in registry_causvid (issue 082)."""
    assert registry_records._record_causvid is registry_causvid._record_causvid
    assert registry_records._describe_causvid is registry_causvid._describe_causvid
    assert model_registry._record_causvid is registry_causvid._record_causvid
    assert model_registry._describe_causvid is registry_causvid._describe_causvid


def test_causvid_spec_row_points_at_family_builders() -> None:
    """MODEL_SPECS causvid row calls the single-sourced builders (issue 082)."""
    spec = model_registry.MODEL_SPECS["causvid"]
    assert spec.name == "causvid"
    assert spec.record_builder is registry_causvid._record_causvid
    assert spec.success_message is registry_causvid._describe_causvid
