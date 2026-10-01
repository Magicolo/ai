"""SFX family split surface tests (issue 082).

Behavior-preservation contract for the `voyage/registry_sfx.py`
per-family extraction: the MMAudio code/vocoder/CLIP pins +
record/describe builders live once in the new family module, and both
`voyage.registry_records` and `voyage.model_registry` re-export the
identical objects. This row carries no EXPECTED ingest hash (the
manifest record carries no sha — same open residual as the
inspector/CausVid rows, see `registry_records.py`).
"""

from __future__ import annotations

import voyage.model_registry as model_registry
import voyage.registry_records as registry_records
import voyage.registry_sfx as registry_sfx

SFX_PIN_NAMES = (
    "MMAUDIO_CODE_COMMIT",
    "MMAUDIO_CODE_COMMIT_SHORT",
    "MMAUDIO_HF_REPO",
    "MMAUDIO_HF_REVISION",
    "MMAUDIO_SUBDIR",
    "MMAUDIO_WEIGHT_FILES",
    "MMAUDIO_EXT_FILES",
    "MMAUDIO_SMALL_MIN_BYTES",
    "MMAUDIO_MEDIUM_MIN_BYTES",
    "MMAUDIO_LARGE_MIN_BYTES",
    "MMAUDIO_VAE_MIN_BYTES",
    "MMAUDIO_SYNCHFORMER_MIN_BYTES",
    "MMAUDIO_LICENSE",
    "MMAUDIO_LICENSE_URL",
    "MMAUDIO_VOCODER_REPO",
    "MMAUDIO_VOCODER_REVISION",
    "MMAUDIO_VOCODER_SUBDIR",
    "MMAUDIO_VOCODER_ALLOW",
    "MMAUDIO_VOCODER_MIN_BYTES",
    "MMAUDIO_VOCODER_LICENSE",
    "MMAUDIO_CLIP_REPO",
    "MMAUDIO_CLIP_REVISION",
    "MMAUDIO_CLIP_SUBDIR",
    "MMAUDIO_CLIP_ALLOW",
    "MMAUDIO_CLIP_MIN_BYTES",
    "MMAUDIO_CLIP_LICENSE",
)


def test_sfx_pins_are_single_sourced() -> None:
    """SFX pins live once, in registry_sfx (issue 082)."""
    for name in SFX_PIN_NAMES:
        assert getattr(registry_records, name) is getattr(registry_sfx, name)
        assert getattr(model_registry, name) == getattr(registry_sfx, name)


def test_sfx_builders_are_single_sourced() -> None:
    """SFX record/describe helpers live once, in registry_sfx (issue 082)."""
    assert registry_records._record_sfx is registry_sfx._record_sfx
    assert registry_records._describe_sfx is registry_sfx._describe_sfx
    assert model_registry._record_sfx is registry_sfx._record_sfx
    assert model_registry._describe_sfx is registry_sfx._describe_sfx


def test_sfx_spec_row_points_at_family_builders() -> None:
    """MODEL_SPECS sfx-mmaudio row calls the single-sourced builders (issue 082)."""
    spec = model_registry.MODEL_SPECS["sfx-mmaudio"]
    assert spec.name == "sfx-mmaudio"
    assert spec.record_builder is registry_sfx._record_sfx
    assert spec.success_message is registry_sfx._describe_sfx
