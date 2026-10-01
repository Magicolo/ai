"""Real-ESRGAN family split surface tests (issue 082).

Behavior-preservation contract for the `voyage/registry_realesrgan.py`
per-family extraction: the Real-ESRGAN pins + record/describe builders
live once in the new family module, and both `voyage.registry_records`
and `voyage.model_registry` re-export the identical objects.
"""

from __future__ import annotations

import voyage.model_registry as model_registry
import voyage.registry_realesrgan as registry_realesrgan
import voyage.registry_records as registry_records

REALESRGAN_PIN_NAMES = (
    "REALESRGAN_HF_REPO",
    "REALESRGAN_HF_REVISION",
    "REALESRGAN_SUBDIR",
    "REALESRGAN_ANIME_FILE",
    "REALESRGAN_ANIME_MIN_BYTES",
    "REALESRGAN_UPSTREAM_URL",
    "REALESRGAN_LICENSE",
    "REALESRGAN_LICENSE_URL",
    "EXPECTED_REALESRGAN_SHA256",
)


def test_realesrgan_pins_are_single_sourced() -> None:
    """Real-ESRGAN pins live once, in registry_realesrgan (issue 082)."""
    for name in REALESRGAN_PIN_NAMES:
        assert getattr(registry_records, name) is getattr(registry_realesrgan, name)
        assert getattr(model_registry, name) == getattr(registry_realesrgan, name)


def test_realesrgan_builders_are_single_sourced() -> None:
    """Real-ESRGAN record/describe helpers live once, in registry_realesrgan (issue 082)."""
    assert registry_records._record_realesrgan is registry_realesrgan._record_realesrgan
    assert registry_records._describe_realesrgan is registry_realesrgan._describe_realesrgan
    assert model_registry._record_realesrgan is registry_realesrgan._record_realesrgan
    assert model_registry._describe_realesrgan is registry_realesrgan._describe_realesrgan


def test_realesrgan_spec_row_points_at_family_builders() -> None:
    """MODEL_SPECS realesrgan-anime row calls the single-sourced builders (issue 082)."""
    spec = model_registry.MODEL_SPECS["realesrgan-anime"]
    assert spec.name == "realesrgan-anime"
    assert spec.record_builder is registry_realesrgan._record_realesrgan
    assert spec.success_message is registry_realesrgan._describe_realesrgan
