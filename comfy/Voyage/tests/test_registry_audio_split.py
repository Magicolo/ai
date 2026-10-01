"""Audio family split surface tests (issue 082).

Behavior-preservation contract for the `voyage/registry_audio.py`
per-family extraction: the ACE-Step music-stack pins + record/describe
builders live once in the new family module, and both
`voyage.registry_records` and `voyage.model_registry` re-export the
identical objects.
"""

from __future__ import annotations

import voyage.model_registry as model_registry
import voyage.registry_audio as registry_audio
import voyage.registry_records as registry_records

AUDIO_PIN_NAMES = (
    "ACE_MAIN_REPO",
    "ACE_MAIN_REVISION",
    "ACE_MAIN_SUBDIR",
    "ACE_CHECKPOINTS_SUBDIR",
    "ACE_MAIN_ALLOW",
    "ACE_TURBO_MIN_BYTES",
    "ACE_LM17_MIN_BYTES",
    "ACE_MAIN_LICENSE",
    "ACE_LM_REPO",
    "ACE_LM_REVISION",
    "ACE_LM_SUBDIR",
    "ACE_LM_ALLOW",
    "ACE_LM_MIN_BYTES",
    "_ACE_CHECKPOINTS_RELATIVE",
    "_ACE_LM_RELATIVE",
)


def test_audio_pins_are_single_sourced() -> None:
    """Audio pins live once, in registry_audio (issue 082)."""
    for name in AUDIO_PIN_NAMES:
        assert getattr(registry_records, name) is getattr(registry_audio, name)
        assert getattr(model_registry, name) == getattr(registry_audio, name)


def test_audio_builders_are_single_sourced() -> None:
    """Audio record/describe helpers live once, in registry_audio (issue 082)."""
    assert registry_records._record_audio is registry_audio._record_audio
    assert registry_records._describe_audio is registry_audio._describe_audio
    assert model_registry._record_audio is registry_audio._record_audio
    assert model_registry._describe_audio is registry_audio._describe_audio


def test_audio_spec_row_points_at_family_builders() -> None:
    """MODEL_SPECS audio-acestep row calls the single-sourced builders (issue 082)."""
    spec = model_registry.MODEL_SPECS["audio-acestep"]
    assert spec.name == "audio-acestep"
    assert spec.record_builder is registry_audio._record_audio
    assert spec.success_message is registry_audio._describe_audio
