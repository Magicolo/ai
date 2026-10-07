"""Registry split surface tests (issue 082).

Behavior-preservation contract for the `voyage/model_registry.py`
per-family extraction: `MODEL_SPECS` keeps its keys and rows, and every
record builder / describe helper / pin re-exported from `voyage` is the
identical object as the new `voyage/registry_records.py` home module.
"""

from __future__ import annotations

import voyage.model_registry as model_registry
import voyage.registry_records as registry_records

EXPECTED_SPEC_KEYS = (
    "director-qwen8b",
    "director-qwen4b-awq",
    "director-qwen35-gguf",
    "inspector-qwen35",
    "audio-acestep",
    "audio-sonicmaster",
    "sfx-mmaudio",
    "ltxv-2b",
    "causvid",
    "ltx25",
    "ltx23",
    "film",
    "rife",
    "realesrgan-anime",
)

RECORD_BUILDERS = (
    "_record_director",
    "_record_director_awq",
    "_record_inspector",
    "_record_audio",
    "_record_mastering",
    "_record_ltxv",
    "_record_causvid",
    "_record_ltx25",
    "_record_ltx23",
    "_record_sfx",
    "_record_film",
    "_record_rife",
    "_record_realesrgan",
)

DESCRIBE_HELPERS = (
    "_describe_director",
    "_describe_director_awq",
    "_describe_inspector",
    "_describe_audio",
    "_describe_mastering",
    "_describe_ltxv",
    "_describe_sfx",
    "_describe_causvid",
    "_describe_ltx25",
    "_describe_ltx23",
    "_describe_film",
    "_describe_rife",
    "_describe_realesrgan",
)

PIN_NAMES = (
    "WAN21_HF_REPO",
    "QWEN_HF_REPO",
    "QWEN_HF_REVISION",
    "QWEN4B_AWQ_HF_REPO",
    "QWEN4B_AWQ_HF_REVISION",
    "QWEN35_HF_REPO",
    "QWEN35_HF_REVISION",
    "ACE_MAIN_REPO",
    "ACE_MAIN_REVISION",
    "LTXV_HF_REPO",
    "CAUSVID_HF_REPO",
    "LTX25_DIT_REPO",
    "LTX25_TE_REPO",
    "LTX23_DIT_REPO",
    "LTX23_TE_REPO",
    "FILM_HF_REPO",
)


def test_model_specs_keys_unchanged() -> None:
    """The registry assembly keeps all bundles (issue 082 + Track A mastering)."""
    assert tuple(sorted(model_registry.MODEL_SPECS)) == tuple(sorted(EXPECTED_SPEC_KEYS))


def test_record_builders_are_single_sourced() -> None:
    """Record builders live once, in registry_records (issue 082)."""
    for name in RECORD_BUILDERS:
        assert getattr(model_registry, name) is getattr(registry_records, name)


def test_describe_helpers_are_single_sourced() -> None:
    """Describe helpers live once, in registry_records (issue 082)."""
    for name in DESCRIBE_HELPERS:
        assert getattr(model_registry, name) is getattr(registry_records, name)


def test_pins_are_single_sourced() -> None:
    """Pin constants live once, in registry_records (issue 082)."""
    for name in PIN_NAMES:
        assert getattr(model_registry, name) == getattr(registry_records, name)


def test_spec_rows_point_at_record_builders() -> None:
    """MODEL_SPECS rows call the single-sourced builders (issue 082)."""
    for key, spec in model_registry.MODEL_SPECS.items():
        assert spec.name == key
        if key == "director-qwen35-gguf":
            # Sidecar-track row (DESIGN §140 llama entry): pins + builders
            # live in model_registry until the family-module move to
            # registry_director lands — assert identity at that home.
            assert spec.record_builder is model_registry._record_director_gguf
            assert spec.success_message is model_registry._describe_director_gguf
            continue
        assert spec.record_builder is getattr(registry_records, f"_record_{spec_key(key)}")
        assert spec.success_message is getattr(registry_records, f"_describe_{spec_key(key)}")


def test_registry_all_covers_surface() -> None:
    """The seam __all__ is the explicit export contract (mypy reads it)."""
    surface = {name for name in dir(model_registry) if not name.startswith("__")}
    assert set(model_registry.__all__) <= surface
    for name in ("MODEL_SPECS", "download_model", "verify_model"):
        assert name in model_registry.__all__


def spec_key(model_key: str) -> str:
    """MODEL_SPECS key → builder suffix (ltxv-2b → ltxv)."""
    mapping = {
        "director-qwen8b": "director",
        "director-qwen4b-awq": "director_awq",
        "inspector-qwen35": "inspector",
        "audio-acestep": "audio",
        "audio-sonicmaster": "mastering",
        "sfx-mmaudio": "sfx",
        "ltxv-2b": "ltxv",
        "causvid": "causvid",
        "ltx25": "ltx25",
        "ltx23": "ltx23",
        "film": "film",
        "rife": "rife",
        "realesrgan-anime": "realesrgan",
    }
    return mapping[model_key]
