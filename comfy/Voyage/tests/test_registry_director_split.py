"""Director family split surface tests (issue 082).

Behavior-preservation contract for the `voyage/registry_director.py`
per-family extraction: the Qwen3-8B + Qwen3-4B-AWQ + shared MiniLM pins +
record/describe builders live once in the new family module, and both
`voyage.registry_records` and `voyage.model_registry` re-export the
identical objects. This row carries no EXPECTED ingest hash (the
manifest record carries no sha — same open residual as the
inspector/audio/sfx/CausVid rows, see `registry_records.py`).

Single-source note (batch-15 decision): the shared `MINILM_*` pins move
into `voyage.registry_director` alongside the QWEN rows (not duplicated,
not left behind in `registry_records`), so both `_record_director`
builders resolve the embedding pins from the same module.
"""

from __future__ import annotations

import voyage.model_registry as model_registry
import voyage.registry_director as registry_director
import voyage.registry_records as registry_records

DIRECTOR_PIN_NAMES = (
    "QWEN_HF_REPO",
    "QWEN_HF_REVISION",
    "QWEN_SUBDIR",
    "QWEN_ALLOW",
    "QWEN_MIN_BYTES",
    "QWEN_LICENSE",
    "QWEN_LICENSE_URL",
    "QWEN4B_AWQ_HF_REPO",
    "QWEN4B_AWQ_HF_REVISION",
    "QWEN4B_AWQ_SUBDIR",
    "QWEN4B_AWQ_ALLOW",
    "QWEN4B_AWQ_MIN_BYTES",
    "QWEN4B_AWQ_LICENSE",
    "QWEN4B_AWQ_LICENSE_URL",
    "MINILM_HF_REPO",
    "MINILM_HF_REVISION",
    "MINILM_SUBDIR",
    "MINILM_ALLOW",
    "MINILM_MIN_BYTES",
    "MINILM_LICENSE",
)


def test_director_pins_are_single_sourced() -> None:
    """Director pins live once, in registry_director (issue 082)."""
    for name in DIRECTOR_PIN_NAMES:
        assert getattr(registry_records, name) is getattr(registry_director, name)
        assert getattr(model_registry, name) == getattr(registry_director, name)


def test_director_builders_are_single_sourced() -> None:
    """Director record/describe helpers live once, in registry_director (issue 082)."""
    assert registry_records._record_director is registry_director._record_director
    assert registry_records._describe_director is registry_director._describe_director
    assert registry_records._record_director_awq is registry_director._record_director_awq
    assert registry_records._describe_director_awq is registry_director._describe_director_awq
    assert model_registry._record_director is registry_director._record_director
    assert model_registry._describe_director is registry_director._describe_director
    assert model_registry._record_director_awq is registry_director._record_director_awq
    assert model_registry._describe_director_awq is registry_director._describe_director_awq


def test_director_spec_rows_point_at_family_builders() -> None:
    """MODEL_SPECS director rows call the single-sourced builders (issue 082)."""
    spec = model_registry.MODEL_SPECS["director-qwen8b"]
    assert spec.name == "director-qwen8b"
    assert spec.record_builder is registry_director._record_director
    assert spec.success_message is registry_director._describe_director
    spec_awq = model_registry.MODEL_SPECS["director-qwen4b-awq"]
    assert spec_awq.name == "director-qwen4b-awq"
    assert spec_awq.record_builder is registry_director._record_director_awq
    assert spec_awq.success_message is registry_director._describe_director_awq
