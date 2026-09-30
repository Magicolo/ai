"""FILM family split surface tests (issue 082).

Behavior-preservation contract for the `voyage/registry_film.py`
per-family extraction: the FILM pins + record/describe builders live
once in the new family module, and both `voyage.registry_records`
and `voyage.model_registry` re-export the identical objects.
"""

from __future__ import annotations

import voyage.model_registry as model_registry
import voyage.registry_film as registry_film
import voyage.registry_records as registry_records

FILM_PIN_NAMES = (
    "FILM_HF_REPO",
    "FILM_HF_REVISION",
    "FILM_SUBDIR",
    "FILM_FILE",
    "FILM_REPO_PATH",
    "FILM_MIN_BYTES",
    "FILM_LICENSE",
    "FILM_LICENSE_URL",
    "EXPECTED_FILM_SHA256",
)


def test_film_pins_are_single_sourced() -> None:
    """FILM pins live once, in registry_film (issue 082)."""
    for name in FILM_PIN_NAMES:
        assert getattr(registry_records, name) is getattr(registry_film, name)
        assert getattr(model_registry, name) == getattr(registry_film, name)


def test_film_builders_are_single_sourced() -> None:
    """FILM record/describe helpers live once, in registry_film (issue 082)."""
    assert registry_records._record_film is registry_film._record_film
    assert registry_records._describe_film is registry_film._describe_film
    assert model_registry._record_film is registry_film._record_film
    assert model_registry._describe_film is registry_film._describe_film


def test_film_spec_row_points_at_family_builders() -> None:
    """MODEL_SPECS film row calls the single-sourced builders (issue 082)."""
    spec = model_registry.MODEL_SPECS["film"]
    assert spec.name == "film"
    assert spec.record_builder is registry_film._record_film
    assert spec.success_message is registry_film._describe_film
