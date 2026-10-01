"""Issue 166 (batch 13): registry weights resolve to loader-ready paths.

`resolve_augment_weights` is the production seam between the registry
(pinned FILM + Real-ESRGAN weights) and the torch loaders in
`voyage.workers.augment_worker`: it maps a models dir to existing,
floor-sized weight paths, or `None` per leg when the weights are absent
(the caller keeps the ffmpeg fallback — default-off unless provisioned).

All resolution legs are torch-free (missing/empty/undersized legs run in
the slim gates image); the provisioned load-through leg skips loudly
without torch+weights (run it in `voyage-video` with the models volume
mounted). CPU-only throughout.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from voyage.augment import AugmentWeights, resolve_augment_weights
from voyage.registry_film import FILM_MIN_BYTES, FILM_REPO_PATH
from voyage.registry_realesrgan import (
    REALESRGAN_ANIME_FILE,
    REALESRGAN_ANIME_MIN_BYTES,
    REALESRGAN_SUBDIR,
)

_REALESRGAN_RELATIVE = f"{REALESRGAN_SUBDIR}/{REALESRGAN_ANIME_FILE}"
"""Registry-relative path of the pinned anime-6B `.pth` under a models dir."""


def _write_sparse(path: Path, size_bytes: int) -> Path:
    """Create a sparse file of `size_bytes` (instant, no RAM/disk cost)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size_bytes)
    return path


def test_missing_models_dir_resolves_to_skip() -> None:
    """A nonexistent models dir resolves to all-None (graceful skip, never raises)."""
    resolved = resolve_augment_weights(Path("/nonexistent-voyage-models-dir"))
    assert resolved == AugmentWeights(film=None, realesrgan=None)


def test_empty_weight_files_read_as_not_provisioned(tmp_path: Path) -> None:
    """Zero-byte files count as absent (a torn download must skip, never load)."""
    _write_sparse(tmp_path / FILM_REPO_PATH, 0)
    _write_sparse(tmp_path / _REALESRGAN_RELATIVE, 0)
    resolved = resolve_augment_weights(tmp_path)
    assert resolved == AugmentWeights(film=None, realesrgan=None)


def test_undersized_weight_files_read_as_not_provisioned(tmp_path: Path) -> None:
    """Files below the registry floors count as absent (truncated fetch skips)."""
    _write_sparse(tmp_path / FILM_REPO_PATH, FILM_MIN_BYTES - 1)
    _write_sparse(tmp_path / _REALESRGAN_RELATIVE, REALESRGAN_ANIME_MIN_BYTES - 1)
    resolved = resolve_augment_weights(tmp_path)
    assert resolved == AugmentWeights(film=None, realesrgan=None)


def test_floor_sized_weight_files_resolve_to_paths(tmp_path: Path) -> None:
    """Files at the registry floors resolve to their loader-ready paths."""
    film_path = _write_sparse(tmp_path / FILM_REPO_PATH, FILM_MIN_BYTES)
    realesrgan_path = _write_sparse(tmp_path / _REALESRGAN_RELATIVE, REALESRGAN_ANIME_MIN_BYTES)
    resolved = resolve_augment_weights(tmp_path)
    assert resolved == AugmentWeights(film=film_path, realesrgan=realesrgan_path)


def test_string_models_dir_is_accepted(tmp_path: Path) -> None:
    """A `str` models dir resolves identically to its `Path` form."""
    _write_sparse(tmp_path / FILM_REPO_PATH, FILM_MIN_BYTES)
    resolved = resolve_augment_weights(str(tmp_path))
    assert resolved.film == tmp_path / FILM_REPO_PATH
    assert resolved.realesrgan is None


def test_non_path_models_dir_fails_loud() -> None:
    """A non-path models dir raises `TypeError` (never misread as a dir)."""
    with pytest.raises(TypeError, match="models_dir"):
        resolve_augment_weights(123)  # type: ignore[arg-type]


def _find_provisioned_weight(relative_path: str) -> Path | None:
    """First existing provisioned copy of `relative_path`, or None (skip signal)."""
    candidates = [
        os.environ.get("VOYAGE_MODELS", ""),
        "/models",
        str(Path.home() / ".cache" / "voyage-models"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        found = Path(candidate) / relative_path
        try:
            if found.is_file() and found.stat().st_size > 0:
                return found
        except OSError:
            continue
    return None


def test_provisioned_weights_load_through_resolved_paths() -> None:
    """Resolved provisioned weights strict-load and run synthetic tensors.

    Red pre-wiring (issue 166): no production seam resolves registry
    paths to the loaders. Green: the resolver finds both provisioned
    weights and each strict-loads plus runs a tiny CPU tensor. Skips
    (not fails) without torch+weights.
    """
    if importlib.util.find_spec("torch") is None:
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    film = _find_provisioned_weight(FILM_REPO_PATH)
    realesrgan = _find_provisioned_weight(_REALESRGAN_RELATIVE)
    if film is None or realesrgan is None:
        pytest.skip("needs provisioned film + realesrgan weights (models volume absent)")
    import torch

    from voyage.workers import augment_worker

    resolved = resolve_augment_weights(film.parent.parent)
    assert resolved.film == film
    assert resolved.realesrgan == realesrgan
    assert resolved.film is not None
    assert resolved.realesrgan is not None
    frame = torch.rand(3, 16, 16)
    upscaled = augment_worker.upscale_frames([frame], resolved.realesrgan, scale=2, device="cpu")
    assert upscaled[0].shape == (3, 32, 32)
    mid = augment_worker.interpolate_pair(frame, torch.rand(3, 16, 16), resolved.film, device="cpu")
    assert mid.shape == (3, 16, 16)
