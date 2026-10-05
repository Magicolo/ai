"""Batched FILM interpolation: `interpolate_mids` validation + equivalence.

`interpolate_mids` evaluates every adjacent pair at all blend moments with
one flow per pair (via `forward_multi_timestep`), replacing the old
per-pair `interpolate_pair` loops at the 4 caller sites (augment
`enhance_frames`, interp poller, morph, seam). Validation runs torch-free
(moments, pair batch, frame count, weights) so the slim gates image covers
it; provisioned legs prove exact equivalence with the `interpolate_pair`
loop on CPU fp32 (every FILM op is per-sample — no batchnorm — so batching
is exact on deterministic devices).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from voyage.workers import augment_worker


def _torch_available() -> bool:
    """Whether real torch imports (slim gates image: no — provisioned legs skip)."""
    return importlib.util.find_spec("torch") is not None


def test_interpolate_mids_rejects_non_list_moments() -> None:
    """Moments must be a list or tuple of blend times (torch-free)."""
    with pytest.raises(TypeError, match="list or tuple"):
        augment_worker.interpolate_mids([], "/models/film.safetensors", moments="0.5")  # type: ignore[arg-type]


def test_interpolate_mids_rejects_empty_moments() -> None:
    """At least one blend moment is required (torch-free)."""
    with pytest.raises(ValueError, match="at least one blend moment"):
        augment_worker.interpolate_mids([], "/models/film.safetensors", moments=[])


def test_interpolate_mids_rejects_out_of_range_moment() -> None:
    """Blend moments inherit the [0, 1] range check (torch-free)."""
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        augment_worker.interpolate_mids([], "/models/film.safetensors", moments=[1.5])


def test_interpolate_mids_rejects_bad_pair_batch() -> None:
    """Pair batch must be a positive int — bools and zeros fail (torch-free)."""
    with pytest.raises(TypeError, match="must be an int"):
        augment_worker.interpolate_mids(
            [], "/models/film.safetensors", moments=[0.5], pair_batch=True
        )
    with pytest.raises(ValueError, match="must be positive"):
        augment_worker.interpolate_mids([], "/models/film.safetensors", moments=[0.5], pair_batch=0)


def test_interpolate_mids_rejects_non_callable_on_pair() -> None:
    """`on_pair` must be callable or None (torch-free)."""
    with pytest.raises(TypeError, match="callable or None"):
        augment_worker.interpolate_mids(
            [],
            "/models/film.safetensors",
            moments=[0.5],
            on_pair="fired",  # type: ignore[arg-type]
        )


def test_interpolate_mids_rejects_fewer_than_two_frames() -> None:
    """One frame has no adjacent pair (torch-free bare length check)."""
    with pytest.raises(ValueError, match="at least two frames"):
        augment_worker.interpolate_mids(["only"], "/models/film.safetensors", moments=[0.5])


def test_interpolate_mids_missing_weights_fails_torch_free(tmp_path: Path) -> None:
    """Absent weights raise before any torch import (provisioning error)."""
    missing = tmp_path / "film_net_fp16.safetensors"
    with pytest.raises(NotImplementedError, match="weights missing"):
        augment_worker.interpolate_mids(["frame-a", "frame-b"], missing, moments=[0.5])


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


def _gradient_frames(count: int) -> list[Any]:
    """Deterministic 16x16 RGB gradient frames (CPU, no PIL/numpy needed)."""
    import torch

    frames = []
    for offset in range(count):
        rows = torch.linspace(0.0, 1.0, 16).unsqueeze(1).expand(16, 16)
        tilt = (rows + offset / max(count, 1)) % 1.0
        frames.append(torch.stack([tilt, rows, 1.0 - rows]))
    return frames


def test_provisioned_mids_match_pair_loop() -> None:
    """Batched mids equal the per-pair loop exactly (CPU fp32, single moment)."""
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    if importlib.util.find_spec("safetensors") is None:
        pytest.skip("needs safetensors (run in voyage-video)")
    from voyage.registry_film import FILM_REPO_PATH

    film = _find_provisioned_weight(FILM_REPO_PATH)
    if film is None:
        pytest.skip("needs provisioned film weights (models volume absent)")
    assert film is not None
    import torch

    augment_worker.evict_augment_models()
    frames = _gradient_frames(4)
    expected = [
        augment_worker.interpolate_pair(frames[index], frames[index + 1], film, device="cpu")
        for index in range(3)
    ]
    actual = augment_worker.interpolate_mids(frames, film, moments=[0.5], device="cpu")
    assert len(actual) == 3
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


def test_provisioned_mids_pair_major_multi_moment() -> None:
    """Multi-moment output is pair-major, batch-1 equals batch-2, on_pair fires in order."""
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    if importlib.util.find_spec("safetensors") is None:
        pytest.skip("needs safetensors (run in voyage-video)")
    from voyage.registry_film import FILM_REPO_PATH

    film = _find_provisioned_weight(FILM_REPO_PATH)
    if film is None:
        pytest.skip("needs provisioned film weights (models volume absent)")
    assert film is not None
    import torch

    augment_worker.evict_augment_models()
    frames = _gradient_frames(4)
    moments = [0.25, 0.5, 0.75]
    fired: list[tuple[int, int]] = []
    batched = augment_worker.interpolate_mids(
        frames,
        film,
        moments=moments,
        device="cpu",
        on_pair=lambda index, total: fired.append((index, total)),
    )
    single = augment_worker.interpolate_mids(
        frames, film, moments=moments, pair_batch=1, device="cpu"
    )
    assert len(batched) == 9
    assert fired == [(0, 3), (1, 3), (2, 3)]
    for got, want in zip(batched, single, strict=True):
        assert torch.equal(got, want)
    for pair_index in range(3):
        for moment_index, moment in enumerate(moments):
            want = augment_worker.interpolate_pair(
                frames[pair_index], frames[pair_index + 1], film, moment=moment, device="cpu"
            )
            assert torch.equal(batched[pair_index * 3 + moment_index], want)


def test_provisioned_mids_report_timings() -> None:
    """Timings carry both load and inference milliseconds."""
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    if importlib.util.find_spec("safetensors") is None:
        pytest.skip("needs safetensors (run in voyage-video)")
    from voyage.registry_film import FILM_REPO_PATH

    film = _find_provisioned_weight(FILM_REPO_PATH)
    if film is None:
        pytest.skip("needs provisioned film weights (models volume absent)")
    assert film is not None

    augment_worker.evict_augment_models()
    timings: dict[str, float] = {}
    augment_worker.interpolate_mids(
        _gradient_frames(3), film, moments=[0.5], device="cpu", timings=timings
    )
    assert set(timings) == {"load_ms", "infer_ms"}
    assert timings["load_ms"] >= 0.0
    assert timings["infer_ms"] >= 0.0
    augment_worker.evict_augment_models()
