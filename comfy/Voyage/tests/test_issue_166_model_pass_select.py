"""Issue 166 final slice: present-legs tensor selection (CPU-safe).

Provisioned legs select the tensor chunk encode (`enhance_frames` via
SRVGG + FILM on the chunk device); absent legs keep the ffmpeg fallback
byte-identical. Fallback legs run torch-free in the slim gates image;
provisioned legs skip loudly without torch+weights (run them in
`voyage-video` with the models volume).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from voyage.augment import AugmentWeights
from voyage.persistence import read_effective_config


def _torch_available() -> bool:
    """Whether real torch imports (slim gates image: no)."""
    return importlib.util.find_spec("torch") is not None


def test_model_pass_selector_exists_and_defaults_off() -> None:
    """TDD red anchor: the present-legs selector does not exist yet."""
    from voyage import augment as augment_module

    assert hasattr(augment_module, "model_pass_active")
    assert augment_module.model_pass_active(AugmentWeights(film=None, realesrgan=None)) is False


def test_model_pass_selector_needs_a_leg() -> None:
    """The selector is True when at least one leg is provisioned."""
    from voyage.augment import model_pass_active

    both = AugmentWeights(film=Path("/models/film"), realesrgan=Path("/models/esrgan"))
    assert model_pass_active(both) is True
    assert model_pass_active(AugmentWeights(film=Path("/models/film"), realesrgan=None)) is True
    assert model_pass_active(AugmentWeights(film=None, realesrgan=Path("/models/esrgan"))) is True
    assert model_pass_active(AugmentWeights(film=None, realesrgan=None)) is False


def test_model_pass_selector_rejects_bad_inputs() -> None:
    """Wrong-typed weights fail loud before any model work."""
    from voyage.augment import model_pass_active

    with pytest.raises(TypeError, match="weights"):
        model_pass_active({"film": None})  # type: ignore[arg-type]


def test_finalize_present_legs_selects_tensor_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Demanded work + legs calls the tensor model pass (not the vf-only fallback).

    The pass is trigger-gated: demanded work (`upscale > 1`) on the
    768x432@24 fake source flags reencode work (a 1/1 request would take
    the stream-copy fast path — see
    tests/test_native_skips_model_pass.py). Both legs provisioned select
    the interleaved durable pass (one call, both legs on the 2060).
    """
    import shutil

    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    from tests.conftest import initialize_run_directory
    from voyage.media import finalize_run
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="select166", style="pastel neon line-art, peaceful")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()

    seen: dict[str, Any] = {}

    def _fake_resolve(base: Path | str) -> AugmentWeights:
        return AugmentWeights(
            film=Path(str(base) + "/film"),
            realesrgan=Path(str(base) + "/esrgan"),
            rife=Path(str(base) + "/rife"),
        )

    def _fake_durable_pass(
        run_dir_arg: Path,
        usable: list[Path],
        *,
        weights: AugmentWeights,
        out_width: int,
        out_height: int,
        source_fps: float,
        upscale_factor: int = 2,
        multiplier: int = 4,
        chunk_frames: int = 32,
        crf: int = 15,
        preset: str = "veryfast",
        device: str = "cuda:1",
        work_dir: Path,
        interp_backend: str = "rife",
        upscale_device: str | None = None,
        interp_device: str | None = None,
        timings: dict[str, float] | None = None,
        progress: Any = None,
    ) -> tuple[Path, int]:
        seen["upscale_called"] = True
        seen["interp_called"] = True
        seen["segments"] = len(usable)
        work_dir.mkdir(parents=True, exist_ok=True)
        intermediate = work_dir / "model_intermediate.mp4"
        shutil.copy(usable[0] / "video.mp4", intermediate)
        return (intermediate, int(round(source_fps)))

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _fake_resolve)
    monkeypatch.setattr(finalize_module, "run_durable_model_pass", _fake_durable_pass)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out = tmp_path / "tensor-selected.mp4"
    finalize_run(
        run_dir,
        out,
        upscale=2,
        interpolate=1,
        models_dir=models_dir,
    )
    assert out.exists() and out.stat().st_size > 0
    assert seen.get("upscale_called") is True
    assert seen.get("interp_called") is True
    assert seen.get("segments") == 1


def test_finalize_absent_legs_never_enhances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Demanded work but absent legs never touches the tensor model pass (fallback)."""
    import voyage.augment as augment_module
    from tests.conftest import initialize_run_directory
    from voyage.media import finalize_run
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="select166", style="pastel neon line-art, peaceful")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()

    def _forbidden_model_pass(
        segment_videos: list[Path],
        weights: AugmentWeights,
        *,
        source_fps: float,
        upscale_factor: int = 2,
        multiplier: int = 4,
        crf: int = 15,
        preset: str = "veryfast",
        work_dir: Path,
        chunk_frames: int = 32,
        devices: tuple[str, ...] | None = None,
    ) -> tuple[Path, int]:
        raise AssertionError("tensor model pass must not run when legs are absent")

    monkeypatch.setattr(augment_module, "run_finalize_model_pass", _forbidden_model_pass)
    empty_models = tmp_path / "empty-models"
    empty_models.mkdir()
    out = tmp_path / "fallback-absent.mp4"
    finalize_run(
        run_dir,
        out,
        upscale=2,
        interpolate=1,
        models_dir=empty_models,
    )
    assert out.exists() and out.stat().st_size > 0


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


def test_provisioned_selector_matches_resolved_legs() -> None:
    """Resolved provisioned legs select; empty dir does not (needs weights)."""
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    from voyage.augment import model_pass_active, resolve_augment_weights
    from voyage.registry_film import FILM_REPO_PATH
    from voyage.registry_realesrgan import REALESRGAN_ANIME_FILE, REALESRGAN_SUBDIR

    film = _find_provisioned_weight(FILM_REPO_PATH)
    realesrgan = _find_provisioned_weight(f"{REALESRGAN_SUBDIR}/{REALESRGAN_ANIME_FILE}")
    if film is None or realesrgan is None:
        pytest.skip("needs provisioned film + realesrgan weights (models volume absent)")
    assert film is not None
    assert realesrgan is not None
    resolved = resolve_augment_weights(film.parent.parent)
    assert model_pass_active(resolved) is True
    assert model_pass_active(AugmentWeights(film=None, realesrgan=None)) is False
