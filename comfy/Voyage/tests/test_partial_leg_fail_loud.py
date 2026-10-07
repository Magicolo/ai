"""Issue 215: partial augment stacks fail loud (never the unbounded legacy flow).

A half-provisioned stack (interp leg xor ESRGAN leg) used to route into the
legacy all-at-once `run_finalize_model_pass`, which decodes every segment to
PNGs and loads every tensor at once (~400 GB staging at 256 segments). The
durable sidecar keys ledgers on both legs and cannot take partial runs, so
`finalize_run` now refuses them with a `MediaError` naming both remedies
(provision the missing leg, or set upscale=1/interpolate=1 for the bounded
ffmpeg fallback). CPU-only: the legacy and durable passes are both stubbed
forbidden — the test pins the code path, never pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.augment import AugmentWeights
from voyage.persistence import read_effective_config


def _commit_fake_run(tmp_path: Path) -> Path:
    """Commit one fake-backend segment; returns the run dir."""
    from tests.conftest import initialize_run_directory
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="partial215", style="pastel neon line-art, peaceful")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    return run_dir


def _run_finalize_expecting_partial_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    weights: AugmentWeights,
    missing_match: str,
) -> None:
    """`finalize_run` with partial `weights` raises before any model pass."""
    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    import voyage.augment_parallel as parallel_module
    import voyage.media as media_module
    from voyage.errors import MediaError
    from voyage.media import finalize_run

    def _fake_resolve(base: Path | str) -> AugmentWeights:
        del base
        return weights

    def _forbidden_durable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("durable pass must not run for partial legs")

    def _forbidden_legacy(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("legacy all-at-once pass must not run for partial legs")

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _fake_resolve)
    monkeypatch.setattr(finalize_module, "run_durable_model_pass", _forbidden_durable)
    monkeypatch.setattr(augment_module, "run_finalize_model_pass", _forbidden_legacy)
    monkeypatch.setattr(parallel_module, "parallel_model_pass_armed", lambda **kwargs: False)
    monkeypatch.setattr(media_module, "model_music_parallel_armed", lambda **kwargs: False)

    run_dir = _commit_fake_run(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with pytest.raises(MediaError, match=missing_match):
        finalize_run(
            run_dir,
            tmp_path / "partial.mp4",
            upscale=2,
            interpolate=2,
            models_dir=models_dir,
        )


def test_esrgan_only_finalize_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RIFE/backend interp leg missing: fail loud, never silently unbounded."""
    _run_finalize_expecting_partial_error(
        tmp_path,
        monkeypatch,
        AugmentWeights(film=None, rife=None, realesrgan=Path("/models/esrgan")),
        "rife interp weights",
    )


def test_interp_only_finalize_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ESRGAN leg missing: fail loud, never silently unbounded."""
    _run_finalize_expecting_partial_error(
        tmp_path,
        monkeypatch,
        AugmentWeights(film=None, rife=Path("/models/rife"), realesrgan=None),
        "realesrgan weights",
    )


def test_partial_error_names_both_remedies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The error tells the operator how out: provision or drop to ffmpeg."""
    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    import voyage.augment_parallel as parallel_module
    import voyage.media as media_module
    from voyage.errors import MediaError
    from voyage.media import finalize_run

    monkeypatch.setattr(
        augment_module,
        "resolve_augment_weights",
        lambda base: AugmentWeights(film=None, rife=None, realesrgan=Path("/models/x")),
    )
    monkeypatch.setattr(
        finalize_module,
        "run_durable_model_pass",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no durable")),
    )
    monkeypatch.setattr(parallel_module, "parallel_model_pass_armed", lambda **kwargs: False)
    monkeypatch.setattr(media_module, "model_music_parallel_armed", lambda **kwargs: False)

    run_dir = _commit_fake_run(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with pytest.raises(MediaError) as excinfo:
        finalize_run(
            run_dir,
            tmp_path / "partial.mp4",
            upscale=2,
            interpolate=2,
            models_dir=models_dir,
        )
    message = str(excinfo.value)
    assert "rife interp weights" in message
    assert "upscale=1/interpolate=1" in message
    assert "ffmpeg fallback" in message
