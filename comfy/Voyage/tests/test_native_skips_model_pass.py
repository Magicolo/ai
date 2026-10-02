"""Floors trigger augmentation: matching sources skip the tensor model pass.

Why this file exists: the augment floors (24fps / 1216x704 by default)
are the trigger for augmentation work. `use_model_pass=True` with
provisioned legs must still take the stream-copy fast path when the
augment plan flags nothing (`needs_reencode` False) — otherwise every
native ltx25 finalize would pay for a no-op enhance. A lift (geometry
or fps below the plan target) must still select the tensor path.
"""

from pathlib import Path
from typing import Any

import pytest

from voyage.augment import AugmentWeights


def _commit_fake_run(tmp_path: Path, run_id: str) -> Path:
    """Commit one fake-backend segment; returns the run dir."""
    from tests.conftest import initialize_run_directory
    from voyage import paths
    from voyage.config import load_config
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id=run_id, style="pastel neon line-art, peaceful")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    return run_dir


def _stub_weights(base: Path | str) -> AugmentWeights:
    """Legs that exist on paper only — never probed, only recorded."""
    return AugmentWeights(film=Path(str(base) + "/film"), realesrgan=Path(str(base) + "/esrgan"))


def test_finalize_matching_source_skips_tensor_model_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Knob-on + legs present but nothing to lift: no tensor run."""
    import voyage.augment as augment_module
    from voyage.media import finalize_run

    run_dir = _commit_fake_run(tmp_path, "notrigger")

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
        raise AssertionError("tensor model pass must not run when the plan flags no work")

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _stub_weights)
    monkeypatch.setattr(augment_module, "run_finalize_model_pass", _forbidden_model_pass)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out = tmp_path / "native-passthrough.mp4"
    # Fake segments are 768x432@24; matching request + disabled floors
    # means the plan flags nothing, so legs present must not matter.
    finalize_run(
        run_dir,
        out,
        width=768,
        height=432,
        fps=24,
        min_fps=0,
        min_width=0,
        min_height=0,
        use_model_pass=True,
        models_dir=models_dir,
    )
    assert out.exists() and out.stat().st_size > 0


def test_finalize_lift_still_selects_tensor_model_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same legs + a real lift: the tensor path still engages (stubbed)."""
    import shutil

    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    from voyage.media import finalize_run

    run_dir = _commit_fake_run(tmp_path, "trigger")

    seen: dict[str, Any] = {}

    def _recording_model_pass(
        run_dir_arg: Path,
        usable: list[Path],
        *,
        out_width: int,
        out_height: int,
        source_fps: float,
        weights: AugmentWeights,
        upscale_factor: int = 2,
        multiplier: int = 4,
        chunk_frames: int = 32,
        crf: int = 15,
        preset: str = "veryfast",
        device: str = "cuda:1",
        work_dir: Path,
        timings: dict[str, float] | None = None,
    ) -> tuple[Path, int]:
        seen["model_pass_called"] = True
        seen["segments"] = len(usable)
        work_dir.mkdir(parents=True, exist_ok=True)
        intermediate = work_dir / "model_intermediate.mp4"
        shutil.copy(usable[0] / "video.mp4", intermediate)
        return (intermediate, int(round(source_fps)))

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _stub_weights)
    monkeypatch.setattr(finalize_module, "run_durable_model_pass", _recording_model_pass)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out = tmp_path / "lifted-tensor.mp4"
    # Requesting above the 768x432@24 source flags reencode work, so the
    # tensor path must engage; the stub intermediate carries source
    # geometry and the presentation vf scales it to the request.
    finalize_run(
        run_dir,
        out,
        width=1216,
        height=704,
        fps=24,
        min_fps=0,
        min_width=0,
        min_height=0,
        use_model_pass=True,
        models_dir=models_dir,
    )
    assert out.exists() and out.stat().st_size > 0
    assert seen.get("model_pass_called") is True
    assert seen.get("segments") == 1
