"""Issue 166 finalize threading: explicit quality multipliers, 1/1 default (DESIGN §140).

`upscale`/`interpolate` (both default 1) thread `AugmentConfig` ->
`resolve_config` -> CLI (`--upscale`/`--interpolate`/`--presentation-fps`)
-> `FinalizeOptions` / `resolve_finalize_settings` -> `finalize_run`
(which consults `resolve_augment_weights` and keeps the ffmpeg path when
legs are absent or no work is demanded). Identity proofs (CPU-only, real
ffmpeg via the fake-backend commit path): 1/1 output == explicit 1/1
output (the ffmpeg-only fast path), and demanded work with absent
weights still ships via the ffmpeg fallback (never an error). The
present-weights tensor encode is the GPU-box residual (see the issue).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage.augment import AugmentWeights
from voyage.config import AugmentConfig, ProjectConfig, Unset
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _base_config() -> ProjectConfig:
    return ProjectConfig(style="knob-166-probe")


def _commit(run_dir: Path, count: int) -> None:
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        for _ in range(count):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def _sha256(candidate: Path) -> str:
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def test_multipliers_default_one_everywhere() -> None:
    """Every layer defaults to shipping the source as-is (DESIGN §140).

    1/1 means no work on either axis (ffmpeg path, never an error), so
    the default is safe on weight-free boxes.
    """
    from voyage.media import FinalizeOptions, resolve_finalize_settings

    assert AugmentConfig().upscale == 1
    assert AugmentConfig().interpolate == 1
    assert AugmentConfig().presentation_fps is None
    assert FinalizeOptions().upscale == 1
    assert FinalizeOptions().interpolate == 1
    assert FinalizeOptions().presentation_fps is None
    assert _base_config().augment.upscale == 1
    assert _base_config().augment.interpolate == 1
    resolved = resolve_finalize_settings(
        options=None,
        skip_bad=None,
        sample_rate=None,
        channels=None,
        overlap_fraction=None,
        overlap_cap_seconds=None,
        upscale=None,
        interpolate=None,
        crf=None,
        preset=None,
    )
    assert resolved.upscale == 1
    assert resolved.interpolate == 1
    assert resolved.settings.upscale == 1
    assert resolved.settings.interpolate == 1


def test_default_preset_ships_one_one(tmp_path: Path) -> None:
    """The preset ships with 1/1 multipliers."""
    from voyage.config import preset_config

    config = preset_config("knob166", "line art", 7)
    assert config.augment.upscale == 1
    assert config.augment.interpolate == 1


def test_multipliers_reject_bad_values() -> None:
    """Out-of-vocabulary multipliers fail loud before any media work."""
    from voyage.media import FinalizeOptions, resolve_finalize_settings

    with pytest.raises(ValueError, match="upscale"):
        FinalizeOptions(upscale=3)
    with pytest.raises(ValueError, match="interpolate"):
        FinalizeOptions(interpolate=0)
    with pytest.raises(ValueError, match="upscale"):
        resolve_finalize_settings(
            options=None,
            skip_bad=None,
            sample_rate=None,
            channels=None,
            overlap_fraction=None,
            overlap_cap_seconds=None,
            upscale=3,
            interpolate=None,
            crf=None,
            preset=None,
        )


def test_finalize_run_rejects_bad_multiplier(tmp_path: Path) -> None:
    """`finalize_run` validates the multipliers before touching segments."""
    from voyage.media import finalize_run

    with pytest.raises(ValueError, match="upscale"):
        finalize_run(
            tmp_path / "run",
            tmp_path / "out.mp4",
            upscale="yes",  # type: ignore[arg-type]
        )


def test_config_threads_multipliers() -> None:
    """`resolve_config` carries explicit multipliers into `[augment]`."""
    from voyage.config import resolve_config

    assert resolve_config(_base_config(), upscale=2).augment.upscale == 2
    assert resolve_config(_base_config(), interpolate=2).augment.interpolate == 2
    assert resolve_config(_base_config(), upscale=None).augment.upscale == 1
    assert resolve_config(_base_config(), interpolate=Unset).augment.interpolate == 1
    resolved = resolve_config(_base_config(), upscale=2, interpolate=4)
    assert resolved.augment.upscale == 2
    assert resolved.augment.interpolate == 4


def test_cli_flag_parity_and_mapping() -> None:
    """Finalize-time verbs accept `--upscale/--interpolate`; overrides map them."""
    from voyage.cli import build_parser
    from voyage.cli_core import _augment_overrides

    parser = build_parser()
    args = parser.parse_args(["configure", "calm", "--segments", "1"])
    assert args.upscale is None
    assert args.interpolate is None
    flagged = parser.parse_args(
        ["configure", "calm", "--segments", "1", "--upscale", "2", "--interpolate", "2"]
    )
    assert flagged.upscale == 2
    assert flagged.interpolate == 2
    assert _augment_overrides(argparse.Namespace()) == {}
    assert _augment_overrides(argparse.Namespace(upscale=None)) == {}
    assert _augment_overrides(argparse.Namespace(upscale=Unset)) == {}
    assert _augment_overrides(argparse.Namespace(upscale=2)) == {"upscale": 2}
    assert _augment_overrides(
        argparse.Namespace(upscale=2, interpolate=4, presentation_fps=32)
    ) == {"upscale": 2, "interpolate": 4, "presentation_fps": 32}


def test_finalize_one_one_byte_identical_to_explicit(tmp_path: Path) -> None:
    """Default 1/1 ships the same bytes as explicit 1/1 (same native encode)."""
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="knob166", style="pastel neon line-art, peaceful")
    _commit(run_dir, 1)
    out_default = tmp_path / "off-default.mp4"
    out_explicit = tmp_path / "off-explicit.mp4"
    finalize_run(run_dir, out_default)
    finalize_run(run_dir, out_explicit, upscale=1, interpolate=1)
    assert _sha256(out_default) == _sha256(out_explicit)


def test_finalize_demanded_work_absent_weights_falls_back(tmp_path: Path) -> None:
    """Demanded work with an empty models dir falls back to the ffmpeg bytes."""
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="knob166", style="pastel neon line-art, peaceful")
    _commit(run_dir, 1)
    empty_models = tmp_path / "empty-models"
    empty_models.mkdir()
    out = tmp_path / "demanded-absent.mp4"
    finalize_run(run_dir, out, upscale=2, models_dir=empty_models)
    assert out.exists() and out.stat().st_size > 0


def test_finalize_consults_seam_only_when_demanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1/1 never touches the registry seam; demanded work resolves once."""
    import voyage.augment as augment_module
    from voyage.config import preset_config, resolve_config
    from voyage.media import finalize_run
    from voyage.persistence import create_run_dir

    calls: list[str] = []

    def _recording(base: Path | str) -> AugmentWeights:
        calls.append(str(base))
        return AugmentWeights(film=None, realesrgan=None)

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _recording)
    run_dir = tmp_path / "run"
    stored = resolve_config(
        preset_config("knob166", "pastel neon line-art, peaceful", 7, video_backend="fake"),
        upscale=1,
        interpolate=1,
    )
    create_run_dir(run_dir, stored)
    _commit(run_dir, 1)
    empty_models = tmp_path / "empty-models"
    empty_models.mkdir()
    finalize_run(run_dir, tmp_path / "seam-off.mp4")
    assert calls == []
    finalize_run(run_dir, tmp_path / "seam-on.mp4", upscale=2, models_dir=empty_models)
    assert calls == [str(empty_models)]
