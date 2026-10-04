"""Issue 166 finalize threading: model-augment knob, default-on (DESIGN §140).

`use_model_pass` (default True) threads `AugmentConfig` -> `resolve_config`
-> CLI (`--use-model-pass`) + TUI form -> `FinalizeOptions` /
`resolve_finalize_settings` -> `finalize_run` (which consults
`resolve_augment_weights` and keeps the ffmpeg vf path when legs are
absent). Identity proofs (CPU-only, real ffmpeg via the fake-backend
commit path): knob-off output == pre-knob output, and knob-on with
absent weights == knob-off (same bytes). The present-weights tensor
encode is the GPU-box residual (see the issue).
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
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        for _ in range(count):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def _sha256(candidate: Path) -> str:
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def test_knob_defaults_on_everywhere() -> None:
    """Every layer defaults to the model pass when provisioned (DESIGN §140).

    Absent legs still read as the ffmpeg path (never an error), so the
    default is safe on weight-free boxes — `--no-augment` opts out.
    """
    from voyage.media import FinalizeOptions, resolve_finalize_settings

    assert AugmentConfig().use_model_pass is True
    assert FinalizeOptions().use_model_pass is True
    assert _base_config().augment.use_model_pass is True
    resolved = resolve_finalize_settings(
        options=None,
        skip_bad=None,
        sample_rate=None,
        channels=None,
        overlap_fraction=None,
        overlap_cap_seconds=None,
        min_fps=None,
        min_width=None,
        min_height=None,
        crf=None,
        preset=None,
    )
    assert resolved.use_model_pass is True
    assert resolved.settings.use_model_pass is True


def test_default_toml_leaves_knob_on(tmp_path: Path) -> None:
    """The preset ships with the knob on."""
    from voyage.config import preset_config

    config = preset_config("knob166", "line art", 7)
    assert config.augment.use_model_pass is True


def test_knob_rejects_non_bool() -> None:
    """Wrong-typed knob values fail loud before any media work."""
    from voyage.media import FinalizeOptions, resolve_finalize_settings

    with pytest.raises(TypeError, match="use_model_pass"):
        FinalizeOptions(use_model_pass="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="use_model_pass"):
        resolve_finalize_settings(
            options=None,
            skip_bad=None,
            sample_rate=None,
            channels=None,
            overlap_fraction=None,
            overlap_cap_seconds=None,
            min_fps=None,
            min_width=None,
            min_height=None,
            crf=None,
            preset=None,
            use_model_pass="yes",  # type: ignore[arg-type]
        )


def test_finalize_run_rejects_non_bool_knob(tmp_path: Path) -> None:
    """`finalize_run` validates the knob before touching segments."""
    from voyage.media import finalize_run

    with pytest.raises(TypeError, match="use_model_pass"):
        finalize_run(
            tmp_path / "run",
            tmp_path / "out.mp4",
            use_model_pass="yes",  # type: ignore[arg-type]
        )


def test_config_threads_knob() -> None:
    """`resolve_config` carries an explicit knob into `[augment]`."""
    from voyage.config import resolve_config

    assert resolve_config(_base_config(), use_model_pass=True).augment.use_model_pass is True
    assert resolve_config(_base_config(), use_model_pass=False).augment.use_model_pass is False
    assert resolve_config(_base_config(), use_model_pass=None).augment.use_model_pass is True
    assert resolve_config(_base_config(), use_model_pass=Unset).augment.use_model_pass is True
    resolved = resolve_config(_base_config(), use_model_pass=True, min_fps=60)
    assert resolved.augment.use_model_pass is True
    assert resolved.augment.min_fps == 60


def test_cli_flag_parity_and_mapping() -> None:
    """All finalizing verbs accept `--use-model-pass`; overrides map it."""
    from voyage.cli import build_parser
    from voyage.cli_core import _augment_overrides

    parser = build_parser()
    for verb in (
        ["finalize", "--run", "r", "--output", "o.mp4"],
        ["generate", "--style", "x", "--duration", "5s"],
        ["run", "--run", "r"],
        ["stop", "--run", "r"],
    ):
        assert parser.parse_args(verb).use_model_pass is None
        flagged = [*verb, "--use-model-pass"]
        assert parser.parse_args(flagged).use_model_pass is True
    assert _augment_overrides(argparse.Namespace()) == {}
    assert _augment_overrides(argparse.Namespace(use_model_pass=None)) == {}
    assert _augment_overrides(argparse.Namespace(use_model_pass=Unset)) == {}
    assert _augment_overrides(argparse.Namespace(use_model_pass=True)) == {"use_model_pass": True}
    assert _augment_overrides(argparse.Namespace(use_model_pass=True, no_augment=True)) == {
        "min_fps": 0,
        "min_resolution": "0",
        "use_model_pass": False,
    }


def test_tui_threads_knob() -> None:
    """The TUI form carries the knob: checked (default) = on, unchecked = stored wins."""
    from voyage.tui_state import GenerateFormState, to_generate_namespace

    assert GenerateFormState().use_model_pass is True
    namespace = to_generate_namespace(GenerateFormState(style="x", backend="fake"))
    assert namespace.use_model_pass is True
    unchecked = to_generate_namespace(
        GenerateFormState(style="x", backend="fake", use_model_pass=False)
    )
    assert unchecked.use_model_pass is Unset


def test_tui_knob_settings_round_trip(tmp_path: Path) -> None:
    """Last-settings persistence keeps the checked box."""
    from voyage.tui_state import GenerateFormState, load_last_settings, save_last_settings

    settings_file = tmp_path / "tui-last.toml"
    state = GenerateFormState(style="x", use_model_pass=True)
    save_last_settings(state, settings_file)
    assert load_last_settings(settings_file) == state


def test_finalize_knob_off_byte_identical_to_default(tmp_path: Path) -> None:
    """Explicit `use_model_pass=False` ships the pre-knob bytes (fast path)."""
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="knob166", style="pastel neon line-art, peaceful")
    _commit(run_dir, 1)
    out_default = tmp_path / "off-default.mp4"
    out_explicit = tmp_path / "off-explicit.mp4"
    finalize_run(run_dir, out_default, min_fps=0, min_width=0, min_height=0)
    finalize_run(run_dir, out_explicit, min_fps=0, min_width=0, min_height=0, use_model_pass=False)
    assert _sha256(out_default) == _sha256(out_explicit)


def test_finalize_knob_on_absent_weights_byte_identical_to_off(tmp_path: Path) -> None:
    """Knob on with an empty models dir falls back to the ffmpeg bytes."""
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="knob166", style="pastel neon line-art, peaceful")
    _commit(run_dir, 1)
    empty_models = tmp_path / "empty-models"
    empty_models.mkdir()
    out_off = tmp_path / "knob-off.mp4"
    out_on = tmp_path / "knob-on-absent.mp4"
    finalize_run(run_dir, out_off)
    finalize_run(run_dir, out_on, use_model_pass=True, models_dir=empty_models)
    assert _sha256(out_off) == _sha256(out_on)


def test_finalize_consults_seam_only_when_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Knob off never touches the registry seam; knob on resolves once."""
    import voyage.augment as augment_module
    from voyage.media import finalize_run

    calls: list[str] = []

    def _recording(base: Path | str) -> AugmentWeights:
        calls.append(str(base))
        return AugmentWeights(film=None, realesrgan=None)

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _recording)
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="knob166", style="pastel neon line-art, peaceful")
    _commit(run_dir, 1)
    empty_models = tmp_path / "empty-models"
    empty_models.mkdir()
    finalize_run(run_dir, tmp_path / "seam-off.mp4")
    assert calls == []
    finalize_run(run_dir, tmp_path / "seam-on.mp4", use_model_pass=True, models_dir=empty_models)
    assert calls == [str(empty_models)]
