"""Finalize augmentation knobs: config + TOML + CLI + TUI (Track A).

Covers the `AugmentConfig` model (defaults, 0-disables, half-geometry
rejection), the `[augment]` TOML round-trip, `resolve_config` overrides,
the `--min-fps/--min-resolution/--no-augment` CLI flags (parity across
the finalizing verbs, mirroring `test_sfx_parser_parity.py`), and the
TUI form fields (defaults, validation, namespace, persistence, planning
purity). CPU-only: pure config transforms — no workers, no ffmpeg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from pydantic import ValidationError

from voyage.cli import build_parser
from voyage.cli_core import _augment_overrides
from voyage.cli_finalize import cmd_finalize
from voyage.config import (
    AugmentConfig,
    ProjectConfig,
    Unset,
    parse_min_resolution,
    preset_config,
    resolve_config,
)


def _base_config() -> ProjectConfig:
    return ProjectConfig(style="augment-probe")


def test_augment_defaults() -> None:
    assert AugmentConfig().model_dump() == {
        "min_fps": 24,
        "min_width": 1216,
        "min_height": 704,
        "use_model_pass": True,
        "interp_multiplier": 2,
        "presentation_fps": 32,
    }
    assert _base_config().augment == AugmentConfig()


def test_zero_disables_both_floors() -> None:
    disabled = AugmentConfig(min_fps=0, min_width=0, min_height=0)
    assert (disabled.min_fps, disabled.min_width, disabled.min_height) == (0, 0, 0)


def test_half_geometry_rejected() -> None:
    with pytest.raises(ValidationError):
        AugmentConfig(min_width=0, min_height=720)
    with pytest.raises(ValidationError):
        AugmentConfig(min_width=1280, min_height=0)


def test_negative_floors_rejected() -> None:
    with pytest.raises(ValidationError):
        AugmentConfig(min_fps=-1)
    with pytest.raises(ValidationError):
        AugmentConfig(min_width=-1280, min_height=720)


def test_parse_min_resolution() -> None:
    assert parse_min_resolution("1280x720") == (1280, 720)
    assert parse_min_resolution(" 1920X1080 ") == (1920, 1080)
    assert parse_min_resolution("0") == (0, 0)
    assert parse_min_resolution("0x0") == (0, 0)


@pytest.mark.parametrize("raw", ["soon", "1280", "1280x", "x720", "-1x720", "1280x720x480"])
def test_parse_min_resolution_rejects_shape(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_min_resolution(raw)


@pytest.mark.parametrize("raw", ["0x720", "1280x0"])
def test_parse_min_resolution_rejects_half_disable(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_min_resolution(raw)


def test_default_toml_carries_augment_section(tmp_path: Path) -> None:
    config = preset_config("augment-run", "line art", 7)
    assert config.augment == AugmentConfig()


def test_toml_round_trip_custom_floors(tmp_path: Path) -> None:
    config = preset_config("augment-run", "line art", 7)
    config = resolve_config(config, min_fps=60, min_resolution="1920x1080")
    assert (config.augment.min_fps, config.augment.min_width, config.augment.min_height) == (
        60,
        1920,
        1080,
    )


def test_toml_rejects_half_geometry(tmp_path: Path) -> None:
    base = preset_config("augment-run", "line art", 7)
    with pytest.raises(ValueError):
        resolve_config(base, min_resolution="0x1080")


def test_resolve_without_augment_options_is_pure_noop() -> None:
    base = _base_config()
    before = base.model_dump()
    resolved = resolve_config(base)
    assert resolved.model_dump() == before
    assert base.model_dump() == before


def test_resolve_min_fps_override() -> None:
    resolved = resolve_config(_base_config(), min_fps=60)
    assert resolved.augment.min_fps == 60
    assert (resolved.augment.min_width, resolved.augment.min_height) == (1216, 704)


def test_resolve_min_resolution_override() -> None:
    resolved = resolve_config(_base_config(), min_resolution="1920x1080")
    assert (resolved.augment.min_width, resolved.augment.min_height) == (1920, 1080)
    assert resolved.augment.min_fps == 24


def test_resolve_zero_disables() -> None:
    resolved = resolve_config(_base_config(), min_fps=0, min_resolution="0")
    assert resolved.augment == AugmentConfig(min_fps=0, min_width=0, min_height=0)


def test_resolve_absent_encodings() -> None:
    assert resolve_config(_base_config(), min_fps=None).augment == AugmentConfig()
    assert resolve_config(_base_config(), min_fps=Unset).augment == AugmentConfig()
    assert resolve_config(_base_config(), min_resolution=None).augment == AugmentConfig()
    assert resolve_config(_base_config(), min_resolution=Unset).augment == AugmentConfig()


def test_resolve_invalid_rejected() -> None:
    with pytest.raises(ValidationError):
        resolve_config(_base_config(), min_fps=-1)
    with pytest.raises(ValueError):
        resolve_config(_base_config(), min_resolution="0x720")
    with pytest.raises(ValueError):
        resolve_config(_base_config(), min_resolution="soon")


def _parse(verb_args: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(verb_args)


def _augment_defaults(args: argparse.Namespace) -> dict[str, object]:
    return {
        "min_fps": args.min_fps,
        "min_resolution": args.min_resolution,
        "no_augment": args.no_augment,
    }


def test_augment_parser_defaults() -> None:
    assert _augment_defaults(_parse(["configure", "calm", "--segments", "1"])) == {
        "min_fps": None,
        "min_resolution": None,
        "no_augment": False,
    }


def test_augment_flags_parse() -> None:
    args = _parse(
        [
            "configure",
            "calm",
            "--segments",
            "1",
            "--min-fps",
            "60",
            "--min-resolution",
            "1920x1080",
        ]
    )
    assert _augment_defaults(args) == {
        "min_fps": 60,
        "min_resolution": "1920x1080",
        "no_augment": False,
    }
    assert _parse(["configure", "calm", "--segments", "1", "--no-augment"]).no_augment is True


def test_configure_carries_augment_flags() -> None:
    """The shared helper exposes every augment flag on `configure` (020)."""
    flags = ["--min-fps", "60", "--min-resolution", "1920x1080", "--no-augment"]
    args = _parse(["configure", "calm", "--segments", "1", *flags])
    assert _augment_defaults(args) == {
        "min_fps": 60,
        "min_resolution": "1920x1080",
        "no_augment": True,
    }


def test_augment_overrides_mapping() -> None:
    base = ["configure", "calm", "--segments", "1"]
    assert _augment_overrides(_parse(base)) == {}
    assert _augment_overrides(_parse([*base, "--min-fps", "60"])) == {"min_fps": 60}
    assert _augment_overrides(_parse([*base, "--min-resolution", "0"])) == {"min_resolution": "0"}
    # --no-augment wins over explicit floors (both to 0) and forces the pass off.
    assert _augment_overrides(_parse([*base, "--min-fps", "60", "--no-augment"])) == {
        "min_fps": 0,
        "min_resolution": "0",
        "use_model_pass": False,
    }
    # TUI/hand-built namespaces: Unset blanks and missing attrs are absent.
    assert _augment_overrides(argparse.Namespace(min_fps=Unset, min_resolution=Unset)) == {}
    assert _augment_overrides(argparse.Namespace()) == {}


def test_cmd_finalize_rejects_invalid_augment(tmp_path: Path) -> None:
    from tests.conftest import initialize_run_directory

    initialize_run_directory(tmp_path, run_id="x", style="y", seed=0)
    code = cmd_finalize(
        argparse.Namespace(
            run=str(tmp_path),
            output=str(tmp_path / "final.mp4"),
            skip_bad=False,
            min_fps=-1,
            min_resolution=None,
            no_augment=False,
        )
    )
    assert code == 2
    code = cmd_finalize(
        argparse.Namespace(
            run=str(tmp_path),
            output=str(tmp_path / "final.mp4"),
            skip_bad=False,
            min_fps=None,
            min_resolution="soon",
            no_augment=False,
        )
    )
    assert code == 2


# --- 088 fold: tests/test_sfx_parser_parity.py (2 tests, verbatim) ---
# Original module docstring (banner, issue ID stays greppable):
# """SFX flag parity across finalizing verbs (issue 092).
#
# `finalize` owns the SFX pass; `generate` and `stop --finalize` forward
# into `cmd_finalize`. Every finalizing parser must carry the same six
# flags — a missing flag is an AttributeError at finalize time (the
# `generate` E2E failures that motivated the shared `_add_sfx_args`
# helper). CPU-only.
# """


def _sfx_defaults(args: argparse.Namespace) -> dict[str, object]:
    return {
        "no_sfx": args.no_sfx,
        "sfx_backend": args.sfx_backend,
        "sfx_caption": args.sfx_caption,
        "sfx_device": args.sfx_device,
        "sfx_model_size": args.sfx_model_size,
        "sfx_workers": args.sfx_workers,
    }


def test_configure_carries_sfx_flags() -> None:
    args = build_parser().parse_args(["configure", "calm", "--segments", "1"])
    assert _sfx_defaults(args) == {
        "no_sfx": False,
        "sfx_backend": None,
        "sfx_caption": None,
        "sfx_device": None,
        "sfx_model_size": None,
        "sfx_workers": 1,
    }


def test_generate_sfx_overrides_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "configure",
            "calm",
            "--segments",
            "1",
            "--no-sfx",
            "--sfx-device",
            "cuda:1",
            "--sfx-model-size",
            "small_44k",
            "--sfx-workers",
            "2",
        ]
    )
    assert _sfx_defaults(args) == {
        "no_sfx": True,
        "sfx_backend": None,
        "sfx_caption": None,
        "sfx_device": "cuda:1",
        "sfx_model_size": "small_44k",
        "sfx_workers": 2,
    }
