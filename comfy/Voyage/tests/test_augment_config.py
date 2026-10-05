"""Finalize explicit-quality knobs: config + CLI + persistence (Track A).

Covers the `AugmentConfig` triple (exact dump, worker-vocab validation),
`resolve_config` overrides, the `--upscale/--interpolate/--presentation-fps`
CLI flags on `configure`, the removed floor flags (fail at the parser),
and the fail-loud legacy-augment-key guard in `read_effective_config`.
CPU-only: pure config transforms — no workers, no ffmpeg.
"""

from __future__ import annotations

import argparse
import json
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
    preset_config,
    resolve_config,
)


def _base_config() -> ProjectConfig:
    return ProjectConfig(style="augment-probe")


def test_augment_defaults() -> None:
    assert AugmentConfig().model_dump() == {
        "upscale": 1,
        "interpolate": 1,
        "presentation_fps": None,
    }
    assert _base_config().augment == AugmentConfig()


@pytest.mark.parametrize("multiplier", [1, 2, 4])
def test_upscale_accepts_worker_vocab(multiplier: int) -> None:
    assert AugmentConfig(upscale=multiplier).upscale == multiplier


@pytest.mark.parametrize("multiplier", [0, 3, 5, -1])
def test_upscale_rejects_outside_vocab(multiplier: int) -> None:
    with pytest.raises(ValidationError):
        AugmentConfig(upscale=multiplier)


@pytest.mark.parametrize("multiplier", [1, 2, 4])
def test_interpolate_accepts_worker_vocab(multiplier: int) -> None:
    assert AugmentConfig(interpolate=multiplier).interpolate == multiplier


@pytest.mark.parametrize("multiplier", [0, 3, 5, -1])
def test_interpolate_rejects_outside_vocab(multiplier: int) -> None:
    with pytest.raises(ValidationError):
        AugmentConfig(interpolate=multiplier)


def test_presentation_fps_defaults_to_unset() -> None:
    assert AugmentConfig().presentation_fps is None
    assert AugmentConfig(presentation_fps=32).presentation_fps == 32


def test_presentation_fps_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        AugmentConfig(presentation_fps=0)
    with pytest.raises(ValidationError):
        AugmentConfig(presentation_fps=-1)


def test_preset_carries_default_augment() -> None:
    config = preset_config("augment-run", "line art", 7)
    assert config.augment == AugmentConfig()


def test_resolve_without_augment_options_is_pure_noop() -> None:
    base = _base_config()
    before = base.model_dump()
    resolved = resolve_config(base)
    assert resolved.model_dump() == before
    assert base.model_dump() == before


def test_resolve_upscale_override() -> None:
    resolved = resolve_config(_base_config(), upscale=2)
    assert resolved.augment.upscale == 2
    assert resolved.augment.interpolate == 1
    assert resolved.augment.presentation_fps is None


def test_resolve_interpolate_override() -> None:
    resolved = resolve_config(_base_config(), interpolate=4)
    assert resolved.augment.interpolate == 4
    assert resolved.augment.upscale == 1


def test_resolve_presentation_fps_override() -> None:
    resolved = resolve_config(_base_config(), presentation_fps=32)
    assert resolved.augment.presentation_fps == 32
    assert (resolved.augment.upscale, resolved.augment.interpolate) == (1, 1)


def test_resolve_absent_encodings() -> None:
    assert resolve_config(_base_config(), upscale=None).augment == AugmentConfig()
    assert resolve_config(_base_config(), upscale=Unset).augment == AugmentConfig()
    assert resolve_config(_base_config(), interpolate=None).augment == AugmentConfig()
    assert resolve_config(_base_config(), interpolate=Unset).augment == AugmentConfig()
    assert resolve_config(_base_config(), presentation_fps=None).augment == AugmentConfig()
    assert resolve_config(_base_config(), presentation_fps=Unset).augment == AugmentConfig()


def test_resolve_invalid_rejected() -> None:
    with pytest.raises(ValidationError):
        resolve_config(_base_config(), upscale=0)
    with pytest.raises(ValidationError):
        resolve_config(_base_config(), upscale=3)
    with pytest.raises(ValidationError):
        resolve_config(_base_config(), interpolate=5)
    with pytest.raises(ValidationError):
        resolve_config(_base_config(), presentation_fps=0)


def _parse(verb_args: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(verb_args)


def _augment_defaults(args: argparse.Namespace) -> dict[str, object]:
    return {
        "upscale": args.upscale,
        "interpolate": args.interpolate,
        "presentation_fps": args.presentation_fps,
    }


def test_augment_parser_defaults() -> None:
    assert _augment_defaults(_parse(["configure", "calm", "--segments", "1"])) == {
        "upscale": None,
        "interpolate": None,
        "presentation_fps": None,
    }


def test_augment_flags_parse() -> None:
    args = _parse(
        [
            "configure",
            "calm",
            "--segments",
            "1",
            "--upscale",
            "2",
            "--interpolate",
            "4",
            "--presentation-fps",
            "32",
        ]
    )
    assert _augment_defaults(args) == {
        "upscale": 2,
        "interpolate": 4,
        "presentation_fps": 32,
    }


def test_configure_carries_augment_flags() -> None:
    """The shared helper exposes every augment flag on `configure` (020)."""
    flags = ["--upscale", "2", "--interpolate", "4", "--presentation-fps", "32"]
    args = _parse(["configure", "calm", "--segments", "1", *flags])
    assert _augment_defaults(args) == {
        "upscale": 2,
        "interpolate": 4,
        "presentation_fps": 32,
    }


@pytest.mark.parametrize("flag", ["--no-augment", "--min-fps", "--min-resolution"])
def test_removed_augment_flags_no_longer_parse(flag: str) -> None:
    """Deleted floor flags fail at the parser (never swallow into defaults)."""
    with pytest.raises(SystemExit):
        _parse(["configure", "calm", "--segments", "1", flag])


def test_augment_overrides_mapping() -> None:
    base = ["configure", "calm", "--segments", "1"]
    assert _augment_overrides(_parse(base)) == {}
    assert _augment_overrides(_parse([*base, "--upscale", "2"])) == {"upscale": 2}
    assert _augment_overrides(_parse([*base, "--interpolate", "4"])) == {"interpolate": 4}
    assert _augment_overrides(_parse([*base, "--presentation-fps", "32"])) == {
        "presentation_fps": 32
    }
    # TUI/hand-built namespaces: Unset blanks and missing attrs are absent.
    assert (
        _augment_overrides(
            argparse.Namespace(upscale=Unset, interpolate=Unset, presentation_fps=Unset)
        )
        == {}
    )
    assert _augment_overrides(argparse.Namespace()) == {}


def test_cmd_finalize_rejects_invalid_augment(tmp_path: Path) -> None:
    from tests.conftest import initialize_run_directory

    initialize_run_directory(tmp_path, run_id="x", style="y", seed=0)
    code = cmd_finalize(
        argparse.Namespace(
            run=str(tmp_path),
            output=str(tmp_path / "final.mp4"),
            skip_bad=False,
            upscale=3,
            interpolate=None,
            presentation_fps=None,
        )
    )
    assert code == 2
    code = cmd_finalize(
        argparse.Namespace(
            run=str(tmp_path),
            output=str(tmp_path / "final.mp4"),
            skip_bad=False,
            upscale=None,
            interpolate=None,
            presentation_fps=0,
        )
    )
    assert code == 2


def test_read_effective_config_rejects_legacy_augment_keys(tmp_path: Path) -> None:
    """Pre-multiplier manifests fail loud (re-configure, never silent read)."""
    from tests.conftest import initialize_run_directory
    from voyage.errors import StateError
    from voyage.persistence import read_effective_config

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="legacy", style="s", seed=0)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest["augment"], dict)
    manifest["augment"]["min_fps"] = 24
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StateError, match="augment keys"):
        read_effective_config(run_dir)


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
