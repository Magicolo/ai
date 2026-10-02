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

from voyage.cli import _augment_overrides, build_parser, cmd_finalize
from voyage.config import (
    AugmentConfig,
    ProjectConfig,
    Unset,
    default_config_toml,
    load_config,
    parse_min_resolution,
    resolve_config,
)
from voyage.tui_state import (
    GenerateFormState,
    field_errors,
    load_last_settings,
    plan_summary,
    save_last_settings,
    to_generate_namespace,
)


def _base_config() -> ProjectConfig:
    return ProjectConfig(style="augment-probe")


def test_augment_defaults() -> None:
    assert AugmentConfig().model_dump() == {
        "min_fps": 32,
        "min_width": 1280,
        "min_height": 720,
        "use_model_pass": True,
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
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = run_dir / "voyage.toml"
    config_path.write_text(default_config_toml("augment-run", "line art", 7), encoding="utf-8")
    assert "[augment]" in config_path.read_text(encoding="utf-8")
    config, _digest = load_config(config_path)
    assert config.augment == AugmentConfig()


def test_toml_round_trip_custom_floors(tmp_path: Path) -> None:
    config_path = tmp_path / "voyage.toml"
    config_path.write_text(default_config_toml("augment-run", "line art", 7), encoding="utf-8")
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("min_fps = 32", "min_fps = 60")
    text = text.replace("min_width = 1280", "min_width = 1920")
    text = text.replace("min_height = 720", "min_height = 1080")
    config_path.write_text(text, encoding="utf-8")
    config, _digest = load_config(config_path)
    assert (config.augment.min_fps, config.augment.min_width, config.augment.min_height) == (
        60,
        1920,
        1080,
    )


def test_toml_rejects_half_geometry(tmp_path: Path) -> None:
    from voyage.errors import ConfigurationError

    config_path = tmp_path / "voyage.toml"
    config_path.write_text(default_config_toml("augment-run", "line art", 7), encoding="utf-8")
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("min_width = 1280", "min_width = 0")
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(config_path)


def test_resolve_without_augment_options_is_pure_noop() -> None:
    base = _base_config()
    before = base.model_dump()
    resolved = resolve_config(base)
    assert resolved.model_dump() == before
    assert base.model_dump() == before


def test_resolve_min_fps_override() -> None:
    resolved = resolve_config(_base_config(), min_fps=60)
    assert resolved.augment.min_fps == 60
    assert (resolved.augment.min_width, resolved.augment.min_height) == (1280, 720)


def test_resolve_min_resolution_override() -> None:
    resolved = resolve_config(_base_config(), min_resolution="1920x1080")
    assert (resolved.augment.min_width, resolved.augment.min_height) == (1920, 1080)
    assert resolved.augment.min_fps == 32


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
    for verb in (
        ["finalize", "--run", "r", "--output", "o.mp4"],
        ["generate", "--style", "x", "--duration", "5s"],
        ["run", "--run", "r"],
    ):
        assert _augment_defaults(_parse(verb)) == {
            "min_fps": None,
            "min_resolution": None,
            "no_augment": False,
        }


def test_augment_flags_parse() -> None:
    args = _parse(
        [
            "finalize",
            "--run",
            "r",
            "--output",
            "o.mp4",
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
    assert _parse(["run", "--run", "r", "--no-augment"]).no_augment is True


def test_run_and_generate_share_augment_flags() -> None:
    """The shared helper keeps all three verbs in lockstep (mirrors 020)."""
    flags = ["--min-fps", "60", "--min-resolution", "1920x1080", "--no-augment"]
    run_args = _parse(["run", "--run", "r", *flags])
    gen_args = _parse(["generate", "--duration", "5s", "--style", "calm", *flags])
    fin_args = _parse(["finalize", "--run", "r", "--output", "o.mp4", *flags])
    assert _augment_defaults(run_args) == _augment_defaults(gen_args) == _augment_defaults(fin_args)


def test_augment_overrides_mapping() -> None:
    assert _augment_overrides(_parse(["run", "--run", "r"])) == {}
    assert _augment_overrides(_parse(["run", "--run", "r", "--min-fps", "60"])) == {"min_fps": 60}
    assert _augment_overrides(_parse(["run", "--run", "r", "--min-resolution", "0"])) == {
        "min_resolution": "0"
    }
    # --no-augment wins over explicit floors (both to 0) and forces the pass off.
    assert _augment_overrides(_parse(["run", "--run", "r", "--min-fps", "60", "--no-augment"])) == {
        "min_fps": 0,
        "min_resolution": "0",
        "use_model_pass": False,
    }
    # TUI/hand-built namespaces: Unset blanks and missing attrs are absent.
    assert _augment_overrides(argparse.Namespace(min_fps=Unset, min_resolution=Unset)) == {}
    assert _augment_overrides(argparse.Namespace()) == {}


def test_cmd_finalize_rejects_invalid_augment(tmp_path: Path) -> None:
    (tmp_path / "voyage.toml").write_text(
        'schema_version = 1\nrun_id = "x"\nstyle = "y"\nseed = 0\n', encoding="utf-8"
    )
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


def test_tui_augment_defaults() -> None:
    state = GenerateFormState()
    assert state.min_fps == "32"
    assert state.min_resolution == "1280x720"


def test_tui_namespace_carries_augment_attrs() -> None:
    """Issue 097 class: the TUI namespace must satisfy cmd_generate's finalize block.

    Issue 023: untouched-at-default augment floors emit Unset (stored TOML wins),
    never the concrete form defaults.
    """
    namespace = to_generate_namespace(GenerateFormState(style="x", backend="fake", name="augns"))
    assert namespace.min_fps is Unset
    assert namespace.min_resolution is Unset
    assert namespace.no_augment is False


def test_tui_blank_augment_fields_emit_unset() -> None:
    """Blank TUI augment fields emit Unset (issue 045), never None."""
    namespace = to_generate_namespace(GenerateFormState(style="x", min_fps="", min_resolution=""))
    assert namespace.min_fps is Unset
    assert namespace.min_resolution is Unset


def test_tui_zero_disables_a_floor() -> None:
    namespace = to_generate_namespace(GenerateFormState(style="x", min_fps="0", min_resolution="0"))
    assert namespace.min_fps == 0
    assert namespace.min_resolution == "0"


@pytest.mark.parametrize("raw", ["-1", "soon", "3.5"])
def test_tui_min_fps_rejects_bad_values(raw: str) -> None:
    assert "min_fps" in field_errors(GenerateFormState(style="x", min_fps=raw))


@pytest.mark.parametrize("raw", ["32", "0", "60"])
def test_tui_min_fps_accepts_non_negative(raw: str) -> None:
    assert "min_fps" not in field_errors(GenerateFormState(style="x", min_fps=raw))


@pytest.mark.parametrize("raw", ["soon", "1280", "0x720", "1280x0", "-1x720"])
def test_tui_min_resolution_rejects_bad_values(raw: str) -> None:
    assert "min_resolution" in field_errors(GenerateFormState(style="x", min_resolution=raw))


@pytest.mark.parametrize("raw", ["1280x720", "0", "1920x1080"])
def test_tui_min_resolution_accepts_good_values(raw: str) -> None:
    assert "min_resolution" not in field_errors(GenerateFormState(style="x", min_resolution=raw))


def test_tui_augment_fields_leave_planning_pure() -> None:
    """Augment knobs never touch plan math (planning stays CLI truth)."""
    plain = plan_summary(GenerateFormState(style="x"))
    floored = plan_summary(GenerateFormState(style="x", min_fps="60", min_resolution="0"))
    assert floored == plain


def test_tui_augment_settings_round_trip(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    state = GenerateFormState(style="x", min_fps="60", min_resolution="0")
    save_last_settings(state, settings_file)
    assert load_last_settings(settings_file) == state


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


def test_all_finalizing_verbs_carry_sfx_flags() -> None:
    parser = build_parser()
    generate = parser.parse_args(["generate", "--style", "x", "--duration", "5s"])
    finalize = parser.parse_args(["finalize", "--run", "r", "--output", "o.mp4"])
    stop = parser.parse_args(["stop", "--run", "r"])
    assert _sfx_defaults(generate) == _sfx_defaults(finalize) == _sfx_defaults(stop)


def test_generate_sfx_overrides_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--style",
            "x",
            "--duration",
            "5s",
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
