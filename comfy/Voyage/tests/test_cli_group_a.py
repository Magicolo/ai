"""Group A CLI surface fixes (Voyage issues 110, 112, 143, 147, 149, 185).

Why this module exists: CLI-boundary findings about validation order
and flag coverage. Each test below pins the single-source contract
for one issue: the generate-driven finalize handoff (147),
validate-before-mutate ordering (110/143/185), the duration grammar
(112), and the run-name routing (149). CPU/fake only; workers are
monkeypatched — no GPU, no model weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from voyage.cli import build_parser, main, parse_duration

_STYLE = "pastel neon line-art, peaceful"


def test_generate_finalize_inherits_console_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generate-driven finalize honors --verbose/--no-color (147)."""
    import voyage.cli_finalize as finalize_module
    import voyage.cli_generate as gen_ops

    captured: dict[str, argparse.Namespace] = {}

    def _capture_finalize(namespace: argparse.Namespace) -> int:
        captured["namespace"] = namespace
        return 0

    class _FakeSupervisor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_segments(self, count: object) -> list[str]:
            return ["000000"]

    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "configure",
                "console-probe",
                "--backend",
                "fake",
                "--segments",
                "1",
                "--style",
                _STYLE,
                "--seed",
                "11",
            ]
        )
        == 0
    )
    monkeypatch.setattr(finalize_module, "cmd_finalize", _capture_finalize)
    monkeypatch.setattr(gen_ops, "Supervisor", _FakeSupervisor)
    monkeypatch.setattr(gen_ops, "validate_run", lambda run_dir: [])
    assert main(["generate", "console-probe", "--verbose", "--no-color"]) == 0
    assert captured["namespace"].verbose is True
    assert captured["namespace"].no_color is True


def test_generate_rejects_bad_take_seconds_before_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid numeric overrides exit 2 with no orphan run dir (110)."""
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "configure",
                "orphan-probe",
                "--backend",
                "fake",
                "--segments",
                "1",
                "--style",
                _STYLE,
                "--take-seconds",
                "5",
            ]
        )
        == 2
    )
    assert "take_seconds" in capsys.readouterr().err
    assert not (tmp_path / "output" / "orphan-probe").exists()


def test_cmd_generate_rejects_blank_style_without_litter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blank style exits 2 before the first mkdir (143)."""
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "configure",
                "blank-probe",
                "--backend",
                "fake",
                "--style",
                "   ",
                "--segments",
                "1",
            ]
        )
        == 2
    )
    assert "style" in capsys.readouterr().err
    assert not (tmp_path / "output" / "blank-probe").exists()


def test_cmd_generate_missing_attributes_exit_without_litter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hand-built namespaces get exit 2, not AttributeError + partial dir (185)."""
    from voyage.cli_generate import cmd_generate

    monkeypatch.chdir(tmp_path)
    assert cmd_generate(argparse.Namespace()) == 2
    assert not (tmp_path / "output").exists()
    # Missing seed randomizes instead of exiting 2 (mock downstream past creation).
    import voyage.models_ensure as ensure_module
    from voyage.persistence import read_effective_config

    monkeypatch.setattr(ensure_module, "ensure_models", lambda *a, **k: 0)
    assert (
        main(
            [
                "configure",
                "group-a",
                "--backend",
                "fake",
                "--segments",
                "1",
                "--style",
                _STYLE,
            ]
        )
        == 0
    )
    config = read_effective_config(tmp_path / "output" / "group-a")
    assert isinstance(config.seed, int)
    assert (tmp_path / "output" / "group-a" / "manifest.json").exists()


def test_parse_duration_rejects_mixed_signs() -> None:
    """Per-component signs are subtractive typos, not arithmetic (112)."""
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration("2m-30s")
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration("1h-30m")
    with pytest.raises(ValueError, match="must be positive"):
        parse_duration("-5s")


def test_parse_duration_rejects_bare_trailing_number() -> None:
    """`1h30` must not silently bind 30 to seconds (112)."""
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration("1h30")
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration("1m30")
    assert parse_duration("90") == 90.0
    assert parse_duration("1h30m") == 5400.0
    assert parse_duration("1h2m3.5s") == pytest.approx(3723.5)


def test_parse_duration_interior_space_names_the_rule() -> None:
    """Whitespace rejection says no-spaces instead of bare examples (112)."""
    with pytest.raises(ValueError, match="no spaces"):
        parse_duration("1m 30s")


def test_generate_parser_accepts_plan_extension_shorthand() -> None:
    """Approved (b): --segments/--duration extend the stored plan additively."""
    args = build_parser().parse_args(["generate", "probe", "--segments", "3"])
    assert args.name == "probe"
    assert args.segments == 3
    assert args.duration is None
    args = build_parser().parse_args(["generate", "probe", "--duration", "5s"])
    assert args.duration == 5.0
    assert args.segments is None


def test_generate_output_help_names_name_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--output help matches the post-migration routing (149)."""
    with pytest.raises(SystemExit) as exc_info:
        main(["generate", "--help"])
    assert exc_info.value.code == 0
    assert "output/<name>" in capsys.readouterr().out
