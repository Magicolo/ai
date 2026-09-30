"""Group A CLI/TUI surface fixes (Voyage issues 109-112, 115-116, 143-149, 179, 182, 185).

Why this module exists: sixteen CLI/TUI-boundary findings share one
theme — two surfaces (CLI parsers vs TUI namespaces vs cross-verb
handoff namespaces) disagreeing about normalization, validation order,
or flag coverage. Each test below pins the single-source contract for
one issue: the stop/finalize handoff (109/179/147), validate-before-
mutate ordering (110/143/185), the TUI live-validation floor (111),
the duration grammar (112), CUDA preflight coverage (115), and the
run-name migration (116/148/149) plus TUI namespace parity for the
download/SFX opt-outs (145/182). CPU/fake only; torch availability is
stubbed, workers are monkeypatched — no GPU, no model weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from voyage.cli import build_parser, main, parse_duration
from voyage.cli_run_ops import cmd_init
from voyage.tui_state import GenerateFormState, field_errors, to_generate_namespace

_STYLE = "pastel neon line-art, peaceful"


def _init_args(output: Path, style: str = _STYLE) -> argparse.Namespace:
    return argparse.Namespace(
        output=str(output),
        run_id="group-a",
        name="group-a",
        style=style,
        seed=11,
        force=False,
        backend="fake",
        director="deterministic",
        director_device="cpu",
    )


def test_stop_finalize_end_to_end_without_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stop --finalize` finalizes instead of AttributeError (109)."""
    import voyage.cli_finalize as finalize_module

    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "init",
                "--output",
                str(run_dir),
                "--run-id",
                "group-a",
                "--style",
                _STYLE,
                "--backend",
                "fake",
            ]
        )
        == 0
    )

    def _fake_finalize_run(run: Path, output: Path, **kwargs: object) -> None:
        del run, kwargs
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake-final")

    monkeypatch.setattr(finalize_module, "finalize_run", _fake_finalize_run)
    assert main(["stop", "--run", str(run_dir), "--finalize"]) == 0


def test_stop_parser_offers_full_finalize_surface() -> None:
    """Stop carries skip/augment/console flags like every finalizing verb (109/179)."""
    namespace = build_parser().parse_args(
        [
            "stop",
            "--run",
            str(Path("any")),
            "--finalize",
            "--verbose",
            "--no-color",
            "--skip-bad",
            "--min-fps",
            "32",
        ]
    )
    assert namespace.skip_bad is True
    assert namespace.verbose is True
    assert namespace.no_color is True
    assert namespace.min_fps == 32


def test_generate_finalize_inherits_console_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finalize third of `generate` honors --verbose/--no-color (147)."""
    import voyage.cli as cli_module

    captured: dict[str, argparse.Namespace] = {}

    def _capture_finalize(namespace: argparse.Namespace) -> int:
        captured["namespace"] = namespace
        return 0

    monkeypatch.setattr(cli_module, "cmd_finalize", _capture_finalize)
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "generate",
                "--backend",
                "fake",
                "--duration",
                "2s",
                "--style",
                _STYLE,
                "--name",
                "console-probe",
                "--seed",
                "11",
                "--verbose",
                "--no-color",
            ]
        )
        == 0
    )
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
                "generate",
                "--backend",
                "fake",
                "--duration",
                "2s",
                "--style",
                _STYLE,
                "--name",
                "orphan-probe",
                "--take-seconds",
                "5",
            ]
        )
        == 2
    )
    assert "take_seconds" in capsys.readouterr().err
    assert not (tmp_path / "output" / "orphan-probe").exists()


def test_cmd_init_rejects_blank_style_without_litter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Blank style exits 2 before the first mkdir (143)."""
    run_dir = tmp_path / "run"
    assert cmd_init(_init_args(run_dir, style="   ")) == 2
    assert "style" in capsys.readouterr().err
    assert not run_dir.exists()


def test_cmd_init_missing_attributes_exit_without_litter(tmp_path: Path) -> None:
    """Hand-built namespaces get exit 2, not AttributeError + partial dir (185)."""
    base = {
        "output": str(tmp_path / "run"),
        "run_id": "group-a",
        "name": "group-a",
        "force": True,
        "backend": "fake",
    }
    run_dir = tmp_path / "run"
    assert cmd_init(argparse.Namespace(**{**base, "seed": 11})) == 2  # missing style
    assert not run_dir.exists()
    assert cmd_init(argparse.Namespace(**{**base, "style": _STYLE})) == 2  # missing seed
    assert not run_dir.exists()


def test_tui_rejects_take_seconds_at_or_below_ahead_window() -> None:
    """Live validation enforces take_seconds > ahead_seconds (111)."""
    assert "take_seconds" in field_errors(
        GenerateFormState(style="x", name="probe", take_seconds="5")
    )
    assert "take_seconds" in field_errors(
        GenerateFormState(style="x", name="probe", take_seconds="20")
    )
    assert "take_seconds" not in field_errors(
        GenerateFormState(style="x", name="probe", take_seconds="20.1")
    )
    assert "take_seconds" not in field_errors(GenerateFormState(style="x", name="probe"))


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


def test_benchmark_video_fast_fails_without_cuda_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Benchmark video preflights torch like run/generate do (115)."""
    import voyage.cli as cli_module
    import voyage.cli_observe as observe_module

    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "init",
                "--output",
                str(run_dir),
                "--run-id",
                "group-a",
                "--style",
                _STYLE,
                "--backend",
                "ltxv",
            ]
        )
        == 0
    )
    monkeypatch.setattr(cli_module, "_torch_available", lambda: False)

    def _no_workers(*args: object, **kwargs: object) -> object:
        raise AssertionError("workers must never start past a failed preflight")

    monkeypatch.setattr(observe_module, "Supervisor", _no_workers)
    assert main(["benchmark", "video", "--run", str(run_dir)]) == 1
    assert "CUDA" in capsys.readouterr().err


def test_soak_fast_fails_without_cuda_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Soak preflights torch before printing its rule line (115)."""
    import voyage.cli as cli_module
    import voyage.cli_observe as observe_module

    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "init",
                "--output",
                str(run_dir),
                "--run-id",
                "group-a",
                "--style",
                _STYLE,
                "--backend",
                "ltxv",
            ]
        )
        == 0
    )
    monkeypatch.setattr(cli_module, "_torch_available", lambda: False)

    def _no_workers(*args: object, **kwargs: object) -> object:
        raise AssertionError("workers must never start past a failed preflight")

    monkeypatch.setattr(observe_module, "Supervisor", _no_workers)
    assert main(["soak", "--run", str(run_dir), "--segments", "1"]) == 1
    assert "CUDA" in capsys.readouterr().err


def test_effective_run_id_strips_padding_on_both_surfaces() -> None:
    """Padded names resolve identically on CLI and TUI (116)."""
    from voyage.cli import _effective_run_id

    assert _effective_run_id(argparse.Namespace(name=" boba ", run_id="voyage")) == "boba"
    namespace = to_generate_namespace(GenerateFormState(style="x", name=" boba "))
    assert namespace.name == "boba"
    assert namespace.output == str(Path("output") / "boba")


def test_tui_run_dir_for_honors_name_over_run_id() -> None:
    """The TUI watcher follows _effective_run_id, not run_id (148)."""
    from voyage.tui import VoyageApp

    namespace = argparse.Namespace(output=None, run_id="legacy", name="boba")
    assert VoyageApp._run_dir_for(VoyageApp(), namespace) == Path("output/boba").resolve()
    explicit = argparse.Namespace(output="custom", run_id="legacy", name="boba")
    assert VoyageApp._run_dir_for(VoyageApp(), explicit) == Path("custom").resolve()


def test_generate_output_help_names_name_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--output help matches the post-migration routing (149)."""
    with pytest.raises(SystemExit) as exc_info:
        main(["generate", "--help"])
    assert exc_info.value.code == 0
    assert "output/<name>" in capsys.readouterr().out


def test_tui_namespace_carries_no_download_opt_out() -> None:
    """TUI Generate can request verify-only runs (145)."""
    assert to_generate_namespace(GenerateFormState(style="x", name="probe")).no_download is False
    assert (
        to_generate_namespace(
            GenerateFormState(style="x", name="probe", no_download=True)
        ).no_download
        is True
    )


def test_tui_namespace_carries_sfx_opt_out() -> None:
    """TUI Generate can skip the finalize-time SFX pass (182)."""
    assert to_generate_namespace(GenerateFormState(style="x", name="probe")).no_sfx is False
    assert (
        to_generate_namespace(GenerateFormState(style="x", name="probe", no_sfx=True)).no_sfx
        is True
    )
