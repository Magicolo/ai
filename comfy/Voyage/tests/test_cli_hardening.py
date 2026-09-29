"""CLI hardening tests: paths, counts, overrides, help (CPU/fake).

Covers issues 008/049/051/057/061/073/079 at the argparse boundary:
traversal `--run-id` rejected, non-positive counts exit 2, invalid
numeric overrides exit 2 on stderr (no traceback), director/models
typos fail at parse time, and `status` shows novelty + slowest stage.
Fake backends only; no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from voyage import paths
from voyage.cli import (
    build_parser,
    is_flat_folder_name,
    main,
    parse_duration,
    resolve_run_dir,
)
from voyage.config import default_config_toml, load_config

_STYLE = "pastel neon line-art, peaceful"


def _init_fake_run(run_dir: Path, run_id: str = "hardening") -> None:
    assert (
        main(
            [
                "init",
                "--output",
                str(run_dir),
                "--run-id",
                run_id,
                "--style",
                _STYLE,
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("voyage", True),
        ("gen-default", True),
        ("  padded  ", True),
        ("", False),
        ("   ", False),
        ("a/b", False),
        ("a\\b", False),
        ("..", False),
        ("../../tmp/evil-run", False),
        ("output/x", False),
        (".", False),
        ("con", False),
        ("NUL.txt", False),
        ("com1", False),
        ("a.b", True),
        ("comet", True),
    ],
)
def test_is_flat_folder_name(value: str, expected: bool) -> None:
    assert is_flat_folder_name(value) is expected


def test_init_rejects_traversal_run_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Crafted --run-id cannot escape the run tree (008)."""
    target = tmp_path / "run"
    assert (
        main(
            [
                "init",
                "--output",
                str(target),
                "--run-id",
                "../../tmp/evil-run",
                "--style",
                _STYLE,
                "--force",
            ]
        )
        == 2
    )
    assert "flat folder" in capsys.readouterr().err
    assert not target.exists()


def test_generate_rejects_traversal_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`generate` validates --run-id before creating anything (008)."""
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
                "--run-id",
                "a/b",
            ]
        )
        == 2
    )
    assert "flat folder" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_init_accepts_absolute_output(tmp_path: Path) -> None:
    """An explicit absolute --output inside the tree works (008)."""
    target = tmp_path / "sub" / "run"
    assert (
        main(
            [
                "init",
                "--output",
                str(target),
                "--run-id",
                "fine",
                "--style",
                _STYLE,
            ]
        )
        == 0
    )
    assert (target / paths.CONFIG_FILENAME).exists()


def test_resolve_run_dir_returns_absolute() -> None:
    """The shared helper normalizes to an absolute path (008/057)."""
    assert resolve_run_dir("output/some-run") == (Path.cwd() / "output/some-run").resolve()
    assert resolve_run_dir("output/some-run").is_absolute()


def test_init_then_run_with_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relative init + relative run commits (057 doubling regression)."""
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "init",
                "--output",
                "rel-run",
                "--run-id",
                "rel",
                "--style",
                _STYLE,
            ]
        )
        == 0
    )
    assert (tmp_path / "rel-run" / paths.CONFIG_FILENAME).exists()
    assert main(["run", "--run", "rel-run", "--segments", "1"]) == 0
    segment = tmp_path / "rel-run" / paths.SEGMENTS_DIRNAME / "000000"
    assert (segment / paths.DONE_MARKER).exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--blocks", "0"],
        ["--blocks", "-2"],
        ["--take-seconds", "-1"],
        ["--beats-per-segment", "0"],
        ["--drift-every-n", "0"],
    ],
)
def test_run_rejects_invalid_numeric_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    """Bad overrides fail cleanly on stderr (061, no traceback)."""
    run_dir = tmp_path / "run"
    _init_fake_run(run_dir)
    assert main(["run", "--run", str(run_dir), "--segments", "1", *extra]) == 2
    assert "invalid numeric override" in capsys.readouterr().err


def test_generate_rejects_invalid_numeric_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`generate` override typos also exit 2 with stderr (061)."""
    output = tmp_path / "gen"
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
                "--output",
                str(output),
                "--run-id",
                "gen",
                "--blocks",
                "0",
            ]
        )
        == 2
    )
    assert "invalid numeric override" in capsys.readouterr().err


def test_run_rejects_bogus_director(tmp_path: Path) -> None:
    """Director typos fail at parse time (073)."""
    run_dir = tmp_path / "run"
    _init_fake_run(run_dir)
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--run", str(run_dir), "--segments", "1", "--director", "bogus"])
    assert exc_info.value.code == 2


def test_generate_parser_director_choices() -> None:
    """Generate accepts qwen, rejects QWEN at parse time (073)."""
    parser = build_parser()
    args = parser.parse_args(["generate", "--duration", "2s", "--style", _STYLE])
    assert args.director == "qwen"
    args = parser.parse_args(
        ["generate", "--duration", "2s", "--style", _STYLE, "--director", "qwen"]
    )
    assert args.director == "qwen"
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["generate", "--duration", "2s", "--style", _STYLE, "--director", "QWEN"])
    assert exc_info.value.code == 2


def test_run_accepts_deterministic_director(tmp_path: Path) -> None:
    """The allowlisted deterministic director still runs (073)."""
    run_dir = tmp_path / "run"
    _init_fake_run(run_dir)
    assert (
        main(
            [
                "run",
                "--run",
                str(run_dir),
                "--segments",
                "1",
                "--director",
                "deterministic",
            ]
        )
        == 0
    )


@pytest.mark.parametrize("count", ["0", "-3"])
def test_run_rejects_non_positive_segments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], count: str
) -> None:
    """Zero work is an input error, not a success (079)."""
    run_dir = tmp_path / "run"
    _init_fake_run(run_dir)
    assert main(["run", "--run", str(run_dir), "--segments", count]) == 2
    assert "must be positive" in capsys.readouterr().err


def test_soak_rejects_non_positive_segments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Soak with zero segments exits 2 before loading work (079)."""
    run_dir = tmp_path / "run"
    _init_fake_run(run_dir)
    assert main(["soak", "--run", str(run_dir), "--segments", "0"]) == 2
    assert "must be positive" in capsys.readouterr().err


def test_benchmark_rejects_bad_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Benchmark counts fail CLI-side, not worker-side (079/060)."""
    run_dir = tmp_path / "run"
    _init_fake_run(run_dir)
    assert main(["benchmark", "video", "--run", str(run_dir), "--warmup", "-1"]) == 2
    assert "warmup" in capsys.readouterr().err
    assert main(["benchmark", "video", "--run", str(run_dir), "--measured", "0"]) == 2
    assert "measured" in capsys.readouterr().err
    assert main(["benchmark", "end-to-end", "--segments", "0"]) == 2
    assert "must be positive" in capsys.readouterr().err


def test_models_download_rejects_unknown_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Models typos fail at parse time with argparse choices (051)."""
    with pytest.raises(SystemExit) as exc_info:
        main(["models", "download", "bogus-target"])
    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_models_unknown_target_goes_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Programmatic unknown targets report on stderr (051)."""
    from voyage.cli import cmd_models

    args = argparse.Namespace(models_action="download", models_target="bogus", models_dir=None)
    assert cmd_models(args) == 2
    assert "unknown models target" in capsys.readouterr().err


def test_parse_duration_accepts_documented_forms() -> None:
    """Help examples all parse: hours, fractional, combined, bare (051)."""
    assert parse_duration("1h") == 3600.0
    assert parse_duration("2.5m") == 150.0
    assert parse_duration("1h2m3.5s") == pytest.approx(3723.5)
    assert parse_duration(" 90 ") == 90.0


def test_parse_duration_negative_hits_positivity_branch() -> None:
    """Signed values parse, then fail as non-positive (051)."""
    with pytest.raises(ValueError, match="must be positive"):
        parse_duration("-5s")


def test_stop_finalize_uses_resolved_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative `stop --run` must not double the finalize path (051)."""
    import voyage.cli as cli_module

    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "init",
                "--output",
                "rel-run",
                "--run-id",
                "rel",
                "--style",
                _STYLE,
            ]
        )
        == 0
    )
    captured: dict[str, str] = {}

    def fake_finalize(args: argparse.Namespace) -> int:
        captured["output"] = str(args.output)
        return 0

    monkeypatch.setattr(cli_module, "cmd_finalize", fake_finalize)
    assert main(["stop", "--run", "rel-run", "--finalize"]) == 0
    assert captured["output"] == str((tmp_path / "rel-run" / "final.mp4").resolve())


def test_cuda_blame_names_audio_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio-only CUDA blames audio, not video (051)."""
    import importlib.util

    from voyage.cli import _require_cuda_stack

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    (tmp_path / "voyage.toml").write_text(
        default_config_toml("blame", _STYLE, 11), encoding="utf-8"
    )
    config, _digest = load_config(tmp_path / "voyage.toml")
    config.audio.backend = "acestep"
    assert _require_cuda_stack(config) is False
    err = capsys.readouterr().err
    assert "audio 'acestep'" in err
    assert "video 'fake'" not in err


def test_status_shows_novelty_before_first_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fresh runs report no concepts + age qualifier (049-status)."""
    run_dir = tmp_path / "run"
    _init_fake_run(run_dir)
    assert main(["status", "--run", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "Novelty: no concepts yet" in out
    assert "not running" in out


def test_status_shows_novelty_and_slowest_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Committed runs show the novelty verdict + bottleneck (049-status)."""
    run_dir = tmp_path / "run"
    _init_fake_run(run_dir)
    assert main(["run", "--run", str(run_dir), "--segments", "1"]) == 0
    capsys.readouterr()
    assert main(["status", "--run", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "Novelty: accepted (record " in out
    assert "Stages (last commit 000000)" in out
    assert "Slowest stage:" in out


def test_status_running_hides_age_qualifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """RUNNING runs report plain uptime; paused runs report age (049-status)."""
    run_dir = tmp_path / "run"
    _init_fake_run(run_dir)
    assert main(["resume", "--run", str(run_dir)]) == 0
    capsys.readouterr()
    assert main(["status", "--run", str(run_dir)]) == 0
    assert "(age since init" not in capsys.readouterr().out


def test_slowest_stage_picks_max() -> None:
    """Slowest-stage helper names the bottleneck, ignores junk (049-status)."""
    from voyage.cli import _slowest_stage

    assert _slowest_stage({"video": 1.2, "audio": 0.4}) == "video (1.2s)"
    assert _slowest_stage({}) is None
    assert _slowest_stage({"video": "fast"}) is None


def test_models_info_reports_bundles_and_license(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`models info` names bundles, pins, license, and verify (051)."""
    assert main(["models", "info"]) == 0
    out = capsys.readouterr().out
    for expected in (
        "longlive2-bf16",
        "ltxv-2b",
        "causvid",
        "model_registry.py",
        "CC BY-NC-SA 4.0",
        "models verify",
    ):
        assert expected in out, expected
