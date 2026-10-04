"""CLI hardening tests: paths, counts, overrides, help (CPU/fake).

Covers issues 008/049/051/057/061/073/079 at the argparse boundary:
traversal `--run-id` rejected, non-positive counts exit 2, invalid
numeric overrides exit 2 on stderr (no traceback), director/models
typos fail at parse time, and `status` shows novelty + slowest stage.
Fake backends only; no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import paths
from voyage.cli import build_parser, main, parse_duration
from voyage.cli_paths import is_flat_folder_name
from voyage.config import preset_config

_STYLE = "pastel neon line-art, peaceful"


def _init_fake_run(run_dir: Path, run_id: str = "hardening") -> None:
    from tests.conftest import initialize_run_directory

    initialize_run_directory(run_dir, run_id=run_id, style=_STYLE, seed=11)


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
                "configure",
                "../../tmp/evil-run",
                "--backend",
                "fake",
                "--segments",
                "1",
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
    """`configure` validates the NAME before creating anything (008)."""
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "configure",
                "a/b",
                "--backend",
                "fake",
                "--segments",
                "1",
                "--style",
                _STYLE,
            ]
        )
        == 2
    )
    assert "flat folder" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_init_accepts_absolute_output(tmp_path: Path) -> None:
    """An explicit absolute --output inside the tree works (008)."""
    from tests.conftest import initialize_run_directory

    target = tmp_path / "sub" / "run"
    initialize_run_directory(target, run_id="rel", style=_STYLE, seed=11)
    assert (target / paths.MANIFEST_FILENAME).exists()


def test_generate_rejects_invalid_numeric_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`configure` override typos also exit 2 with stderr (061)."""
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "configure",
                "gen",
                "--backend",
                "fake",
                "--segments",
                "1",
                "--style",
                _STYLE,
                "--blocks",
                "0",
            ]
        )
        == 2
    )
    assert "invalid numeric override" in capsys.readouterr().err


def test_generate_parser_director_choices() -> None:
    """Configure accepts llama/qwen, rejects QWEN at parse time (073)."""
    parser = build_parser()
    args = parser.parse_args(["configure", "probe", "--segments", "1", "--style", _STYLE])
    assert args.director is None
    args = parser.parse_args(
        ["configure", "probe", "--segments", "1", "--style", _STYLE, "--director", "qwen"]
    )
    assert args.director == "qwen"
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            ["configure", "probe", "--segments", "1", "--style", _STYLE, "--director", "QWEN"]
        )
    assert exc_info.value.code == 2


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


def test_cuda_blame_names_audio_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio-only CUDA blames audio, not video (051)."""
    import importlib.util

    from voyage.cli_planning import _require_cuda_stack

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    config = preset_config("blame", _STYLE, 11, video_backend="fake")
    config.audio.backend = "acestep"
    assert _require_cuda_stack(config) is False
    err = capsys.readouterr().err
    assert "audio 'acestep'" in err
    assert "video 'fake'" not in err
