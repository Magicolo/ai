"""Output-tree containment warnings (issue 024).

`--run-id` traversal is rejected at the CLI layer (008); `--output` and
`--final-video` cannot be rejected the same way — absolute outside-tree
paths are legitimate (documented `/tmp` flows, tmp_path-based suites) —
so they warn on stderr instead. The warning is the typo guard: a bare
`--output /` can no longer silently scatter writes with exit 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.cli import main
from voyage.cli_paths import (
    is_outside_output_dir,
    output_root,
    warn_if_outside_output_dir,
)

_STYLE = "pastel neon line-art, peaceful"


def test_output_root_is_cwd_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The containment root tracks the current working directory (024)."""
    monkeypatch.chdir(tmp_path)
    assert output_root() == (tmp_path / "output").resolve()


def test_inside_output_is_contained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Paths under `./output/` pass quietly (024)."""
    monkeypatch.chdir(tmp_path)
    assert not is_outside_output_dir(tmp_path / "output" / "run")
    assert not is_outside_output_dir("output/run")


def test_outside_output_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absolute and dotdot-escaping paths are flagged (024)."""
    monkeypatch.chdir(tmp_path)
    assert is_outside_output_dir(tmp_path / "run")
    assert is_outside_output_dir("/tmp/evil-run")
    assert is_outside_output_dir(tmp_path / ".." / "evil")


def test_warn_reports_flag_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The warning names the flag and stays silent when contained (024)."""
    monkeypatch.chdir(tmp_path)
    assert warn_if_outside_output_dir(tmp_path / "run", flag="--output") is True
    assert "--output" in capsys.readouterr().err
    assert warn_if_outside_output_dir(tmp_path / "output" / "run", flag="--output") is False
    assert capsys.readouterr().err == ""


def test_init_outside_output_warns_but_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Warn-only: outside-tree init still exits 0 (documented /tmp flows) (024)."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "run"
    assert (
        main(
            [
                "init",
                "--output",
                str(target),
                "--run-id",
                "warned",
                "--style",
                _STYLE,
                "--backend",
                "fake",
            ]
        )
        == 0
    )
    assert "outside" in capsys.readouterr().err
    assert (target / "voyage.toml").exists()


def test_init_inside_output_is_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Contained init prints no containment warning (024)."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "output" / "run"
    assert (
        main(
            [
                "init",
                "--output",
                str(target),
                "--run-id",
                "quiet",
                "--style",
                _STYLE,
                "--backend",
                "fake",
            ]
        )
        == 0
    )
    assert "outside" not in capsys.readouterr().err
