"""Single run-dir resolver + output-root override + final-video warn (issue 206).

`--run` used to resolve any string with no flatness, absoluteness, or
containment check while `--name` was traversal-guarded; `output_root()`
was cwd-coupled with no override; `configure --final-video` outside the
output tree never warned (only finalize `--output` did); `run.sh`
ignored `$XDG_CACHE_HOME`; `qualify.sh` artifacts collided same-day.
All against tmp paths, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.cli_paths import output_root, resolve_run_dir, resolve_run_ref


def test_both_flags_is_misuse(capsys: pytest.CaptureFixture[str]) -> None:
    """Passing both --run and --name resolves to nothing (exit 2)."""
    assert resolve_run_dir(run="/tmp/anywhere", name="jango") is None
    assert "only one" in capsys.readouterr().err


def test_neither_flag_is_misuse(capsys: pytest.CaptureFixture[str]) -> None:
    """Passing neither flag resolves to nothing (exit 2)."""
    assert resolve_run_dir(run=None, name=None) is None
    assert "required" in capsys.readouterr().err


def test_empty_strings_read_as_absent(capsys: pytest.CaptureFixture[str]) -> None:
    """Empty strings behave like missing flags (benchmark default)."""
    assert resolve_run_dir(run="", name="") is None
    assert "required" in capsys.readouterr().err


def test_name_resolves_under_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A flat --name lands under the cwd output root."""
    monkeypatch.chdir(tmp_path)
    assert resolve_run_dir(run=None, name="jango") == (tmp_path / "output" / "jango").resolve()


def test_name_traversal_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    """A traversal --name resolves to nothing."""
    assert resolve_run_dir(run=None, name="../../evil") is None
    assert "flat folder name" in capsys.readouterr().err


def test_run_relative_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    """A relative --run is refused (absolute-path invariant)."""
    assert resolve_run_dir(run="output/jango", name=None) is None
    assert "absolute" in capsys.readouterr().err


def test_run_absolute_outside_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An absolute --run outside output/ resolves but warns (warn-only)."""
    monkeypatch.chdir(tmp_path)
    outside = (tmp_path / ".." / "evil-run").resolve()
    assert resolve_run_dir(run=str(outside), name=None) == outside
    err = capsys.readouterr().err
    assert "outside" in err
    assert "--run" in err


def test_run_absolute_inside_is_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An absolute --run inside output/ resolves quietly."""
    monkeypatch.chdir(tmp_path)
    inside = tmp_path / "output" / "jango"
    assert resolve_run_dir(run=str(inside), name=None) == inside.resolve()
    assert capsys.readouterr().err == ""


def test_resolve_run_ref_matches_resolve_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy alias delegates without behavior drift."""
    monkeypatch.chdir(tmp_path)
    inside = str(tmp_path / "output" / "jango")
    assert resolve_run_ref(run=inside, name=None) == resolve_run_dir(run=inside, name=None)
    assert resolve_run_ref(run=None, name="jango") == resolve_run_dir(run=None, name="jango")


def test_output_root_honors_voyage_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """VOYAGE_OUTPUT pins the root; unset keeps the cwd default."""
    monkeypatch.chdir(tmp_path)
    assert output_root() == (tmp_path / "output").resolve()
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("VOYAGE_OUTPUT", str(elsewhere))
    assert output_root() == elsewhere.resolve()
    monkeypatch.delenv("VOYAGE_OUTPUT")
    assert output_root() == (tmp_path / "output").resolve()


def test_configure_final_video_outside_warns_but_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An outside-tree --final-video warns (but configures) at create."""
    import voyage.models_ensure as ensure_module
    from voyage.cli import main
    from voyage.persistence import read_manifest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ensure_module, "ensure_models", lambda *args, **kwargs: 0)
    outside_final = str((tmp_path / ".." / "final.mp4").resolve())
    assert (
        main(
            [
                "configure",
                "warned",
                "--style",
                "pastel neon line-art, peaceful",
                "--backend",
                "fake",
                "--segments",
                "1",
                "--seed",
                "11",
                "--final-video",
                outside_final,
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "outside" in err
    assert "--final-video" in err
    assert read_manifest(tmp_path / "output" / "warned")["final_video"] == outside_final


def test_configure_final_video_inside_is_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A contained --final-video configures with no warning."""
    import voyage.models_ensure as ensure_module
    from voyage.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ensure_module, "ensure_models", lambda *args, **kwargs: 0)
    assert (
        main(
            [
                "configure",
                "quiet",
                "--style",
                "pastel neon line-art, peaceful",
                "--backend",
                "fake",
                "--segments",
                "1",
                "--seed",
                "11",
                "--final-video",
                "output/quiet/custom.mp4",
            ]
        )
        == 0
    )
    assert "outside" not in capsys.readouterr().err
