"""`--name` shortcut + auto-refinalize on `run` (user request).

`--name jango` resolves to the default output path (`output/jango`)
on every run verb; extending a run re-finalizes `final.mp4` when (and
only when) new segments were committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from voyage.cli_paths import resolve_run_ref


def test_name_resolves_under_default_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_run_ref(run=None, name="jango")
    assert resolved == (tmp_path / "output" / "jango").resolve()


def test_both_run_and_name_is_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert resolve_run_ref(run="a", name="b") is None
    assert "only one of" in capsys.readouterr().err


def test_bad_name_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert resolve_run_ref(run=None, name="../evil") is None
    assert capsys.readouterr().err != ""


def test_run_passthrough_and_neither() -> None:
    assert resolve_run_ref(run="rel-dir", name=None) == Path("rel-dir").resolve()
    assert resolve_run_ref(run=None, name=None) is None
    assert resolve_run_ref(run="", name=None) is None


def _parse(argv: list[str]) -> argparse.Namespace:
    from voyage.cli import build_parser

    return build_parser().parse_args(argv)


def test_all_run_verbs_accept_name() -> None:
    from voyage import cli

    assert _parse(["run", "--name", "jango"]).func is cli.cmd_run
    assert _parse(["status", "--name", "jango"]).func is cli.cmd_status
    assert _parse(["pause", "--name", "jango"]).func is cli.cmd_pause
    assert _parse(["resume", "--name", "jango"]).func is cli.cmd_resume
    assert _parse(["stop", "--name", "jango"]).func is cli.cmd_stop
    assert _parse(["validate", "--name", "jango"]).func is cli.cmd_validate
    finalized = _parse(["finalize", "--name", "jango", "--output", "f.mp4"])
    assert finalized.func is cli.cmd_finalize
    assert _parse(["sfx", "--name", "jango"]).func is cli.cmd_sfx
    assert _parse(["soak", "--name", "jango", "--segments", "1"]).func is cli.cmd_soak
    assert _parse(["inspect", "segments", "--name", "jango"]).func is cli.cmd_inspect
    assert _parse(["benchmark", "video", "--name", "jango"]).func is cli.cmd_benchmark


def test_run_flag_no_longer_required_at_parse() -> None:
    args = _parse(["run", "--name", "jango"])
    assert args.run is None
    assert args.name == "jango"


def _run_namespace(run_dir: Path, **overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "run": str(run_dir),
        "name": None,
        "segments": 1,
        "draft": False,
        "director": None,
        "director_device": None,
        "blocks": None,
        "take_seconds": None,
        "quantization": None,
        "beats_per_segment": None,
        "drift_every_n": None,
        "music_caption": None,
        "video_caption": None,
        "min_fps": None,
        "min_resolution": None,
        "no_augment": False,
        "use_model_pass": None,
        "verbose": False,
        "no_color": True,
        "no_finalize": False,
        "skip_bad": False,
        "no_sfx": True,
        "sfx_backend": None,
        "sfx_caption": None,
        "sfx_device": None,
        "sfx_model_size": None,
        "sfx_workers": 1,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_run_refinalizes_when_segments_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.cli_run_ops as run_ops
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "jango"
    initialize_run_directory(run_dir, run_id="jango")
    calls: list[argparse.Namespace] = []

    class _FakeSupervisor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_segments(self, count: object) -> list[str]:
            return ["000002"]

    def _fake_finalize(ns: argparse.Namespace) -> int:
        calls.append(ns)
        return 0

    monkeypatch.setattr(run_ops, "Supervisor", _FakeSupervisor)
    monkeypatch.setattr("voyage.cli.cmd_finalize", _fake_finalize)
    assert run_ops.cmd_run(_run_namespace(run_dir)) == 0
    (finalize_ns,) = calls
    assert finalize_ns.output == str(run_dir / "final.mp4")
    assert finalize_ns.run == str(run_dir)


def test_run_skips_refinalize_without_new_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.cli_run_ops as run_ops
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "jango"
    initialize_run_directory(run_dir, run_id="jango")

    class _FakeSupervisor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_segments(self, count: object) -> list[str]:
            return []

    def _forbidden(ns: argparse.Namespace) -> int:
        raise AssertionError("no new segments — must not refinalize")

    monkeypatch.setattr(run_ops, "Supervisor", _FakeSupervisor)
    monkeypatch.setattr("voyage.cli.cmd_finalize", _forbidden)
    assert run_ops.cmd_run(_run_namespace(run_dir)) == 0


def test_run_no_finalize_flag_skips_refinalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.cli_run_ops as run_ops
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "jango"
    initialize_run_directory(run_dir, run_id="jango")

    class _FakeSupervisor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_segments(self, count: object) -> list[str]:
            return ["000002"]

    def _forbidden(ns: argparse.Namespace) -> int:
        raise AssertionError("--no-finalize must skip the refinalize step")

    monkeypatch.setattr(run_ops, "Supervisor", _FakeSupervisor)
    monkeypatch.setattr("voyage.cli.cmd_finalize", _forbidden)
    assert run_ops.cmd_run(_run_namespace(run_dir, no_finalize=True)) == 0


def test_resume_never_refinalizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli_status as status_ops
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "jango"
    initialize_run_directory(run_dir, run_id="jango")

    def _forbidden(ns: argparse.Namespace) -> int:
        raise AssertionError("resume commits nothing — must not refinalize")

    monkeypatch.setattr("voyage.cli.cmd_finalize", _forbidden)
    assert status_ops.cmd_resume(argparse.Namespace(run=str(run_dir), name=None)) == 0
