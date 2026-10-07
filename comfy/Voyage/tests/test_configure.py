"""`configure <NAME>` verb: init or update a run manifest (two-verb CLI).

The manifest (`manifest.json`) holds the full effective config plus the
planned segment count; `generate` later reconciles the directory against
it. Creation requires style + one of --segments/--duration; updates
touch only provided flags. Shrinking the count trims overflow segments;
a backend change on a committed run is refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from voyage import paths


@pytest.fixture(autouse=True)
def _never_touch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voyage.models_ensure.ensure_models", lambda *a, **k: 0)


def test_run_manifest_is_now_manifest_json() -> None:
    assert paths.MANIFEST_FILENAME == "manifest.json"


def _configure_namespace(name: str, **overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "name": name,
        "backend": None,
        "from_run": None,
        "duration": None,
        "segments": None,
        "style": None,
        "seed": None,
        "final_video": None,
        "skip_bad": False,
        "no_download": False,
        "force": False,
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
        "upscale": None,
        "interpolate": None,
        "presentation_fps": None,
        "no_sfx": False,
        "sfx_backend": None,
        "sfx_caption": None,
        "sfx_device": None,
        "sfx_model_size": None,
        # Parser-faithful: --sfx-workers defaults to None (absent), so an
        # omitted flag never clobbers a stored value on updates (issue 203).
        "sfx_workers": None,
        "verbose": False,
        "no_color": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_configure_creates_manifest_with_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    args = _configure_namespace("jango", style="dark harbors", segments=3, seed=7, no_download=True)
    assert cli_configure.cmd_configure(args) == 0
    manifest = json.loads((tmp_path / "output" / "jango" / "manifest.json").read_text())
    assert manifest["segments"] == 3
    assert manifest["style"] == "dark harbors"
    assert manifest["seed"] == 7
    assert manifest["video"]["backend"] == "ltx25"


def test_configure_create_needs_style_and_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    assert cli_configure.cmd_configure(_configure_namespace("a", segments=2)) == 2
    assert cli_configure.cmd_configure(_configure_namespace("b", style="s")) == 2
    assert (
        cli_configure.cmd_configure(_configure_namespace("c", style="s", segments=2, duration=5.0))
        == 2
    )


def test_configure_update_touches_only_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("j", style="first", segments=4, seed=7, no_download=True)
        )
        == 0
    )
    assert (
        cli_configure.cmd_configure(_configure_namespace("j", style="second", no_download=True))
        == 0
    )
    manifest = json.loads((tmp_path / "output" / "j" / "manifest.json").read_text())
    assert manifest["style"] == "second"
    assert manifest["segments"] == 4
    assert manifest["seed"] == 7


def _commit_three_segments(run_dir: Path) -> None:
    """Scaffold 3 committed segments with readable manifests (100f each)."""
    for index in ("000000", "000001", "000002"):
        segment = run_dir / "segments" / index
        segment.mkdir(parents=True, exist_ok=True)
        (segment / "DONE").write_text("done\n", encoding="utf-8")
        (segment / "manifest.json").write_text(
            json.dumps(
                {
                    "format": 1,
                    "transition": {},
                    "prompt_plan": {},
                    "audio_state": {},
                    "world_state": {},
                    "metrics": {"frames": 100},
                    "checksums": {},
                }
            ),
            encoding="utf-8",
        )


def test_configure_shrink_trims_overflow_and_recounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.conftest import initialize_run_directory
    from voyage import cli_configure
    from voyage.persistence import read_state

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "shrink"
    initialize_run_directory(run_dir, run_id="shrink")
    _commit_three_segments(run_dir)
    state = read_state(run_dir)
    state.committed_segments = 3
    state.next_segment_number = 3
    state.timeline_frames = 300
    from voyage.persistence import write_state

    write_state(run_dir, state)
    # Manifest must exist for the update path: write one via a create call.
    assert (
        cli_configure.cmd_configure(
            _configure_namespace(
                "shrink", style="s", segments=5, seed=1, force=True, no_download=True
            )
        )
        == 0
    )
    assert (
        cli_configure.cmd_configure(_configure_namespace("shrink", segments=2, no_download=True))
        == 0
    )
    assert not (run_dir / "segments" / "000002").exists()
    assert (run_dir / "segments" / "000001").exists()
    updated = read_state(run_dir)
    assert updated.committed_segments == 2
    assert updated.next_segment_number == 2
    assert updated.timeline_frames == 200


def test_configure_refuses_backend_change_on_committed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.conftest import initialize_run_directory
    from voyage import cli_configure
    from voyage.persistence import read_state, write_state

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "locked"
    initialize_run_directory(run_dir, run_id="locked")
    assert (
        cli_configure.cmd_configure(
            _configure_namespace(
                "locked", style="s", segments=2, seed=1, force=True, no_download=True
            )
        )
        == 0
    )
    state = read_state(run_dir)
    state.committed_segments = 1
    write_state(run_dir, state)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("locked", backend="ltxv", no_download=True)
        )
        == 2
    )


def test_old_run_manifest_json_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from voyage.errors import StateError
    from voyage.persistence import read_effective_config

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "legacy"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StateError, match="manifest.json"):
        read_effective_config(run_dir)


def _create_source_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "src") -> Path:
    """Configure a tuned source run; returns its run dir."""
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    args = _configure_namespace(
        name,
        style="dark harbors",
        segments=5,
        seed=7,
        backend="ltxv",
        blocks=3,
        take_seconds=30.0,
        no_download=True,
    )
    assert cli_configure.cmd_configure(args) == 0
    return tmp_path / "output" / name


def test_configure_from_copies_style_tuning_and_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage import cli_configure

    _create_source_run(tmp_path, monkeypatch)
    monkeypatch.setattr("voyage.seeds.random_master_seed", lambda: 999)
    monkeypatch.chdir(tmp_path)
    args = _configure_namespace("child", from_run="src", no_download=True)
    assert cli_configure.cmd_configure(args) == 0
    manifest = json.loads((tmp_path / "output" / "child" / "manifest.json").read_text())
    assert manifest["name"] == "child"
    assert manifest["style"] == "dark harbors"
    assert manifest["seed"] == 999
    assert manifest["video"]["backend"] == "ltxv"
    assert manifest["video"]["blocks_per_segment"] == 3
    assert manifest["audio"]["take_seconds"] == 30.0
    assert manifest["segments"] == 5


def test_configure_from_explicit_flags_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from voyage import cli_configure

    _create_source_run(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    args = _configure_namespace(
        "child",
        from_run="src",
        style="fresh style",
        backend="fake",
        segments=2,
        seed=11,
        no_download=True,
    )
    assert cli_configure.cmd_configure(args) == 0
    manifest = json.loads((tmp_path / "output" / "child" / "manifest.json").read_text())
    assert manifest["style"] == "fresh style"
    assert manifest["video"]["backend"] == "fake"
    assert manifest["segments"] == 2
    assert manifest["seed"] == 11


def test_configure_from_missing_source_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    args = _configure_namespace("child", from_run="nope", style="s", segments=2, no_download=True)
    assert cli_configure.cmd_configure(args) == 1
    assert not (tmp_path / "output" / "child").exists()


def test_configure_from_on_existing_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage import cli_configure

    _create_source_run(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    first = _configure_namespace("child", style="s", segments=2, seed=1, no_download=True)
    assert cli_configure.cmd_configure(first) == 0
    again = _configure_namespace("child", from_run="src", no_download=True)
    assert cli_configure.cmd_configure(again) == 2
    manifest = json.loads((tmp_path / "output" / "child" / "manifest.json").read_text())
    assert manifest["style"] == "s"
