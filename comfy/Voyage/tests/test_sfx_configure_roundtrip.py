"""`configure --sfx-*` flags round-trip into the stored run config (issue 203).

CPU-only: parser defaults, `cli_core._sfx_overrides` mapping,
`resolve_config` SFX branches, and `cmd_configure` create/update
persistence through `read_effective_config` — every `--sfx-*` flag
either errors or round-trips, never silently drops. The generate-side
sentinel contract (`_finalize_run_dir` passes None so the manifest
rules) is pinned without running finalize.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from voyage.cli import build_parser


@pytest.fixture(autouse=True)
def _never_touch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voyage.models_ensure.ensure_models", lambda *a, **k: 0)


def _configure_namespace(name: str, **overrides: object) -> argparse.Namespace:
    """Minimal `configure` namespace (mirrors tests/test_configure.py)."""
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
        "scene_cut_every_n": None,
        "music_caption": None,
        "video_caption": None,
        "upscale": None,
        "interpolate": None,
        "presentation_fps": None,
        "interp_backend": None,
        "no_sfx": False,
        "sfx_backend": None,
        "sfx_caption": None,
        "sfx_device": None,
        "sfx_model_size": None,
        "sfx_workers": None,
        "verbose": False,
        "no_color": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_sfx_parser_defaults_are_absent() -> None:
    """All five `--sfx-*` flags default to None (absent), like every sibling."""
    args = build_parser().parse_args(["configure", "calm", "--segments", "1"])
    assert args.sfx_backend is None
    assert args.sfx_caption is None
    assert args.sfx_device is None
    assert args.sfx_model_size is None
    assert args.sfx_workers is None


def test_sfx_flags_parse_explicit_values() -> None:
    """Explicit `--sfx-*` values survive argparse (choices enforced)."""
    args = build_parser().parse_args(
        [
            "configure",
            "calm",
            "--segments",
            "1",
            "--sfx-backend",
            "mmaudio",
            "--sfx-caption",
            "distant rumble",
            "--sfx-device",
            "cuda:1",
            "--sfx-model-size",
            "small_44k",
            "--sfx-workers",
            "2",
        ]
    )
    assert args.sfx_backend == "mmaudio"
    assert args.sfx_caption == "distant rumble"
    assert args.sfx_device == "cuda:1"
    assert args.sfx_model_size == "small_44k"
    assert args.sfx_workers == 2


def test_sfx_workers_rejects_three_at_parse() -> None:
    """`--sfx-workers 3` errors at parse time (choices), never silently."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["configure", "calm", "--segments", "1", "--sfx-workers", "3"])


def test_sfx_overrides_absent_is_empty() -> None:
    """Missing/None SFX attrs map to no overrides (inherit stored)."""
    from voyage.cli_core import _sfx_overrides

    assert _sfx_overrides(argparse.Namespace()) == {}
    assert _sfx_overrides(_configure_namespace("x")) == {}


def test_sfx_overrides_forward_provided() -> None:
    """Provided SFX flags map to resolve_config kwargs verbatim."""
    from voyage.cli_core import _sfx_overrides

    args = _configure_namespace(
        "x",
        sfx_backend="mmaudio",
        sfx_caption="distant rumble",
        sfx_device="cuda:1",
        sfx_model_size="small_44k",
        sfx_workers=2,
    )
    assert _sfx_overrides(args) == {
        "sfx_backend": "mmaudio",
        "sfx_caption": "distant rumble",
        "sfx_device": "cuda:1",
        "sfx_model_size": "small_44k",
        "sfx_workers": 2,
    }


def test_resolve_config_sfx_overrides_roundtrip() -> None:
    """All five SFX overrides land on SfxConfig through resolve_config."""
    from voyage.config import preset_config, resolve_config

    base = preset_config("demo", "pastel neon line-art, peaceful", 1)
    resolved = resolve_config(
        base,
        sfx_backend="fake",
        sfx_caption="distant rumble",
        sfx_device="cpu",
        sfx_model_size="small_44k",
        sfx_workers=2,
    )
    assert resolved.sfx.backend == "fake"
    assert resolved.sfx.sfx_caption == "distant rumble"
    assert resolved.sfx.device == "cpu"
    assert resolved.sfx.model_size == "small_44k"
    assert resolved.sfx.num_workers == 2
    assert base.sfx.num_workers == 1  # pure: input untouched


def test_resolve_config_explicit_sfx_wins_over_backend_preset() -> None:
    """Backend preset applies first; explicit SFX flags win (issue 022 order)."""
    from voyage.config import preset_config, resolve_config

    base = preset_config("demo", "pastel neon line-art, peaceful", 1)
    resolved = resolve_config(base, backend="fake", sfx_backend="mmaudio", sfx_device="cuda:0")
    assert resolved.video.backend == "fake"
    assert resolved.sfx.backend == "mmaudio"
    assert resolved.sfx.device == "cuda:0"


def test_resolve_config_rejects_bad_sfx_values() -> None:
    """Garbage SFX overrides fail loud at resolve time, never corrupt runs."""
    from voyage.config import preset_config, resolve_config

    base = preset_config("demo", "pastel neon line-art, peaceful", 1)
    with pytest.raises(ValidationError):
        resolve_config(base, sfx_workers=3)
    with pytest.raises(ValidationError):
        resolve_config(base, sfx_workers=True)
    with pytest.raises(ValidationError):
        resolve_config(base, sfx_backend="sdxl")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        resolve_config(base, sfx_model_size="tiny_8k")  # type: ignore[arg-type]


def test_sfx_config_defaults_keep_legacy_runs_byte_identical() -> None:
    """New SfxConfig fields default to the legacy behavior (no pin, 1 worker)."""
    from voyage.config import SfxConfig

    assert SfxConfig().sfx_caption is None
    assert SfxConfig().num_workers == 1


def _manifest_sfx(tmp_path: Path, name: str) -> dict[str, object]:
    manifest = json.loads((tmp_path / "output" / name / "manifest.json").read_text())
    sfx = manifest["sfx"]
    assert isinstance(sfx, dict)
    return sfx


def test_configure_create_roundtrips_all_sfx_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every `--sfx-*` flag persists through the manifest (no silent drop)."""
    from voyage import cli_configure
    from voyage.persistence import read_effective_config

    monkeypatch.chdir(tmp_path)
    args = _configure_namespace(
        "sfxrun",
        style="dark harbors",
        segments=2,
        seed=7,
        sfx_backend="mmaudio",
        sfx_caption="distant rumble",
        sfx_device="cuda:1",
        sfx_model_size="small_44k",
        sfx_workers=2,
    )
    assert cli_configure.cmd_configure(args) == 0
    sfx = _manifest_sfx(tmp_path, "sfxrun")
    assert sfx["backend"] == "mmaudio"
    assert sfx["sfx_caption"] == "distant rumble"
    assert sfx["device"] == "cuda:1"
    assert sfx["model_size"] == "small_44k"
    assert sfx["num_workers"] == 2
    effective = read_effective_config(tmp_path / "output" / "sfxrun")
    assert effective.sfx.backend == "mmaudio"
    assert effective.sfx.sfx_caption == "distant rumble"
    assert effective.sfx.device == "cuda:1"
    assert effective.sfx.model_size == "small_44k"
    assert effective.sfx.num_workers == 2


def test_configure_update_absent_sfx_flags_keep_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An update without SFX flags keeps stored values (absent stays absent)."""
    from voyage import cli_configure
    from voyage.persistence import read_effective_config

    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace(
                "keep",
                style="dark harbors",
                segments=2,
                seed=7,
                sfx_backend="mmaudio",
                sfx_device="cuda:1",
                sfx_model_size="small_44k",
                sfx_caption="distant rumble",
                sfx_workers=2,
            )
        )
        == 0
    )
    assert cli_configure.cmd_configure(_configure_namespace("keep", style="darker harbors")) == 0
    effective = read_effective_config(tmp_path / "output" / "keep")
    assert effective.style == "darker harbors"
    assert effective.sfx.backend == "mmaudio"
    assert effective.sfx.device == "cuda:1"
    assert effective.sfx.model_size == "small_44k"
    assert effective.sfx.sfx_caption == "distant rumble"
    assert effective.sfx.num_workers == 2


def test_configure_update_single_sfx_flag_touches_only_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Updating one SFX flag leaves the other four stored values alone."""
    from voyage import cli_configure
    from voyage.persistence import read_effective_config

    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace(
                "tune",
                style="dark harbors",
                segments=2,
                seed=7,
                sfx_backend="mmaudio",
                sfx_device="cuda:0",
                sfx_model_size="large_44k_v2",
                sfx_caption="distant rumble",
                sfx_workers=2,
            )
        )
        == 0
    )
    assert cli_configure.cmd_configure(_configure_namespace("tune", sfx_workers=1)) == 0
    effective = read_effective_config(tmp_path / "output" / "tune")
    assert effective.sfx.num_workers == 1
    assert effective.sfx.backend == "mmaudio"
    assert effective.sfx.device == "cuda:0"
    assert effective.sfx.model_size == "large_44k_v2"
    assert effective.sfx.sfx_caption == "distant rumble"


def test_finalize_run_dir_passes_sfx_sentinels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`generate` finalizes with None SFX sentinels so the manifest rules."""
    from voyage import cli_generate

    captured: dict[str, object] = {}

    def _stub_finalize(namespace: argparse.Namespace) -> int:
        captured.update(vars(namespace))
        return 0

    monkeypatch.setattr("voyage.cli_finalize.cmd_finalize", _stub_finalize)
    run_dir = tmp_path / "output" / "sentinel"
    run_dir.mkdir(parents=True)
    manifest: dict[str, object] = {"final_video": None, "skip_bad": False, "no_sfx": False}
    assert cli_generate._finalize_run_dir(run_dir, manifest, argparse.Namespace()) == 0
    assert captured["sfx_backend"] is None
    assert captured["sfx_caption"] is None
    assert captured["sfx_device"] is None
    assert captured["sfx_model_size"] is None
    assert captured["sfx_workers"] is None
    assert captured["sfx_dual_pan"] is None
