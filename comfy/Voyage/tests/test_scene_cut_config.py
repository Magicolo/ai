"""Configurable scene-cut cadence (`--scene-cut-every-n`, default 3).

Pins the configure-verb knob for the periodic fresh visual start:
config default + validation, `resolve_config` override, CLI flag
parsing, configure round-trip persistence, and the supervisor helper
cadence (every Nth segment cuts, first segment never does).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from voyage.config import VoyageConfig, resolve_config
from voyage.supervisor import SCENE_CUT_EVERY_N_SEGMENTS, scene_cut_for_segment


@pytest.fixture(autouse=True)
def _never_touch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voyage.models_ensure.ensure_models", lambda *a, **k: 0)


def test_cadence_default_is_three() -> None:
    assert SCENE_CUT_EVERY_N_SEGMENTS == 3
    assert VoyageConfig().scene_cut_every_n_segments == 3


def test_cadence_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        VoyageConfig(scene_cut_every_n_segments=0)
    with pytest.raises(ValidationError):
        VoyageConfig(scene_cut_every_n_segments=-2)


def test_helper_cuts_every_third_segment() -> None:
    assert [n for n in range(9) if scene_cut_for_segment(n, 3)] == [2, 5, 8]
    assert not scene_cut_for_segment(0, 3)


def test_helper_honors_custom_cadence() -> None:
    assert [n for n in range(7) if scene_cut_for_segment(n, 2)] == [1, 3, 5]
    assert [n for n in range(13) if scene_cut_for_segment(n, 6)] == [5, 11]


def test_helper_never_cuts_on_non_positive_cadence() -> None:
    assert not any(scene_cut_for_segment(n, 0) for n in range(12))
    assert not any(scene_cut_for_segment(n, -3) for n in range(12))


def test_resolve_config_overrides_cadence() -> None:
    from voyage.config import preset_config

    base = preset_config("cadence", "pastel neon line-art, peaceful", 7)
    out = resolve_config(base, scene_cut_every_n_segments=5)
    assert out.voyage.scene_cut_every_n_segments == 5
    assert base.voyage.scene_cut_every_n_segments == 3
    with pytest.raises(ValidationError):
        resolve_config(base, scene_cut_every_n_segments=0)


def test_cli_parses_scene_cut_every_n_flag() -> None:
    from voyage import cli

    args = cli.build_parser().parse_args(["configure", "c1", "--scene-cut-every-n", "5"])
    assert args.scene_cut_every_n == 5
    assert (
        cli.build_parser()
        .parse_args(["configure", "c1", "--style", "s", "--segments", "2"])
        .scene_cut_every_n
        is None
    )


def test_configure_round_trip_persists_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        name="cine",
        style="dark harbors",
        segments=3,
        seed=7,
        scene_cut_every_n=5,
        no_download=True,
        backend=None,
        from_run=None,
        duration=None,
        final_video=None,
        skip_bad=False,
        force=False,
        director=None,
        director_device=None,
        blocks=None,
        take_seconds=None,
        quantization=None,
        beats_per_segment=None,
        drift_every_n=None,
        music_caption=None,
        video_caption=None,
        upscale=None,
        interpolate=None,
        presentation_fps=None,
        no_sfx=False,
        sfx_backend=None,
        sfx_caption=None,
        sfx_device=None,
        sfx_model_size=None,
        sfx_workers=1,
        verbose=False,
        no_color=True,
    )
    assert cli_configure.cmd_configure(args) == 0
    manifest: dict[str, Any] = json.loads(
        (tmp_path / "output" / "cine" / "manifest.json").read_text()
    )
    assert manifest["voyage"]["scene_cut_every_n_segments"] == 5
