"""Issue 243: argparse mutex groups + single-error plan-size triple.

Parsed CLI gets standard argparse usage errors for conflicting flags;
hand-built namespaces still resolve via is_provided; invalid
--segments prints exactly one line (no double error).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from voyage.cli import build_parser
from voyage.cli_configure import _resolve_segments_triple


def test_parser_has_mutex_groups() -> None:
    """All four applicable pairs/triples live in exclusive groups (243)."""
    parser = build_parser()
    subparsers: Any = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )
    choices: Any = getattr(subparsers, "choices", {})
    found: set[str] = set()
    for sub in choices.values():
        groups: Any = getattr(sub, "_mutually_exclusive_groups", [])
        for group in groups:
            actions: Any = getattr(group, "_group_actions", [])
            options = {option for action in actions for option in action.option_strings}
            if {"--segments", "--duration"} <= options:
                found.add("plan")
            if {"--low-definition", "--medium-definition", "--high-definition"} <= options:
                found.add("tiers")
            if {"--prompt-enhance", "--no-prompt-enhance"} <= options:
                found.add("enhance")
            if {"--sfx-dual-pan", "--no-sfx-dual-pan"} <= options:
                found.add("dualpan")
    assert {"plan", "tiers", "enhance", "dualpan"} <= found


def test_parser_rejects_conflicting_plan_flags(capsys: pytest.CaptureFixture[str]) -> None:
    """Parsed --segments + --duration exits 2 with argparse usage (243)."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["configure", "x", "--segments", "2", "--duration", "5s"])
    assert exc.value.code == 2


def test_parser_rejects_conflicting_enhance_flags() -> None:
    """Parsed enhance pair exits 2 with argparse usage (243)."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            ["configure", "x", "--prompt-enhance", "--no-prompt-enhance", "--style", "s"]
        )
    assert exc.value.code == 2


def test_triple_separates_absent_from_invalid() -> None:
    """Triple distinguishes absent (ok,None) from invalid (not-ok, error) (243)."""
    ok, value, error = _resolve_segments_triple(argparse.Namespace(), 24, 48)
    assert (ok, value, error) == (True, None, None)
    ok, value, error = _resolve_segments_triple(argparse.Namespace(segments=0), 24, 48)
    assert ok is False and value is None and error is not None and "must be positive" in error
    ok, value, error = _resolve_segments_triple(
        argparse.Namespace(segments=2, duration=5.0), 24, 48
    )
    assert ok is False and "only one" in (error or "")
    ok, value, error = _resolve_segments_triple(argparse.Namespace(segments=3), 24, 48)
    assert (ok, value) == (True, 3)


def test_configure_segments_zero_prints_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """configure --segments 0 prints exactly one stderr line (243)."""
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("voyage.models_ensure.ensure_models", lambda *a, **k: 0)
    args = argparse.Namespace(
        name="zero",
        backend=None,
        from_run=None,
        duration=None,
        segments=0,
        style="s",
        seed=7,
        final_video=None,
        skip_bad=False,
        no_download=True,
        force=False,
        director=None,
        director_device=None,
        blocks=None,
        take_seconds=None,
        quantization=None,
        beats_per_segment=None,
        drift_every_n=None,
        scene_cut_every_n=None,
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
        sfx_workers=None,
        prompt_enhance=None,
        no_prompt_enhance=None,
        sfx_dual_pan=None,
        no_sfx_dual_pan=None,
        low_definition=False,
        medium_definition=False,
        high_definition=False,
        verbose=False,
        no_color=True,
        quiet=False,
    )
    assert cli_configure.cmd_configure(args) == 2
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.startswith("error:")]
    assert len(lines) == 1
    assert "must be positive" in lines[0]


def test_is_provided_still_works_for_hand_built_namespaces() -> None:
    """Hand-built namespaces without attrs read as absent (243)."""
    from voyage.cli_core import _prompt_enhance_overrides, _sfx_dual_pan_overrides

    assert _prompt_enhance_overrides(argparse.Namespace()) == {}
    assert _sfx_dual_pan_overrides(argparse.Namespace()) == {}
