"""Handoff-contract tests for issues 003 / 020 / 022 (cli/config/tui_state scope).

Why this module exists: three Rank-1 findings all concern the
validate/generate handoff owned by this track — `validate_run` must
enforce the 0.6 s A/V budget (003, read-only, error strings never
raises), `default_config_toml` must survive C0-control free-text
(020, single escaper), and `cmd_generate` must forward the caption
pins into its inner `cmd_run` call (022). One module so the owned
test surface stays in the owned file; sibling tracks own the
commit-side (supervisor), registry, and media internals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import pytest

pytest.importorskip("hypothesis", reason="property tests need Hypothesis")

import tomllib
from hypothesis import given
from hypothesis import strategies as strategies
from hypothesis.strategies import DataObject

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.cli import validate_run
from voyage.config import default_config_toml, load_config

_SURROGATE_CATEGORY: tuple[Literal["Cs"], ...] = ("Cs",)


def _commit_one(run_dir: Path) -> list[str]:
    from voyage.supervisor import Supervisor

    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    return Supervisor(run_dir, config).run_segments(1)


def test_validate_passes_aligned_segment(tmp_path: Path) -> None:
    """Aligned commit stays VALID (003 read-only gate, happy path)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    assert _commit_one(run_dir) == ["000000"]
    assert validate_run(run_dir) == []


def test_validate_rejects_misaligned_stored_durations(tmp_path: Path) -> None:
    """Crafted 2 s vs 10 s stored durations must fail validate (003)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    assert _commit_one(run_dir) == ["000000"]
    metrics_path = run_dir / "segments" / "000000" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["video"]["duration"] = 2.0
    payload["audio"]["duration"] = 10.0
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_run(run_dir)
    assert any("drift" in error and "000000" in error for error in errors)


def test_toml_escaper_survives_bel_and_esc() -> None:
    """Deterministic pins: BEL/ESC styles must parse and survive (020)."""
    for style in ("style\x07bell", "style\x1besc", 'quote"back\\slash'):
        text = default_config_toml("probe", style, 0)
        parsed = tomllib.loads(text)
        assert parsed["style"] == style


def test_toml_escaper_survives_del_and_nul() -> None:
    """DEL/NUL pins: the full-C0 escaper must cover them too (020)."""
    for style in ("style\x7fdel", "style\x00nul"):
        text = default_config_toml("probe", style, 0)
        parsed = tomllib.loads(text)
        assert parsed["style"] == style


@given(strategies.data())
def test_toml_round_trip_holds_for_control_text(data: DataObject) -> None:
    """Property: any C0-control style survives the TOML round-trip (020)."""
    alphabet = strategies.characters(blacklist_categories=_SURROGATE_CATEGORY, max_codepoint=0x7F)
    raw = data.draw(strategies.text(alphabet=alphabet, max_size=24))
    style = f"x{raw}y"  # non-empty even when the draw is empty/blank
    text = default_config_toml("probe", style, 0)
    assert tomllib.loads(text)["style"] == style


def test_generate_forwards_caption_pins_to_inner_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract: generate --music/--video-caption must reach cmd_run (022)."""
    import voyage.cli as cli_module
    import voyage.models_ensure as ensure_module

    run_dir = tmp_path / "run"
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--backend",
            "fake",
            "--duration",
            "2s",
            "--style",
            "pastel neon line-art, peaceful",
            "--output",
            str(run_dir),
            "--run-id",
            "pins",
            "--seed",
            "11",
            "--music-caption",
            "brass fanfare",
            "--video-caption",
            "red dune",
        ]
    )
    assert isinstance(args, argparse.Namespace)
    captured: dict[str, argparse.Namespace] = {}

    def _fake_run(inner: argparse.Namespace) -> int:
        captured["namespace"] = inner
        return 0

    monkeypatch.setattr(cli_module, "cmd_run", _fake_run)
    monkeypatch.setattr(cli_module, "cmd_finalize", lambda _final: 0)
    monkeypatch.setattr(cli_module, "validate_run", lambda _run: [])
    monkeypatch.setattr(cli_module, "_warn_if_no_cuda", lambda _config: None)
    monkeypatch.setattr(cli_module, "check_free_space", lambda _a, _b: 0.0)
    monkeypatch.setattr(ensure_module, "ensure_models", lambda *_a, **_k: 0)
    assert cli_module.cmd_generate(args) == 0
    inner = captured["namespace"]
    assert getattr(inner, "music_caption", None) == "brass fanfare"
    assert getattr(inner, "video_caption", None) == "red dune"
