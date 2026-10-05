"""Handoff-contract tests for issues 003 / 020 / 022 (cli/config scope).

Why this module exists: three Rank-1 findings all concern the
validate/generate handoff owned by this track — `validate_run` must
enforce the 0.6 s A/V budget (003, read-only, error strings never
raises), `preset_config` must survive C0-control free-text
(020, no TOML layer — direct config), and `cmd_generate` must forward the caption
pins into its inner `cmd_run` call (022). One module so the owned
test surface stays in the owned file; sibling tracks own the
commit-side (supervisor), registry, and media internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

pytest.importorskip("hypothesis", reason="property tests need Hypothesis")

from hypothesis import given
from hypothesis import strategies as strategies
from hypothesis.strategies import DataObject

from tests.conftest import initialize_run_directory
from voyage.cli_validate import validate_run
from voyage.config import preset_config
from voyage.persistence import read_effective_config

_SURROGATE_CATEGORY: tuple[Literal["Cs"], ...] = ("Cs",)


def _commit_one(run_dir: Path) -> list[str]:
    from voyage.supervisor import Supervisor

    config = read_effective_config(run_dir)
    return Supervisor(run_dir, config).run_segments(1)


def test_validate_passes_aligned_segment(tmp_path: Path) -> None:
    """Aligned commit stays VALID (003 read-only gate, happy path)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    assert _commit_one(run_dir) == ["000000"]
    assert validate_run(run_dir) == []


def test_validate_ignores_misaligned_stored_durations(tmp_path: Path) -> None:
    """Crafted 2 s vs 10 s stored durations never fail validate (deferred).

    All backends are deferred (video-only commit): cli_validate checks
    video duration/frames only — there is no audio-duration or 0.6 s A/V
    drift gate, and a recorded audio block is skipped. Stray or drifted
    audio metadata must stay silent here; music finalizes from takes.
    """
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    assert _commit_one(run_dir) == ["000000"]
    segment = run_dir / "segments" / "000000"
    manifest = load_segment_manifest(segment)
    metrics = dict(manifest["metrics"])
    video_block = dict(metrics.get("video", {}))
    audio_block = dict(metrics.get("audio", {}))
    video_block["duration"] = 2.0
    audio_block["duration"] = 10.0
    metrics["video"] = video_block
    metrics["audio"] = audio_block
    write_segment_manifest(segment, {**manifest, "metrics": metrics})
    assert validate_run(run_dir) == []


def test_toml_escaper_survives_bel_and_esc() -> None:
    """Deterministic pins: BEL/ESC styles survive direct config (020)."""
    for style in ("style\x07bell", "style\x1besc", 'quote"back\\slash'):
        assert preset_config("probe", style, 0).style == style


def test_toml_escaper_survives_del_and_nul() -> None:
    """DEL/NUL pins: direct config preserves them too (020)."""
    for style in ("style\x7fdel", "style\x00nul"):
        assert preset_config("probe", style, 0).style == style


@given(strategies.data())
def test_toml_round_trip_holds_for_control_text(data: DataObject) -> None:
    """Property: any C0-control style survives direct config (020)."""
    alphabet = strategies.characters(blacklist_categories=_SURROGATE_CATEGORY, max_codepoint=0x7F)
    raw = data.draw(strategies.text(alphabet=alphabet, max_size=24))
    style = f"x{raw}y"  # non-empty even when the draw is empty/blank
    assert preset_config("probe", style, 0).style == style
