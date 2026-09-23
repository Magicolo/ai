"""Inspector wiring tests: flag default, piggyback merge, amendment safety.

The enabled-path tests run real fake-backend commits (real ffmpeg media),
so sample → summarize → merge is genuine. The `inspected` VALUE is
intentionally not asserted: the slim gates image has no torch (VLM leg
skips), while the director image may load the real inspector — the merge
shape holds in both. The inspected-True leg is covered in test_inspector.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from voyage import paths
from voyage.config import ExperimentalConfig, default_config_toml, load_config
from voyage.errors import ProposalRejected
from voyage.models import StyleSpec
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_state,
    write_manifest,
    write_state,
)
from voyage.prompts import (
    apply_feedback_amendments,
    check_prompt_against_style,
    feedback_amendments,
)
from voyage.supervisor import Supervisor

VISUAL_METRIC_KEYS = (
    "motion_energy",
    "visual_complexity",
    "semantic_change_rate",
    "palette_distance",
    "style_similarity",
    "scene_boundary_strength",
)


def _init_run(run_dir: Path, *, inspector: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    toml = default_config_toml("iwire", "pastel neon line-art, peaceful", 7)
    if inspector:
        toml = toml.replace("visual_inspector = false", "visual_inspector = true")
    (run_dir / paths.CONFIG_FILENAME).write_text(toml, encoding="utf-8")
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


def _read_metrics(run_dir: Path, segment_id: str) -> dict[str, object]:
    raw = json.loads(
        (paths.segment_dir(run_dir, segment_id) / "metrics.json").read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict)
    return raw


def test_experimental_flag_defaults_off() -> None:
    assert ExperimentalConfig().visual_inspector is False
    assert "visual_inspector = false" in default_config_toml("x", "pastel", 1)


def test_disabled_run_writes_no_visual_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir, inspector=False)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
    finally:
        supervisor.stop_workers()
    assert "visual" not in _read_metrics(run_dir, "000000")
    assert read_state(run_dir).committed_segments == 1


def test_enabled_run_merges_visual_section(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir, inspector=True)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert config.experimental.visual_inspector is True
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
        assert supervisor.commit_one_segment() == "000001"
    finally:
        supervisor.stop_workers()
    merged = _read_metrics(run_dir, "000000")
    # Original keys survive the read-modify-write merge.
    for key in ("video", "audio", "frames"):
        assert key in merged, key
    visual = merged["visual"]
    assert isinstance(visual, dict)
    assert isinstance(visual.get("inspected"), bool)
    assert isinstance(visual.get("scene_summary"), str)
    assert isinstance(visual.get("amendments"), list)
    summary = visual["metrics"]
    assert isinstance(summary, dict)
    for key in VISUAL_METRIC_KEYS:
        assert key in summary, key
    assert read_state(run_dir).committed_segments == 2


def test_amendments_never_trip_style_markers() -> None:
    style = StyleSpec(prompt="pastel neon line-art, peaceful")
    extremes = [
        {"motion_energy": 0.0, "visual_complexity": 0.0, "semantic_change_rate": 1.0},
        {"motion_energy": 1.0, "visual_complexity": 1.0, "semantic_change_rate": 0.0},
        {"palette_distance": 1.0, "style_similarity": 0.0, "scene_boundary_strength": 1.0},
    ]
    for measured in extremes:
        amended = apply_feedback_amendments(
            "a calm neon landscape", feedback_amendments(measured, style)
        )
        try:
            check_prompt_against_style(amended, style)
        except ProposalRejected as exc:
            raise AssertionError(f"amendment tripped style markers: {measured}") from exc
