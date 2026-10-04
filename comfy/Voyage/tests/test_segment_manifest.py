"""Per-segment manifest pruning: manifest.json + tmpdir slices (DESIGN §56).

TDD-first contract for the run-file pruning task: commit writes ONE
manifest.json (no individual JSONs), slices stay out of the segment dir,
legacy layouts still load, and the inspect merge updates the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.persistence import read_effective_config


def _commit_one(run_dir: Path) -> Path:
    from voyage.supervisor import Supervisor

    config, _ = read_effective_config(run_dir)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    return run_dir / "segments" / "000000"


def test_commit_writes_manifest_without_individual_jsons(tmp_path: Path) -> None:
    """Commit writes manifest.json + DONE and no individual JSONs."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="manifest")
    segment = _commit_one(run_dir)
    assert (segment / "manifest.json").exists()
    assert (segment / paths.DONE_MARKER).exists()
    for name in (
        "transition.json",
        "prompt_plan.json",
        "audio_state.json",
        "world_state.json",
        "metrics.json",
        "sha256.json",
    ):
        assert not (segment / name).exists(), name
    payload = json.loads((segment / "manifest.json").read_text(encoding="utf-8"))
    assert payload["format"] == 1
    for key in ("transition", "prompt_plan", "audio_state", "world_state", "metrics"):
        assert isinstance(payload[key], dict), key
    assert isinstance(payload["checksums"], dict)
    assert isinstance(payload["checksums"].get("video.mp4"), str)
    assert isinstance(payload["checksums"].get("audio.wav"), str)


def test_slices_land_in_tmpdir_not_segment_dir(tmp_path: Path) -> None:
    """Audio slices never persist inside the committed segment dir."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="slices")
    segment = _commit_one(run_dir)
    leftovers = list(segment.glob("slice_*.wav"))
    assert leftovers == []


def test_legacy_layout_loads_via_helper(tmp_path: Path) -> None:
    """Legacy individual files load through the manifest helper."""
    from voyage.segment_manifest import load_segment_manifest

    segment = tmp_path / "seg"
    segment.mkdir(parents=True)
    (segment / "transition.json").write_text(json.dumps({"phase": "HOLD"}), encoding="utf-8")
    prompt_file = segment / "prompt_plan.json"
    prompt_file.write_text(json.dumps({"segment_id": "000000"}), encoding="utf-8")
    (segment / "audio_state.json").write_text(json.dumps({"take_ids": []}), encoding="utf-8")
    world_file = segment / "world_state.json"
    world_file.write_text(json.dumps({"segment_id": "000000"}), encoding="utf-8")
    (segment / "metrics.json").write_text(json.dumps({"frames": 48}), encoding="utf-8")
    (segment / "sha256.json").write_text(
        json.dumps({"video.mp4": "abc", "audio.wav": "def"}), encoding="utf-8"
    )
    loaded = load_segment_manifest(segment)
    assert loaded["metrics"] == {"frames": 48}
    assert loaded["transition"] == {"phase": "HOLD"}
    assert loaded["checksums"] == {"video.mp4": "abc", "audio.wav": "def"}


def test_inspect_merge_updates_manifest(tmp_path: Path) -> None:
    """Inspect merge rewrites the manifest metrics section."""
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest

    segment = tmp_path / "seg"
    segment.mkdir(parents=True)
    manifest = {
        "format": 1,
        "transition": {},
        "prompt_plan": {},
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": 48},
        "checksums": {"video.mp4": "abc", "audio.wav": "def"},
    }
    write_segment_manifest(segment, manifest)
    loaded = load_segment_manifest(segment)
    updated_metrics = {**loaded["metrics"], "visual": {"inspected": False}}
    write_segment_manifest(segment, {**loaded, "metrics": updated_metrics})
    reloaded = load_segment_manifest(segment)
    assert reloaded["metrics"]["visual"] == {"inspected": False}
    assert reloaded["checksums"] == {"video.mp4": "abc", "audio.wav": "def"}


def test_legacy_inspect_merge_refreshes_metrics_checksum(tmp_path: Path) -> None:
    """Legacy path: metrics rewrite refreshes the sha256.json entry."""
    from voyage.hashing import sha256_file
    from voyage.segment_manifest import load_metrics, update_manifest_metrics

    segment = tmp_path / "seg"
    segment.mkdir(parents=True)
    (segment / "metrics.json").write_text(json.dumps({"frames": 48}), encoding="utf-8")
    (segment / "sha256.json").write_text(
        json.dumps({"video.mp4": "abc", "metrics.json": "stale"}), encoding="utf-8"
    )
    update_manifest_metrics(segment, {"frames": 48, "visual": {"inspected": False}})
    assert load_metrics(segment)["visual"] == {"inspected": False}
    recorded = json.loads((segment / "sha256.json").read_text(encoding="utf-8"))
    assert recorded["metrics.json"] == sha256_file(segment / "metrics.json")
    assert recorded["video.mp4"] == "abc"


def test_legacy_inspect_merge_without_checksum_file(tmp_path: Path) -> None:
    """Legacy path with no sha256.json: metrics rewrite alone, no crash."""
    from voyage.segment_manifest import load_metrics, update_manifest_metrics

    segment = tmp_path / "seg"
    segment.mkdir(parents=True)
    update_manifest_metrics(segment, {"frames": 48})
    assert load_metrics(segment) == {"frames": 48}
    assert not (segment / "sha256.json").exists()
