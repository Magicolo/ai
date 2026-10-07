"""Run-manifest schema versioning (issue 226).

`build_manifest` stamps `schema_version = paths.SCHEMA_VERSION`;
`read_manifest` accepts a missing key as legacy v1 and fails loud on
unknown-future or non-integer versions, so the next breaking change
needs no new content-sniff carve-out. No GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import paths
from voyage.config import ProjectConfig
from voyage.errors import StateError
from voyage.persistence import (
    build_manifest,
    read_effective_config,
    read_manifest,
    record_final_coverage,
    write_manifest,
)


def _pinned_config() -> ProjectConfig:
    return ProjectConfig(style="pastel neon line-art, peaceful")


def test_build_manifest_stamps_schema_version() -> None:
    """Fresh manifests carry the live schema version."""
    manifest = build_manifest(_pinned_config())
    assert manifest["schema_version"] == paths.SCHEMA_VERSION


def test_round_trip_reads_current_version(tmp_path: Path) -> None:
    """A stamped manifest reads back cleanly."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_manifest(run_dir, build_manifest(_pinned_config()))
    assert read_manifest(run_dir)["schema_version"] == paths.SCHEMA_VERSION
    assert read_effective_config(run_dir).style == "pastel neon line-art, peaceful"


def test_future_schema_version_fails_loud(tmp_path: Path) -> None:
    """An unknown-future manifest refuses instead of misreading silently."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = build_manifest(_pinned_config())
    manifest["schema_version"] = paths.SCHEMA_VERSION + 999
    write_manifest(run_dir, manifest)
    with pytest.raises(StateError, match="schema_version"):
        read_manifest(run_dir)
    with pytest.raises(StateError, match="schema_version"):
        read_effective_config(run_dir)


def test_missing_schema_version_reads_as_legacy(tmp_path: Path) -> None:
    """Pre-226 manifests (no key) keep reading: the only shipped past is v1."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = build_manifest(_pinned_config())
    del manifest["schema_version"]
    write_manifest(run_dir, manifest)
    assert "schema_version" not in read_manifest(run_dir)
    assert read_effective_config(run_dir).style == "pastel neon line-art, peaceful"


def test_non_integer_schema_version_fails_loud(tmp_path: Path) -> None:
    """A corrupt version stamp reads as StateError, never a guess."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = build_manifest(_pinned_config())
    manifest["schema_version"] = "v2"
    write_manifest(run_dir, manifest)
    with pytest.raises(StateError, match="schema_version"):
        read_manifest(run_dir)


def test_final_coverage_preserves_schema_version(tmp_path: Path) -> None:
    """The freshness-stamp rewrite keeps the version key it found."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_manifest(run_dir, build_manifest(_pinned_config()))
    record_final_coverage(run_dir, presented_frames=96, segments=1)
    assert read_manifest(run_dir)["schema_version"] == paths.SCHEMA_VERSION
