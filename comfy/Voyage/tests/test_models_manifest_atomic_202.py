"""Models manifest merges atomically with taxonomy errors (issue 202).

Why this module exists: `_merge_manifest_record` wrote
`models/manifest.json` with a direct in-place `write_text` — SIGKILL
mid-write left torn JSON that later verifies misread — while every run
manifest/state went through `atomic_write_json`. The merge now stages
through the shared atomic path (inheriting the issue-222 mode pins)
and unreadable manifests read as `StateError`, matching
`persistence.read_manifest` (issue 002).
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from voyage import model_registry
from voyage.errors import StateError


def test_merge_manifest_record_merges_and_roundtrips(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_registry._merge_manifest_record(models_dir, "film", {"repo": "r1"})
    model_registry._merge_manifest_record(models_dir, "rife", {"repo": "r2"})
    stored = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored == {"film": {"repo": "r1"}, "rife": {"repo": "r2"}}
    assert list(models_dir.glob("*.partial")) == []


def test_merge_manifest_record_routes_through_atomic_write_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, object]] = []
    real = model_registry.atomic_write_json

    def _recording(destination: Path, payload: object) -> None:
        calls.append((destination, payload))
        real(destination, payload)

    monkeypatch.setattr(model_registry, "atomic_write_json", _recording)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_registry._merge_manifest_record(models_dir, "film", {"repo": "r1"})
    assert len(calls) == 1
    assert calls[0][0] == models_dir / "manifest.json"


def test_merge_manifest_record_new_manifest_is_0644(tmp_path: Path) -> None:
    """Composition with issue 222: fresh models manifests land 0644."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_registry._merge_manifest_record(models_dir, "film", {"repo": "r1"})
    assert stat.S_IMODE((models_dir / "manifest.json").stat().st_mode) == 0o644


def test_merge_manifest_record_torn_manifest_raises_state_error(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    torn = models_dir / "manifest.json"
    torn.write_text("{torn", encoding="utf-8")
    with pytest.raises(StateError, match="invalid models manifest"):
        model_registry._merge_manifest_record(models_dir, "film", {"repo": "r1"})
    assert torn.read_text(encoding="utf-8") == "{torn"


def test_merge_manifest_record_non_object_raises_state_error(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(StateError, match="not a JSON object"):
        model_registry._merge_manifest_record(models_dir, "film", {"repo": "r1"})


def test_merge_manifest_record_failed_write_keeps_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fault-injected crash between temp-stage and replace keeps history."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_registry._merge_manifest_record(models_dir, "film", {"repo": "r1"})
    before = (models_dir / "manifest.json").read_bytes()

    def _crash_after_staging(destination: Path, payload: object) -> None:
        litter = destination.parent / (destination.name + ".staged.partial")
        litter.write_bytes(b"staged-then-killed")
        raise RuntimeError("SIGKILL mid-write")

    monkeypatch.setattr(model_registry, "atomic_write_json", _crash_after_staging)
    with pytest.raises(RuntimeError, match="SIGKILL"):
        model_registry._merge_manifest_record(models_dir, "rife", {"repo": "r2"})
    assert (models_dir / "manifest.json").read_bytes() == before
