"""Torn manifest never crashes ensure (issue 197).

Why this module exists: a killed parallel download can tear
`manifest.json` — the exact race the repair pass heals. `JSONDecodeError`
subclasses `ValueError` (disjoint from `OSError`), so a repair that
catches only `OSError` propagates the tear out of `ensure_models` after
every weight verified. The repair must treat torn as missing and fail
loud with a clean error (contrast 077's silent-repair half: hash-less
success must never report ready).
"""

from __future__ import annotations

from pathlib import Path

from voyage.models_ensure import RequiredModel, _read_manifest_keys, _repair_manifest


def test_read_manifest_keys_torn_returns_none(tmp_path: Path) -> None:
    """Torn JSON degrades to None (validate's territory), never raises."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text("{torn", encoding="utf-8")
    assert _read_manifest_keys(models_dir) is None


def test_repair_manifest_torn_fails_loud_not_crash(tmp_path: Path) -> None:
    """Torn manifest lists entries still-missing (clean error, no traceback)."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text("{torn", encoding="utf-8")
    entry = RequiredModel(spec="film", models_dir=models_dir)
    still_missing = _repair_manifest([entry])
    assert still_missing == [entry]
