"""LTXV text-encoder load-time enforcement (issue 238).

CPU-only: the checklist gate is stdlib + registry — no torch, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import model_registry
from voyage.workers import video_ltxv


def _write_te_snapshot(models: Path) -> Path:
    """Minimal checklist-satisfying PixArt TE snapshot (tokenizer + encoder)."""
    te_dir = models / model_registry.LTXV_TE_SUBDIR
    tokenizer_dir = te_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    encoder_dir = te_dir / "text_encoder"
    encoder_dir.mkdir(parents=True)
    (encoder_dir / "config.json").write_text("{}", encoding="utf-8")
    return te_dir


def test_te_gate_passes_for_checklist_clean_snapshot(tmp_path: Path) -> None:
    models = tmp_path / "models"
    expected = _write_te_snapshot(models)
    video_ltxv._assert_te_snapshot_enforced(models, str(expected))


def test_te_gate_refuses_stale_snapshot(tmp_path: Path) -> None:
    """A hand-rolled dir (no tokenizer/encoder checklist) fails loud."""
    models = tmp_path / "models"
    stale = models / model_registry.LTXV_TE_SUBDIR
    stale.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="stale or incomplete"):
        video_ltxv._assert_te_snapshot_enforced(models, str(stale))


def test_te_gate_refuses_foreign_source(tmp_path: Path) -> None:
    """A hub id (the unreachable fallback shape) never loads."""
    models = tmp_path / "models"
    _write_te_snapshot(models)
    with pytest.raises(RuntimeError, match="stale or incomplete"):
        video_ltxv._assert_te_snapshot_enforced(models, model_registry.LTXV_TE_REPO)
