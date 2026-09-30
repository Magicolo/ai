"""Doctor probing: torch/disk/models facts degrade honestly (issue 048).

All run without a GPU: torch presence is simulated via `find_spec` +
`sys.modules` so both the torch-less slim image and the CUDA worker
image paths are covered.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest

from voyage import doctor
from voyage.doctor import check_director_python, check_models, director_python, probe

EXPECTED_KEYS = {
    "python",
    "ffmpeg",
    "ffprobe",
    "ffmpeg_version",
    "nvidia_smi",
    "gpus",
    "torch_cuda",
    "disk_free_gib",
    "models",
    "models_ok",
    "director_python",
    "director_python_exists",
}


def test_probe_carries_extended_keys() -> None:
    """`probe()` reports torch/disk/models facts on top of the old six."""
    facts = probe()
    assert EXPECTED_KEYS <= set(facts)
    assert isinstance(facts["python"], str)
    assert isinstance(facts["gpus"], list)
    assert facts["torch_cuda"] is None or isinstance(facts["torch_cuda"], bool)
    assert facts["disk_free_gib"] is None or isinstance(facts["disk_free_gib"], float)
    assert isinstance(facts["models"], dict)
    assert isinstance(facts["models_ok"], bool)


def test_probe_torch_absent_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No torch installed (slim image) → `torch_cuda` is None, never raises."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert probe()["torch_cuda"] is None


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    fake_cuda = types.SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())


def test_probe_torch_present_reports_cuda_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch importable (worker image) → real `cuda.is_available()` value."""
    _install_fake_torch(monkeypatch, True)
    assert probe()["torch_cuda"] is True
    _install_fake_torch(monkeypatch, False)
    assert probe()["torch_cuda"] is False


def test_probe_torch_broken_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A torch whose CUDA query explodes still degrades to None."""
    broken_cuda = types.SimpleNamespace(
        is_available=lambda: (_ for _ in ()).throw(RuntimeError("no driver"))
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=broken_cuda))
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    assert probe()["torch_cuda"] is None


def test_probe_disk_failure_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable root fs degrades the disk fact instead of raising."""

    def _fail(_path: object) -> object:
        raise OSError("no stat")

    monkeypatch.setattr(shutil, "disk_usage", _fail)
    assert probe()["disk_free_gib"] is None


def test_check_models_empty_dir_fails_all_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty models dir reports missing checks (never exit-0-green)."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setenv(doctor.MODELS_DIR_ENVIRONMENT_VARIABLE, str(models_dir))
    summary = check_models()
    assert summary["dir"] == str(models_dir)
    assert summary["exists"] is True
    assert summary["manifest"] is False
    assert summary["checks"]
    assert all(check["ok"] is False for check in summary["checks"].values())
    assert probe()["models_ok"] is False


def test_check_models_missing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonexistent models dir reports exists=False with failing checks."""
    missing = tmp_path / "no-such-models"
    monkeypatch.setenv(doctor.MODELS_DIR_ENVIRONMENT_VARIABLE, str(missing))
    summary = check_models()
    assert summary["exists"] is False
    assert summary["manifest"] is False
    assert all(check["ok"] is False for check in summary["checks"].values())


def test_director_python_unset_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var (slim image / host) → None path, exists False, never raises."""
    monkeypatch.delenv(doctor.DIRECTOR_PYTHON_ENVIRONMENT_VARIABLE, raising=False)
    assert director_python() is None
    summary = check_director_python()
    assert summary == {"path": None, "exists": False}
    facts = probe()
    assert facts["director_python"] is None
    assert facts["director_python_exists"] is False


def test_director_python_present_reports_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set env var → path reported; exists tracks the filesystem."""
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv(doctor.DIRECTOR_PYTHON_ENVIRONMENT_VARIABLE, str(interpreter))
    assert director_python() == interpreter
    assert check_director_python()["exists"] is True
    facts = probe()
    assert facts["director_python"] == str(interpreter)
    assert facts["director_python_exists"] is True
    interpreter.unlink()
    assert check_director_python()["exists"] is False
