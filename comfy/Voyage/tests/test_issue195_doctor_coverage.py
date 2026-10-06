"""Doctor covers all 11 default-CUDA registry stacks (issue 195).

Why this module exists: `check_models` hardcoded six verifiers while
`MODEL_SPECS` + `models verify` + `models_ensure.required_specs` all
cover ten — the five newest default-CUDA stacks (film, realesrgan-anime, rife,
sfx-mmaudio, director-qwen4b-awq) were invisible, so `doctor` reported
`models_ok: true` on a box missing ~16 GB of default-path weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import doctor
from voyage.doctor import check_models

_EXPECTED_NEW_STACKS = (
    "film",
    "realesrgan-anime",
    "rife",
    "sfx-mmaudio",
    "director-qwen4b-awq",
)


def test_check_models_covers_new_default_stacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All five post-Qwen default stacks appear as named checks."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setenv(doctor.MODELS_DIR_ENVIRONMENT_VARIABLE, str(models_dir))
    summary = check_models()
    for name in _EXPECTED_NEW_STACKS:
        assert name in summary["checks"], f"{name} missing from check_models"


def test_check_models_empty_dir_fails_new_stacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty dir reports the new stacks not-ok (never silent-ready)."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setenv(doctor.MODELS_DIR_ENVIRONMENT_VARIABLE, str(models_dir))
    summary = check_models()
    for name in _EXPECTED_NEW_STACKS:
        assert summary["checks"][name]["ok"] is False
