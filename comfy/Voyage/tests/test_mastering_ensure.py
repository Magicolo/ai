"""SonicMaster Track D e2e pins: mastering venv/image/docs + ensure seam.

Scope (Track D): worker venv/image + docs + pins ONLY — the registry spec,
supervisor wiring, and GPU proof land in later tracks. These tests pin what
Track D owns:

- both worker Dockerfiles carry the isolated `/opt/venvs/mastering` stack
  with the exact pins (and leave the ACE/SFX pins undisturbed);
- both build scripts smoke-assert the venv;
- `docs/MODELS.md` documents the gated VAE + provisioning;
- the `VOYAGE_MASTERING_PYTHON` executable seam resolves (doctor helpers +
  the generic `SubprocessWorker(executable=...)` plumbing mastering reuses);
- the generic registry download/verify wrappers fail loud offline (the
  mastering-specific spec is deferred to Track C, so the suite pins the
  fail-loud contract instead of a spec that does not exist yet).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voyage import doctor
from voyage.doctor import check_mastering_python, mastering_python, probe

REPO_ROOT = Path(__file__).resolve().parent.parent
"""Voyage root (tests run with any CWD; pins resolve from the file)."""

LTX_DOCKERFILE = REPO_ROOT / "worker" / "Dockerfile.ltx"
VIDEO_DOCKERFILE = REPO_ROOT / "worker" / "Dockerfile.video"
BUILD_LTX = REPO_ROOT / "scripts" / "build-ltx.sh"
BUILD_VIDEO = REPO_ROOT / "scripts" / "build-video.sh"
MODELS_DOC = REPO_ROOT / "docs" / "MODELS.md"

_MASTERING_VENV = "/opt/venvs/mastering"
_MASTERING_INTERPRETER = "/opt/venvs/mastering/bin/python"
_MASTERING_PINS = (
    "torch==2.4.0",
    "cu124",
    "transformers==4.44.0",
    "diffusers==0.30.0",
    "soundfile==0.14.0",
    "safetensors==0.8.0",
    "huggingface_hub==0.36.2",
    "pydantic==2.10.6",
)
"""Exact pip pins both images bake (hub<1.0 per transformers 4.44)."""


def test_ltx_dockerfile_carries_mastering_venv() -> None:
    """`Dockerfile.ltx` bakes the isolated mastering stack on python3.11."""
    text = LTX_DOCKERFILE.read_text(encoding="utf-8")
    assert "python3.11 -m venv /opt/venvs/mastering" in text
    for pin in _MASTERING_PINS:
        assert pin in text, pin
    assert "VOYAGE_MASTERING_PYTHON=/opt/venvs/mastering/bin/python" in text


def test_ltx_dockerfile_leaves_ace_sfx_pins_undisturbed() -> None:
    """Track D adds a venv; it never re-pins the ACE/SFX/director stacks."""
    text = LTX_DOCKERFILE.read_text(encoding="utf-8")
    assert "/opt/venvs/acestep" in text
    assert "/opt/venvs/sfx" in text
    assert "torch==2.14.0" in text
    assert "transformers==4.57.6" in text
    assert "transformers==5.17.0" in text


def test_video_dockerfile_carries_mastering_venv() -> None:
    """`Dockerfile.video` carries the same stack (finalize runs here too)."""
    text = VIDEO_DOCKERFILE.read_text(encoding="utf-8")
    assert "python3.10 -m venv /opt/venvs/mastering" in text
    for pin in _MASTERING_PINS:
        assert pin in text, pin
    assert "VOYAGE_MASTERING_PYTHON=/opt/venvs/mastering/bin/python" in text


def test_video_dockerfile_leaves_main_env_pins_undisturbed() -> None:
    """Track D adds a venv; the video main env + director venv keep pins."""
    text = VIDEO_DOCKERFILE.read_text(encoding="utf-8")
    assert "torch==2.8.0" in text
    assert "/opt/venvs/director" in text
    assert "transformers==4.57.6" in text


def test_mastering_venv_avoids_unbuildable_python() -> None:
    """No python3.13 interpreter install: torch 2.4.0+cu124 ships no cp313 wheel.

    The cu124 index carries cp38-cp312 only (verified live); pairing the
    requested 3.13 with this torch pin would fail pip resolution at build
    time, so Track D reuses each image's existing interpreter instead.
    (The Dockerfile comments name the rejected version in prose, so this
    pins the install form, not the prose.)
    """
    for dockerfile in (LTX_DOCKERFILE, VIDEO_DOCKERFILE):
        text = dockerfile.read_text(encoding="utf-8")
        assert "python3.13 -m venv" not in text
        assert "python3.13-venv" not in text
        assert "python3.13-dev" not in text


def test_build_scripts_smoke_mastering_venv() -> None:
    """Both build scripts gate on the mastering venv (build-ltx.sh pattern)."""
    for script in (BUILD_LTX, BUILD_VIDEO):
        text = script.read_text(encoding="utf-8")
        assert "/opt/venvs/mastering/bin/python" in text, script.name
        assert "transformers.__version__ == '4.44.0'" in text, script.name
        assert "diffusers.__version__ == '0.30.0'" in text, script.name
        assert "mastering venv ok" in text, script.name


def test_models_docs_cover_sonicmaster() -> None:
    """`docs/MODELS.md` notes the gated VAE + provisioning (Track A owns pins)."""
    text = MODELS_DOC.read_text(encoding="utf-8")
    assert "SonicMaster" in text
    assert "gated" in text
    assert "VAE" in text
    assert "VOYAGE_MASTERING_PYTHON" in text
    assert "--no-download" in text
    assert "Track A" in text


def test_mastering_python_unset_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var (slim image / host) → None path, exists False, never raises."""
    monkeypatch.delenv(doctor.MASTERING_PYTHON_ENVIRONMENT_VARIABLE, raising=False)
    assert mastering_python() is None
    assert check_mastering_python() == {"path": None, "exists": False}
    facts = probe()
    assert facts["mastering_python"] is None
    assert facts["mastering_python_exists"] is False


def test_mastering_python_present_reports_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set env var → path reported; exists tracks the filesystem."""
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv(doctor.MASTERING_PYTHON_ENVIRONMENT_VARIABLE, str(interpreter))
    assert mastering_python() == interpreter
    assert check_mastering_python()["exists"] is True
    facts = probe()
    assert facts["mastering_python"] == str(interpreter)
    assert facts["mastering_python_exists"] is True
    interpreter.unlink()
    assert check_mastering_python()["exists"] is False


def test_executable_seam_prepends_venv_bin_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """The venv interpreter lands on PATH (Stage C JIT contract, reused here)."""
    from voyage.rpc import _spawn_env

    monkeypatch.setenv("PATH", os.pathsep.join(("/usr/bin", "/bin")))
    env = _spawn_env(_MASTERING_INTERPRETER)
    assert env is not None
    assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(_MASTERING_INTERPRETER)
    assert _spawn_env(None) is None


def test_subprocess_worker_honors_alternate_executable(tmp_path: Path) -> None:
    """`SubprocessWorker(executable=...)` keeps the mastering interpreter (no spawn)."""
    from voyage.rpc import SubprocessWorker

    worker = SubprocessWorker(
        "voyage.workers.fake",
        tmp_path,
        tmp_path / "mastering-worker.log",
        init_op=None,
        executable=_MASTERING_INTERPRETER,
    )
    assert worker._executable == _MASTERING_INTERPRETER
    assert worker._alternate_executable == _MASTERING_INTERPRETER


def test_generic_verify_fails_loud_on_empty_dir(tmp_path: Path) -> None:
    """Verify on an empty models dir fails closed (offline, no hub traffic)."""
    from voyage import model_registry

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ok, message = model_registry.verify_model(models_dir, "film")
    assert ok is False
    assert "missing" in message


def test_generic_download_rejects_unknown_spec(tmp_path: Path) -> None:
    """Unknown specs raise before any network (no silent fallback, ever)."""
    from voyage import model_registry

    with pytest.raises(ValueError, match="unknown model spec"):
        model_registry.download_model(tmp_path / "models", "mastering-no-such-spec")


def test_mastering_spec_registered() -> None:
    """The `audio-sonicmaster` spec lives in the registry (Track C pins)."""
    from voyage import model_registry

    assert "audio-sonicmaster" in model_registry.MODEL_SPECS


def test_mastering_wrappers_exist() -> None:
    """Download/verify wrappers exist behind the generic table-driven pair."""
    from voyage import model_registry

    assert callable(model_registry.download_mastering_models)
    assert callable(model_registry.verify_mastering_models)


def test_mastering_verify_fails_loud_on_empty_dir(tmp_path: Path) -> None:
    """Verify on an empty models dir fails closed (offline, no hub traffic)."""
    from voyage import model_registry

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ok, message = model_registry.verify_mastering_models(models_dir)
    assert ok is False
    assert "missing" in message


def test_mastering_ensure_gate() -> None:
    """CUDA backends ensure the mastering stack by default; `--no-master` drops it."""
    from voyage.config import ProjectConfig, with_video_backend
    from voyage.models_ensure import required_specs

    config: ProjectConfig = with_video_backend(
        ProjectConfig(style="pastel neon line-art, peaceful"), "ltxv"
    )
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert "audio-sonicmaster" in specs
    specs = {
        item.spec for item in required_specs(config, sfx_enabled=False, mastering_enabled=False)
    }
    assert "audio-sonicmaster" not in specs
