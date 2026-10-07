"""Load-time sha gates for the LTX stacks (issues 208/209).

The sessions verify every recorded weight against the download manifest
before any weight load (fail closed, no missing-manifest opt-in) — the
per-file-dict sibling of the causvid single-sha call site. CPU-only: tiny
stub bytes stand in for the multi-GB weights, and the fail-closed paths
raise before any torch/ComfyUI import, so no GPU stubs are needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from voyage.registry_ltx25 import verify_recorded_shas

_DIT_REL = "ltx25/LTX-2.5-Distilled-Q3_K_M.gguf"
_TE_REL = "ltx25/gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf"


def _write_stack(models: Path, dit_bytes: bytes = b"dit-bytes") -> dict[str, Path]:
    """Minimal two-file stack plus a matching manifest entry."""
    dit_path = models / _DIT_REL
    te_path = models / _TE_REL
    dit_path.parent.mkdir(parents=True, exist_ok=True)
    dit_path.write_bytes(dit_bytes)
    te_path.write_bytes(b"te-bytes")
    paths = {_DIT_REL: dit_path, _TE_REL: te_path}
    (models / "manifest.json").write_text(
        json.dumps(
            {
                "ltx25": {
                    "checkpoint_shas": {
                        relative: hashlib.sha256(path.read_bytes()).hexdigest()
                        for relative, path in paths.items()
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return paths


def test_verify_recorded_shas_accepts_matching_manifest(tmp_path: Path) -> None:
    """The gate stays green on untampered weights."""
    models = tmp_path / "models"
    paths = _write_stack(models)
    verify_recorded_shas(models, "ltx25", paths)


def test_verify_recorded_shas_rejects_tampered_bytes(tmp_path: Path) -> None:
    """One flipped byte after ensure fails the load (no execution)."""
    models = tmp_path / "models"
    paths = _write_stack(models)
    paths[_DIT_REL].write_bytes(b"dit-BYTES")
    with pytest.raises(ValueError, match="mismatch"):
        verify_recorded_shas(models, "ltx25", paths)


def test_verify_recorded_shas_fails_closed_without_manifest(tmp_path: Path) -> None:
    """No manifest means no baseline — the load is refused, never passed through."""
    models = tmp_path / "models"
    dit_path = models / _DIT_REL
    dit_path.parent.mkdir(parents=True, exist_ok=True)
    dit_path.write_bytes(b"dit-bytes")
    with pytest.raises(ValueError, match="no manifest"):
        verify_recorded_shas(models, "ltx25", {_DIT_REL: dit_path})


def test_verify_recorded_shas_fails_closed_without_entry(tmp_path: Path) -> None:
    """A manifest for another stack is not a baseline for this one."""
    models = tmp_path / "models"
    paths = _write_stack(models)
    (models / "manifest.json").write_text(json.dumps({"film": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no manifest entry"):
        verify_recorded_shas(models, "ltx25", paths)


def test_verify_recorded_shas_fails_closed_without_per_file_sha(tmp_path: Path) -> None:
    """A recorded set missing one file cannot gate that file's load."""
    models = tmp_path / "models"
    paths = _write_stack(models)
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    entry = manifest["ltx25"]
    assert isinstance(entry, dict)
    shas = entry["checkpoint_shas"]
    assert isinstance(shas, dict)
    del shas[_TE_REL]
    (models / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="no recorded sha256"):
        verify_recorded_shas(models, "ltx25", paths)


def test_verify_recorded_shas_fails_closed_on_torn_manifest(tmp_path: Path) -> None:
    """Torn JSON is validate_run's territory — the load is still refused here."""
    models = tmp_path / "models"
    paths = _write_stack(models)
    (models / "manifest.json").write_text("{torn", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable manifest"):
        verify_recorded_shas(models, "ltx25", paths)


def _write_session_stack(models: Path, subdir: str, files: dict[str, bytes]) -> None:
    """Stub weight files for a session-construction probe (no manifest)."""
    for relative, payload in files.items():
        candidate = models / subdir / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(payload)


def test_ltxv_session_refuses_load_without_manifest(tmp_path: Path) -> None:
    """Issue 209: LTXVSession fails before any torch import when unprovisioned."""
    from voyage.workers import video_ltxv

    models = tmp_path / "models"
    _write_session_stack(
        models,
        video_ltxv.LTXV_SUBDIR,
        {video_ltxv.DIT_FILENAME: b"dit", video_ltxv.UPSC_FILENAME: b"up"},
    )
    with pytest.raises(ValueError, match="no manifest"):
        video_ltxv.LTXVSession(models, "cuda:0")


def test_ltx25_session_refuses_load_without_manifest(tmp_path: Path) -> None:
    """Issue 209: LTX25Session fails before any torch import when unprovisioned."""
    from voyage.registry_ltx25 import LTX25_DIT_FILE, LTX25_SUBDIR, LTX25_TE_FILE
    from voyage.workers import video_ltx25

    models = tmp_path / "models"
    _write_session_stack(models, LTX25_SUBDIR, {LTX25_DIT_FILE: b"dit", LTX25_TE_FILE: b"te"})
    with pytest.raises(ValueError, match="no manifest"):
        video_ltx25.LTX25Session(models, "cuda:0", tmp_path / "scratch")


def test_ltx23_session_refuses_load_without_manifest(tmp_path: Path) -> None:
    """Issue 209: LTX23Session fails before any torch import when unprovisioned."""
    from voyage.registry_ltx23 import (
        LTX23_DIT_FILE,
        LTX23_DIT_SUBFOLDER,
        LTX23_SUBDIR,
        LTX23_TE_FILE,
    )
    from voyage.workers import video_ltx23

    models = tmp_path / "models"
    _write_session_stack(
        models,
        LTX23_SUBDIR,
        {f"{LTX23_DIT_SUBFOLDER}/{LTX23_DIT_FILE}": b"dit", LTX23_TE_FILE: b"te"},
    )
    with pytest.raises(ValueError, match="no manifest"):
        video_ltx23.LTX23Session(models, "cuda:0", tmp_path / "scratch")


@pytest.mark.parametrize(
    ("spec_name", "manifest_key", "worker_name"),
    [
        ("ltx25", "ltx25", "video_ltx25"),
        ("ltx23", "ltx23", "video_ltx23"),
        ("ltxv-2b", "ltxv", "video_ltxv"),
    ],
)
def test_stack_gate_covers_every_expected_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec_name: str,
    manifest_key: str,
    worker_name: str,
) -> None:
    """Issues 208/209: each session gate verifies exactly the ingest-pinned set."""
    import importlib

    from voyage import model_registry

    worker = importlib.import_module(f"voyage.workers.{worker_name}")
    spec = model_registry.MODEL_SPECS[spec_name]
    models = tmp_path / "models"
    for expected in spec.expected_hashes:
        candidate = models / expected.relative_path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(b"honest-bytes")
    (models / "manifest.json").write_text(
        json.dumps(
            {
                manifest_key: {
                    "checkpoint_shas": {
                        expected.relative_path: hashlib.sha256(b"honest-bytes").hexdigest()
                        for expected in spec.expected_hashes
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    seen: list[str] = []
    monkeypatch.setattr(
        model_registry,
        "verify_checkpoint_sha256",
        lambda path, sha: seen.append(str(path)),
    )
    worker._verify_stack_manifest(models)
    verified = {str(Path(path).relative_to(models).as_posix()) for path in seen}
    assert verified == {expected.relative_path for expected in spec.expected_hashes}


@pytest.mark.parametrize(
    ("spec_name", "manifest_key", "worker_name"),
    [
        ("ltx25", "ltx25", "video_ltx25"),
        ("ltx23", "ltx23", "video_ltx23"),
        ("ltxv-2b", "ltxv", "video_ltxv"),
    ],
)
def test_stack_gate_trips_on_post_provision_tampering(
    tmp_path: Path, spec_name: str, manifest_key: str, worker_name: str
) -> None:
    """Every stack gate refuses a swapped file (no silent execution)."""
    import importlib

    from voyage import model_registry

    worker = importlib.import_module(f"voyage.workers.{worker_name}")
    spec = model_registry.MODEL_SPECS[spec_name]
    models = tmp_path / "models"
    for expected in spec.expected_hashes:
        candidate = models / expected.relative_path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(b"honest-bytes")
    (models / "manifest.json").write_text(
        json.dumps(
            {
                manifest_key: {
                    "checkpoint_shas": {
                        expected.relative_path: hashlib.sha256(b"honest-bytes").hexdigest()
                        for expected in spec.expected_hashes
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    first = models / spec.expected_hashes[0].relative_path
    first.write_bytes(b"tampered-bytes")
    with pytest.raises(ValueError, match="mismatch"):
        worker._verify_stack_manifest(models)
