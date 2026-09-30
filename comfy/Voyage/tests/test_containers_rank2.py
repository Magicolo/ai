"""Rank-2 container hardening batch: issues 069 + 075 + 076 + 077.

Why one module: the four issues share a verification shape (text-scan the
build files, unit-test the manifest repair) and a single gate run. Dockerfile
and .dockerignore assertions scan file text on purpose — no image build is
needed (or wanted: the video image is 36 GB and the directive forbids full
builds). `models_ensure` race/repair behavior gets real unit tests against
temporary model roots: a parallel-download race is simulated by a downloader
that verifies fine but never merges its manifest record (last-writer-wins
loss), and repair failure by a read-only models directory.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from voyage.config import ProjectConfig, with_video_backend
from voyage.console import VoyageConsole

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"
VIDEO_DOCKERFILE_PATH = REPO_ROOT / "worker" / "Dockerfile.video"

FROZEN_VIDEO_APT: dict[str, str] = {
    "python3.10": "3.10.12-1~22.04.18",
    "python3.10-venv": "3.10.12-1~22.04.18",
    "python3-pip": "22.0.2+dfsg-1ubuntu0.7",
    "git": "1:2.34.1-1ubuntu1.17",
    "gcc": "4:11.2.0-1ubuntu1",
    "g++": "4:11.2.0-1ubuntu1",
    "ffmpeg": "7:4.4.2-0ubuntu0.22.04.1",
    "python3.10-dev": "3.10.12-1~22.04.18",
}
"""Exact apt versions frozen from the live video image 2026-09-30
(`docker run --rm voyage-video:latest dpkg-query -W ...`, CPU-only probe —
no GPU needed for a version query). A silent Dockerfile edit breaks the
pin test below on purpose: bumps update both files together.
"""


def _ignore_entries() -> list[str]:
    """Non-comment, non-blank .dockerignore lines (the effective patterns)."""
    return [
        line.strip()
        for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_dockerignore_covers_pytest_and_coverage_artifacts() -> None:
    """Gate residue (`.coverage`, `coverage.xml`, `.pytest_cache/`) stays out."""
    entries = _ignore_entries()
    assert ".coverage" in entries
    assert "coverage.xml" in entries
    assert ".pytest_cache/" in entries


def test_dockerignore_covers_dev_caches_and_env_artifacts() -> None:
    """Explicit dot-cache trio + venv/egg-info/env (069: no bare-glob trust)."""
    entries = _ignore_entries()
    for pattern in (
        ".mypy_cache/",
        ".ruff_cache/",
        ".hypothesis/",
        ".venv/",
        "*.egg-info/",
        ".env",
    ):
        assert pattern in entries


def test_dockerignore_scope_comment_matches_copy_graph() -> None:
    """Header no longer claims `tests/` is COPYed (slim/video never COPY it)."""
    content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    assert "voyage/ and tests/" not in content
    assert "issue 054" in content


def _non_comment_text(content: str) -> str:
    """Dockerfile text minus `#` comment lines (prose may cite old modes)."""
    return "\n".join(line for line in content.splitlines() if not line.strip().startswith("#"))


def test_video_image_has_no_world_writable_code_dirs() -> None:
    """`chmod 777` on the PYTHONPATH tree is gone (075: code-planting hole)."""
    content = _non_comment_text(VIDEO_DOCKERFILE_PATH.read_text(encoding="utf-8"))
    assert "chmod 777" not in content
    assert " 777 " not in content


def test_video_image_locks_opt_trees_to_owner_write() -> None:
    """Both shim link farms stay owner-writable, group/other read-execute."""
    content = VIDEO_DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "chmod 755 /opt/longlive" in content
    assert "chmod 755 /opt/causvid/wan_models" in content


def test_video_image_still_precreates_runtime_links() -> None:
    """The skip-when-correct steady state both shims rely on (075 guard rail)."""
    content = VIDEO_DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "ln -sfn /models/Wan2.1-T2V-1.3B /opt/causvid/wan_models/Wan2.1-T2V-1.3B" in content
    assert "ln -sfn /models/wan_models /opt/longlive/wan_models" in content


def test_video_image_pins_every_apt_package() -> None:
    """All eight apt packages carry `=` pins at the frozen versions (076)."""
    content = VIDEO_DOCKERFILE_PATH.read_text(encoding="utf-8")
    joined = content.replace("\\\n", " ")
    for package, version in FROZEN_VIDEO_APT.items():
        assert f"{package}={version}" in joined
    assert "stays unversioned on purpose" not in content


def _stub_spec(manifest_key: str, record: dict[str, Any]) -> Any:
    """A registry row with a trivial record builder (no weight files needed)."""
    from voyage.model_registry import ModelSpec

    return ModelSpec(
        name="stub",
        manifest_key=manifest_key,
        snapshots=(),
        files=(),
        record_builder=lambda _models_dir: dict(record),
        checks=(),
        success_message=lambda _models_dir: "stub OK",
    )


def _stub_config() -> ProjectConfig:
    return with_video_backend(ProjectConfig(style="pastel neon line-art, peaceful"), "ltxv")


def test_repair_restores_race_dropped_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A manifest that lost one parallel-download record is re-merged (077)."""
    import voyage.model_registry as registry
    from voyage.models_ensure import RequiredModel, _repair_manifest

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text('{"other": {"x": 1}}\n', encoding="utf-8")
    monkeypatch.setitem(registry.MODEL_SPECS, "stub-a", _stub_spec("stub-a", {"sha": "abc"}))
    entry = RequiredModel(spec="stub-a", models_dir=models_dir)
    assert _repair_manifest([entry]) == []
    import json

    merged = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    assert merged["stub-a"] == {"sha": "abc"}
    assert merged["other"] == {"x": 1}


def test_repair_returns_entries_it_cannot_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Best-effort silence becomes a loud return value: OSError stays listed."""
    import voyage.model_registry as registry
    from voyage.models_ensure import RequiredModel, _repair_manifest

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text('{"other": {"x": 1}}\n', encoding="utf-8")
    monkeypatch.setitem(registry.MODEL_SPECS, "stub-a", _stub_spec("stub-a", {"sha": "abc"}))
    entry = RequiredModel(spec="stub-a", models_dir=models_dir)
    models_dir.chmod(0o555)
    try:
        assert _repair_manifest([entry]) == [entry]
    finally:
        models_dir.chmod(0o755)


def test_repair_treats_torn_manifest_as_unrepairable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Corrupt JSON is never overwritten blindly — the entry stays missing."""
    import voyage.model_registry as registry
    from voyage.models_ensure import RequiredModel, _repair_manifest

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text("{invalid\n", encoding="utf-8")
    monkeypatch.setitem(registry.MODEL_SPECS, "stub-a", _stub_spec("stub-a", {"sha": "abc"}))
    entry = RequiredModel(spec="stub-a", models_dir=models_dir)
    assert _repair_manifest([entry]) == [entry]
    assert (models_dir / "manifest.json").read_text(encoding="utf-8") == "{invalid\n"


def test_ensure_fails_loud_when_repair_cannot_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`models ready` is never printed over a hash-less manifest (077 core)."""
    import voyage.model_registry as registry
    import voyage.models_ensure as ensure

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text('{"other": {"x": 1}}\n', encoding="utf-8")
    monkeypatch.setitem(registry.MODEL_SPECS, "stub-a", _stub_spec("stub-a", {"sha": "abc"}))
    monkeypatch.setattr(
        ensure,
        "required_specs",
        lambda *args, **kwargs: [ensure.RequiredModel(spec="stub-a", models_dir=models_dir)],
    )
    present: set[str] = set()

    def _verify(_dir: Path, _spec: str) -> tuple[bool, str]:
        return ("stub-a" in present, "OK" if "stub-a" in present else "missing")

    monkeypatch.setattr(registry, "verify_model", _verify)

    def _download_without_merge(_dir: Path, _spec: str) -> dict[str, object]:
        present.add(_spec)
        return {}

    monkeypatch.setattr(registry, "download_model", _download_without_merge)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    models_dir.chmod(0o555)
    try:
        assert ensure.ensure_models(_stub_config(), False, console, models_dir) == 1
    finally:
        models_dir.chmod(0o755)
    output = stream.getvalue()
    assert "models ready" not in output
    assert "stub-a" in output


def test_ensure_reports_ready_only_over_a_complete_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy path still succeeds AND leaves every record on disk (077).

    Characterization guard (passes both before and after the repair
    rework — the rework only changes failure paths): pins the
    ready-implies-complete contract for the download-drops-record race.
    """
    import json

    import voyage.model_registry as registry
    import voyage.models_ensure as ensure

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text('{"other": {"x": 1}}\n', encoding="utf-8")
    monkeypatch.setitem(registry.MODEL_SPECS, "stub-a", _stub_spec("stub-a", {"sha": "abc"}))
    monkeypatch.setattr(
        ensure,
        "required_specs",
        lambda *args, **kwargs: [ensure.RequiredModel(spec="stub-a", models_dir=models_dir)],
    )
    present: set[str] = set()

    def _verify(_dir: Path, _spec: str) -> tuple[bool, str]:
        return ("stub-a" in present, "OK" if "stub-a" in present else "missing")

    monkeypatch.setattr(registry, "verify_model", _verify)

    def _download_without_merge(_dir: Path, _spec: str) -> dict[str, object]:
        present.add(_spec)
        return {}

    monkeypatch.setattr(registry, "download_model", _download_without_merge)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    assert ensure.ensure_models(_stub_config(), False, console, models_dir) == 0
    assert "models ready" in stream.getvalue()
    merged = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    assert merged["stub-a"] == {"sha": "abc"}
