"""Supply-chain pin gates (issues 067/070/071/073).

Each gate encodes one investigation finding as a regression test: the video
image's pip rows stay frozen (067), the Wan2.2 base revision stays tracked
until pinned (070), checkpoint hashes are verified at ingest/ensure/load
instead of self-attested (071), and no worker loads weights from a bare hub
id (073). CPU-only: the hub, torch and the LTX stack are stubbed — no
network, no GPU, no checkpoint bytes are ever executed.
"""

from __future__ import annotations

import ast
import json
import shlex
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from voyage import model_registry

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DOCKERFILE = REPO_ROOT / "worker" / "Dockerfile.video"

# Issue 079 deleted the only `-r requirements.txt` row with the longlive2
# backend (its upstream requirements could never be frozen without a GPU-box
# build). No `-r` row remains: any NEW one fails.
_NO_REQUIREMENT_ROWS: list[str] = []

_OPTIONS_WITH_VALUE = frozenset(
    {
        "--index-url",
        "--extra-index-url",
        "-e",
        "--editable",
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-f",
        "--find-links",
        "--trusted-host",
    }
)


def _scan_pip_rows(dockerfile: Path) -> tuple[list[str], list[str]]:
    """Split Dockerfile pip installs into (floating, requirement_file) rows.

    Returns requirement specs without an exact pin (bare names, floors like
    `>=`, ranges) plus every `-r` target. Exact pins (`==`), PEP 508 URLs
    (`@`, `://`, `git+`) and option values are not floating. Comment lines
    are skipped (the re-freeze notes quote `pip install -r` themselves).
    """
    import re

    logical: list[str] = []
    buffer = ""
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
        else:
            logical.append(buffer + line)
            buffer = ""
    if buffer:
        logical.append(buffer)
    floating: list[str] = []
    requirement_files: list[str] = []
    for line in logical:
        if line.lstrip().startswith("#"):
            continue
        for chunk in line.split("&&"):
            _, separator, tail = chunk.partition("pip install")
            if not separator:
                continue
            tokens = shlex.split(re.sub(r"\s+@\s+", "@", tail))
            index = 0
            while index < len(tokens):
                token = tokens[index]
                if token in _OPTIONS_WITH_VALUE:
                    if token in ("-r", "--requirement") and index + 1 < len(tokens):
                        requirement_files.append(tokens[index + 1])
                    index += 2
                    continue
                if token.startswith("-"):
                    index += 1
                    continue
                spec = token.strip("\"'")
                if "==" in spec or "@" in spec or "://" in spec or spec.startswith("git+"):
                    index += 1
                    continue
                floating.append(spec)
                index += 1
    return floating, requirement_files


def test_video_dockerfile_has_no_floating_pins() -> None:
    """Issue 067: every video-image pip row is frozen exact (GPU-box freeze)."""
    floating, requirement_files = _scan_pip_rows(VIDEO_DOCKERFILE)
    assert floating == [], f"floating pip rows in worker/Dockerfile.video: {sorted(floating)}"
    assert requirement_files == _NO_REQUIREMENT_ROWS


def test_wan22_pins_removed_with_longlive2_backend() -> None:
    """Issue 079 closes 070 by deletion: the floating Wan2.2 base is gone."""
    assert not hasattr(model_registry, "WAN_HF_REVISION")
    assert not hasattr(model_registry, "WAN_HF_REPO")
    assert "longlive2-bf16" not in model_registry.MODEL_SPECS


def test_floating_snapshot_set_is_empty() -> None:
    """Issue 070 closed (characterization, not TDD): no floating snapshot remains.

    The only floating row was Wan2.2 (longlive2-bf16); issue 079 deleted the
    backend with it. Fails if a NEW floating row appears.
    """
    floating = {
        (name, snapshot.repo_id)
        for name, spec in model_registry.MODEL_SPECS.items()
        for snapshot in spec.snapshots
        if snapshot.revision is None
    }
    assert floating == set()


def _install_hub_stub(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    """Stub huggingface_hub: snapshot is a no-op, file fetch writes payload."""

    def fake_snapshot_download(**kwargs: Any) -> str:
        return str(kwargs.get("local_dir", ""))

    def fake_hub_download(**kwargs: Any) -> str:
        local_dir = Path(str(kwargs["local_dir"]))
        subfolder = str(kwargs.get("subfolder", ""))
        target = local_dir / subfolder / str(kwargs["filename"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    hub_module = types.ModuleType("huggingface_hub")
    hub_module.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    hub_module.hf_hub_download = fake_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)


def test_download_rejects_tampered_bytes_before_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 071: ingest verifies expected hashes before the manifest merge."""
    _install_hub_stub(monkeypatch, b"tampered-bytes")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        model_registry.download_model(tmp_path, "film")
    assert not (tmp_path / "manifest.json").exists()


def test_verify_model_checks_manifest_recorded_sha(tmp_path: Path) -> None:
    """Issue 071: ensure fails closed when the recorded sha disagrees."""
    candidate = tmp_path / model_registry.FILM_REPO_PATH
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"x" * model_registry.FILM_MIN_BYTES)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"film": {"checkpoint_sha256": "0" * 64}}), encoding="utf-8"
    )
    ok, message = model_registry.verify_model(tmp_path, "film")
    assert not ok
    assert "mismatch" in message


def test_verify_model_accepts_matching_manifest_sha(tmp_path: Path) -> None:
    """Issue 071: the recorded-sha leg stays green on untampered weights."""
    import hashlib

    payload = b"x" * model_registry.FILM_MIN_BYTES
    candidate = tmp_path / model_registry.FILM_REPO_PATH
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(payload)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"film": {"checkpoint_sha256": hashlib.sha256(payload).hexdigest()}}),
        encoding="utf-8",
    )
    ok, message = model_registry.verify_model(tmp_path, "film")
    assert ok, message


def test_verify_against_manifest_closed_by_default(tmp_path: Path) -> None:
    """Issue 071: unknown-manifest + known-key fails closed without opt-in."""
    target = tmp_path / "model.pt"
    target.write_bytes(b"anything")
    with pytest.raises(ValueError, match="VOYAGE_ALLOW_MISSING_MANIFEST"):
        model_registry.verify_checkpoint_against_manifest(tmp_path, "video", target)


def test_verify_against_manifest_explicit_opt_in(tmp_path: Path) -> None:
    """Issue 071 (escape hatch, passes pre-fix too): explicit opt-in passes through."""
    target = tmp_path / "model.pt"
    target.write_bytes(b"anything")
    model_registry.verify_checkpoint_against_manifest(
        tmp_path, "video", target, allow_missing_manifest=True
    )


def test_verify_against_manifest_env_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 071: external volumes opt in via env without code changes."""
    target = tmp_path / "model.pt"
    target.write_bytes(b"anything")
    monkeypatch.setenv("VOYAGE_ALLOW_MISSING_MANIFEST", "1")
    model_registry.verify_checkpoint_against_manifest(tmp_path, "video", target)


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


def test_resolve_te_source_uses_present_snapshot_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 073: a provisioned TE snapshot resolves locally, no download."""
    from voyage.workers import video_ltxv

    models = tmp_path / "models"
    expected = _write_te_snapshot(models)

    def _boom(models_dir: Path, spec_name: str) -> dict[str, Any]:
        raise AssertionError("present snapshot must not download")

    monkeypatch.setattr(model_registry, "download_model", _boom)
    assert video_ltxv._resolve_te_source(models) == str(expected)


def test_resolve_te_source_downloads_absent_snapshot_into_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 073: a missing TE snapshot is fetched into /models, then local."""
    from voyage.workers import video_ltxv

    models = tmp_path / "models"
    models.mkdir()
    calls: list[tuple[str, str]] = []

    def _fake_download(models_dir: Path, spec_name: str) -> dict[str, Any]:
        calls.append((str(models_dir), spec_name))
        _write_te_snapshot(models_dir)
        return {}

    monkeypatch.setattr(model_registry, "download_model", _fake_download)
    resolved = video_ltxv._resolve_te_source(models)
    assert resolved == str(models / model_registry.LTXV_TE_SUBDIR)
    assert calls == [(str(models), "ltxv-2b")]


def _install_ltxv_session_stubs(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any]) -> None:
    """Slim-safe LTXV stack: record the TE from_pretrained source + flags."""
    torch_stub = types.SimpleNamespace(set_grad_enabled=lambda enabled: None, bfloat16="bf16")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    class _Placed:
        def to(self, *args: Any, **kwargs: Any) -> _Placed:
            return self

    def _make_module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    inference = _make_module("ltx_video.inference")
    inference.create_transformer = lambda *args, **kwargs: _Placed()  # type: ignore[attr-defined]
    inference.create_latent_upsampler = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    autoencoder = _make_module("ltx_video.models.autoencoders.causal_video_autoencoder")
    autoencoder.CausalVideoAutoencoder = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *args, **kwargs: _Placed()
    )
    patchifier = _make_module("ltx_video.models.transformers.symmetric_patchifier")
    patchifier.SymmetricPatchifier = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    pipeline_module = _make_module("ltx_video.pipelines.pipeline_ltx_video")
    pipeline_module.LTXVideoPipeline = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    pipeline_module.LTXMultiScalePipeline = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: object()
    )
    scheduler_module = _make_module("ltx_video.schedulers.rf")
    scheduler_module.RectifiedFlowScheduler = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *args, **kwargs: object()
    )
    for parent in (
        "ltx_video",
        "ltx_video.models",
        "ltx_video.models.autoencoders",
        "ltx_video.models.transformers",
        "ltx_video.pipelines",
        "ltx_video.schedulers",
    ):
        _make_module(parent)

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> _Tokenizer:
            seen["tokenizer_source"] = source
            seen["tokenizer_kwargs"] = kwargs
            return cls()

    class _TextEncoder(_Placed):
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> _TextEncoder:
            seen["encoder_source"] = source
            seen["encoder_kwargs"] = kwargs
            return cls()

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.T5Tokenizer = _Tokenizer  # type: ignore[attr-defined]
    transformers_stub.T5EncoderModel = _TextEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)


def test_ltxv_session_loads_te_from_local_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 073: the session loads the TE from the pinned snapshot, offline-first."""
    from voyage.workers import video_ltxv

    models = tmp_path / "models"
    ltxv_dir = models / model_registry.LTXV_SUBDIR
    ltxv_dir.mkdir(parents=True)
    (ltxv_dir / video_ltxv.DIT_FILENAME).write_bytes(b"dit")
    (ltxv_dir / video_ltxv.UPSC_FILENAME).write_bytes(b"upscaler")
    expected_te = _write_te_snapshot(models)
    seen: dict[str, Any] = {}
    _install_ltxv_session_stubs(monkeypatch, seen)
    video_ltxv.LTXVSession(models, "cuda:0")
    assert seen["tokenizer_source"] == str(expected_te)
    assert seen["encoder_source"] == str(expected_te)
    assert seen["tokenizer_kwargs"]["local_files_only"] is True
    assert seen["encoder_kwargs"]["local_files_only"] is True
    assert seen["tokenizer_kwargs"]["revision"] == model_registry.LTXV_TE_REVISION
    assert seen["encoder_kwargs"]["revision"] == model_registry.LTXV_TE_REVISION


def _bare_hub_loads(path: Path) -> list[str]:
    """from_pretrained calls rooted at a hub id (literal with `/`, *_REPO names)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and node.args
        ):
            first = node.args[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and "/" in first.value
            ):
                offenders.append(f"{path.name}:{node.lineno}: literal {first.value!r}")
            elif isinstance(first, ast.Name) and first.id.endswith(
                ("_REPO_ID", "_HF_REPO", "_REPO")
            ):
                offenders.append(f"{path.name}:{node.lineno}: hub-id name {first.id}")
    return offenders


def test_no_bare_hub_id_from_pretrained_in_workers() -> None:
    """Issue 073 (grep-gate): workers load from resolved local paths, never hub ids."""
    workers_dir = REPO_ROOT / "voyage" / "workers"
    offenders = [
        finding
        for candidate in sorted(workers_dir.glob("*.py"))
        for finding in _bare_hub_loads(candidate)
    ]
    assert offenders == [], f"bare hub-id from_pretrained in workers: {offenders}"
