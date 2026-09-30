"""Director snapshots resolve to the single /models copy (volume-deleted safe).

The director worker used to load Qwen/MiniLM/Qwen3.5 by hub id from the
ephemeral HF cache only, so a deleted ~/.cache/voyage-models silently
degraded to deterministic/skip even though `generate` had just ensured
the snapshots. These tests pin the fix: known repo ids resolve to
<models_dir>/<subdir>, a missing snapshot is fetched into /models
(single copy, hub id stays the fallback), and custom ids keep working.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from voyage import model_registry
from voyage.config import default_config_toml, load_config
from voyage.supervisor import Supervisor
from voyage.workers import director as director_worker


def _touch_sparse(path: Path, size: int) -> None:
    """Create a size-counting but disk-free file (st_size without blocks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.truncate(size)


def _write_qwen_snapshot(models: Path) -> Path:
    """Minimal checklist-satisfying Qwen3-8B snapshot (sparse shards)."""
    qwen = models / model_registry.QWEN_SUBDIR
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        (qwen / name).parent.mkdir(parents=True, exist_ok=True)
        (qwen / name).write_text("{}", encoding="utf-8")
    per_shard, remainder = divmod(model_registry.QWEN_MIN_BYTES, 5)
    for index in range(1, 6):
        size = per_shard + (remainder if index == 5 else 0)
        _touch_sparse(qwen / f"model-0000{index}-of-00005.safetensors", size)
    return qwen


def test_resolve_snapshot_maps_director_pair() -> None:
    qwen = model_registry.resolve_snapshot(model_registry.QWEN_HF_REPO)
    assert qwen is not None
    assert (qwen.spec_name, qwen.relative_dir) == (
        "director-qwen8b",
        model_registry.QWEN_SUBDIR,
    )
    minilm = model_registry.resolve_snapshot(model_registry.MINILM_HF_REPO)
    assert minilm is not None
    assert (minilm.spec_name, minilm.relative_dir) == (
        "director-qwen8b",
        model_registry.MINILM_SUBDIR,
    )


def test_resolve_snapshot_maps_inspector() -> None:
    ref = model_registry.resolve_snapshot(model_registry.QWEN35_HF_REPO)
    assert ref is not None
    assert (ref.spec_name, ref.relative_dir) == (
        "inspector-qwen35",
        model_registry.QWEN35_SUBDIR,
    )


def test_resolve_snapshot_maps_any_known_repo() -> None:
    """Custom-id policy: every snapshot repo in the registry maps, not
    just the director rows (a video TE id must not fall back to cache)."""
    ref = model_registry.resolve_snapshot(model_registry.LTXV_TE_REPO)
    assert ref is not None
    assert (ref.spec_name, ref.relative_dir) == (
        "ltxv-2b",
        model_registry.LTXV_TE_SUBDIR,
    )


def test_resolve_snapshot_unknown_returns_none() -> None:
    assert model_registry.resolve_snapshot("someone/something-else") is None


def test_snapshot_present_true_when_checklist_satisfied(tmp_path: Path) -> None:
    models = tmp_path / "models"
    _write_qwen_snapshot(models)
    ref = model_registry.resolve_snapshot(model_registry.QWEN_HF_REPO)
    assert ref is not None
    assert model_registry.snapshot_present(models, ref) is True


def test_snapshot_present_false_when_weights_missing(tmp_path: Path) -> None:
    models = tmp_path / "models"
    qwen = models / model_registry.QWEN_SUBDIR
    qwen.mkdir(parents=True)
    (qwen / "config.json").write_text("{}", encoding="utf-8")
    ref = model_registry.resolve_snapshot(model_registry.QWEN_HF_REPO)
    assert ref is not None
    assert model_registry.snapshot_present(models, ref) is False


def test_snapshot_present_false_for_empty_dir(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    ref = model_registry.resolve_snapshot(model_registry.QWEN_HF_REPO)
    assert ref is not None
    assert model_registry.snapshot_present(models, ref) is False


def test_resolve_model_source_passes_local_dirs_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E2E local paths (e.g. /models/Qwen3.5-9B) never hit the downloader."""

    def _boom(models_dir: Path, spec_name: str) -> dict[str, Any]:
        raise AssertionError("must not download for a local dir")

    monkeypatch.setattr(model_registry, "download_model", _boom)
    assert director_worker._resolve_model_source(str(tmp_path)) == str(tmp_path)


def test_resolve_model_source_passes_unknown_ids_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom ids outside the registry keep today's hub/cache behavior."""

    def _boom(models_dir: Path, spec_name: str) -> dict[str, Any]:
        raise AssertionError("must not download for an unknown id")

    monkeypatch.setattr(model_registry, "download_model", _boom)
    assert (
        director_worker._resolve_model_source("someone/something-else") == "someone/something-else"
    )


def test_resolve_model_source_uses_present_snapshot_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    expected = _write_qwen_snapshot(models)
    monkeypatch.setattr(director_worker, "_CONFIG", {"backend": "qwen", "models_dir": str(models)})

    def _boom(models_dir: Path, spec_name: str) -> dict[str, Any]:
        raise AssertionError("present snapshot must not download")

    monkeypatch.setattr(model_registry, "download_model", _boom)
    assert director_worker._resolve_model_source(model_registry.QWEN_HF_REPO) == str(expected)


def test_resolve_model_source_downloads_absent_snapshot_into_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(director_worker, "_CONFIG", {"backend": "qwen", "models_dir": str(models)})
    calls: list[tuple[str, str]] = []

    def _fake_download(models_dir: Path, spec_name: str) -> dict[str, Any]:
        calls.append((str(models_dir), spec_name))
        _write_qwen_snapshot(models_dir)
        return {}

    monkeypatch.setattr(model_registry, "download_model", _fake_download)
    resolved = director_worker._resolve_model_source(model_registry.QWEN_HF_REPO)
    assert resolved == str(models / model_registry.QWEN_SUBDIR)
    assert calls == [(str(models), "director-qwen8b")]


def test_resolve_model_source_download_failure_raises_and_restores_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed fetch must reach the caller's fallback chain (deterministic
    / skip) with the error — and never leak the HF_HUB_OFFLINE override."""
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(director_worker, "_CONFIG", {"backend": "qwen", "models_dir": str(models)})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def _fail(models_dir: Path, spec_name: str) -> dict[str, Any]:
        assert os.environ.get("HF_HUB_OFFLINE") == "0"
        raise OSError("network down")

    monkeypatch.setattr(model_registry, "download_model", _fail)
    with pytest.raises(OSError, match="network down"):
        director_worker._resolve_model_source(model_registry.QWEN_HF_REPO)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_models_dir_prefers_init_payload_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        director_worker,
        "_CONFIG",
        {"backend": "qwen", "models_dir": str(tmp_path / "vol")},
    )
    monkeypatch.setenv("VOYAGE_MODELS_DIR", str(tmp_path / "env"))
    assert director_worker._models_dir() == str(tmp_path / "vol")


def test_models_dir_falls_back_to_env_then_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(director_worker, "_CONFIG", {"backend": "qwen"})
    monkeypatch.setenv("VOYAGE_MODELS_DIR", str(tmp_path / "env"))
    assert director_worker._models_dir() == str(tmp_path / "env")
    monkeypatch.delenv("VOYAGE_MODELS_DIR")
    assert director_worker._models_dir() == "/models"


def test_handle_init_records_models_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fresh dict (auto-restored): handle_init mutates the global _CONFIG, and
    # a leaked models_dir would redirect later tests' resolution.
    monkeypatch.setattr(director_worker, "_CONFIG", {"backend": "qwen"})
    director_worker.handle_init({"models_dir": "/vol/models"})
    assert director_worker._CONFIG["models_dir"] == "/vol/models"
    director_worker.handle_init({"backend": "qwen"})
    assert director_worker._CONFIG["models_dir"] == "/vol/models"
    with pytest.raises(TypeError, match="models_dir"):
        director_worker.handle_init({"models_dir": 123})


def _stub_torch_and_transformers(monkeypatch: pytest.MonkeyPatch, seen: dict[str, str]) -> None:
    """Slim-safe loader plumbing: record the from_pretrained source id."""
    monkeypatch.setattr(director_worker, "_require_module", lambda _name: None)
    torch_stub = types.SimpleNamespace(bfloat16="bf16", float32="fp32")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    class _Tokenizer:
        eos_token_id = 0

        def __init__(self, source: str) -> None:
            seen["tokenizer"] = source

        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> _Tokenizer:
            return cls(source)

    class _Model:
        def __init__(self, source: str) -> None:
            seen["model"] = source

        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> _Model:
            return cls(source)

        def eval(self) -> _Model:
            return self

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = _Tokenizer  # type: ignore[attr-defined]
    transformers_stub.AutoModelForCausalLM = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)


def test_load_qwen_loads_from_resolved_snapshot_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    monkeypatch.setattr(director_worker, "_CONFIG", {"backend": "qwen", "models_dir": str(models)})
    monkeypatch.setattr(director_worker, "_QWEN", {})
    resolved = str(models / model_registry.QWEN_SUBDIR)
    monkeypatch.setattr(director_worker, "_resolve_model_source", lambda model_id: resolved)
    seen: dict[str, str] = {}
    _stub_torch_and_transformers(monkeypatch, seen)
    director_worker._load_qwen(model_registry.QWEN_HF_REPO, device="cpu")
    assert seen == {"tokenizer": resolved, "model": resolved}


def test_load_embedder_loads_from_resolved_snapshot_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    monkeypatch.setattr(director_worker, "_CONFIG", {"backend": "qwen", "models_dir": str(models)})
    monkeypatch.setattr(director_worker, "_EMBEDDER", {})
    monkeypatch.setattr(director_worker, "_require_module", lambda _name: None)
    resolved = str(models / model_registry.MINILM_SUBDIR)
    monkeypatch.setattr(director_worker, "_resolve_model_source", lambda model_id: resolved)
    seen: dict[str, str] = {}
    module = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, source: str, **kwargs: Any) -> None:
            seen["model"] = source

    module.SentenceTransformer = _SentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    director_worker._load_embedder(model_registry.MINILM_HF_REPO)
    assert seen == {"model": resolved}


def test_load_inspector_loads_from_resolved_snapshot_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    monkeypatch.setattr(director_worker, "_CONFIG", {"backend": "qwen", "models_dir": str(models)})
    monkeypatch.setattr(director_worker, "_INSPECTOR", {})
    monkeypatch.setattr(director_worker, "_require_module", lambda _name: None)
    resolved = str(models / model_registry.QWEN35_SUBDIR)
    monkeypatch.setattr(director_worker, "_resolve_model_source", lambda model_id: resolved)
    seen: dict[str, str] = {}

    class _Processor:
        def __init__(self, source: str) -> None:
            seen["processor"] = source

        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> _Processor:
            return cls(source)

    class _Model:
        def __init__(self, source: str) -> None:
            seen["model"] = source

        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> _Model:
            return cls(source)

        def eval(self) -> _Model:
            return self

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoProcessor = _Processor  # type: ignore[attr-defined]
    transformers_stub.AutoModelForMultimodalLM = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    torch_stub = types.SimpleNamespace(bfloat16="bf16", float32="fp32")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    director_worker._load_inspector(model_registry.QWEN35_HF_REPO)
    assert seen == {"processor": resolved, "model": resolved}


def test_supervisor_passes_models_dir_to_director_worker(tmp_path: Path) -> None:
    from voyage import paths

    run_dir = tmp_path / "run"
    (run_dir / paths.LOGS_DIRNAME).mkdir(parents=True)
    (run_dir / paths.CONFIG_FILENAME).write_text(
        default_config_toml("director-models", "probe", 11), encoding="utf-8"
    )
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    assert supervisor._director._init_payload == {
        "models_dir": config.video.models_dir,
        "device": config.director.device,
    }
