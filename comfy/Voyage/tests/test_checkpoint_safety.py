"""Supply-chain safety: manifest merge, checkpoint hashes, torch.load flags.

Covers issues 006a (manifest clobber), 005 (torch.load RCE), and 086
(models_dir_layout coverage). CPU-only: torch and the hub are stubbed —
no checkpoint bytes are ever executed and no network is touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from voyage import model_registry
from voyage.workers import video_causvid, video_longlive


class _RecordingTorch:
    """Minimal torch stand-in that records load calls (never executes bytes)."""

    bfloat16 = "bfloat16"

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.load_calls: list[dict[str, Any]] = []
        self.cuda = types.SimpleNamespace(empty_cache=lambda: None)

    def device(self, name: str) -> str:
        return name

    def set_grad_enabled(self, enabled: bool) -> None:
        del enabled

    def load(self, source: Any, **kwargs: Any) -> Any:
        self.load_calls.append({"source": source, **kwargs})
        return self.payload


def _install_torch_stub(monkeypatch: pytest.MonkeyPatch, payload: Any) -> _RecordingTorch:
    fake_torch = _RecordingTorch(payload)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def _install_hub_stub(
    monkeypatch: pytest.MonkeyPatch, checkpoint_bytes: bytes = b"fake-weights"
) -> None:
    """Stub huggingface_hub: snapshot is a no-op, file fetch writes bytes."""

    def fake_snapshot_download(**kwargs: Any) -> str:
        return str(kwargs.get("local_dir", ""))

    def fake_hub_download(**kwargs: Any) -> str:
        local_dir = Path(str(kwargs["local_dir"]))
        subfolder = str(kwargs.get("subfolder", ""))
        target = local_dir / subfolder / str(kwargs["filename"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(checkpoint_bytes)
        return str(target)

    hub_module = types.ModuleType("huggingface_hub")
    hub_module.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    hub_module.hf_hub_download = fake_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)


def _read_manifest(models_dir: Path) -> dict[str, Any]:
    loaded = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_longlive_download_preserves_foreign_manifest_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 006a: video download merges instead of clobbering siblings."""
    _install_hub_stub(monkeypatch)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"director": {"repo": "Qwen/Qwen3-8B"}}), encoding="utf-8"
    )
    record = model_registry.download_longlive2_bf16(tmp_path)
    manifest = _read_manifest(tmp_path)
    assert manifest["director"] == {"repo": "Qwen/Qwen3-8B"}
    assert manifest["video"]["repo"] == model_registry.LONGLIVE_HF_REPO
    assert record == manifest


def test_causvid_download_records_checkpoint_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 005: causvid manifest carries the hash the worker verifies."""
    _install_hub_stub(monkeypatch, b"fake-causvid-snapshot")
    record = model_registry.download_causvid_models(tmp_path)
    checkpoint = tmp_path / model_registry.CAUSVID_SUBDIR / model_registry.CAUSVID_CHECKPOINT_FILE
    expected = hashlib.sha256(b"fake-causvid-snapshot").hexdigest()
    assert checkpoint.exists()
    assert record["causvid"]["checkpoint_sha256"] == expected
    assert _read_manifest(tmp_path)["causvid"]["checkpoint_sha256"] == expected


def test_verify_checkpoint_sha256_accepts_matching_hash(tmp_path: Path) -> None:
    target = tmp_path / "model_bf16.pt"
    target.write_bytes(b"weights-bytes")
    expected = hashlib.sha256(b"weights-bytes").hexdigest()
    model_registry.verify_checkpoint_sha256(target, expected)


def test_verify_checkpoint_sha256_rejects_mismatch_and_empty(tmp_path: Path) -> None:
    target = tmp_path / "model_bf16.pt"
    target.write_bytes(b"weights-bytes")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        model_registry.verify_checkpoint_sha256(target, "0" * 64)
    with pytest.raises(ValueError, match="no recorded sha256"):
        model_registry.verify_checkpoint_sha256(target, "")


def test_verify_against_manifest_passes_through_without_manifest(tmp_path: Path) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"anything")
    model_registry.verify_checkpoint_against_manifest(tmp_path, "video", target)


def test_verify_against_manifest_fails_closed_on_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "model_bf16.pt"
    target.write_bytes(b"real-bytes")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"video": {"checkpoint_sha256": "1" * 64}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        model_registry.verify_checkpoint_against_manifest(tmp_path, "video", target)


def test_recovery_tape_load_uses_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 005: tape loads never unpickle code (flags pinned via stub)."""
    fake_torch = _install_torch_stub(monkeypatch, {"profile": "longlive2-bf16-fp8"})
    tape_path = tmp_path / "recovery.pt"
    tape_path.write_bytes(b"never-executed-bytes")
    tape = video_longlive._load_recovery_tape(str(tape_path))
    assert tape == {"profile": "longlive2-bf16-fp8"}
    assert len(fake_torch.load_calls) == 1
    assert fake_torch.load_calls[0]["weights_only"] is True
    assert fake_torch.load_calls[0]["map_location"] == "cpu"


def test_recovery_tape_load_rejects_non_tape_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch = _install_torch_stub(monkeypatch, {"profile": "x"})
    impostor = tmp_path / "recovery.json"
    impostor.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to load"):
        video_longlive._load_recovery_tape(str(impostor))
    with pytest.raises(ValueError, match="refusing to load"):
        video_longlive._load_recovery_tape(str(tmp_path / "missing.pt"))
    assert fake_torch.load_calls == []


def test_recovery_tape_load_rejects_non_dict_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_torch_stub(monkeypatch, ["not", "a", "dict"])
    tape_path = tmp_path / "recovery.pt"
    tape_path.write_bytes(b"never-executed-bytes")
    with pytest.raises(ValueError, match="must be a dict"):
        video_longlive._load_recovery_tape(str(tape_path))


def test_generator_container_load_verifies_then_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 005: generator path hashes first, loads weights-only second."""
    checkpoint = tmp_path / "model_bf16.pt"
    checkpoint.write_bytes(b"generator-bytes")
    expected = hashlib.sha256(b"generator-bytes").hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"video": {"checkpoint_sha256": expected}}), encoding="utf-8"
    )
    fake_torch = _install_torch_stub(monkeypatch, {"generator": {}})
    container = video_longlive.load_generator_container(checkpoint, tmp_path)
    assert container == {"generator": {}}
    assert len(fake_torch.load_calls) == 1
    assert fake_torch.load_calls[0]["weights_only"] is True


def test_generator_container_load_fails_closed_on_tampered_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model_bf16.pt"
    checkpoint.write_bytes(b"tampered-bytes")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"video": {"checkpoint_sha256": "2" * 64}}), encoding="utf-8"
    )
    fake_torch = _install_torch_stub(monkeypatch, {"generator": {}})
    with pytest.raises(ValueError, match="sha256 mismatch"):
        video_longlive.load_generator_container(checkpoint, tmp_path)
    assert fake_torch.load_calls == []


def _install_text_encoder_stubs(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Stub the wan_5b text-encoder tree behind CpuUmt5Encoder."""
    seen_states: list[Any] = []

    class _FakeModel:
        def eval(self) -> _FakeModel:
            return self

        def load_state_dict(self, state: Any) -> None:
            seen_states.append(state)

    wan_package = types.ModuleType("wan_5b")
    modules_package = types.ModuleType("wan_5b.modules")
    t5_module = types.ModuleType("wan_5b.modules.t5")
    t5_module.umt5_xxl = lambda **kwargs: _FakeModel()  # type: ignore[attr-defined]
    tokenizers_module = types.ModuleType("wan_5b.modules.tokenizers")
    tokenizers_module.HuggingfaceTokenizer = lambda **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wan_5b", wan_package)
    monkeypatch.setitem(sys.modules, "wan_5b.modules", modules_package)
    monkeypatch.setitem(sys.modules, "wan_5b.modules.t5", t5_module)
    monkeypatch.setitem(sys.modules, "wan_5b.modules.tokenizers", tokenizers_module)
    return seen_states


def test_text_encoder_state_dict_load_uses_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 005: T5 .pth is a plain state dict — weights-only suffices."""
    fake_torch = _install_torch_stub(monkeypatch, {"encoder.block.0.weight": "tensor"})
    seen_states = _install_text_encoder_stubs(monkeypatch)
    wan_dir = tmp_path / "wan_models" / model_registry.WAN_SUBDIR
    (wan_dir / "google" / "umt5-xxl").mkdir(parents=True)
    (wan_dir / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"never-executed-bytes")
    video_longlive.CpuUmt5Encoder(wan_dir, "cpu")
    assert len(fake_torch.load_calls) == 1
    assert fake_torch.load_calls[0]["weights_only"] is True
    assert seen_states == [{"encoder.block.0.weight": "tensor"}]


def _write_causvid_stack(models_dir: Path, checkpoint_bytes: bytes) -> Path:
    checkpoint = models_dir / model_registry.CAUSVID_SUBDIR / model_registry.CAUSVID_CHECKPOINT_FILE
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(checkpoint_bytes)
    base_dir = models_dir / model_registry.WAN21_SUBDIR
    base_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "diffusion_pytorch_model.safetensors",
        "config.json",
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
    ):
        (base_dir / name).write_bytes(b"stub")
    tokenizer_dir = base_dir / "google" / "umt5-xxl"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_bytes(b"{}")
    return checkpoint


def _install_causvid_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the CUDA-only CausVid pipeline tree (init path only)."""

    class _FakeGenerator:
        def load_state_dict(self, state: Any, strict: bool = True) -> None:
            del state, strict

    class _FakeTextEncoder:
        def to(self, device: str) -> _FakeTextEncoder:
            del device
            return self

    class _FakePipeline:
        def __init__(self, config: Any, device: str) -> None:
            del config, device
            self.generator = _FakeGenerator()
            self.text_encoder = _FakeTextEncoder()

        def to(self, device: str, dtype: Any) -> _FakePipeline:
            del device, dtype
            return self

    causvid_package = types.ModuleType("causvid")
    models_package = types.ModuleType("causvid.models")
    wan_package = types.ModuleType("causvid.models.wan")
    inference_module = types.ModuleType("causvid.models.wan.causal_inference")
    inference_module.InferencePipeline = _FakePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "causvid", causvid_package)
    monkeypatch.setitem(sys.modules, "causvid.models", models_package)
    monkeypatch.setitem(sys.modules, "causvid.models.wan", wan_package)
    monkeypatch.setitem(sys.modules, "causvid.models.wan.causal_inference", inference_module)
    omegaconf_module = types.ModuleType("omegaconf")
    omegaconf_module.OmegaConf = types.SimpleNamespace(  # type: ignore[attr-defined]
        load=lambda path: types.SimpleNamespace(num_frame_per_block=3)
    )
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_module)


def test_causvid_checkpoint_load_verifies_then_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 005: DMD snapshot hashes first, loads weights-only second."""
    checkpoint = _write_causvid_stack(tmp_path, b"dmd-snapshot-bytes")
    expected = hashlib.sha256(b"dmd-snapshot-bytes").hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"causvid": {"checkpoint_sha256": expected}}), encoding="utf-8"
    )
    fake_torch = _install_torch_stub(monkeypatch, {"generator": {}})
    _install_causvid_stubs(monkeypatch)
    config_path = tmp_path / "wan_causal_dmd.yaml"
    config_path.write_text("num_frame_per_block: 3\n", encoding="utf-8")
    video_causvid.CausvidSession(tmp_path, "cuda:0", config_path=config_path)
    assert len(fake_torch.load_calls) == 1
    assert fake_torch.load_calls[0]["weights_only"] is True
    assert fake_torch.load_calls[0]["source"] == str(checkpoint)


def test_causvid_checkpoint_load_fails_closed_on_tampered_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_causvid_stack(tmp_path, b"tampered-bytes")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"causvid": {"checkpoint_sha256": "3" * 64}}), encoding="utf-8"
    )
    fake_torch = _install_torch_stub(monkeypatch, {"generator": {}})
    _install_causvid_stubs(monkeypatch)
    config_path = tmp_path / "wan_causal_dmd.yaml"
    config_path.write_text("num_frame_per_block: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        video_causvid.CausvidSession(tmp_path, "cuda:0", config_path=config_path)
    assert fake_torch.load_calls == []


def test_models_dir_layout_covers_shipped_stacks(tmp_path: Path) -> None:
    """Issue 086: every downloader tree is addressable from the helper."""
    layout = model_registry.models_dir_layout(tmp_path)
    assert layout["causvid_dir"] == str(tmp_path / model_registry.CAUSVID_SUBDIR)
    assert layout["wan21_dir"] == str(tmp_path / model_registry.WAN21_SUBDIR)
    assert layout["inspector_dir"] == str(tmp_path / model_registry.QWEN35_SUBDIR)
    assert layout["ltxv_text_encoder_dir"] == str(tmp_path / model_registry.LTXV_TE_SUBDIR)
