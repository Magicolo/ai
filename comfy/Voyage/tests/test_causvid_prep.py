"""Stream D CausVid prep scaffolding tests (DESIGN §5.4, TASK §30.1).

CPU-only: registry pins resolve, verify fails on a missing dir, verify
passes against a tmp fixture, and the upstream-notes doc exists. No worker,
no weights, no GPU — the worker slice (Stream C) owns live behavior.
"""

from __future__ import annotations

from pathlib import Path

from voyage import model_registry
from voyage.model_registry import verify_causvid_models

_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_hex40(value: str) -> bool:
    return len(value) == 40 and all(char in _HEX_DIGITS for char in value)


def _sparse_file(path: Path, size: int) -> None:
    """Create a size-byte sparse stand-in.

    Verify gates on `st_size`, never on content, so a truncated (sparse)
    file exercises the size check without writing gigabytes in a CPU-only
    prep test. Small descriptor files are written for real by the caller.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def test_causvid_pins_resolve() -> None:
    assert model_registry.CAUSVID_HF_REPO == "tianweiy/CausVid"
    assert _is_hex40(model_registry.CAUSVID_HF_REVISION)
    assert _is_hex40(model_registry.CAUSVID_COMMIT)
    assert model_registry.CAUSVID_COMMIT.startswith(model_registry.CAUSVID_COMMIT_SHORT)
    assert model_registry.CAUSVID_SUBDIR == "causvid"
    assert model_registry.CAUSVID_CHECKPOINT_FILE == (
        f"{model_registry.CAUSVID_CHECKPOINT_SUBDIR}/{model_registry.CAUSVID_CHECKPOINT_NAME}"
    )
    assert "BY-NC-SA" in model_registry.CAUSVID_LICENSE
    assert model_registry.WAN21_HF_REPO == "Wan-AI/Wan2.1-T2V-1.3B"
    assert _is_hex40(model_registry.WAN21_HF_REVISION)
    assert model_registry.WAN21_ALLOW
    assert any(pattern.endswith(".pth") for pattern in model_registry.WAN21_ALLOW)
    assert model_registry.WAN21_LICENSE == "Apache 2.0"


def test_verify_causvid_missing_dir_fails(tmp_path: Path) -> None:
    ok, message = verify_causvid_models(tmp_path / "models")
    assert not ok
    assert "missing" in message


def test_verify_causvid_fixture_passes(tmp_path: Path) -> None:
    models = tmp_path / "models"
    _sparse_file(
        models / model_registry.CAUSVID_SUBDIR / model_registry.CAUSVID_CHECKPOINT_FILE,
        model_registry.CAUSVID_CKPT_MIN_BYTES + 8,
    )
    wan21_dir = models / model_registry.WAN21_SUBDIR
    _sparse_file(
        wan21_dir / "diffusion_pytorch_model.safetensors",
        model_registry.WAN21_DIT_MIN_BYTES + 8,
    )
    _sparse_file(wan21_dir / "Wan2.1_VAE.pth", model_registry.WAN21_VAE_MIN_BYTES + 8)
    _sparse_file(
        wan21_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        model_registry.WAN21_T5_MIN_BYTES + 8,
    )
    (wan21_dir / "config.json").write_text("{}", encoding="utf-8")
    tokenizer_dir = wan21_dir / "google" / "umt5-xxl"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    ok, message = verify_causvid_models(models)
    assert ok, message
    assert "causvid OK" in message


def test_upstream_causvid_notes_doc_exists() -> None:
    notes = Path(__file__).resolve().parent.parent / "docs" / "UPSTREAM_CAUSVID_NOTES.md"
    text = notes.read_text(encoding="utf-8")
    assert model_registry.CAUSVID_COMMIT in text
    assert model_registry.CAUSVID_HF_REVISION in text
    assert model_registry.WAN21_HF_REVISION in text
    assert "BY-NC-SA" in text
    assert "16 fps" in text
