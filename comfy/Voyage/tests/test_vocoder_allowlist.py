"""Issue 072: MMAudio vocoder snapshot must be data-only (no executable fetches).

The nvidia BigVGAN snapshot previously allowed `*.py` plus a whole
`alias_free_activation/*` subtree. The loader (`mmaudio_sfx._load_feature_utils`)
resolves only data files via `from_pretrained` with the class already imported
from the pinned `/opt/mmaudio` clone — snapshot `.py` files are never imported
(`trust_remote_code` is never set). This module pins the data-only contract:
the allow-list carries exactly the consumed files, and verify fails loud when
any `.py` lands under the vocoder dir. CPU-only: sparse truncates + tmp_path,
no hub, no torch, no weights.
"""

from __future__ import annotations

from pathlib import Path

from voyage import model_registry


def _write_sparse(candidate: Path, size_bytes: int) -> None:
    """Create a sparse file of exactly `size_bytes` (instant, no data write)."""
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open("wb") as handle:
        handle.truncate(size_bytes)


def _write_minimal_sfx_tree(models_dir: Path) -> None:
    """Checklist-satisfying SFX tree (sparse weights + presence-only configs)."""
    spec = model_registry.MODEL_SPECS["sfx-mmaudio"]
    for check in spec.checks:
        if not isinstance(check, model_registry.RequiredFile):
            continue
        candidate = models_dir / check.relative_path
        if check.min_bytes > 0:
            _write_sparse(candidate, check.min_bytes)
        else:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text("{}", encoding="utf-8")


def test_vocoder_allow_list_is_data_only() -> None:
    """Allow-list carries exactly the consumed data files (no code globs)."""
    allowed = tuple(model_registry.MMAUDIO_VOCODER_ALLOW)
    assert "config.json" in allowed
    assert "bigvgan_generator.pt" in allowed
    for pattern in allowed:
        assert not pattern.endswith(".py"), f"executable pattern {pattern!r}"
        assert ".py" not in pattern, f"executable pattern {pattern!r}"
    assert not any("alias_free_activation" in pattern for pattern in allowed)


def test_verify_sfx_rejects_vocoder_python(tmp_path: Path) -> None:
    """A `.py` under the vocoder dir fails verify loud (unexpected executable)."""
    _write_minimal_sfx_tree(tmp_path)
    vocoder_dir = tmp_path / model_registry.MMAUDIO_VOCODER_SUBDIR
    (vocoder_dir / "modeling_bigvgan.py").write_text("# unexpected", encoding="utf-8")
    ok, message = model_registry.verify_sfx_models(tmp_path)
    assert ok is False
    assert ".py" in message


def test_verify_sfx_rejects_nested_vocoder_python(tmp_path: Path) -> None:
    """Nested executable fetches fail too (old `alias_free_activation/*` subtree)."""
    _write_minimal_sfx_tree(tmp_path)
    nested = tmp_path / model_registry.MMAUDIO_VOCODER_SUBDIR / "alias_free_activation"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "activations.py").write_text("# unexpected", encoding="utf-8")
    ok, message = model_registry.verify_sfx_models(tmp_path)
    assert ok is False
    assert ".py" in message


def test_verify_sfx_accepts_clean_vocoder(tmp_path: Path) -> None:
    """A code-free vocoder dir keeps verify green (data-only baseline)."""
    _write_minimal_sfx_tree(tmp_path)
    ok, message = model_registry.verify_sfx_models(tmp_path)
    assert ok, message
