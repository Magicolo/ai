"""Models-dir layout coverage for the 245 rows (issue 245).

CPU-only: pure path joins over a tmp models dir — no downloads, no GPU.
"""

from __future__ import annotations

from pathlib import Path

from voyage import model_registry


def test_layout_names_rife_dir_alongside_film(tmp_path: Path) -> None:
    """Both interpolation specs resolve even though they share one dir."""
    layout = model_registry.models_dir_layout(tmp_path)
    assert layout["film_dir"] == str(tmp_path / model_registry.FILM_SUBDIR)
    assert layout["rife_dir"] == str(tmp_path / model_registry.RIFE_SUBDIR)
    assert layout["rife_dir"] == layout["film_dir"]


def test_layout_names_director_gguf_and_awq_dirs(tmp_path: Path) -> None:
    layout = model_registry.models_dir_layout(tmp_path)
    assert layout["director_gguf_dir"] == str(tmp_path / model_registry.QWEN35_GGUF_SUBDIR)
    assert layout["director_awq_dir"] == str(tmp_path / model_registry.QWEN4B_AWQ_SUBDIR)


def test_layout_covers_245_spec_relative_dirs(tmp_path: Path) -> None:
    """Every relative dir of the three 245 specs appears in the layout."""
    layout_values = set(model_registry.models_dir_layout(tmp_path).values())
    for spec_name in ("rife", "director-qwen35-gguf", "director-qwen4b-awq"):
        spec = model_registry.MODEL_SPECS[spec_name]
        for snapshot in spec.snapshots:
            assert str(tmp_path / snapshot.relative_dir) in layout_values
        for filereq in spec.files:
            # Single-file rows land at relative_dir/subfolder/filename
            # (either segment may be empty) — the containing dir must be
            # a listed layout root.
            parent = (tmp_path / filereq.relative_dir / filereq.subfolder / filereq.filename).parent
            assert str(parent) in layout_values
