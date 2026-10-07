"""Issue 288: CausVid validates overlap against the config block size + fps-first.

`handle_init` validated `overlap_frames` against the module constant
while the session re-validated against the YAML config's
`num_frame_per_block` — a divergent config passed init and failed after
minutes of model load. Init now reads the block size from the config
(cheap stdlib regex, no torch/omegaconf), and `handle_generate_blocks`
rejects a bad fps before the session check like ltxv/ltx25.

All stdlib (slim gates image).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voyage.workers.video_causvid import (
    DEFAULT_OVERLAP_FRAMES,
    NUM_FRAME_PER_BLOCK,
    config_block_size,
)


def test_config_block_size_reads_yaml(tmp_path: Path) -> None:
    """The pinned key parses with indentation and trailing comments."""
    config = tmp_path / "wan_causal_dmd.yaml"
    config.write_text(
        "image_or_video_shape: [1, 21, 16, 60, 104]\n  num_frame_per_block: 6  # wide blocks\n",
        encoding="utf-8",
    )
    assert config_block_size(config) == 6


def test_config_block_size_falls_back_to_constant(tmp_path: Path) -> None:
    """Missing file/key/value reads as the session's own default."""
    assert config_block_size(tmp_path / "absent.yaml") == NUM_FRAME_PER_BLOCK
    bare = tmp_path / "bare.yaml"
    bare.write_text("image_or_video_shape: [1, 21, 16, 60, 104]\n", encoding="utf-8")
    assert config_block_size(bare) == NUM_FRAME_PER_BLOCK
    garbage = tmp_path / "garbage.yaml"
    garbage.write_text("num_frame_per_block: lots\n", encoding="utf-8")
    assert config_block_size(garbage) == NUM_FRAME_PER_BLOCK
    zero = tmp_path / "zero.yaml"
    zero.write_text("num_frame_per_block: 0\n", encoding="utf-8")
    assert config_block_size(zero) == NUM_FRAME_PER_BLOCK


def test_init_validates_against_config_block_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlap valid per constant but not per config fails before any load."""
    from voyage.workers import video_causvid

    monkeypatch.setattr(video_causvid, "_SESSION", None)
    config = tmp_path / "wan_causal_dmd.yaml"
    config.write_text("num_frame_per_block: 6\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "models_dir": str(tmp_path),
        "device": "cuda:0",
        "overlap_frames": 3,
        "config_path": str(config),
    }
    with pytest.raises(ValueError, match="divisible by num_frame_per_block"):
        video_causvid.handle_init(payload)


def test_init_accepts_config_consistent_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config-consistent overlap passes validation (then fails past it)."""
    from voyage.workers import video_causvid

    monkeypatch.setattr(video_causvid, "_SESSION", None)
    config = tmp_path / "wan_causal_dmd.yaml"
    config.write_text("num_frame_per_block: 3\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "models_dir": str(tmp_path),
        "device": "cuda:0",
        "overlap_frames": DEFAULT_OVERLAP_FRAMES,
        "config_path": str(config),
    }
    # Overlap passes; the missing weight stack fails next (fail-fast order).
    with pytest.raises(FileNotFoundError, match="missing"):
        video_causvid.handle_init(payload)


def test_generate_blocks_rejects_bad_fps_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad fps raises ValueError before the not-initialized RuntimeError."""
    from voyage.workers import video_causvid

    monkeypatch.setattr(video_causvid, "_SESSION", None)
    with pytest.raises(ValueError, match="fps must be positive"):
        video_causvid.handle_generate_blocks({"fps": -1})
    with pytest.raises(RuntimeError, match="not initialized"):
        video_causvid.handle_generate_blocks({"fps": 16})
