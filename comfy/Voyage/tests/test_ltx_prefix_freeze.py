"""Stage-2 prefix-freeze wiring for the LTX workers (DESIGN §140).

The `LTXVLatentUpsampler` drops `noise_mask`, so the stage-2 refine
repaints the frozen continuation prefix. Both workers re-attach the mask
through the shared `ltx_mask_utils.LTXPrefixFreeze` pack node (node 57,
prefix branch only, K derived from the carry). The graph-shape assertions
below run in the slim gates image (dicts only, no torch); the pack node's
tensor behavior is covered by torch tests that skip without torch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from voyage.workers import video_ltx23, video_ltx25


def _pack_parent() -> Path | None:
    """Locate the `custom_nodes` dir holding the pack (host checkout or image)."""
    override = os.environ.get("VOYAGE_LTX_MASK_PACK")
    candidates = ([Path(override)] if override else []) + [
        Path(__file__).resolve().parents[2] / "Comfy" / "custom_nodes",
        Path("/opt/comfyui/custom_nodes"),
    ]
    for candidate in candidates:
        if (candidate / "ltx_mask_utils" / "prefix_freeze.py").is_file():
            return candidate
    return None


def _freeze_node():
    torch = pytest.importorskip("torch", reason="pack node needs torch (absent from slim image)")
    pack_parent = _pack_parent()
    if pack_parent is None:
        pytest.skip("ltx_mask_utils pack not mounted (host checkout or /opt/comfyui)")
    if str(pack_parent) not in sys.path:
        sys.path.insert(0, str(pack_parent))
    from ltx_mask_utils.prefix_freeze import LTXPrefixFreeze

    return LTXPrefixFreeze, torch


def _prefix_names(count: int) -> list[str]:
    return [f"frame_{index:02d}.png" for index in range(count)]


def test_ltx25_prefix_branch_carries_freeze_node() -> None:
    graph = video_ltx25.build_mode_a_graph(
        prompt="harbor",
        seed=7,
        save_prefix="seg000001-b0",
        prefix_filenames=_prefix_names(25),
    )
    assert len(graph) == 29 + 25 + 2 + 1
    freeze = graph["57"]
    assert freeze["class_type"] == "LTXPrefixFreeze"
    assert freeze["inputs"]["samples"] == ["18", 0]
    assert freeze["inputs"]["prefix_latent_frames"] == 4
    assert graph["19"]["inputs"]["video_latent"] == ["57", 0]


def test_ltx25_fresh_graph_has_no_freeze_node() -> None:
    graph = video_ltx25.build_mode_a_graph(prompt="harbor", seed=7, save_prefix="seg000001-b0")
    assert len(graph) == 29
    assert "57" not in graph
    assert graph["19"]["inputs"]["video_latent"] == ["18", 0]


def test_ltx25_freeze_k_derives_from_carry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_LTX_CARRY", "9")
    graph = video_ltx25.build_mode_a_graph(
        prompt="harbor",
        seed=7,
        save_prefix="seg000001-b0",
        prefix_filenames=_prefix_names(9),
    )
    assert graph["57"]["inputs"]["prefix_latent_frames"] == 2


def test_ltx23_prefix_branch_carries_freeze_node() -> None:
    graph = video_ltx23.build_mode_a_graph(
        prompt="harbor",
        seed=7,
        save_prefix="seg000001-b0",
        prefix_filenames=_prefix_names(25),
    )
    freeze = graph["57"]
    assert freeze["class_type"] == "LTXPrefixFreeze"
    assert freeze["inputs"]["samples"] == ["18", 0]
    assert freeze["inputs"]["prefix_latent_frames"] == 4
    assert graph["19"]["inputs"]["video_latent"] == ["57", 0]


def test_ltx23_fresh_graph_has_no_freeze_node() -> None:
    graph = video_ltx23.build_mode_a_graph(prompt="harbor", seed=7, save_prefix="seg000001-b0")
    assert "57" not in graph
    assert graph["19"]["inputs"]["video_latent"] == ["18", 0]


def test_freeze_builds_5d_mask() -> None:
    LTXPrefixFreeze, torch = _freeze_node()
    (out,) = LTXPrefixFreeze().freeze({"samples": torch.zeros(1, 128, 7, 12, 7)}, 2)
    mask = out["noise_mask"]
    assert mask.shape == (1, 1, 7, 1, 1)
    assert mask.dtype == torch.float32
    assert bool((mask[:, :, :2] == 0.0).all())
    assert bool((mask[:, :, 2:] == 1.0).all())


def test_freeze_zero_prefix_is_full_denoise() -> None:
    LTXPrefixFreeze, torch = _freeze_node()
    (out,) = LTXPrefixFreeze().freeze({"samples": torch.zeros(1, 128, 7, 12, 7)}, 0)
    assert bool((out["noise_mask"] == 1.0).all())


def test_freeze_rejects_out_of_range() -> None:
    LTXPrefixFreeze, torch = _freeze_node()
    latent = {"samples": torch.zeros(1, 128, 7, 12, 7)}
    with pytest.raises(ValueError, match="prefix_latent_frames"):
        LTXPrefixFreeze().freeze(latent, 8)
    with pytest.raises(ValueError, match="prefix_latent_frames"):
        LTXPrefixFreeze().freeze(latent, -1)
