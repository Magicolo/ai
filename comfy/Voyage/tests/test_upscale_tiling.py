"""Spatial tiling for the Real-ESRGAN upscale leg (DESIGN §140 GPU defaults).

Live incident (this change): a single 1216x704 frame through RRDBNet-x4
needs ~5.5 GiB — batch-halving bottoms out at one frame and still OOMs
the 6 GB 2060 (`torch.OutOfMemoryError` in `conv_first`), while 768x512
peaks at 3.29 GiB and fits. The upscale leg therefore tiles frames above
a pixel budget (2D grid with overlap margins cropped after upscale, the
official Real-ESRGAN recipe) instead of failing; frames below the budget
keep the exact direct path (byte-identical everywhere else).

`tile_grid` + the budget predicate are stdlib-pure and tested in gates;
the tiled-vs-direct numeric equivalence needs torch + weights, so it
runs wherever those exist (video image, idle-GPU manual runs) and skips
in the slim gate image.
"""

from __future__ import annotations

import os

import pytest


def test_tile_grid_covers_without_gaps() -> None:
    from voyage.workers.augment_worker import tile_grid

    tiles = tile_grid(1216, 704, tile=512)
    assert tiles[0][:2] == (0, 0)
    assert tiles[-1][2:] == (1216, 704)
    covered = 0
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= 1216
        assert 0 <= y0 < y1 <= 704
        assert x1 - x0 <= 512 and y1 - y0 <= 512
        covered += (x1 - x0) * (y1 - y0)
    # Exact partition: areas sum to the frame (no gaps, no overlaps —
    # overlap lives in the model's input context, cropped after upscale).
    assert covered == 1216 * 704
    # Full coverage: every 64px probe point sits in exactly one tile.
    for probe_y in range(0, 704, 64):
        for probe_x in range(0, 1216, 64):
            hits = sum(1 for x0, y0, x1, y1 in tiles if x0 <= probe_x < x1 and y0 <= probe_y < y1)
            assert hits == 1


def test_tile_grid_small_frame_is_single_tile() -> None:
    from voyage.workers.augment_worker import tile_grid

    assert tile_grid(512, 512, tile=512) == [(0, 0, 512, 512)]
    assert tile_grid(768, 512, tile=512) == [(0, 0, 384, 512), (384, 0, 768, 512)]


def test_tiling_budget_separates_measured_fit_from_oom() -> None:
    """1216x704 (measured OOM) tiles; 768x512 and 832x480 (measured fit,
    3.29 GiB peak) keep the direct path."""
    from voyage.workers.augment_worker import needs_upscale_tiling

    assert needs_upscale_tiling(1216, 704) is True
    assert needs_upscale_tiling(768, 512) is False
    assert needs_upscale_tiling(832, 480) is False


def test_tiled_upscale_matches_direct_within_tolerance() -> None:
    """Forced tiling on a small frame must not change the pixels (seams
    would show as grid artifacts in the shipped video)."""
    torch = pytest.importorskip("torch")
    from voyage.workers.augment_worker import upscale_frames

    weights = "/models/realesrgan/RealESRGAN_x4plus_anime_6B.pth"
    if not os.path.exists(weights):
        pytest.skip("augment weights absent (needs /models)")
    frame = torch.rand(3, 256, 256)
    direct = upscale_frames([frame], weights, scale=2, device="cpu")[0]
    tiled = upscale_frames([frame], weights, scale=2, device="cpu", tile=128)[0]
    assert tiled.shape == direct.shape
    assert float((tiled - direct).abs().max()) < 0.02
