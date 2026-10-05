"""LTX23 continuation strength parameter (Track B1/C) — ltx23 only.

build_mode_a_graph must accept strength (default 1.0 frozen prefix)
and write it into node 56 LTXVImgToVideoInplace.inputs.strength.
ltx25 moved to Extend chaining (see test_ltx_extend_chain.py).
"""

from voyage.workers.video_ltx23 import build_mode_a_graph as build_23


def _prefix() -> list[str]:
    return [f"p{i:02d}.png" for i in range(25)]


def test_ltx23_default_strength_is_one() -> None:
    graph = build_23(prompt="p", seed=1, save_prefix="s", prefix_filenames=_prefix())
    assert graph["56"]["inputs"]["strength"] == 1.0


def test_ltx23_strength_override() -> None:
    graph = build_23(prompt="p", seed=1, save_prefix="s", prefix_filenames=_prefix(), strength=0.6)
    assert graph["56"]["inputs"]["strength"] == 0.6
