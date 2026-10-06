"""LTX25 experiment profile: fast geometry + carry/strength overrides (Phase 1).

Production defaults must mirror the baked constants exactly (env unset);
VOYAGE_LTX_FAST=1 selects the fast-iteration profile (768x448 commit,
49f windows, 9f carry); VOYAGE_LTX_CARRY overrides the carry with
contract validation; VOYAGE_LTX_STRENGTH overrides the frozen-prefix
strength (ltx23 parity). All pure — no GPU session needed.
"""

import pytest

from voyage.workers.video_ltx25 import (
    COMMIT_HEIGHT,
    COMMIT_WIDTH,
    CONDITIONING_TAIL_FRAMES,
    SEGMENT_TARGET_FRAMES,
    STAGE1_HEIGHT,
    STAGE1_WIDTH,
    build_mode_a_graph,
    continuation_strength_from_env,
    extend_conditioning_tail,
    resolve_experiment_profile,
)


def test_production_defaults_mirror_baked_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_LTX_FAST", raising=False)
    monkeypatch.delenv("VOYAGE_LTX_CARRY", raising=False)
    profile = resolve_experiment_profile()
    assert profile.stage1_width == STAGE1_WIDTH
    assert profile.stage1_height == STAGE1_HEIGHT
    assert profile.commit_width == COMMIT_WIDTH
    assert profile.commit_height == COMMIT_HEIGHT
    assert profile.target_frames == SEGMENT_TARGET_FRAMES
    assert profile.tail_frames == CONDITIONING_TAIL_FRAMES


def test_fast_profile_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_LTX_FAST", "1")
    monkeypatch.delenv("VOYAGE_LTX_CARRY", raising=False)
    profile = resolve_experiment_profile()
    assert (profile.stage1_width, profile.stage1_height) == (384, 224)
    assert (profile.commit_width, profile.commit_height) == (768, 448)
    assert profile.commit_width % 64 == 0
    assert profile.commit_height % 64 == 0
    assert (profile.target_frames - 1) % 8 == 0
    assert (profile.tail_frames - 1) % 8 == 0
    assert profile.target_frames == 49
    assert profile.tail_frames == 9


def test_carry_override_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_LTX_FAST", raising=False)
    monkeypatch.setenv("VOYAGE_LTX_CARRY", "17")
    profile = resolve_experiment_profile()
    assert profile.tail_frames == 17
    assert profile.target_frames == SEGMENT_TARGET_FRAMES


def test_carry_override_valid_on_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_LTX_FAST", "1")
    monkeypatch.setenv("VOYAGE_LTX_CARRY", "17")
    profile = resolve_experiment_profile()
    assert (profile.target_frames, profile.tail_frames) == (49, 17)


@pytest.mark.parametrize("raw_carry", ["16", "0", "-7", "257", "265", "nine", "9.0", ""])
def test_carry_override_invalid(monkeypatch: pytest.MonkeyPatch, raw_carry: str) -> None:
    monkeypatch.delenv("VOYAGE_LTX_FAST", raising=False)
    monkeypatch.setenv("VOYAGE_LTX_CARRY", raw_carry)
    with pytest.raises(ValueError, match="VOYAGE_LTX_CARRY"):
        resolve_experiment_profile()


def test_carry_override_fast_mid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_LTX_FAST", "1")
    monkeypatch.setenv("VOYAGE_LTX_CARRY", "25")
    profile = resolve_experiment_profile()
    assert profile.tail_frames == 25


def test_strength_default_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_LTX_STRENGTH", raising=False)
    assert continuation_strength_from_env() == 1.0


@pytest.mark.parametrize("raw_strength", ["0", "-0.5", "1.5", "nan", "abc", ""])
def test_strength_invalid(monkeypatch: pytest.MonkeyPatch, raw_strength: str) -> None:
    monkeypatch.setenv("VOYAGE_LTX_STRENGTH", raw_strength)
    with pytest.raises(ValueError, match="VOYAGE_LTX_STRENGTH"):
        continuation_strength_from_env()


def test_graph_honors_fast_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_LTX_FAST", "1")
    monkeypatch.delenv("VOYAGE_LTX_CARRY", raising=False)
    prefix = [f"p{i:02d}.png" for i in range(9)]
    graph = build_mode_a_graph(prompt="p", seed=1, save_prefix="s", prefix_filenames=prefix)
    assert graph["8"]["inputs"]["width"] == 384
    assert graph["8"]["inputs"]["height"] == 224
    assert graph["8"]["inputs"]["length"] == 49
    assert graph["9"]["inputs"]["frames_number"] == 49
    assert graph["56"]["inputs"]["strength"] == 1.0


def test_graph_strength_param(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_LTX_FAST", raising=False)
    monkeypatch.delenv("VOYAGE_LTX_CARRY", raising=False)
    prefix = [f"p{i:02d}.png" for i in range(25)]
    graph = build_mode_a_graph(
        prompt="p", seed=1, save_prefix="s", prefix_filenames=prefix, strength=0.6
    )
    assert graph["56"]["inputs"]["strength"] == 0.6


def test_graph_prefix_count_follows_carry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_LTX_FAST", raising=False)
    monkeypatch.setenv("VOYAGE_LTX_CARRY", "17")
    with pytest.raises(ValueError, match="prefix needs 17 frames"):
        build_mode_a_graph(
            prompt="p",
            seed=1,
            save_prefix="s",
            prefix_filenames=[f"p{i:02d}.png" for i in range(25)],
        )


def test_extend_tail_fresh_is_novel_suffix() -> None:
    assert extend_conditioning_tail(None, [f"n{i}" for i in range(49)], 9) == [
        f"n{i}" for i in range(40, 49)
    ]


def test_extend_tail_production_is_novel_suffix() -> None:
    old = [f"o{i}" for i in range(25)]
    novel = [f"n{i}" for i in range(232)]
    assert extend_conditioning_tail(old, novel, 25) == novel[-25:]


def test_extend_tail_wide_carry_keeps_window_tail() -> None:
    # Fast window: 25-frame prefix + 24 novel; the window tail is the last
    # prefix frame plus the whole novel (no silent shortening to 24).
    old = [f"o{i}" for i in range(25)]
    novel = [f"n{i}" for i in range(24)]
    assert extend_conditioning_tail(old, novel, 25) == ["o24", *[f"n{i}" for i in range(24)]]


def test_extend_tail_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        extend_conditioning_tail(None, [], 9)
    with pytest.raises(ValueError, match="positive"):
        extend_conditioning_tail(None, ["n0"], 0)
