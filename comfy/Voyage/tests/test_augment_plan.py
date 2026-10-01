"""Track B: finalize-stage fps/resolution floors (pure plan tests).

`plan_augmentation` is the CPU fallback path later GPU model passes plug
into: it computes the presentation box/fps from probed source +
requested target + floors, so ffmpeg rendering (`finalize_run`) just
executes the plan. All tests here are pure math — no ffmpeg, no run
directory — pinning the four contract cases (CausVid lift, LTXV cover,
already-above unchanged, 0 disables) plus the boundary pins the vf
wiring relies on (presentation floor survives 0, drops never
minterpolate, unknown sources force re-encode).
"""

from __future__ import annotations

import pytest

from voyage.media import (
    AUGMENT_DEFAULT_MIN_FPS,
    AUGMENT_DEFAULT_MIN_HEIGHT,
    AUGMENT_DEFAULT_MIN_WIDTH,
    FinalizeOptions,
    plan_augmentation,
)


def test_augment_defaults_match_coming_config() -> None:
    assert AUGMENT_DEFAULT_MIN_FPS == 32
    assert AUGMENT_DEFAULT_MIN_WIDTH == 1280
    assert AUGMENT_DEFAULT_MIN_HEIGHT == 720
    options = FinalizeOptions()
    assert (options.min_fps, options.min_width, options.min_height) == (32, 1280, 720)


def test_causvid_native_lifts_to_presentation() -> None:
    """CausVid 832x480@16 -> 1280x720@32 with motion-interpolation lift."""
    plan = plan_augmentation(832, 480, 16.0, 832, 480, 16, 32, 1280, 720)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1280, 720, 32)
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True


def test_ltxv_native_covers_presentation_preserving_aspect() -> None:
    """LTXV 768x512@24 -> 1280x720@32; the vf fits + pads (never stretches).

    The 3:2 source cannot fill the 16:9 box pixel-for-pixel, so the plan
    keeps the covering box and the scale-to-fit + pad chain letterboxes —
    aspect preserved, floors covered.
    """
    plan = plan_augmentation(768, 512, 24.0, 768, 512, 24, 32, 1280, 720)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1280, 720, 32)
    assert plan.out_w >= 1280 and plan.out_h >= 720
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True


def test_already_above_floors_unchanged() -> None:
    """A target already covering the floors keeps its geometry, no lift."""
    plan = plan_augmentation(1920, 1080, 32.0, 1920, 1080, 32, 32, 1280, 720)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1920, 1080, 32)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is False


def test_zero_disables_floors() -> None:
    """0 disables the new floors — native geometry passes through."""
    plan = plan_augmentation(768, 432, 24.0, 768, 432, 24, 0, 0, 0)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (768, 432, 24)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is False


def test_zero_fps_floor_still_respects_presentation_min() -> None:
    """0 disables the 32fps floor, not the shipped-video 24fps guarantee."""
    plan = plan_augmentation(832, 480, 16.0, 832, 480, 16, 0, 0, 0)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (832, 480, 24)
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True


def test_none_disables_like_zero() -> None:
    plan = plan_augmentation(768, 432, 24.0, 768, 432, 24, None, None, None)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (768, 432, 24)
    assert plan.needs_reencode is False


def test_fps_drop_never_minterpolates_but_still_reencodes() -> None:
    """A 60fps source finalized at 32fps uses the plain fps filter."""
    plan = plan_augmentation(1280, 720, 60.0, 1280, 720, 24, 32, 1280, 720)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1280, 720, 32)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_unknown_source_forces_reencode_without_lift() -> None:
    """Unprobable fps/geometry can never take the stream-copy fast path."""
    plan = plan_augmentation(0, 0, 0.0, 1280, 720, 24, 32, 1280, 720)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1280, 720, 32)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_rejects_nonpositive_target_or_requested() -> None:
    with pytest.raises(ValueError, match="target geometry"):
        plan_augmentation(832, 480, 16.0, 0, 480, 16, 32, 1280, 720)
    with pytest.raises(ValueError, match="requested fps"):
        plan_augmentation(832, 480, 16.0, 832, 480, 0, 32, 1280, 720)


def test_rejects_negative_floors() -> None:
    with pytest.raises(ValueError, match="floors"):
        plan_augmentation(832, 480, 16.0, 832, 480, 16, -1, 1280, 720)
    with pytest.raises(ValueError, match="min_fps"):
        FinalizeOptions(min_fps=-1)
    with pytest.raises(ValueError, match="min_width"):
        FinalizeOptions(min_width=-1)
    with pytest.raises(ValueError, match="min_height"):
        FinalizeOptions(min_height=-1)


def test_source_above_target_and_floors_is_preserved() -> None:
    """Segments above target+floors are never downscaled (floors minimum).

    A 2432x1408 source with a 1216x704 target and 1280x720 floors must ship
    at its native spec — the floors are a minimum quality requirement, not
    a ceiling, and downscaling would waste the rendered compute.
    """
    plan = plan_augmentation(2432, 1408, 32.0, 1216, 704, 32, 32, 1280, 720)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (2432, 1408, 32)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is False


def test_floors_disabled_preserves_above_target_source() -> None:
    """`--min-* 0` must not downscale segments above the config target.

    With floors disabled, a 1536x1024 source against a 768x512 target ships
    at native spec. (The pre-hardening plan downscaled to the target here.)
    """
    plan = plan_augmentation(1536, 1024, 24.0, 768, 512, 24, 0, 0, 0)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1536, 1024, 24)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is False


def test_partial_axis_each_dimension_resolves_independently() -> None:
    """Per-axis max: a wide-but-short source keeps its width, lifts height.

    1920x700 against a 1216x704 target with 1280x720 floors ships 1920x720:
    width from the source, height from the floor.
    """
    plan = plan_augmentation(1920, 700, 24.0, 1216, 704, 24, 32, 1280, 720)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1920, 720, 32)
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True
