"""Explicit-quality augment plan (pure plan tests).

`plan_augmentation` is pure math over the probed source + explicit
quality multipliers, so ffmpeg rendering (`finalize_run`) just executes
the plan. All tests here are pure math — no ffmpeg, no run directory —
pinning the contract cases (native passthrough, upscale math,
interpolate math, CausVid 16fps ships native, fps-drop re-encodes
without minterpolate, unprobable sources raise, bad multipliers
rejected).
"""

from __future__ import annotations

import pytest

from voyage.config import AugmentConfig
from voyage.errors import MediaError
from voyage.media import FinalizeOptions, plan_augmentation


def test_augment_defaults_are_explicit_unity() -> None:
    assert (AugmentConfig().upscale, AugmentConfig().interpolate) == (1, 1)
    assert AugmentConfig().presentation_fps is None
    options = FinalizeOptions()
    assert (options.upscale, options.interpolate) == (1, 1)
    assert options.presentation_fps is None


def test_ltx25_native_passes_through_at_unity() -> None:
    """ltx25 native 1216x704@24 ships untouched at 1/1 (stream-copy fast path)."""
    plan = plan_augmentation(1216, 704, 24.0)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1216, 704, 24)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is False


def test_causvid_native_ships_native_at_unity() -> None:
    """No floors: CausVid 832x480@16 at 1/1 ships 16fps (no lift)."""
    plan = plan_augmentation(832, 480, 16.0)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (832, 480, 16)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is False


def test_upscale_doubles_both_axes() -> None:
    """768x512 x2 = 1536x1024; geometry change forces a re-encode (no lift)."""
    plan = plan_augmentation(768, 512, 24.0, upscale=2)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1536, 1024, 24)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_upscale_4x_quadruples_both_axes() -> None:
    plan = plan_augmentation(768, 512, 24.0, upscale=4)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (3072, 2048, 24)
    assert plan.needs_reencode is True


def test_interpolate_quadruples_fps() -> None:
    """24fps x4 = 96fps; the fps lift arms the minterpolate path."""
    plan = plan_augmentation(768, 512, 24.0, interpolate=4)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (768, 512, 96)
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True


def test_interpolate_doubles_fps() -> None:
    plan = plan_augmentation(768, 432, 24.0, interpolate=2)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (768, 432, 48)
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True


def test_presentation_fps_pins_rate() -> None:
    """24fps x2 presented at 32fps: out is pinned, lift flags (slow-mo setup)."""
    plan = plan_augmentation(768, 512, 24.0, interpolate=2, presentation_fps=32)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (768, 512, 32)
    assert plan.needs_minterpolate is True
    assert plan.needs_reencode is True


def test_fps_drop_never_minterpolates_but_still_reencodes() -> None:
    """A 60fps source presented at 32fps uses the plain fps filter."""
    plan = plan_augmentation(1280, 720, 60.0, presentation_fps=32)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1280, 720, 32)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_model_interpolate_suppresses_minterpolate() -> None:
    """When the model pass already interpolated, ffmpeg adds no lift.

    24fps x2 content (48fps interpolated rate) presented at 32fps
    stretches the timeline (slow motion) instead of synthesizing frames.
    """
    plan = plan_augmentation(
        768, 512, 24.0, interpolate=2, presentation_fps=32, model_interpolate=2
    )
    assert (plan.out_w, plan.out_h, plan.out_fps) == (768, 512, 32)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_unprobable_dims_always_raise() -> None:
    """Unprobable dims can never take the stream-copy fast path (fail loud)."""
    with pytest.raises(MediaError, match="unprobable source dims"):
        plan_augmentation(0, 0, 24.0)
    with pytest.raises(MediaError, match="unprobable source dims"):
        plan_augmentation(0, 0, 0.0, presentation_fps=32)


def test_unprobable_fps_needs_presentation_pin() -> None:
    """Unprobable fps raises unless presentation_fps pins the rate."""
    with pytest.raises(MediaError, match="unprobable source fps"):
        plan_augmentation(1280, 720, 0.0)
    plan = plan_augmentation(1280, 720, 0.0, presentation_fps=32)
    assert (plan.out_w, plan.out_h, plan.out_fps) == (1280, 720, 32)
    assert plan.needs_minterpolate is False
    assert plan.needs_reencode is True


def test_rejects_bad_upscale_and_interpolate() -> None:
    with pytest.raises(ValueError, match="upscale"):
        plan_augmentation(832, 480, 16.0, upscale=3)
    with pytest.raises(ValueError, match="upscale"):
        plan_augmentation(832, 480, 16.0, upscale=0)
    with pytest.raises(ValueError, match="interpolate"):
        plan_augmentation(832, 480, 16.0, interpolate=0)
    with pytest.raises(ValueError, match="model_interpolate"):
        plan_augmentation(832, 480, 16.0, model_interpolate=0)


def test_finalize_options_rejects_bad_quality() -> None:
    with pytest.raises(ValueError, match="upscale"):
        FinalizeOptions(upscale=3)
    with pytest.raises(ValueError, match="interpolate"):
        FinalizeOptions(interpolate=0)
    with pytest.raises(ValueError, match="presentation_fps"):
        FinalizeOptions(presentation_fps=0)
