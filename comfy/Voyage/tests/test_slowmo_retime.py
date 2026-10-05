"""Slow-mo retiming: setpts video stretch + SFX bounds scaling (DESIGN §140).

When an explicit `--presentation-fps` combines with `interpolate > 1`
so that `stretch = source_fps * interpolate / out_fps != 1`, the tensor
presentation stage must RETIME (setpts — keep every FILM frame, stretch the
timeline), never decimate (a bare `fps=` filter drops frames to hold the
duration). The deferred/stretched music mix and the SFX bounds follow the
same factor; without an explicit presentation fps everything stays on the
legacy timeline (byte-identical).

Always-deferred finalize is ledger-only: `build_final_audio` blends the
takes ledger and fails loud without it — the old joint-fallback concat,
`_atempo_stages`, and single-segment shortcut helpers were deleted, so
their pins live in `tests/test_deferred_audio.py` now.
"""

from __future__ import annotations

import pytest


def test_slowmo_video_active_needs_tensor_presentation_and_stretch() -> None:
    from voyage.media import slowmo_video_active

    assert slowmo_video_active(True, 32, 1.5) is True
    assert slowmo_video_active(False, 32, 1.5) is False
    assert slowmo_video_active(True, None, 1.5) is False
    assert slowmo_video_active(True, 32, 1.0) is False
    assert slowmo_video_active(True, 48, 1.0) is False


def test_tensor_presentation_vf_slowmo_prefixes_setpts() -> None:
    from voyage.media import tensor_presentation_vf

    vf = tensor_presentation_vf(1216, 704, 32, 1.5, slowmo=True)
    assert vf.startswith("setpts=1.5*PTS,")
    assert "fps=32" in vf
    assert "scale=1216:704" in vf


def test_tensor_presentation_vf_legacy_has_no_setpts() -> None:
    from voyage.media import tensor_presentation_vf

    vf = tensor_presentation_vf(1216, 704, 24, 4.0, slowmo=False)
    assert "setpts" not in vf
    assert vf == (
        "scale=1216:704:force_original_aspect_ratio=decrease,"
        "pad=1216:704:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24"
    )


def test_scale_bounds_to_timeline_identity_within_tolerance() -> None:
    from voyage.sfx_finalize import _scale_bounds_to_timeline

    bounds = [(0.0, 4.0, "a"), (4.0, 8.0, "b")]
    assert _scale_bounds_to_timeline(bounds, 8.0) == bounds


def test_scale_bounds_to_timeline_stretches_uniformly() -> None:
    from voyage.sfx_finalize import _scale_bounds_to_timeline

    bounds = [(0.0, 4.0, "a"), (4.0, 8.0, "b")]
    scaled = _scale_bounds_to_timeline(bounds, 12.0)
    assert scaled == [(0.0, 6.0, "a"), (6.0, 12.0, "b")]


def test_presentation_stretch_matches_slowmo_factor() -> None:
    from voyage.media import presentation_stretch

    assert presentation_stretch(24.0, 2, 32) == 1.5
    assert presentation_stretch(24.0, 4, 24) == 4.0
    assert presentation_stretch(16.0, 1, 24) == 16.0 / 24


def test_presentation_stretch_unprobable_source_is_no_stretch() -> None:
    """An unprobable first segment (fps 0.0) must not raise.

    `finalize_run` used to call `slowmo_factor` unconditionally, which
    rejects non-positive fps — a finalize that previously proceeded down
    the re-encode path would crash with an unrelated ValueError. The
    tensor path can never arm on fps 0.0, so 1.0 is exactly right.
    """
    from voyage.media import presentation_stretch

    assert presentation_stretch(0.0, 2, 32) == 1.0
    assert presentation_stretch(-1.0, 4, 24) == 1.0


def test_scale_bounds_to_timeline_rejects_empty() -> None:
    from voyage.errors import MediaError
    from voyage.sfx_finalize import _scale_bounds_to_timeline

    with pytest.raises(MediaError):
        _scale_bounds_to_timeline([], 12.0)


def test_tensor_path_armed_needs_every_leg() -> None:
    from typing import Any

    from voyage.media import tensor_path_armed

    full: dict[str, Any] = {
        "model_selected": True,
        "weights_present": True,
        "source_fps": 24.0,
        "needs_reencode": True,
        "devices_available": True,
    }
    assert tensor_path_armed(**full) is True
    assert tensor_path_armed(**{**full, "model_selected": False}) is False
    assert tensor_path_armed(**{**full, "weights_present": False}) is False
    assert tensor_path_armed(**{**full, "source_fps": 0.0}) is False
    assert tensor_path_armed(**{**full, "needs_reencode": False}) is False
    assert tensor_path_armed(**{**full, "devices_available": False}) is False
