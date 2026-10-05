"""Issue 083 TDD: unify media/augment finalize surface (failing-first).

Pins the single-home contracts before the implementation lands:
- `voyage.media.interpolated_frame_count` is the shared augment
  frame-count helper (single plan-math home, no TODO/duplication).
- CRF bounds agree across the two modules (one codec ladder).
- Scalar-vs-`options=` settings resolution agrees (one contract).
"""

from __future__ import annotations


def test_interpolated_frame_count_single_home() -> None:
    import voyage.augment as augment
    import voyage.media as media

    assert callable(media.interpolated_frame_count)
    assert media.interpolated_frame_count is augment.interpolated_frame_count
    assert media.interpolated_frame_count(5, 4) == 17


def test_crf_bounds_agree() -> None:
    import voyage.augment as augment
    import voyage.media as media

    assert (media.FINALIZE_CRF_MINIMUM, media.FINALIZE_CRF_MAXIMUM) == (
        augment.CRF_MINIMUM,
        augment.CRF_MAXIMUM,
    )


def test_resolve_finalize_settings_scalar_options_parity() -> None:
    import voyage.media as media

    scalar = media.resolve_finalize_settings(
        options=None,
        skip_bad=False,
        sample_rate=48000,
        channels=2,
        overlap_fraction=0.10,
        overlap_cap_seconds=0.5,
        upscale=None,
        interpolate=None,
        presentation_fps=None,
        crf=None,
        preset=None,
    )
    explicit = media.resolve_finalize_settings(
        options=media.FinalizeOptions(),
        skip_bad=False,
        sample_rate=48000,
        channels=2,
        overlap_fraction=0.10,
        overlap_cap_seconds=0.5,
        upscale=None,
        interpolate=None,
        presentation_fps=None,
        crf=None,
        preset=None,
    )
    assert scalar == explicit
