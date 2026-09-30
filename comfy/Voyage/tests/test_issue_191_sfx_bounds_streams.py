"""Issue 191: `segment_sfx_bounds` must read the video stream, not `streams[0]`.

Two silent defects shift every downstream SFX caption: the duration fallback
reads whatever ffprobe lists first (routinely the audio stream on muxed
segments), and a total probe failure tiles `(cursor, cursor)` zero-bounds so
the cursor never advances. Both fail loud after the fix; `fps <= 0` is
rejected at both bounds functions instead of splitting ZeroDivisionError vs
silent-zero.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import paths
from voyage.errors import MediaError


def _stub_probe(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> None:
    import voyage.sfx_finalize as sfx_module

    def _fake(path: Path) -> dict[str, object]:
        return dict(payload)

    monkeypatch.setattr(sfx_module, "probe", _fake)


def test_audio_first_listing_uses_video_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audio-first stream listing still yields the video stream's duration."""
    from voyage.sfx_finalize import segment_sfx_bounds

    segment = tmp_path / "segments" / "000000"
    segment.mkdir(parents=True)
    _stub_probe(
        monkeypatch,
        {
            "format": {"duration": 0},
            "streams": [
                {"codec_type": "audio", "nb_frames": "300"},
                {"codec_type": "video", "nb_frames": "96"},
            ],
        },
    )
    (bounds,) = segment_sfx_bounds(tmp_path, [segment], 24)
    assert bounds[0] == pytest.approx(0.0)
    assert bounds[1] == pytest.approx(96.0 / 24.0)


def test_all_unprobable_segment_raises_instead_of_tiling_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No container duration and no video frame count is a loud MediaError."""
    import voyage.sfx_finalize as sfx_module
    from voyage.sfx_finalize import segment_sfx_bounds

    segment = tmp_path / "segments" / "000000"
    segment.mkdir(parents=True)

    def _boom(path: Path) -> dict[str, object]:
        raise MediaError("ffprobe failed")

    monkeypatch.setattr(sfx_module, "probe", _boom)
    with pytest.raises(MediaError):
        segment_sfx_bounds(tmp_path, [segment], 24)


def test_non_positive_fps_rejected_at_sfx_bounds(tmp_path: Path) -> None:
    """`fps <= 0` raises MediaError, never ZeroDivisionError or silent zero."""
    from voyage.sfx_finalize import segment_sfx_bounds

    segment = tmp_path / "segments" / "000000"
    segment.mkdir(parents=True)
    with pytest.raises(MediaError):
        segment_sfx_bounds(tmp_path, [segment], 0)


def test_non_positive_fps_rejected_at_segment_timeline(tmp_path: Path) -> None:
    """The music-path twin validates its divisor too (was ZeroDivisionError)."""
    from voyage.media import _segment_timeline
    from voyage.segment_manifest import write_segment_manifest

    segment = paths.segment_dir(tmp_path, "000000")
    segment.mkdir(parents=True, exist_ok=True)
    write_segment_manifest(
        segment,
        {
            "metrics": {"frames": 96},
            "transition": {},
            "prompt_plan": {},
            "audio_state": {},
            "world_state": {},
            "checksums": {},
        },
    )
    with pytest.raises(MediaError):
        _segment_timeline([segment], 0)
