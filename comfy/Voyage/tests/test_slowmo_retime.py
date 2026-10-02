"""Slow-mo retiming: setpts video stretch + SFX bounds scaling (DESIGN §140).

When an explicit `--presentation-fps` combines with `interp_multiplier > 1`
so that `stretch = source_fps * multiplier / out_fps != 1`, the tensor
presentation stage must RETIME (setpts — keep every FILM frame, stretch the
timeline), never decimate (a bare `fps=` filter drops frames to hold the
duration). The deferred/stretched music mix and the SFX bounds follow the
same factor; without an explicit presentation fps everything stays on the
legacy timeline (byte-identical).
"""

from __future__ import annotations

from pathlib import Path

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


def _sine_wav(dest: Path, seconds: float) -> Path:
    """Real ffmpeg sine tone (joint-backend segment audio stand-in)."""
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"sine render failed: {proc.stderr[-500:]}"
    return dest


def _joint_segment(parent: Path, name: str, *, frames: int, fps: int = 24) -> Path:
    """Committed joint-backend segment: sine audio.wav + manifest frames.

    No takes ledger is written on purpose — joint backends (ltx25/ltx23)
    commit the worker's audio directly, so `build_final_audio` takes the
    fallback concat path for them.
    """
    import json

    segment = parent / name
    segment.mkdir(parents=True, exist_ok=True)
    _sine_wav(segment / "audio.wav", frames / fps)
    manifest = {
        "format": 1,
        "transition": {},
        "prompt_plan": {},
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": frames},
        "checksums": {},
    }
    (segment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (segment / "DONE").write_text("done\n", encoding="utf-8")
    return segment


def _probed_audio_seconds(path: Path) -> float:
    import subprocess

    proc = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"ffprobe failed: {proc.stderr[-500:]}"
    return float(proc.stdout.strip())


def test_build_final_audio_joint_single_segment_stretches(tmp_path: Path) -> None:
    """Joint single segment + slow-mo: the mix must cover the stretched video.

    ltx25/ltx23 commit joint audio (no takes ledger), so the old shortcut
    shipped the 1x preview under a 1.5x video and `-shortest` trimmed
    frames. The stretched copy must land at ~1.5x duration.
    """
    from voyage.media_audio import build_final_audio

    run_dir = tmp_path
    segment = _joint_segment(run_dir / "segments", "000000", frames=96)
    out = build_final_audio(run_dir, [segment], run_dir / "tmp", 24, 48000, 2, stretch=1.5)
    assert abs(_probed_audio_seconds(out) - 6.0) < 0.3


def test_build_final_audio_joint_fallback_concat_stretches(tmp_path: Path) -> None:
    """Joint multi-segment fallback concat + slow-mo: stretched, not short."""
    from voyage.media_audio import build_final_audio

    run_dir = tmp_path
    segments = [
        _joint_segment(run_dir / "segments", f"{index:06d}", frames=96) for index in range(2)
    ]
    out = build_final_audio(run_dir, segments, run_dir / "tmp", 24, 48000, 2, stretch=1.5)
    assert abs(_probed_audio_seconds(out) - 12.0) < 0.3


def test_build_final_audio_joint_fallback_no_stretch_is_plain(tmp_path: Path) -> None:
    """stretch=1.0 keeps the legacy sample-exact concat (byte behavior)."""
    from voyage.media_audio import build_final_audio

    run_dir = tmp_path
    segments = [
        _joint_segment(run_dir / "segments", f"{index:06d}", frames=96) for index in range(2)
    ]
    out = build_final_audio(run_dir, segments, run_dir / "tmp", 24, 48000, 2, stretch=1.0)
    assert abs(_probed_audio_seconds(out) - 8.0) < 0.1


def test_atempo_stages_keep_every_stage_in_range() -> None:
    """ffmpeg atempo accepts one stage in [0.5, 2.0] — split outside that."""
    from voyage.media_audio import _atempo_stages

    assert _atempo_stages(2.0 / 3.0) == [pytest.approx(2.0 / 3.0)]
    assert _atempo_stages(1.0) == [1.0]
    assert _atempo_stages(0.25) == [0.5, 0.5]
    assert _atempo_stages(2.0) == [2.0]
    assert _atempo_stages(4.0) == [2.0, 2.0]
