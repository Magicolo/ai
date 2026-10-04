"""Group E1 (media/augment) resolutions: failing-first pins for 096/138/188/189/190/192/194.

Each test below failed against the pre-fix tree (or pins already-fixed
behavior as characterization) and passes after the E1 implementation.
All finalize legs run on fake backends (real ffmpeg, CPU-only, no GPU).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage.errors import MediaError
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _committed_run(run_dir: Path, count: int) -> None:
    initialize_run_directory(run_dir, run_id="e1")
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.run_segments(count)
    finally:
        supervisor.stop_workers()


def _stub_probe(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    import voyage.media_audio as media_audio_module

    def _fake_probe(_path: Path) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(media_audio_module, "probe", _fake_probe)


def _video_payload(
    rate: str, frames: str, duration: float, width: int = 1280, height: int = 720
) -> dict[str, Any]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "width": width,
                "height": height,
                "avg_frame_rate": rate,
                "nb_frames": frames,
            }
        ],
        "format": {"duration": duration},
    }


# Issue 096: validate_video fps/frames quirks.


def test_096_zero_over_zero_fps_raises_media_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`avg_frame_rate "0/0"` must be a MediaError, not a ZeroDivisionError."""
    from voyage.media import validate_video

    _stub_probe(monkeypatch, _video_payload("0/0", "29", 1.0))
    with pytest.raises(MediaError, match="fps"):
        validate_video(tmp_path / "clip.mp4", 1280, 720, 30)


def test_096_unparseable_fps_raises_media_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voyage.media import validate_video

    _stub_probe(monkeypatch, _video_payload("zzz", "29", 1.0))
    with pytest.raises(MediaError, match="fps"):
        validate_video(tmp_path / "clip.mp4", 1280, 720, 30)


def test_096_unknown_frame_count_short_duration_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`nb_frames == 0` with too little duration-estimate fails the gate."""
    from voyage.media import validate_video

    _stub_probe(monkeypatch, _video_payload("30/1", "0", 0.2))
    with pytest.raises(MediaError, match="too few frames"):
        validate_video(tmp_path / "clip.mp4", 1280, 720, 30, min_frames=29)


def test_096_unknown_frame_count_sufficient_duration_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`nb_frames == 0` with a duration covering min_frames is legitimate."""
    from voyage.media import validate_video

    _stub_probe(monkeypatch, _video_payload("30/1", "0", 2.0))
    info = validate_video(tmp_path / "clip.mp4", 1280, 720, 30, min_frames=29)
    assert info["duration"] == pytest.approx(2.0)


def test_096_na_frame_count_treated_as_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`nb_frames "N/A"` (mkv-style) takes the duration path, not ValueError."""
    from voyage.media import validate_video

    _stub_probe(monkeypatch, _video_payload("30/1", "N/A", 0.2))
    with pytest.raises(MediaError, match="too few frames"):
        validate_video(tmp_path / "clip.mp4", 1280, 720, 30, min_frames=29)


# Issues 138 + 188: skip_bad triage contract (single contract, both files).


def test_138_missing_audio_strict_raises(tmp_path: Path) -> None:
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    _committed_run(run_dir, 2)
    (run_dir / "segments" / "000001" / "audio.wav").unlink()
    with pytest.raises(MediaError, match="missing audio.wav"):
        finalize_run(run_dir, tmp_path / "strict.mp4")


@pytest.mark.slow
def test_138_missing_audio_lenient_skips_segment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    _committed_run(run_dir, 2)
    (run_dir / "segments" / "000001" / "audio.wav").unlink()
    out = finalize_run(run_dir, tmp_path / "lenient.mp4", skip_bad=True)
    assert out.exists()
    assert "skipping 000001" in capsys.readouterr().out


@pytest.mark.slow
def test_138_missing_video_lenient_skips_segment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    _committed_run(run_dir, 2)
    (run_dir / "segments" / "000001" / "video.mp4").unlink()
    out = finalize_run(run_dir, tmp_path / "lenient.mp4", skip_bad=True)
    assert out.exists()
    assert "skipping 000001" in capsys.readouterr().out


def test_188_numbering_gap_strict_raises(tmp_path: Path) -> None:
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    _committed_run(run_dir, 3)
    import shutil

    shutil.rmtree(run_dir / "segments" / "000001")
    with pytest.raises(MediaError, match="numbering gap"):
        finalize_run(run_dir, tmp_path / "strict.mp4")


@pytest.mark.slow
def test_188_numbering_gap_lenient_warns_and_finalizes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    _committed_run(run_dir, 3)
    import shutil

    shutil.rmtree(run_dir / "segments" / "000001")
    out = finalize_run(run_dir, tmp_path / "lenient.mp4", skip_bad=True)
    assert out.exists()
    assert "numbering gap" in capsys.readouterr().out


# Issue 189: single-slice audio verification.


def _completed_no_output(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 0, "", "")


def test_189_slice_take_rejects_empty_exit_zero_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exit-0 ffmpeg that writes no slice must fail at slice time."""
    import voyage.media as media_module
    import voyage.media_audio as media_audio_module

    monkeypatch.setattr(media_audio_module, "run_capture", _completed_no_output)
    take = tmp_path / "take.wav"
    take.write_bytes(b"fake-take")
    with pytest.raises(MediaError, match="empty"):
        media_module.slice_take(take, 0.0, 2.0, tmp_path / "slice.wav", 48000, 2)


def test_189_slice_take_accepts_real_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.media as media_module
    import voyage.media_audio as media_audio_module

    def _writes_slice(argv: list[str]) -> subprocess.CompletedProcess[str]:
        Path(argv[-1]).write_bytes(b"fake-slice-payload")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(media_audio_module, "run_capture", _writes_slice)
    take = tmp_path / "take.wav"
    take.write_bytes(b"fake-take")
    dest = media_module.slice_take(take, 0.0, 2.0, tmp_path / "slice.wav", 48000, 2)
    assert dest.exists()


def test_189_assemble_single_slice_rejects_empty_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.media as media_module
    import voyage.media_audio as media_audio_module

    monkeypatch.setattr(media_audio_module, "run_capture", _completed_no_output)
    lonely = tmp_path / "lonely.wav"
    lonely.write_bytes(b"fake-slice")
    with pytest.raises(MediaError, match="empty"):
        media_module.assemble_segment_audio([lonely], tmp_path / "seg.wav", 0.4)


# Issue 190: scalar-vs-options precedence (uniform scalar-wins).


def test_190_explicit_audio_scalars_win_over_options() -> None:
    from voyage.media import FinalizeOptions, resolve_finalize_settings

    options = FinalizeOptions(
        joint_style="blend", overlap_fraction=0.10, sample_rate=48000, channels=2
    )
    resolved = resolve_finalize_settings(
        options=options,
        skip_bad=None,
        sample_rate=44100,
        channels=1,
        overlap_fraction=0.0,
        overlap_cap_seconds=0.25,
        min_fps=None,
        min_width=None,
        min_height=None,
        crf=None,
        preset=None,
    )
    assert resolved.settings.sample_rate == 44100
    assert resolved.settings.channels == 1
    assert resolved.settings.overlap_fraction == 0.0
    assert resolved.settings.overlap_cap_seconds == 0.25
    assert resolved.settings.effective_overlap_fraction() == 0.0


def test_190_none_scalars_keep_options_values() -> None:
    from voyage.media import FinalizeOptions, resolve_finalize_settings

    options = FinalizeOptions(
        joint_style="blend", overlap_fraction=0.20, sample_rate=48000, channels=2
    )
    resolved = resolve_finalize_settings(
        options=options,
        skip_bad=None,
        sample_rate=None,
        channels=None,
        overlap_fraction=None,
        overlap_cap_seconds=None,
        min_fps=None,
        min_width=None,
        min_height=None,
        crf=None,
        preset=None,
    )
    assert resolved.settings.overlap_fraction == 0.20
    assert resolved.settings.sample_rate == 48000


def test_190_skip_bad_scalar_overrides_options() -> None:
    from voyage.media import FinalizeOptions, resolve_finalize_settings

    options = FinalizeOptions(skip_bad=True)
    resolved = resolve_finalize_settings(
        options=options,
        skip_bad=False,
        sample_rate=None,
        channels=None,
        overlap_fraction=None,
        overlap_cap_seconds=None,
        min_fps=None,
        min_width=None,
        min_height=None,
        crf=None,
        preset=None,
    )
    assert resolved.settings.skip_bad is False


# Issue 192: decode-chunk stale-dir hygiene (entry already guarded; pin it).


def test_192_preseeded_dest_dir_raises_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.augment as augment_module
    from voyage.augment import ffmpeg_decode_chunk

    def _must_not_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"ffmpeg must not spawn for a stale dir: {argv}")

    monkeypatch.setattr(augment_module, "run_capture", _must_not_run)
    dest = tmp_path / "chunk_00"
    dest.mkdir(parents=True)
    (dest / "frame_000000.png").write_bytes(b"stale")
    with pytest.raises(MediaError, match="not fresh"):
        ffmpeg_decode_chunk(tmp_path / "source.mp4", dest, 0, 1)


def test_192_decode_count_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Over/under-delivery past the glob must fail loud, not mix silently."""

    import voyage.augment as augment_module
    from voyage.augment import ffmpeg_decode_chunk

    def _short(argv: list[str]) -> subprocess.CompletedProcess[str]:
        dest_dir = Path(argv[-1]).parent
        for index in range(2):
            (dest_dir / f"frame_{index:06d}.png").write_bytes(b"fake-png-payload")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(augment_module, "run_capture", _short)
    with pytest.raises(MediaError, match="expected 4"):
        ffmpeg_decode_chunk(tmp_path / "source.mp4", tmp_path / "chunk_00", 0, 4)


# Issue 194: media-owned half — pure setup-facts helper for the §104 block.


def test_194_presentation_setup_facts_records_floors_and_plan() -> None:
    from voyage.media import plan_augmentation, presentation_setup_facts

    plan = plan_augmentation(768, 512, 24.0, 768, 512, 24, 32, 1280, 720)
    facts = presentation_setup_facts(plan, min_fps=32, min_width=1280, min_height=720)
    assert facts["min_fps"] == 32
    assert facts["min_width"] == 1280
    assert facts["min_height"] == 720
    assert facts["out_w"] == plan.out_w
    assert facts["out_h"] == plan.out_h
    assert facts["out_fps"] == plan.out_fps
    assert facts["needs_reencode"] == plan.needs_reencode
    assert facts["needs_minterpolate"] == plan.needs_minterpolate


def test_194_presentation_setup_facts_floors_off_round_trips() -> None:
    from voyage.media import plan_augmentation, presentation_setup_facts

    plan = plan_augmentation(768, 512, 24.0, 768, 512, 24, 0, 0, 0)
    facts = presentation_setup_facts(plan, min_fps=0, min_width=0, min_height=0)
    assert facts["min_fps"] == 0
    assert facts["out_fps"] == plan.out_fps
