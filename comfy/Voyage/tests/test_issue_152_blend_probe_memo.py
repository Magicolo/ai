"""Issue 152: blend folds must thread known durations (probe-memo) + time each blend.

The pairwise fold topology stays: `test_final_blend_scale` pins at most 2
audio inputs per ffmpeg call (a wide N-input acrossfade graph deadlocked the
scheduler on a live 31-segment run), so the single-graph join is recorded as
residual in the issue log. This module pins the ownable half: every blend
input is probed at most once per join, and each blend reports wall time.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from voyage import paths
from voyage.errors import MediaError


def _sine(dest: Path, seconds: float) -> Path:
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
    assert proc.returncode == 0, proc.stderr[-1000:]
    return dest


def test_blend_fade_formula_is_single_sourced(tmp_path: Path) -> None:
    """The fade helper matches `_blend_pair`'s clamp and fails loud on slivers."""
    from voyage.media import _blend_fade_seconds

    assert _blend_fade_seconds(8.0, 8.0, 1.0) == pytest.approx(1.0)
    assert _blend_fade_seconds(8.0, 1.0, 1.0) == pytest.approx(0.5)
    with pytest.raises(MediaError):
        _blend_fade_seconds(8.0, 8.0, 0.0)


def test_blend_pair_accepts_known_durations_without_probing(tmp_path: Path) -> None:
    """Threaded durations skip both ffprobe calls (fallback kept when absent)."""
    import voyage.media as media_module
    import voyage.media_audio as media_audio_module
    from voyage.media import _blend_pair

    first = _sine(tmp_path / "a.wav", 4.0)
    second = _sine(tmp_path / "b.wav", 4.0)
    dest = tmp_path / "out.wav"

    def _no_probe(path: Path) -> dict[str, object]:
        raise AssertionError(f"probe must not run for {path}")

    real_probe = media_module.probe
    media_audio_module.probe = _no_probe
    try:
        _blend_pair(first, second, dest, 1.0, first_seconds=4.0, second_seconds=4.0)
    finally:
        media_audio_module.probe = real_probe
    assert dest.exists() and dest.stat().st_size > 0


def test_blend_pair_records_wall_milliseconds(tmp_path: Path) -> None:
    """An out-list collects exactly one timing entry per blend call."""
    from voyage.media import _blend_pair

    first = _sine(tmp_path / "a.wav", 4.0)
    second = _sine(tmp_path / "b.wav", 4.0)
    timings: list[float] = []
    _blend_pair(first, second, tmp_path / "out.wav", 1.0, timing_ms=timings)
    assert len(timings) == 1
    assert timings[0] >= 0.0


def test_assemble_fold_probes_each_slice_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 slices join with 3 probes total, not 3 + 2 per blend."""
    import voyage.media as media_module
    import voyage.media_audio as media_audio_module
    from voyage.media import assemble_segment_audio

    slices = [_sine(tmp_path / f"s{i}.wav", 4.0) for i in range(3)]
    calls = 0
    real_probe = media_module.probe

    def _counting(path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return real_probe(path)

    monkeypatch.setattr(media_audio_module, "probe", _counting)
    timings: list[float] = []
    out = assemble_segment_audio(slices, tmp_path / "window.wav", 1.0, blend_timings=timings)
    assert out.exists()
    assert calls <= 3
    # Single-graph (issue 152): N slices join in one spawn, so one timing entry.
    assert len(timings) == 1


def _synthetic_music_run(run_dir: Path, segments: int) -> list[Path]:
    """Committed-segment dirs + one take covering the timeline (no supervisor)."""
    import json

    from voyage.segment_manifest import write_segment_manifest

    frames, fps = 96, 24
    timeline = segments * frames / fps
    take_path = run_dir / "audio" / "take_0000.wav"
    _sine(take_path, timeline + 1.0)
    ledger = run_dir / "audio" / "takes.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "take_id": "take_0000",
                "path": str(take_path),
                "caption": "test take",
                "seed": 0,
                "covers_from": 0.0,
                "duration": timeline + 1.0,
                "segment_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    usable: list[Path] = []
    for index in range(segments):
        segment = paths.segment_dir(run_dir, f"{index:06d}")
        segment.mkdir(parents=True, exist_ok=True)
        write_segment_manifest(
            segment,
            {
                "metrics": {"frames": frames},
                "transition": {},
                "prompt_plan": {},
                "audio_state": {},
                "world_state": {},
                "checksums": {},
            },
        )
        usable.append(segment)
    return usable


def test_build_final_audio_reports_per_blend_timings(tmp_path: Path) -> None:
    """3 windows blend with 1 timing entry (single-graph join, issue 152)."""
    from voyage.media import build_final_audio

    run_dir = tmp_path / "run"
    usable = _synthetic_music_run(run_dir, segments=3)
    timings: list[float] = []
    out = build_final_audio(run_dir, usable, tmp_path, 24, 48000, 2, blend_timings=timings)
    assert out.exists()
    assert len(timings) == 1
    assert all(entry >= 0.0 for entry in timings)


def test_build_final_audio_fold_probes_each_window_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 windows join with 3 window probes, not 2 per blend."""
    import voyage.media as media_module
    import voyage.media_audio as media_audio_module
    from voyage.media import build_final_audio

    run_dir = tmp_path / "run"
    usable = _synthetic_music_run(run_dir, segments=3)
    calls = 0
    real_probe = media_module.probe

    def _counting(path: Path) -> dict[str, object]:
        nonlocal calls
        if str(path).endswith(".wav") and "window" in str(path):
            calls += 1
        return real_probe(path)

    monkeypatch.setattr(media_audio_module, "probe", _counting)
    build_final_audio(run_dir, usable, tmp_path, 24, 48000, 2)
    assert calls <= 3


def _sfx_segment(run_dir: Path, seg_id: str, duration: float) -> None:
    """Committed segment with probed video + one SFX caption (no supervisor)."""
    import json
    import subprocess

    from voyage.director import deterministic_decision

    segment = run_dir / "segments" / seg_id
    segment.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=24:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(segment / "video.mp4"),
        ],
        check=True,
    )
    decision = deterministic_decision(0, "reef", "reef", "ESTABLISH", "pastel neon")
    dumped = decision.model_dump()
    dumped["audio"]["sfx_caption"] = "soft wind"
    (segment / "transition.json").write_text(json.dumps(dumped), encoding="utf-8")
    (segment / "DONE").write_text("", encoding="utf-8")


def test_render_sfx_bed_threads_stem_durations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4 stems join with 4 stem probes (not 2 per blend) + 3 timing entries."""
    import voyage.media as media_module
    import voyage.media_audio as media_audio_module
    from voyage.sfx_finalize import render_sfx_bed, segment_sfx_bounds

    run_dir = tmp_path / "run"
    for index in range(5):
        _sfx_segment(run_dir, f"{index:06d}", 5.0)
    usable = [run_dir / "segments" / f"{index:06d}" for index in range(5)]
    final_video = tmp_path / "final_video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=24:duration=25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(final_video),
        ],
        check=True,
    )
    bounds = segment_sfx_bounds(run_dir, usable, 24)
    stem_probes = 0
    real_probe = media_module.probe

    def _counting(path: Path) -> dict[str, object]:
        nonlocal stem_probes
        if run_dir / "audio" / "sfx" in path.parents:
            stem_probes += 1
        return real_probe(path)

    monkeypatch.setattr(media_audio_module, "probe", _counting)
    timings: list[float] = []
    bed = render_sfx_bed(
        run_dir,
        final_video,
        25.0,
        bounds,
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "large_44k_v2",
        11,
        48000,
        2,
        1,
        blend_timings=timings,
    )
    assert bed.exists()
    assert stem_probes <= 4
    # Single-graph (issue 152): N stems join in one spawn, so one timing entry.
    assert len(timings) == 1
    bed_seconds = float(real_probe(bed).get("format", {}).get("duration", 0.0))
    assert bed_seconds == pytest.approx(25.0, abs=0.6)
