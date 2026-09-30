"""Issue 031: finalize concat-copy fast path + take-slice cache.

Native-geometry runs stream-copy committed videos (zero video re-encodes);
the slice cache memos identical (take, start, duration) windows across the
overlap blend. Real ffmpeg throughout (slim image ships it); segments come
from the fake-backend commit path like the existing finalize tests.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import load_config
from voyage.media import (
    _cached_slice_take,
    _segment_video_matches_target,
    _slice_cache_key,
    build_final_audio,
    finalize_run,
    probe,
    validate_video,
)
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, style: str = "pastel neon line-art, peaceful") -> None:
    initialize_run_directory(run_dir, run_id="fastpath", style=style, seed=7)


def _commit_two(run_dir: Path) -> None:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def _sine_take(dest: Path, seconds: float) -> Path:
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


def test_slice_cache_key_rounds_to_ffmpeg_precision() -> None:
    assert _slice_cache_key(Path("/takes/a.wav"), 1.0, 2.0) == _slice_cache_key(
        Path("/takes/a.wav"), 1.0, 2.0
    )
    # Float dust below .6f precision shares the key (same ffmpeg invocation).
    assert _slice_cache_key(Path("/takes/a.wav"), 1.0000004, 2.0) == _slice_cache_key(
        Path("/takes/a.wav"), 1.0, 2.0
    )
    assert _slice_cache_key(Path("/takes/a.wav"), 1.0, 2.0) != _slice_cache_key(
        Path("/takes/a.wav"), 1.0, 2.5
    )
    assert _slice_cache_key(Path("/takes/a.wav"), 1.0, 2.0) != _slice_cache_key(
        Path("/takes/b.wav"), 1.0, 2.0
    )


def test_cached_slice_take_runs_ffmpeg_once_for_identical_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    take = _sine_take(tmp_path / "take.wav", seconds=10.0)
    spawns: list[tuple[float, float]] = []
    import voyage.media as media_module

    real_slice_take = media_module.slice_take

    def _counting(
        take_path: Path,
        start_seconds: float,
        duration_seconds: float,
        dest: Path,
        sample_rate: int,
        channels: int,
    ) -> Path:
        spawns.append((start_seconds, duration_seconds))
        return real_slice_take(
            take_path, start_seconds, duration_seconds, dest, sample_rate, channels
        )

    monkeypatch.setattr(media_module, "slice_take", _counting)
    cache: dict[tuple[str, str, str], Path] = {}
    first = _cached_slice_take(cache, take, 4.0, 2.0, tmp_path / "w0.wav", 48000, 2)
    second = _cached_slice_take(cache, take, 4.0, 2.0, tmp_path / "w1.wav", 48000, 2)
    assert spawns == [(4.0, 2.0)]
    assert first.read_bytes() == second.read_bytes()

    third = _cached_slice_take(cache, take, 5.0, 2.0, tmp_path / "w2.wav", 48000, 2)
    assert len(spawns) == 2
    assert third.exists() and third.stat().st_size > 0


def test_segment_video_matches_native_fake_segments(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    first = paths.segment_dir(run_dir, "000000")
    assert _segment_video_matches_target(first, 768, 432, 24)
    assert not _segment_video_matches_target(first, 1280, 704, 24)
    assert not _segment_video_matches_target(first, 768, 432, 16)
    assert not _segment_video_matches_target(tmp_path / "no-such-segment", 768, 432, 24)


def test_finalize_native_geometry_validates_without_reencode(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    out = tmp_path / "final-copy.mp4"
    # Legacy native path: floors disabled so 768x432@24 fake segments
    # stream-copy (default floors would lift to 1280x720@32).
    assert finalize_run(run_dir, out, min_fps=0, min_width=0, min_height=0).exists()
    probed = validate_video(out, 768, 432, 24)
    assert probed["fps"] == pytest.approx(24.0, abs=0.5)
    duration = float(probe(out).get("format", {}).get("duration", 0.0))
    # 2 fake segments x 48f @24fps = 4.0s of content, carried losslessly.
    assert duration == pytest.approx(4.0, abs=0.15)


def test_build_final_audio_slice_cache_wired(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    usable = [paths.segment_dir(run_dir, "000000"), paths.segment_dir(run_dir, "000001")]
    dest = build_final_audio(run_dir, usable, tmp_path, 24, 48000, 2)
    assert dest.exists() and dest.stat().st_size > 0
    metrics = [
        json.loads(line)
        for line in (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert any(record.get("event") == "segment_committed" for record in metrics)


def test_fastpath_skips_part_reencodes_but_keeps_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy path still blends audio and muxes AAC (video copy only)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    calls: list[Any] = []
    import voyage.media as media_module

    real_run_capture = media_module.run_capture

    def _recording(argv: list[str]) -> Any:
        calls.append(argv)
        return real_run_capture(argv)

    monkeypatch.setattr(media_module, "run_capture", _recording)
    out = tmp_path / "final-rec.mp4"
    assert finalize_run(run_dir, out, min_fps=0, min_width=0, min_height=0).exists()
    video_encodes = [argv for argv in calls if "-c:v" in argv and "libx264" in argv]
    assert video_encodes == []
    copy_calls = [argv for argv in calls if "-c:v" in argv and "copy" in argv]
    assert len(copy_calls) == 1
    audio = next(
        s
        for s in probe(out).get("streams", [])
        if isinstance(s, dict) and s.get("codec_type") == "audio"
    )
    assert audio.get("codec_name") == "aac"
