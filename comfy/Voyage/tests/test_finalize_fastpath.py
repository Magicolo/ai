"""Issue 031: finalize concat-copy fast path + take-slice cache.

Native-geometry runs stream-copy committed videos (zero video re-encodes);
the slice cache memos identical (take, start, duration) windows across the
overlap blend. Real ffmpeg throughout (slim image ships it); segments come
from the fake-backend commit path like the existing finalize tests.
"""

from __future__ import annotations

import concurrent.futures
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


@pytest.mark.slow
def test_segment_video_matches_native_fake_segments(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    first = paths.segment_dir(run_dir, "000000")
    assert _segment_video_matches_target(first, 768, 432, 24)
    assert not _segment_video_matches_target(first, 1280, 704, 24)
    assert not _segment_video_matches_target(first, 768, 432, 16)
    assert not _segment_video_matches_target(tmp_path / "no-such-segment", 768, 432, 24)


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


# --- 088 fold: tests/test_final_blend_scale.py (4 tests, verbatim) ---
# Original module docstring (banner, issue ID stays greppable):
# """Final-blend scale: `build_final_audio` must not hang on many segments.
#
# Live incident (poulah, 31 segments): the final blend built ONE ffmpeg
# invocation chaining 31 `acrossfade` filters. That graph deadlocks the
# filter scheduler (futex wait, zero bytes out — stuck 3+ days), while a
# 2-input acrossfade completes in milliseconds. The blend must therefore
# reduce pairwise (each ffmpeg call <= 2 audio inputs), no matter how many
# segment windows feed it.
# """
# NOTE: the source `_sine_take` is NOT duplicated here — it is
# byte-identical to the `_sine_take` above (verified via diff before the
# fold), so the moved tests reuse it with identical assertions.

FRAMES_PER_SEGMENT = 96
FPS = 24
SEGMENT_SECONDS = FRAMES_PER_SEGMENT / FPS


def _synthetic_run(run_dir: Path, segments: int) -> list[Path]:
    """Fabricate committed-segment dirs + one take covering the timeline.

    `build_final_audio` only reads segment manifest metrics (frame counts)
    and the takes ledger — no video.mp4 or supervisor commit needed.
    """
    from voyage.segment_manifest import write_segment_manifest

    timeline = segments * SEGMENT_SECONDS
    take_path = run_dir / "audio" / "take_0000.wav"
    _sine_take(take_path, timeline + 1.0)
    ledger = run_dir / "audio" / "takes.jsonl"
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
                "metrics": {"frames": FRAMES_PER_SEGMENT},
                "transition": {},
                "prompt_plan": {},
                "audio_state": {},
                "world_state": {},
                "checksums": {},
            },
        )
        usable.append(segment)
    return usable


def _input_count(argv: list[str]) -> int:
    return sum(1 for arg in argv if arg == "-i")


def test_final_blend_never_spawns_wide_acrossfade_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final blend never uses acrossfade; the wide join is manual only.

    Live incident (poulah, 31 segments): one ffmpeg invocation chaining 31
    `acrossfade` filters deadlocked the scheduler. The join is now a SINGLE
    wide manual graph (issue 152: chained pairwise `afade`/`adelay`/`amix`
    stages with `aformat=s32` barriers — byte-identical to the old fold,
    proven at N=31 CPU-only with no hang), so the pin forbids the
    `acrossfade` filter class instead of wide inputs. Real ffmpeg still runs
    underneath (recording wrapper delegates).
    """
    import voyage.media as media_module

    run_dir = tmp_path / "run"
    usable = _synthetic_run(run_dir, segments=8)
    calls: list[list[str]] = []
    real_run_capture = media_module.run_capture

    def _recording(argv: list[str]) -> Any:
        calls.append(list(argv))
        return real_run_capture(argv)

    monkeypatch.setattr(media_module, "run_capture", _recording)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(build_final_audio, run_dir, usable, tmp_path, FPS, 48000, 2)
        try:
            dest = future.result(timeout=180)
        except concurrent.futures.TimeoutError:
            pytest.fail("build_final_audio did not finish in 180s (wide-graph deadlock)")
    assert dest.exists() and dest.stat().st_size > 0
    assert not any("acrossfade" in arg for argv in calls for arg in argv), (
        "acrossfade filter must never appear (deadlock class)"
    )
    wide = [argv for argv in calls if _input_count(argv) > 2]
    assert len(wide) == 1, f"expected one single-graph join, saw {len(wide)} wide call(s)"
    assert _input_count(wide[0]) == 8
    graph = wide[0][wide[0].index("-filter_complex") + 1]
    assert "afade" in graph and "adelay" in graph and "amix=inputs=2" in graph
    assert "aformat=sample_fmts=s32" in graph
    assert "acrossfade" not in graph


def test_final_blend_output_matches_video_timeline(tmp_path: Path) -> None:
    """The pairwise blend stays timeline-exact (no A/V drift)."""
    run_dir = tmp_path / "run"
    usable = _synthetic_run(run_dir, segments=5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(build_final_audio, run_dir, usable, tmp_path, FPS, 48000, 2)
        try:
            dest = future.result(timeout=180)
        except concurrent.futures.TimeoutError:
            pytest.fail("build_final_audio did not finish in 180s (wide-graph deadlock)")
    duration = float(probe(dest).get("format", {}).get("duration", 0.0))
    assert duration == pytest.approx(5 * SEGMENT_SECONDS, abs=0.15)


def test_blend_pair_unequal_lengths(tmp_path: Path) -> None:
    """A long-first pair must keep both inputs (minus the overlap).

    Live incident (poulah window 21 + pairwise accum): ffmpeg acrossfade
    collapses when the FIRST input is much longer than the second —
    86s+1s at d=0.4 came out as ~0.2s instead of ~86.6s, poisoning every
    downstream blend. The pair helper must be exact for unequal lengths.
    """
    from voyage.media import _blend_pair

    long = _sine_take(tmp_path / "long.wav", 20.0)
    short = _sine_take(tmp_path / "short.wav", 1.0)
    dest = tmp_path / "blend.wav"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_blend_pair, long, short, dest, 0.4)
        try:
            out = future.result(timeout=120)
        except concurrent.futures.TimeoutError:
            pytest.fail("_blend_pair hung on unequal lengths")
    duration = float(probe(out).get("format", {}).get("duration", 0.0))
    assert duration == pytest.approx(20.0 + 1.0 - 0.4, abs=0.15)


def test_assemble_unequal_slices(tmp_path: Path) -> None:
    """Window assembly with a take joint near the window edge (poulah seg 21).

    Slices 3.49s + 0.91s at fade 0.4 must yield ~4.0s. The old inline
    acrossfade (long slice first) collapsed this to ~1.05s, and the short
    window then poisoned the whole final blend.
    """
    from voyage.media import assemble_segment_audio

    first = _sine_take(tmp_path / "s0.wav", 3.49)
    second = _sine_take(tmp_path / "s1.wav", 0.91)
    dest = tmp_path / "window.wav"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(assemble_segment_audio, [first, second], dest, 0.4)
        try:
            out = future.result(timeout=120)
        except concurrent.futures.TimeoutError:
            pytest.fail("assemble_segment_audio hung on unequal slices")
    duration = float(probe(out).get("format", {}).get("duration", 0.0))
    assert duration == pytest.approx(3.49 + 0.91 - 0.4, abs=0.15)
