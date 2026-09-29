"""Final-blend scale: `build_final_audio` must not hang on many segments.

Live incident (poulah, 31 segments): the final blend built ONE ffmpeg
invocation chaining 31 `acrossfade` filters. That graph deadlocks the
filter scheduler (futex wait, zero bytes out — stuck 3+ days), while a
2-input acrossfade completes in milliseconds. The blend must therefore
reduce pairwise (each ffmpeg call <= 2 audio inputs), no matter how many
segment windows feed it.
"""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from voyage import paths
from voyage.media import build_final_audio, probe

FRAMES_PER_SEGMENT = 96
FPS = 24
SEGMENT_SECONDS = FRAMES_PER_SEGMENT / FPS


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


def _synthetic_run(run_dir: Path, segments: int) -> list[Path]:
    """Fabricate committed-segment dirs + one take covering the timeline.

    `build_final_audio` only reads segment metrics.json (frame counts) and
    the takes ledger — no video.mp4 or supervisor commit needed.
    """
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
        (segment / "metrics.json").write_text(
            json.dumps({"frames": FRAMES_PER_SEGMENT}), encoding="utf-8"
        )
        usable.append(segment)
    return usable


def _input_count(argv: list[str]) -> int:
    return sum(1 for arg in argv if arg == "-i")


def test_final_blend_never_spawns_wide_acrossfade_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every ffmpeg call in the final blend takes at most 2 audio inputs.

    Fails on the old single-graph N-chain (records an 8-input call);
    passes on the pairwise reduction. Real ffmpeg still runs underneath
    (recording wrapper delegates), so this also exercises the audio path.
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
    wide = [argv for argv in calls if _input_count(argv) > 2]
    assert wide == [], f"{len(wide)} ffmpeg call(s) with >2 inputs (deadlock risk)"


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
