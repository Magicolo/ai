"""Issue 152: single-graph staged join pins (TDD failing-first).

The chained pairwise stages + s32 barrier construction must reproduce the
left-fold byte-for-byte in one spawn (N=31 proof in the issue log), with no
acrossfade anywhere (the deadlocked filter class).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest


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


def _reference_fold(stems: list[Path], workdir: Path, overlap: float) -> Path:
    """Production fold shape via `_blend_pair` (reference for parity)."""
    from voyage.media import _blend_fade_seconds, _blend_pair

    durations = [4.0] * len(stems)
    accum = stems[0]
    accum_seconds = durations[0]
    for index in range(1, len(stems)):
        step = workdir / f"reffold_{index:02d}.wav"
        _blend_pair(
            accum,
            stems[index],
            step,
            overlap,
            first_seconds=accum_seconds,
            second_seconds=durations[index],
        )
        accum_seconds += durations[index] - _blend_fade_seconds(
            accum_seconds, durations[index], overlap
        )
        accum = step
    return accum


def test_single_graph_matches_fold_byte_for_byte(tmp_path: Path) -> None:
    """One spawn replays the fold exactly (N=4, 4 s stems, 1 s overlap)."""
    from voyage.media import _join_audio_single_graph

    stems = [_sine(tmp_path / f"stem{i}.wav", 4.0) for i in range(4)]
    fold = _reference_fold(stems, tmp_path, 1.0)
    staged = _join_audio_single_graph(stems, tmp_path / "staged.wav", 1.0)
    assert staged.stat().st_size == fold.stat().st_size
    assert staged.read_bytes() == fold.read_bytes()


def test_single_graph_uses_one_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N=4 joins in exactly one ffmpeg spawn (O(N) I/O, not N-1)."""
    import voyage.media as media_module
    import voyage.media_audio as media_audio_module
    from voyage.media import _join_audio_single_graph

    stems = [_sine(tmp_path / f"stem{i}.wav", 4.0) for i in range(4)]
    calls: list[list[str]] = []
    real = media_module.run_capture

    def _recording(argv: list[str], timeout: float | None = 600.0) -> Any:
        calls.append(list(argv))
        return real(argv, timeout=timeout)

    monkeypatch.setattr(media_audio_module, "run_capture", _recording)
    _join_audio_single_graph(stems, tmp_path / "one.wav", 1.0)
    video_calls = [argv for argv in calls if argv and argv[0] == "ffmpeg"]
    assert len(video_calls) == 1
    assert sum(1 for arg in video_calls[0] if arg == "-i") == 4


def test_single_graph_has_s32_barriers_and_no_acrossfade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staged graph carries per-stage s32 barriers and never acrossfade."""
    import voyage.media as media_module
    import voyage.media_audio as media_audio_module
    from voyage.media import _join_audio_single_graph

    stems = [_sine(tmp_path / f"stem{i}.wav", 4.0) for i in range(4)]
    graphs: list[str] = []
    real = media_module.run_capture

    def _recording(argv: list[str], timeout: float | None = 600.0) -> Any:
        if "-filter_complex" in argv:
            graphs.append(argv[argv.index("-filter_complex") + 1])
        return real(argv, timeout=timeout)

    monkeypatch.setattr(media_audio_module, "run_capture", _recording)
    _join_audio_single_graph(stems, tmp_path / "bar.wav", 1.0)
    assert len(graphs) == 1
    graph = graphs[0]
    assert "acrossfade" not in graph
    assert graph.count("aformat=sample_fmts=s32") == 2
    assert graph.count("afade") == 6
    assert graph.count("amix=inputs=2") == 3
