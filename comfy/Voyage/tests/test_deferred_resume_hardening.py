"""Deferred take resume hardening (DESIGN §140).

Why: finalize replays director decisions and renders ACE takes — a crash
between operations the old tests never simulated strands the next finalize
(ledger says `keep` but the file is gone; a rendered file has no ledger
line; a torn ledger tail breaks the load fatally; a stale continuation
source leaks). These tests pin the four resume gaps closed in
`voyage/audio_finalize.py` + `voyage/audio/planner.py` (output-truth,
orphan adoption, source prune, torn-tail tolerance) with stdlib sine
renders — no GPU, no workers.
"""

from __future__ import annotations

import array
import json
import math
import wave
from pathlib import Path
from typing import Any

import pytest

from voyage.audio.planner import AudioTake, append_take, load_takes
from voyage.audio_finalize import deferred_render_pending, ensure_deferred_takes
from voyage.errors import StateError


def _write_sine_audio_file(path: Path, duration_seconds: float) -> None:
    """Stdlib sine stub: 48 kHz stereo s16le, exact duration (one write call)."""
    rate: int = 48000
    channels: int = 2
    frequency: float = 440.0
    frame_count: int = int(duration_seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono_samples = array.array(
        "h",
        (
            int(12000.0 * math.sin(2.0 * math.pi * frequency * index / rate))
            for index in range(frame_count)
        ),
    )
    stereo_samples = array.array(
        "h", (sample for value in mono_samples for sample in (value, value))
    )
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(channels)
        audio_file.setsampwidth(2)
        audio_file.setframerate(rate)
        audio_file.writeframes(stereo_samples.tobytes())


def _stub_render(payload: dict[str, Any], output_path: Path) -> None:
    """Render seam: stdlib sine at the payload duration (no GPU)."""
    duration_value: Any = payload["duration_seconds"]
    assert isinstance(duration_value, (int, float))
    _write_sine_audio_file(output_path, float(duration_value))


def _production_segment(
    run_dir: Path, index: int, frames: int, caption: str, energy: float = 0.4
) -> Path:
    """Production manifest shape: decision flattened at `transition`."""
    segment: Path = run_dir / "segments" / f"{index:06d}"
    segment.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "format": 1,
        "transition": {
            "decision_index": index,
            "audio": {"music_caption": caption, "energy": energy},
        },
        "metrics": {"frames": frames},
    }
    (segment / "manifest.json").write_text(json.dumps(manifest))
    return segment


def _small_run(run_dir: Path, caption: str = "dark ambient drone") -> tuple[Path, float, float]:
    """One 1s segment + fast take sizing (5s takes, 2s ahead) for quick files."""
    segment: Path = _production_segment(run_dir, 0, 24, caption)
    return segment, 5.0, 2.0


def _small_sizing(take_seconds: float, ahead_seconds: float) -> dict[str, Any]:
    """Fast sizing dict (abutting chain — no continuation overlap for tiny takes)."""
    return {
        "take_seconds": take_seconds,
        "ahead_seconds": ahead_seconds,
        "chain_overlap_seconds": 0.0,
    }


def test_ledger_hit_but_file_deleted_rerenders(tmp_path: Path) -> None:
    """A `keep` with a deleted take file re-renders instead of nooping."""
    segment, take_seconds, ahead_seconds = _small_run(tmp_path)
    sizing: dict[str, Any] = _small_sizing(take_seconds, ahead_seconds)
    first: list[dict[str, Any]] = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
        **sizing,
    )
    assert len(first) == 1
    assert not deferred_render_pending(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        **sizing,
    )
    ledger: Path = tmp_path / "audio" / "takes.jsonl"
    ledger_lines_before: list[str] = ledger.read_text().strip().splitlines()
    take_file: Path = tmp_path / first[0]["path"]
    assert take_file.is_file()
    take_file.unlink()
    assert deferred_render_pending(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        **sizing,
    )
    render_calls: list[str] = []

    def _counting_render(payload: dict[str, Any], output_path: Path) -> None:
        render_calls.append(str(output_path))
        _stub_render(payload, output_path)

    second: list[dict[str, Any]] = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_counting_render,
        **sizing,
    )
    assert len(render_calls) >= 1
    assert take_file.is_file() and take_file.stat().st_size > 0
    ledger_lines_after: list[str] = ledger.read_text().strip().splitlines()
    assert len(ledger_lines_after) == len(ledger_lines_before)
    assert not deferred_render_pending(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        **sizing,
    )
    assert len(second) >= 1


def test_empty_take_file_rerenders(tmp_path: Path) -> None:
    """A zero-byte take file counts as incomplete and re-renders."""
    segment, take_seconds, ahead_seconds = _small_run(tmp_path)
    sizing: dict[str, Any] = _small_sizing(take_seconds, ahead_seconds)
    first: list[dict[str, Any]] = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
        **sizing,
    )
    take_file: Path = tmp_path / first[0]["path"]
    take_file.write_bytes(b"")
    assert deferred_render_pending(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        **sizing,
    )
    ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
        **sizing,
    )
    assert take_file.stat().st_size > 0


def test_orphan_take_file_adopted_without_render(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crash between render and append leaves an adoptable file (no re-render)."""
    segment, take_seconds, ahead_seconds = _small_run(tmp_path)
    sizing: dict[str, Any] = _small_sizing(take_seconds, ahead_seconds)
    first: list[dict[str, Any]] = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
        **sizing,
    )
    take_file: Path = tmp_path / first[0]["path"]
    assert take_file.is_file()
    ledger: Path = tmp_path / "audio" / "takes.jsonl"
    ledger.unlink()
    assert not ledger.exists()

    def _forbidden_render(payload: dict[str, Any], output_path: Path) -> None:
        raise AssertionError("orphan file must be adopted, never re-rendered")

    adopted: list[dict[str, Any]] = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_forbidden_render,
        **sizing,
    )
    assert len(adopted) == 1
    assert adopted[0]["take_id"] == first[0]["take_id"]
    assert ledger.is_file()
    captured = capsys.readouterr()
    assert "orphan adopted" in captured.err
    assert not deferred_render_pending(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        **sizing,
    )


def test_orphan_with_wrong_duration_rerenders(tmp_path: Path) -> None:
    """An orphan file far from the planned duration never adopts (re-renders)."""
    segment, take_seconds, ahead_seconds = _small_run(tmp_path)
    sizing: dict[str, Any] = _small_sizing(take_seconds, ahead_seconds)
    audio_dir: Path = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wrong_file: Path = audio_dir / "take_0000.wav"
    _write_sine_audio_file(wrong_file, 1.0)
    render_calls: list[str] = []

    def _counting_render(payload: dict[str, Any], output_path: Path) -> None:
        render_calls.append(str(output_path))
        _stub_render(payload, output_path)

    ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_counting_render,
        **sizing,
    )
    assert len(render_calls) >= 1


def test_stale_continuation_sources_pruned(tmp_path: Path) -> None:
    """Stale `*_src` files prune at ensure entry and never block rendering."""
    segment, take_seconds, ahead_seconds = _small_run(tmp_path)
    audio_dir: Path = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    stale_source: Path = audio_dir / "take_0000_src.wav"
    stale_source.write_bytes(b"stale continuation bytes")
    assert stale_source.is_file()
    ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[segment],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
        take_seconds=take_seconds,
        ahead_seconds=ahead_seconds,
        chain_overlap_seconds=0.0,
    )
    assert not stale_source.exists()
    remaining_sources: list[Path] = list(audio_dir.glob("*_src*.wav"))
    assert remaining_sources == []


def test_torn_ledger_tail_skipped(tmp_path: Path) -> None:
    """A kill mid-append leaves a torn last line — skipped, not fatal."""
    ledger: Path = tmp_path / "takes.jsonl"
    take = AudioTake(
        take_id="take_0000",
        path="audio/take_0000.wav",
        caption="dark ambient drone",
        seed=7,
        covers_from=0.0,
        duration=5.0,
        segment_index=0,
    )
    append_take(ledger, take)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write('{"take_id": "take_00')
    loaded: list[AudioTake] = load_takes(ledger)
    assert loaded == [take]


def test_torn_middle_line_fails_loud(tmp_path: Path) -> None:
    """Only the tail may be torn — a torn middle line fails loud."""
    ledger: Path = tmp_path / "takes.jsonl"
    first_take = AudioTake(
        take_id="take_0000",
        path="audio/take_0000.wav",
        caption="dark ambient drone",
        seed=7,
        covers_from=0.0,
        duration=5.0,
        segment_index=0,
    )
    append_take(ledger, first_take)
    second_take = AudioTake(
        take_id="take_0001",
        path="audio/take_0001.wav",
        caption="dark ambient drone evolving",
        seed=8,
        covers_from=5.0,
        duration=5.0,
        segment_index=1,
    )
    append_take(ledger, second_take)
    lines: list[str] = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    ledger.write_text(lines[0] + "\n" + '{"take_id": "take_00' + "\n" + lines[1] + "\n")
    with pytest.raises(ValueError):
        load_takes(ledger)


def test_non_object_ledger_line_fails_loud(tmp_path: Path) -> None:
    """A structurally invalid JSON object (array) fails loud, never skipped."""
    ledger: Path = tmp_path / "takes.jsonl"
    take = AudioTake(
        take_id="take_0000",
        path="audio/take_0000.wav",
        caption="dark ambient drone",
        seed=7,
        covers_from=0.0,
        duration=5.0,
        segment_index=0,
    )
    append_take(ledger, take)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("[]\n")
    with pytest.raises(StateError):
        load_takes(ledger)
