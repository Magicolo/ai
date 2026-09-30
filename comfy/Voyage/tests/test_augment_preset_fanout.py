"""Augment chunk preset knob + warm-first fan-out gate (157 orchestration half).

Pure orchestration coverage — ffmpeg argv shape via a monkeypatched
`run_capture`, thread-ordering via events — no torch, no weights,
CPU-only. The worker half (resident caches, list-halving, unconditional
`empty_cache`) already landed; these tests pin the `augment.py` remainder.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

from voyage import augment
from voyage.augment import (
    AugmentChunk,
    interpolated_frame_count,
    run_augment_chunks,
)


def _capture_argv(monkeypatch: pytest.MonkeyPatch, seen: list[list[str]]) -> None:
    """Replace `run_capture` with an argv recorder that fakes a 1-byte mp4."""

    def fake_run_capture(argv: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        destination = Path(argv[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(augment, "run_capture", fake_run_capture)


def test_encode_chunk_defaults_to_veryfast_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """157 fix candidate 1: chunk encodes match the intermediates' preset
    unless overridden."""
    seen: list[list[str]] = []
    _capture_argv(monkeypatch, seen)
    destination = tmp_path / "chunk_00.mp4"
    assert augment.ffmpeg_encode_chunk(tmp_path / "frame_%06d.png", destination, 32) == destination
    assert "-preset" in seen[0]
    assert seen[0][seen[0].index("-preset") + 1] == "veryfast"


def test_encode_chunk_preset_override_reaches_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """157: an explicit preset threads through to the ffmpeg argv."""
    seen: list[list[str]] = []
    _capture_argv(monkeypatch, seen)
    destination = tmp_path / "chunk_00.mp4"
    augment.ffmpeg_encode_chunk(tmp_path / "frame_%06d.png", destination, 32, preset="ultrafast")
    assert seen[0][seen[0].index("-preset") + 1] == "ultrafast"


def test_encode_chunk_rejects_bad_preset(tmp_path: Path) -> None:
    """157: typos fail at validation time, not as an ffmpeg spawn error."""
    with pytest.raises(ValueError, match="preset must be one of"):
        augment.ffmpeg_encode_chunk(
            tmp_path / "frame_%06d.png", tmp_path / "chunk_00.mp4", 32, preset="ludicrous"
        )
    with pytest.raises(TypeError, match="preset must be a str"):
        augment.ffmpeg_encode_chunk(
            tmp_path / "frame_%06d.png",
            tmp_path / "chunk_00.mp4",
            32,
            preset=1,  # type: ignore[arg-type]
        )


def test_chunk_preset_vocabulary_mirrors_finalize() -> None:
    """157: the chunk allow-list stays identical to the finalize one
    (`media.FINALIZE_PRESETS`) — one vocabulary, two homes (kept apart
    only because `media` already imports `augment`, so reuse would cycle)."""
    from voyage.media import FINALIZE_PRESETS

    assert augment.CHUNK_PRESETS == FINALIZE_PRESETS
    assert augment.CHUNK_PRESET_DEFAULT == "veryfast"


def test_run_augment_chunks_warms_first_chunk_before_fanout() -> None:
    """157 orchestration gate: chunk 0 runs serially first so a resident
    model cache is populated before threads spawn — otherwise every
    thread misses at once and each pays a full load peak."""
    gate = threading.Event()
    release_at: list[float] = []
    entries: dict[int, float] = {}

    def worker(chunk: AugmentChunk, device: str) -> str:
        del device
        if chunk.index == 0:
            assert gate.wait(timeout=30)
            release_at.append(time.monotonic())
            return "first"
        entries[chunk.index] = time.monotonic()
        return f"chunk-{chunk.index}"

    def _release() -> None:
        time.sleep(0.5)
        gate.set()

    threading.Thread(target=_release, daemon=True).start()
    chunks = [
        AugmentChunk(
            index=index,
            start_frame=index * 8,
            source_frames=8,
            expected_frames=interpolated_frame_count(8, 2),
            device="cuda:0" if index % 2 == 0 else "cuda:1",
        )
        for index in range(4)
    ]
    outcomes = run_augment_chunks(chunks, worker)
    assert outcomes == ["first", "chunk-1", "chunk-2", "chunk-3"]
    assert sorted(entries) == [1, 2, 3]
    assert release_at != []
    assert all(entry >= release_at[0] for entry in entries.values())
