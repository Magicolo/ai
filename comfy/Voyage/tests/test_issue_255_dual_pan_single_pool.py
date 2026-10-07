"""Issue 255: dual-pan SFX shares one worker pool across both tracks.

`render_sfx_bed(dual_pan=True)` used to build, start, and tear down a full
MMAudio worker pool per track (2x model-load latency + H2D traffic for one
model rendering two seed streams). Both tracks now render through a single
pool built once in `render_sfx_bed`.

Fake backend + real ffmpeg join (same shape as `test_sfx_dual_pan`).
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any

import pytest

from voyage.sfx_finalize import (
    SFX_DUAL_SEED_OFFSET,
    SFX_RIGHT_LEDGER_NAME,
    SFX_RIGHT_STEM_SUFFIX,
    load_sfx_ledger,
)


class _CountingSfxWorker:
    """Fake worker rendering constant DC stems while counting pool use."""

    starts: list[int] = []
    stops: list[int] = []
    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def start(self) -> None:
        type(self).starts.append(id(self))

    def stop(self) -> None:
        type(self).stops.append(id(self))

    def call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        del op
        type(self).calls.append({"worker": id(self), **dict(payload)})
        out = Path(str(payload["output_path"]))
        rate = int(payload["sample_rate"])
        channels = int(payload["channels"])
        frames = max(1, int(float(payload["duration_seconds"]) * rate))
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(struct.pack(f"<{frames * channels}h", *([1000] * frames * channels)))
        return {"sfx": {"duration_seconds": float(payload["duration_seconds"])}}


def _render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dual_pan: bool) -> tuple[Path, Path]:
    from voyage.sfx_finalize import render_sfx_bed

    _CountingSfxWorker.starts.clear()
    _CountingSfxWorker.stops.clear()
    _CountingSfxWorker.calls.clear()
    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _CountingSfxWorker)
    run_dir = tmp_path / "run"
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    bed = render_sfx_bed(
        run_dir,
        final_video,
        12.0,
        [(0.0, 12.0, "rain")],
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "small_44k",
        11,
        48000,
        2,
        1,
        dual_pan=dual_pan,
    )
    return bed, run_dir


def test_dual_pan_starts_one_pool_for_both_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One start/stop serves both tracks; every call routes to that worker."""
    bed, run_dir = _render(tmp_path, monkeypatch, dual_pan=True)
    assert bed.exists()
    assert len(_CountingSfxWorker.starts) == 1
    assert len(_CountingSfxWorker.stops) == 1
    worker_ids = {call["worker"] for call in _CountingSfxWorker.calls}
    assert worker_ids == {_CountingSfxWorker.starts[0]}
    rendered = [str(call["output_path"]) for call in _CountingSfxWorker.calls]
    assert any(SFX_RIGHT_STEM_SUFFIX not in path for path in rendered)
    assert any(SFX_RIGHT_STEM_SUFFIX in path for path in rendered)
    left_records = load_sfx_ledger(run_dir / "audio" / "sfx" / "sfx.jsonl")
    right_records = load_sfx_ledger(run_dir / "audio" / "sfx" / SFX_RIGHT_LEDGER_NAME)
    assert len(left_records) == len(right_records) == 2
    assert [record["seed"] for record in right_records] == [
        record["seed"] + SFX_DUAL_SEED_OFFSET for record in left_records
    ]


def test_single_track_pool_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-track renders still build exactly one pool and stop it."""
    bed, _run_dir = _render(tmp_path, monkeypatch, dual_pan=False)
    assert bed.exists()
    assert len(_CountingSfxWorker.starts) == 1
    assert len(_CountingSfxWorker.stops) == 1
