"""Unit tests for the Phase 4 audio slow loop (§35/§40): planner decisions,
ledger persistence, and take slice/assembly (real ffmpeg, synthetic tones).
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voyage.audio.planner import (
    AudioPlanner,
    AudioTake,
    append_take,
    load_takes,
)
from voyage.media import assemble_segment_audio, probe, slice_take


def _planner() -> AudioPlanner:
    return AudioPlanner(take_seconds=45.0, ahead_seconds=20.0)


def _recorded(planner: AudioPlanner, covers_from: float = 0.0) -> AudioTake:
    take = AudioTake(
        take_id="take_0000",
        path="/takes/take_0000.wav",
        caption="ambient drift",
        seed=7,
        covers_from=covers_from,
        duration=45.0,
        segment_index=0,
    )
    planner.record(take)
    return take


def test_empty_plan_renders_from_video_time() -> None:
    decision = _planner().plan(video_time=0.0, caption="ambient drift", seed=7, segment_index=0)
    assert decision.action == "render"
    assert decision.take is not None
    assert decision.take.covers_from == 0.0
    assert decision.take.duration == 45.0


def test_keep_when_covered_with_margin() -> None:
    planner = _planner()
    current = _recorded(planner)
    decision = planner.plan(video_time=4.0, caption="ambient drift", seed=7, segment_index=2)
    assert decision.action == "keep"
    assert decision.current == current


def test_chained_render_inside_ahead_window() -> None:
    planner = _planner()
    _recorded(planner)
    decision = planner.plan(video_time=30.0, caption="ambient drift", seed=7, segment_index=15)
    assert decision.action == "render"
    assert decision.take is not None
    assert decision.take.covers_from == 45.0  # chained at coverage end


def test_chain_renders_once_per_joint() -> None:
    """Chaining keys on max coverage, not the present take (kaolin incident).

    A chained take anchors in the future, so every later segment inside the
    same ahead window must keep — checking only the take covering now
    chained a duplicate per segment (~4 takes per 48s joint on kaolin).
    """
    planner = AudioPlanner(take_seconds=48.0, ahead_seconds=20.0)
    first = AudioTake(
        take_id="take_0000",
        path="/takes/take_0000.wav",
        caption="ambient drift",
        seed=7,
        covers_from=0.0,
        duration=48.0,
        segment_index=0,
    )
    planner.record(first)
    # 6s-apart cursors against a 0..48 take: 30.0 chains once at 48.0, and
    # every later cursor in the same ahead window keeps.
    chained = planner.plan(30.0, "ambient drift", 7, 5)
    assert chained.action == "render"
    assert chained.take is not None
    assert chained.take.covers_from == 48.0
    assert chained.take.duration == 48.0
    planner.record(chained.take)
    for cursor, segment in ((36.0, 6), (42.0, 7), (48.0, 8), (54.0, 9), (72.0, 12)):
        decision = planner.plan(cursor, "ambient drift", 7, segment)
        assert decision.action == "keep", f"duplicate chain at cursor {cursor}"
    # Next joint chains exactly once at max coverage end (96.0).
    second = planner.plan(78.0, "ambient drift", 7, 13)
    assert second.action == "render"
    assert second.take is not None
    assert second.take.covers_from == 96.0
    planner.record(second.take)
    assert planner.plan(84.0, "ambient drift", 7, 14).action == "keep"
    assert len(planner.takes) == 3


def test_chained_take_overlaps_tail_with_continuation() -> None:
    """Continuation: a chained take extends, not restarts, its predecessor.

    With a 6s chain overlap the chained take backdates its start by the
    overlap and carries the tail take as current, so the caller can build
    the continuation source from the tail's audio.
    """
    planner = AudioPlanner(take_seconds=48.0, ahead_seconds=20.0, chain_overlap_seconds=6.0)
    first = AudioTake(
        take_id="take_0000",
        path="/takes/take_0000.wav",
        caption="ambient drift",
        seed=7,
        covers_from=0.0,
        duration=48.0,
        segment_index=0,
    )
    planner.record(first)
    decision = planner.plan(30.0, "ambient drift", 7, 5)
    assert decision.action == "render"
    assert decision.take is not None
    assert decision.take.covers_from == 42.0
    assert decision.take.duration == 48.0
    assert decision.current is first
    planner.record(decision.take)
    # Later cursors in the same ahead window keep (no duplicates); the
    # next joint chains exactly once, overlapping again.
    for cursor, segment in ((36.0, 6), (42.0, 7), (48.0, 8), (60.0, 10)):
        assert planner.plan(cursor, "ambient drift", 7, segment).action == "keep"
    second = planner.plan(72.0, "ambient drift", 7, 12)
    assert second.action == "render"
    assert second.take is not None
    assert second.take.covers_from == 84.0
    assert second.current is decision.take


def test_chain_overlap_guards_reject_degenerate_values() -> None:
    with pytest.raises(ValueError, match="chain_overlap_seconds"):
        AudioPlanner(take_seconds=48.0, ahead_seconds=20.0, chain_overlap_seconds=48.0)
    with pytest.raises(ValueError, match="chain_overlap_seconds"):
        AudioPlanner(take_seconds=48.0, ahead_seconds=20.0, chain_overlap_seconds=-1.0)


def test_tail_take_prefers_furthest_coverage_then_newest() -> None:
    planner = AudioPlanner(take_seconds=48.0, ahead_seconds=20.0)
    assert planner.tail_take() is None
    original = AudioTake(
        take_id="take_0000",
        path="/takes/take_0000.wav",
        caption="ambient drift",
        seed=7,
        covers_from=0.0,
        duration=48.0,
        segment_index=0,
    )
    repaint = AudioTake(
        take_id="take_0001",
        path="/takes/take_0001.wav",
        caption="brighter pulse",
        seed=9,
        covers_from=0.0,
        duration=48.0,
        segment_index=5,
    )
    chained = AudioTake(
        take_id="take_0002",
        path="/takes/take_0002.wav",
        caption="ambient drift",
        seed=7,
        covers_from=48.0,
        duration=48.0,
        segment_index=8,
    )
    planner.record(original)
    assert planner.tail_take() is original
    # Same coverage end: the newer take (higher segment_index) wins — it
    # is the one the serving walk plays at the coverage end.
    planner.record(repaint)
    assert planner.tail_take() is repaint
    planner.record(chained)
    assert planner.tail_take() is chained


def test_repaint_on_caption_change_with_unconsumed_region() -> None:
    planner = _planner()
    current = _recorded(planner)
    decision = planner.plan(video_time=10.0, caption="brighter pulse", seed=9, segment_index=5)
    assert decision.action == "repaint"
    assert decision.current == current
    assert decision.take is not None
    assert decision.take.caption == "brighter pulse"
    # Repaint output is timeline-aligned with its source, so the new take
    # must keep the source anchor: slicing from `video_time` then lands in
    # the regenerated region instead of replaying the preserved head.
    assert decision.take.covers_from == current.covers_from


def test_repaint_take_slices_past_the_preserved_head() -> None:
    planner = _planner()
    _recorded(planner)
    decision = planner.plan(video_time=10.0, caption="brighter pulse", seed=9, segment_index=5)
    assert decision.take is not None
    planner.record(decision.take)
    serving = planner.take_for_time(10.0)
    assert serving is not None
    assert serving.take_id == decision.take.take_id
    assert 10.0 - serving.covers_from == 10.0  # regenerated region, not 0


def test_take_ids_increment() -> None:
    planner = _planner()
    first = planner.plan(0.0, "a", 1, 0).take
    assert first is not None and first.take_id == "take_0000"
    planner.record(first)
    second = planner.plan(50.0, "a", 2, 25).take
    assert second is not None and second.take_id == "take_0001"


def test_future_repaint_does_not_serve_earlier_segments() -> None:
    """Replay convergence: a repaint serves only from its own segment on.

    Repaints backdate coverage to the source anchor, so without the
    serving bound the newest repaint (a later segment's caption) would
    serve — and re-trigger repaints at — every earlier cursor, and each
    re-finalize would append more takes forever.
    """
    planner = _planner()
    original = AudioTake(
        take_id="take_0000",
        path="/takes/take_0000.wav",
        caption="ambient drift",
        seed=7,
        covers_from=0.0,
        duration=45.0,
        segment_index=0,
    )
    repaint = AudioTake(
        take_id="take_0001",
        path="/takes/take_0001.wav",
        caption="brighter pulse",
        seed=9,
        covers_from=0.0,
        duration=45.0,
        segment_index=5,
    )
    planner.record(original)
    planner.record(repaint)
    # Unbounded (legacy): the future repaint serves cursor 0.0.
    assert planner.take_for_time(0.0) is repaint
    # Bounded: segment 0 still hears the original take -> keep, not repaint.
    assert planner.take_for_time(0.0, 0) is original
    decision = planner.plan(video_time=0.0, caption="ambient drift", seed=7, segment_index=0)
    assert decision.action == "keep"
    assert decision.current is original
    # From segment 5 on the repaint legitimately serves.
    assert planner.take_for_time(10.0, 5) is repaint


def test_ledger_round_trip(tmp_path: Path) -> None:
    ledger = tmp_path / "takes.jsonl"
    take = AudioTake(
        take_id="take_0003",
        path="/takes/take_0003.wav",
        caption="night pulse",
        seed=11,
        covers_from=90.0,
        duration=45.0,
        segment_index=45,
    )
    append_take(ledger, take)
    loaded = load_takes(ledger)
    assert loaded == [take]
    assert loaded[0].covers_until() == 135.0
    assert load_takes(tmp_path / "missing.jsonl") == []


def _sine(dest: Path, seconds: float, frequency: int = 440) -> Path:
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
            f"sine=frequency={frequency}:duration={seconds}",
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


def _duration(path: Path) -> float:
    return float(probe(path).get("format", {}).get("duration", 0.0) or 0.0)


def test_slice_take_cuts_segment_window(tmp_path: Path) -> None:
    take = _sine(tmp_path / "take.wav", seconds=10.0)
    out = slice_take(take, 4.0, 2.0, tmp_path / "slice.wav", 48000, 2)
    assert abs(_duration(out) - 2.0) < 0.05


def test_assemble_single_slice_copies_through(tmp_path: Path) -> None:
    take = _sine(tmp_path / "take.wav", seconds=10.0)
    only = slice_take(take, 1.0, 2.0, tmp_path / "only.wav", 48000, 2)
    out = assemble_segment_audio([only], tmp_path / "seg.wav", 2.0)
    assert abs(_duration(out) - 2.0) < 0.05


def test_assemble_two_slices_crossfades(tmp_path: Path) -> None:
    first = _sine(tmp_path / "a.wav", seconds=5.0, frequency=440)
    second = _sine(tmp_path / "b.wav", seconds=5.0, frequency=660)
    out = assemble_segment_audio([first, second], tmp_path / "seg.wav", 2.0)
    assert abs(_duration(out) - (5.0 + 5.0 - 2.0)) < 0.1


def test_assemble_clamps_fade_to_half_shortest_slice(tmp_path: Path) -> None:
    first = _sine(tmp_path / "a.wav", seconds=3.0, frequency=440)
    second = _sine(tmp_path / "b.wav", seconds=3.0, frequency=660)
    out = assemble_segment_audio([first, second], tmp_path / "seg.wav", 2.0)
    assert abs(_duration(out) - (3.0 + 3.0 - 1.5)) < 0.1


def test_assemble_tiny_slices_fall_back_to_concat(tmp_path: Path) -> None:
    first = _sine(tmp_path / "a.wav", seconds=0.15, frequency=440)
    second = _sine(tmp_path / "b.wav", seconds=0.15, frequency=660)
    out = assemble_segment_audio([first, second], tmp_path / "seg.wav", 2.0)
    assert abs(_duration(out) - 0.30) < 0.05


def _impulse_take(dest: Path, seconds: float, impulse_sample: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((int(seconds * 48000), 2), dtype=np.int16)
    data[impulse_sample, :] = 30000
    with wave.open(str(dest), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(data.tobytes())
    return dest


def _read_samples(path: Path) -> np.ndarray[Any, Any]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).reshape(-1, 2)  # type: ignore[no-any-return]


def test_slice_take_keeps_sub_millisecond_timestamps(tmp_path: Path) -> None:
    # 29 frames at 24 fps = 1.208333...s. Formatting the seek to 3 decimals
    # truncated it to 1.208s (16 samples) and accumulated ~0.3 ms of A/V
    # drift per segment over an infinite run; the impulse must land at the
    # very start of the slice.
    take = _impulse_take(tmp_path / "take.wav", seconds=1.5, impulse_sample=58000)
    out = slice_take(take, 29 / 24, 0.1, tmp_path / "slice.wav", 48000, 2)
    samples = _read_samples(out)
    assert int(np.argmax(np.abs(samples[:, 0]))) <= 1
