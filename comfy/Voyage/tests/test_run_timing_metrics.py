"""Run-level wall-clock timing metrics (jango analysis follow-up).

Why this file exists: the jango 8.6h generation/finalization could only be
analyzed by mtime reconstruction — metrics.jsonl had per-segment stage
timings but no run bracket (worker boot, startup resume, inter-commit
gaps were invisible), no worker-side residual split, no background
model-pass presence, and no finalize stage timings. These tests pin the
new events: `run_started`/`run_finished`, per-worker `worker_started`
(boot + worker-reported load), `video_startup_resumed`,
`segment_committed.video_unaccounted_seconds`, `prewarm_progress`,
`finalize_completed.final_stages_s`, and per-take `ace_take_rendered`
(observer on the deferred-takes render, flushed next to
`finalize_completed`). All additive — existing keys are untouched.
"""

from __future__ import annotations

import array
import json
import math
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.audio_finalize import ensure_deferred_takes
from voyage.persistence import read_effective_config
from voyage.rpc import SubprocessWorker
from voyage.supervisor import Supervisor


def _metric_events(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                events.append(loaded)
    return events


def _of_kind(events: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return [entry for entry in events if entry.get("event") == event]


def test_run_started_finished_bracket_run_segments(tmp_path: Path) -> None:
    """run_segments emits an open/close bracket around the whole run."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="timing")
    config = read_effective_config(run_dir)
    assert config.video.backend == "fake"
    supervisor = Supervisor(run_dir, config)
    assert supervisor.run_segments(1) == ["000000"]

    events = _metric_events(run_dir)
    started = _of_kind(events, "run_started")
    finished = _of_kind(events, "run_finished")
    assert len(started) == 1
    assert len(finished) == 1
    # Open bracket precedes every commit; close bracket follows it.
    kinds = [entry["event"] for entry in events]
    assert kinds.index("run_started") == 0
    assert kinds.index("run_finished") == len(kinds) - 1
    assert started[0]["backend"] == "fake"
    assert started[0]["segments_planned"] == 1
    assert finished[0]["segments_committed"] == 1
    wall = finished[0]["wall_seconds"]
    assert isinstance(wall, (int, float)) and wall >= 0.0


def test_worker_started_reports_boot_and_load(tmp_path: Path) -> None:
    """Each worker start records supervisor boot wall + worker load time."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="timing")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
    finally:
        supervisor.stop_workers()

    boots = _of_kind(_metric_events(run_dir), "worker_started")
    assert sorted(entry["worker"] for entry in boots) == ["audio", "director", "video"]
    for entry in boots:
        boot = entry["boot_seconds"]
        assert isinstance(boot, (int, float)) and boot >= 0.0
        # Fake workers report no load_seconds — the key stays, value None,
        # so readers never branch on a missing field.
        assert "load_seconds" in entry
        load = entry["load_seconds"]
        assert load is None or (isinstance(load, (int, float)) and load >= 0.0)


def test_segment_committed_carries_video_unaccounted(tmp_path: Path) -> None:
    """The worker-side residual is explicit: 0 <= unaccounted <= video wall."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="timing")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
    finally:
        supervisor.stop_workers()

    committed = _of_kind(_metric_events(run_dir), "segment_committed")
    assert len(committed) == 1
    unaccounted = committed[0]["video_unaccounted_seconds"]
    assert isinstance(unaccounted, (int, float))
    assert 0.0 <= unaccounted <= float(committed[0]["stages"]["video"])


def test_prewarm_progress_metric_emits_ledger_delta(tmp_path: Path) -> None:
    """The console pre-warm note gains a metric twin with the same deltas."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="timing")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor._background = SimpleNamespace(
        ledgered_frames=lambda: (1, 12, 8, 12, 8, 4.5, 9.0),
        last_result=None,
    )
    supervisor._progress = SimpleNamespace(
        note=lambda *args, **kwargs: None,
        verbose=False,
    )
    supervisor._report_background_prewarm("000000")

    progress = _of_kind(_metric_events(run_dir), "prewarm_progress")
    assert len(progress) == 1
    event = progress[0]
    assert event["segment_id"] == "000000"
    assert event["upscale_frames"] == 12
    assert event["interp_frames"] == 8
    assert event["upscale_seconds"] == pytest.approx(4.5)
    assert event["interp_seconds"] == pytest.approx(9.0)


def test_finalize_completed_carries_final_stages(tmp_path: Path) -> None:
    """The console finalize table reaches metrics.jsonl as final_stages_s.

    `no_music=True` keeps this test off the ACE-take path (that leg has
    its own coverage; this pins the event shape, not the music render).
    """
    import voyage.media as media_module

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="timing", style="pastel neon line-art, peaceful")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()

    out = tmp_path / "final-timed.mp4"
    assert media_module.finalize_run(run_dir, out, no_music=True).exists()
    events = _of_kind(_metric_events(run_dir), "finalize_completed")
    assert events, "finalize must emit a finalize_completed metrics event"
    stages = events[-1]["final_stages_s"]
    assert isinstance(stages, dict)
    # Triage always runs; mix/publish always run on this path.
    for key in ("triage", "mix audio", "publish video"):
        value = stages.get(key)
        assert isinstance(value, (int, float)) and value >= 0.0, key


def test_start_captures_init_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The init handshake reply is kept for worker_started.load_seconds."""
    worker = SubprocessWorker(
        "stub-module",
        tmp_path,
        tmp_path / "worker.log",
        init_op="init",
        init_payload={"models_dir": "/models"},
    )

    class _StubProc:
        pass

    monkeypatch.setattr("voyage.rpc.subprocess.Popen", lambda *args, **kwargs: _StubProc())
    monkeypatch.setattr(SubprocessWorker, "call", lambda self, op, payload: {"load_seconds": 1.5})
    worker.start()
    assert worker.last_init_result == {"load_seconds": 1.5}


def test_start_without_init_op_clears_init_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No init handshake means no stale load_seconds from a previous start."""
    worker = SubprocessWorker("stub-module", tmp_path, tmp_path / "worker.log", init_op=None)
    worker.last_init_result = {"load_seconds": 9.0}

    class _StubProc:
        pass

    monkeypatch.setattr("voyage.rpc.subprocess.Popen", lambda *args, **kwargs: _StubProc())
    worker.start()
    assert worker.last_init_result is None


def _write_sine_wav(path: Path, duration_seconds: float) -> None:
    """Stdlib sine stub: 48 kHz stereo s16le, exact duration, one writeframes call."""
    rate, channels, freq = 48000, 2, 440.0
    frames = int(duration_seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = array.array(
        "h",
        (int(12000.0 * math.sin(2.0 * math.pi * freq * index / rate)) for index in range(frames)),
    )
    stereo = array.array("h", (sample for value in mono for sample in (value, value)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(stereo.tobytes())


def _stub_render(payload: dict[str, Any], output_path: Path) -> None:
    """Render seam: stdlib sine at the payload duration (no GPU)."""
    duration = payload["duration_seconds"]
    assert isinstance(duration, (int, float))
    _write_sine_wav(output_path, float(duration))


def _production_segment(
    run_dir: Path, index: int, frames: int, caption: str, energy: float = 0.4
) -> Path:
    """Segment dir with the flattened production manifest shape."""
    segment = run_dir / "segments" / f"{index:06d}"
    segment.mkdir(parents=True)
    manifest = {
        "format": 1,
        "transition": {
            "decision_index": index,
            "audio": {"music_caption": caption, "energy": energy},
        },
        "metrics": {"frames": frames},
    }
    (segment / "manifest.json").write_text(json.dumps(manifest))
    return segment


def test_take_observer_reports_each_render(tmp_path: Path) -> None:
    """Every rendered take reports id/action/covers/duration/wall to the observer."""
    captions = [f"dark ambient drone verse {index}" for index in range(6)]
    segments = [_production_segment(tmp_path, index, 240, captions[index]) for index in range(6)]
    facts: list[dict[str, Any]] = []
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_stub_render,
        music_style="fallback style",
        take_observer=facts.append,
    )
    assert takes, "expected at least one rendered take"
    assert len(facts) == len(takes)
    for fact in facts:
        assert set(fact) == {
            "take_id",
            "action",
            "covers_from",
            "duration_seconds",
            "render_seconds",
            "ts",
        }
        assert isinstance(fact["take_id"], str)
        assert isinstance(fact["render_seconds"], (int, float)) and fact["render_seconds"] >= 0.0
        assert isinstance(fact["ts"], (int, float)) and fact["ts"] > 0.0
    # Facts track the ledger takes one-to-one and in order.
    assert [fact["take_id"] for fact in facts] == [take["take_id"] for take in takes]


def test_take_observer_none_by_default(tmp_path: Path) -> None:
    """Without an observer rendering is unchanged and reports nothing."""
    captions = [f"dark ambient drone {index}" for index in range(2)]
    segments = [_production_segment(tmp_path, index, 240, captions[index]) for index in range(2)]
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_stub_render,
        music_style="fallback style",
    )
    assert takes, "expected at least one rendered take"


def test_ensure_for_finalize_forwards_take_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finalize entry point passes the observer through to the takes walk."""
    from voyage import audio_finalize

    seg0 = _production_segment(tmp_path, 0, 96, "dark drone")
    monkeypatch.setattr(
        audio_finalize, "spawn_fake_render_fn", lambda run_dir: (_stub_render, lambda: None)
    )
    facts: list[dict[str, Any]] = []
    rendered = audio_finalize.ensure_deferred_for_finalize(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        stretch=1.0,
        run_seed=0,
        models_dir=None,
        audio_backend="fake",
        take_observer=facts.append,
    )
    assert rendered is True
    assert facts, "expected observer facts from the finalize render"
    assert all("take_id" in fact and "render_seconds" in fact for fact in facts)


def test_append_take_rendered_events_flushes_one_per_fact(tmp_path: Path) -> None:
    """The finalize flush writes one `ace_take_rendered` line per fact."""
    from voyage.media import _append_take_rendered_events

    metrics_path = tmp_path / "metrics.jsonl"
    facts = [
        {
            "take_id": "take_0000",
            "action": "render",
            "covers_from": 0.0,
            "duration_seconds": 45.0,
            "render_seconds": 12.5,
            "ts": 1700000000.0,
        },
        {
            "take_id": "take_0001",
            "action": "repaint",
            "covers_from": 25.0,
            "duration_seconds": 40.0,
            "render_seconds": 9.0,
            # No ts: the flush stamps one so the line stays analyzable.
        },
    ]
    _append_take_rendered_events(metrics_path, facts)
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    assert [event["event"] for event in events] == ["ace_take_rendered"] * 2
    assert events[0]["take_id"] == "take_0000"
    assert events[0]["ts"] == 1700000000.0
    assert events[1]["take_id"] == "take_0001"
    assert isinstance(events[1]["ts"], (int, float))
