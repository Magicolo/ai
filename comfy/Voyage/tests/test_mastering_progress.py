"""Track C SonicMaster progress plumbing: bars, timings, background drain.

Scope is console progress + timing rows + background plumbing ONLY
(Track B owns `voyage/mastering.py` itself): the foreground
`optional_bar(progress, "mastering chunks")` + per-chunk
`tracker.update` + `set_total`/`set_extra` + `optional_stage` span, the
`mastering_poll_s` / `mastering_chunks_done` / `mastering_frames_done`
timing keys surfaced through `timing_table` via `_model_pass_stage_rows`,
and the `on_mastering_chunk` / `on_mastering_frames` callbacks through
`BackgroundPrewarm` into the supervisor `_enqueue` / `_drain` queue and
the `mastering frames` leg. No registry / media-mux / ensure / Dockerfile
coverage here. CPU-only, no GPU/torch/ffmpeg.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage.augment_background import BackgroundPrewarm
from voyage.augment_finalize import _ensure_model_pass_timings
from voyage.console import (
    RichSegmentProgress,
    VoyageConsole,
    optional_bar,
    optional_stage,
)
from voyage.media import _model_pass_stage_rows
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _live_supervisor(
    tmp_path: Path, stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> Supervisor:
    """Fake-backend supervisor with a recording console (mirrors live-display tests)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="mastering", video_backend="fake")
    config = read_effective_config(run_dir)
    console = VoyageConsole(stream=stream)
    supervisor = Supervisor(run_dir, config, progress=RichSegmentProgress(console))
    return supervisor


def test_mastering_bar_advances_per_chunk_with_extra() -> None:
    """Foreground: one update per chunk, total learned mid-pass, rate on the finish line."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    with optional_bar(console, "mastering chunks", total=4) as tracker:
        assert tracker is not None
        for _ in range(4):
            tracker.update(1)
        tracker.set_total(4)
        tracker.set_extra("12.0 chunks/s")
    output = stream.getvalue()
    assert "✓ mastering chunks (4/4," in output
    assert "12.0 chunks/s)" in output


def test_mastering_bar_quiet_advances_silently() -> None:
    """Quiet consoles count internally but print nothing on success."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream, quiet=True)
    with optional_bar(console, "mastering chunks", total=3) as tracker:
        assert tracker is not None
        tracker.update(3)
        assert tracker.done == 3
    assert stream.getvalue() == ""


def test_mastering_stage_span_reports_and_quiet_suppresses() -> None:
    """Foreground span: start/finish lines normally, silence when quiet."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    with optional_stage(console, "mastering", "2 chunk(s)"):
        pass
    output = stream.getvalue()
    assert "mastering" in output
    quiet_stream = io.StringIO()
    quiet_console = VoyageConsole(stream=quiet_stream, quiet=True)
    with optional_stage(quiet_console, "mastering", "2 chunk(s)"):
        pass
    assert quiet_stream.getvalue() == ""


def test_ensure_model_pass_timings_zero_inits_mastering_keys() -> None:
    """Timing keys exist before any mastering work lands (miners never KeyError)."""
    timings: dict[str, float] = {}
    _ensure_model_pass_timings(timings)
    assert timings["mastering_poll_s"] == 0.0
    assert timings["mastering_chunks_done"] == 0.0
    assert timings["mastering_frames_done"] == 0.0
    # Existing legs keep their keys too (no regression on the shared init).
    assert timings["upscale_poll_s"] == 0.0
    assert timings["interp_poll_s"] == 0.0
    _ensure_model_pass_timings(None)


def test_model_pass_stage_rows_surface_mastering_when_known() -> None:
    """Split-legs rows gain `mastering` only when its seconds are positive."""
    rows = _model_pass_stage_rows(
        {"upscale_poll_s": 82.9, "interp_poll_s": 70.0, "mastering_poll_s": 3.0},
        model_start=0.0,
    )
    assert rows == {"upscale": 82.9, "interpolate": 70.0, "mastering": 3.0}


def test_model_pass_stage_rows_keep_historical_shape_without_mastering() -> None:
    """Absent/zero mastering keeps the pre-Track-C shapes byte-identical."""
    rows = _model_pass_stage_rows({"upscale_poll_s": 82.9, "interp_poll_s": 70.0}, model_start=0.0)
    assert rows == {"upscale": 82.9, "interpolate": 70.0}
    wall = _model_pass_stage_rows({}, model_start=100.0)
    assert list(wall) == ["upscale + interpolate"]


def test_timing_table_surfaces_mastering_row_and_quiet_suppresses() -> None:
    """The finalize elapsed-time report names the mastering seconds."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    console.timing_table("finalize", {"upscale": 1.0, "interpolate": 2.0, "mastering": 3.0})
    assert "mastering 3.0s" in stream.getvalue()
    quiet_stream = io.StringIO()
    quiet_console = VoyageConsole(stream=quiet_stream, quiet=True)
    quiet_console.timing_table("finalize", {"mastering": 3.0})
    assert quiet_stream.getvalue() == ""


def test_background_driver_forwards_mastering_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default driver path carries mastering callbacks into `prewarm_once`."""
    seen: dict[str, Any] = {}

    def _fake_prewarm(run_dir: Path, config: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        return None

    monkeypatch.setattr("voyage.augment_background.prewarm_once", _fake_prewarm)

    def _record_mastering_frames(segment: str, frames: int) -> None:
        del segment, frames

    def _record_mastering_chunk(segment: str, index: int, total: int) -> None:
        del segment, index, total

    driver = BackgroundPrewarm(
        tmp_path,
        SimpleNamespace(),
        on_mastering_frames=_record_mastering_frames,
        on_mastering_chunk=_record_mastering_chunk,
    )
    driver._prewarm_fn(tmp_path, SimpleNamespace())
    assert seen["on_mastering_frames"] is _record_mastering_frames
    assert seen["on_mastering_chunk"] is _record_mastering_chunk
    assert seen["include_interp"] is True


def test_background_driver_reports_zero_mastering_ledger(tmp_path: Path) -> None:
    """No mastering sweep runs yet, so the separate ledger stays zero."""
    driver = BackgroundPrewarm(tmp_path, SimpleNamespace())
    assert driver.ledgered_mastering_frames() == (0, 0, 0.0)


def test_supervisor_drain_advances_mastering_leg_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drained mastering frame events advance the `mastering frames` bar."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    try:
        supervisor._enqueue_prewarm_event(("mastering_frames", "000000", 5))
        supervisor._enqueue_prewarm_event(("mastering_chunk", "000000", 0, 2))
        supervisor._drain_prewarm_queue(block=False)
        assert supervisor._pumped_mastering_frames == 5
        assert supervisor._prewarm_bars["mastering"][1]._done == 5
    finally:
        supervisor._close_model_pass_bar()


def test_supervisor_mastering_leg_ignores_model_pass_knob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mastering bars open even when upscale/interp demand is off (independent knob)."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    monkeypatch.setattr(supervisor, "_model_pass_demanded", lambda: False)
    try:
        supervisor._enqueue_prewarm_event(("mastering_frames", "000000", 2))
        supervisor._drain_prewarm_queue(block=False)
        assert supervisor._pumped_mastering_frames == 2
        assert supervisor._prewarm_bars["mastering"][1]._done == 2
    finally:
        supervisor._close_model_pass_bar()


def test_supervisor_report_notes_pumped_mastering_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-commit report notes pumped mastering frames once, then resets."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    try:
        supervisor._enqueue_prewarm_event(("mastering_frames", "000000", 6))
        supervisor._drain_prewarm_queue(block=False)
        assert supervisor._prewarm_bars["mastering"][1]._done == 6
        supervisor._background = SimpleNamespace(
            ledgered_frames=lambda: (1, 0, 0, 0, 0, 0.0, 0.0),
            ledgered_mastering_frames=lambda: (0, 0, 0.0),
            last_result=None,
        )
        supervisor._report_background_prewarm()
        assert supervisor._pumped_mastering_frames == 0
        assert supervisor._prewarm_bars == {}
        assert "mastering ledgered +6f" in stream.getvalue()
        supervisor._report_background_prewarm()
        assert stream.getvalue().count("mastering ledgered") == 1
    finally:
        supervisor._close_model_pass_bar()


def test_supervisor_report_surfaces_mastering_ledger_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ledgered mastering pass reports frames + seconds like the other legs."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    try:
        supervisor._background = SimpleNamespace(
            ledgered_frames=lambda: (1, 0, 0, 0, 0, 0.0, 0.0),
            ledgered_mastering_frames=lambda: (2, 16, 4.5),
            last_result=None,
        )
        supervisor._report_background_prewarm()
        assert "mastering ledgered +16f in 4.5s" in stream.getvalue()
    finally:
        supervisor._close_model_pass_bar()
