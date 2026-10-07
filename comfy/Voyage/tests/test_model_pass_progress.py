"""Model-pass progress: frame-unit bars, per-leg timing rows, sweep detail.

Why: the console used to show one merged chunk bar at finalize and bare
chunk counts during generation, with upscale vs interp time folded into
one aggregate. Progress now counts source frames per leg (both legs
share one comparable unit), splits upscale/interp seconds into their own
timing rows, and reports the total output frame count at publish.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from voyage.augment_background import BackgroundPrewarm, PrewarmResult, prewarm_once
from voyage.augment_finalize import _format_ranges
from voyage.augment_interp_poller import InterpPollResult
from voyage.augment_upscale_poller import UpscalePollResult
from voyage.console import VoyageConsole
from voyage.media import _model_pass_stage_rows


def test_format_ranges_compacts_runs() -> None:
    assert _format_ranges([]) == ""
    assert _format_ranges([3]) == "3"
    assert _format_ranges([0, 1, 2, 5, 7, 8]) == "0-2, 5, 7-8"
    assert _format_ranges([2, 0, 1, 1]) == "0-2"


def test_bar_extra_rides_the_finish_line() -> None:
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    with console.bar("upscale frames", total=96) as tracker:
        tracker.update(96)
        tracker.set_extra("23.4 frames/s")
    assert "✓ upscale frames (96/96," in stream.getvalue()
    assert "23.4 frames/s)" in stream.getvalue()


def test_bar_without_extra_keeps_historical_format() -> None:
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    with console.bar("sfx windows", total=8) as tracker:
        tracker.update(8)
    line = stream.getvalue()
    assert "✓ sfx windows (8/8," in line
    assert "frames/s" not in line


def _commit_segment(run_dir: Path, segment_id: str = "000000") -> Path:
    """Committed segment dir the prewarm enumeration picks up."""
    import json

    segment_dir = run_dir / "segments" / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "video.mp4").write_bytes(b"fake-video")
    manifest = {
        "format": 1,
        "transition": {},
        "prompt_plan": {},
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": 8},
        "checksums": {"video.mp4": "ck"},
    }
    (segment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    return segment_dir


def test_prewarm_once_times_sweeps_and_counts_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.augment_background as background

    up_result = UpscalePollResult(1, 0, 2, 0, 0, frames_done=64, frames_skipped=0)
    ip_result = InterpPollResult(1, 0, 2, 0, 0, 0, frames_done=64, frames_skipped=0)
    plan = SimpleNamespace(
        realesrgan_path=tmp_path / "esrgan.pth",
        interp_path=tmp_path / "film.safetensors",
        interp_backend="rife",
        weights_key="test-key",
        out_width=1216,
        out_height=704,
        source_fps_key=24,
        upscale_factor=2,
        multiplier=2,
        crf=15,
        preset="veryfast",
        device="cuda:1",
    )
    monkeypatch.setattr(background, "resolve_background_plan", lambda *args: plan)
    monkeypatch.setattr(background, "device_free_gib", lambda *args: None)
    config = SimpleNamespace(
        augment=SimpleNamespace(upscale=2, interpolate=2),
        video=SimpleNamespace(models_dir=tmp_path / "models"),
    )
    _commit_segment(tmp_path)
    result = prewarm_once(
        tmp_path,
        config,
        upscale_poll_fn=lambda *args, **kwargs: up_result,
        interp_poll_fn=lambda *args, **kwargs: ip_result,
    )
    assert result is not None
    assert result.upscale_frames_done == 64
    assert result.interp_frames_done == 64
    assert result.upscale_seconds >= 0.0
    assert result.interp_seconds >= 0.0


def test_driver_accumulates_frames_and_seconds(tmp_path: Path) -> None:
    first = PrewarmResult(1, 2, 0, 3, 0, 0, 64, 96, 12.5, 9.0)
    driver = BackgroundPrewarm(tmp_path, SimpleNamespace(), prewarm_fn=lambda dirs, config: first)
    assert driver.ledgered_totals() == (0, 0, 0)
    assert driver.ledgered_frames() == (0, 0, 0, 0, 0, 0.0, 0.0)
    driver.start()
    try:
        import time

        deadline = time.monotonic() + 10.0
        while driver.ledgered_totals()[0] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        driver.stop()
    assert driver.ledgered_totals() == (1, 2, 3)
    assert driver.ledgered_frames() == (1, 2, 3, 64, 96, 12.5, 9.0)
    assert driver.last_result is first


def test_model_pass_stage_rows_split_legs_when_known() -> None:
    rows = _model_pass_stage_rows({"upscale_poll_s": 82.9, "interp_poll_s": 70.0}, model_start=0.0)
    assert rows == {"upscale": 82.9, "interpolate": 70.0}


def test_model_pass_stage_rows_fall_back_to_wall_time() -> None:
    rows = _model_pass_stage_rows({}, model_start=100.0)
    assert list(rows) == ["upscale + interpolate"]
    assert rows["upscale + interpolate"] >= 0.0
