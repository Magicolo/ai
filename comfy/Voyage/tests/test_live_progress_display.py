"""Live generation/finalize progress: pre-warm callbacks, shared display, pump.

The console must show video plus concurrent director/upscale progress
during generation (interpolation runs at finalize time), and
upscale/interp/sfx/music during finalize (DESIGN §59). The background pre-warm
forwards per-chunk/per-frames events
from the pollers; the supervisor pump drains them into the persistent
model-pass bar while the video render blocks; finalize forks share one
N-stream coordinator so concurrent bars never open competing Live
displays. These tests pin the wiring with stub pollers/drivers (no GPU).
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.conftest import initialize_run_directory
from voyage.augment_background import BackgroundPrewarm, PrewarmResult, prewarm_once
from voyage.console import (
    GenerationDisplay,
    ParallelFinalizeDisplay,
    RichSegmentProgress,
    VoyageConsole,
)
from voyage.media import run_model_pass_and_music_parallel
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _background_plan() -> SimpleNamespace:
    """Stand-in for the resolved background plan (geometry + ledger keys)."""
    return SimpleNamespace(
        device="cuda:1",
        realesrgan_path="/models/realesrgan-anime.pth",
        film_path="/models/film_net_fp16.safetensors",
        weights_key="weights-test",
        out_width=768,
        out_height=512,
        source_fps_key=24,
        upscale_factor=2,
        multiplier=4,
        crf=15,
        preset="fast",
    )


def _poll_result(frames: int) -> SimpleNamespace:
    """Stand-in for a poller sweep result over one leg."""
    return SimpleNamespace(
        segments_seen=1,
        chunks_done=2,
        chunks_skipped=0,
        chunks_waiting=0,
        frames_done=frames,
    )


def test_prewarm_once_forwards_chunk_callbacks_to_pollers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `on_*` callbacks reach the pollers as `on_chunk*` (live pump)."""
    seen: dict[str, Any] = {}

    def _fake_upscale(_run_dir: Path, **kwargs: Any) -> SimpleNamespace:
        seen["upscale_chunk"] = kwargs["on_chunk"]
        seen["upscale_frames"] = kwargs["on_chunk_frames"]
        if kwargs["on_chunk"] is not None:
            kwargs["on_chunk"]("000000", 0, 2)
        if kwargs["on_chunk_frames"] is not None:
            kwargs["on_chunk_frames"]("000000", 5)
        return _poll_result(5)

    def _fake_interp(_run_dir: Path, **kwargs: Any) -> SimpleNamespace:
        seen["interp_chunk"] = kwargs["on_chunk"]
        seen["interp_frames"] = kwargs["on_chunk_frames"]
        if kwargs["on_chunk_frames"] is not None:
            kwargs["on_chunk_frames"]("000000", 7)
        return _poll_result(7)

    monkeypatch.setattr(
        "voyage.augment_background.resolve_background_plan",
        lambda run_dir, config: _background_plan(),
    )
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda device: None)
    upscale_events: list[tuple[str, int, int]] = []
    upscale_frames: list[tuple[str, int]] = []
    interp_events: list[tuple[str, int, int]] = []
    interp_frames: list[tuple[str, int]] = []
    result = prewarm_once(
        tmp_path,
        SimpleNamespace(),
        upscale_poll_fn=_fake_upscale,
        interp_poll_fn=_fake_interp,
        on_upscale_chunk=lambda segment, index, total: upscale_events.append(
            (segment, index, total)
        ),
        on_upscale_frames=lambda segment, frames: upscale_frames.append((segment, frames)),
        on_interp_chunk=lambda segment, index, total: interp_events.append((segment, index, total)),
        on_interp_frames=lambda segment, frames: interp_frames.append((segment, frames)),
    )
    assert result is not None
    assert result.upscale_frames_done == 5
    assert result.interp_frames_done == 7
    assert upscale_events == [("000000", 0, 2)]
    assert upscale_frames == [("000000", 5)]
    assert interp_events == []
    assert interp_frames == [("000000", 7)]


def test_prewarm_once_moot_plan_skips_pollers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No background plan means no sweep and no poller calls (moot pass)."""
    calls: list[str] = []
    monkeypatch.setattr(
        "voyage.augment_background.resolve_background_plan",
        lambda run_dir, config: None,
    )
    result = prewarm_once(
        tmp_path,
        SimpleNamespace(),
        upscale_poll_fn=lambda run_dir, **kwargs: calls.append("upscale"),
        interp_poll_fn=lambda run_dir, **kwargs: calls.append("interp"),
    )
    assert result is None
    assert calls == []


def test_background_driver_default_path_forwards_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default driver path carries upscale callbacks into `prewarm_once`.

    Interpolation runs at finalize time (another agent's scope), so the
    generation-time driver never forwards interp callbacks — it passes
    `include_interp=False` instead.
    """
    seen: dict[str, Any] = {}

    def _fake_prewarm(run_dir: Path, config: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        return None

    monkeypatch.setattr("voyage.augment_background.prewarm_once", _fake_prewarm)

    def _record_upscale_frames(segment: str, frames: int) -> None:
        del segment, frames

    driver = BackgroundPrewarm(
        tmp_path,
        SimpleNamespace(),
        on_upscale_frames=_record_upscale_frames,
    )
    driver._prewarm_fn(tmp_path, SimpleNamespace())
    assert seen["on_upscale_frames"] is _record_upscale_frames
    assert seen["on_upscale_chunk"] is None
    assert seen["include_interp"] is False


def test_background_driver_custom_fn_owns_pass_without_callbacks(
    tmp_path: Path,
) -> None:
    """Custom drivers are opaque: live callbacks cannot flow through them."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _custom_driver(run_dir: Path, config: Any) -> None:
        calls.append(((run_dir, config), {}))
        return None

    driver = BackgroundPrewarm(
        tmp_path,
        SimpleNamespace(),
        prewarm_fn=_custom_driver,
        on_upscale_frames=lambda segment, frames: None,
    )
    driver._prewarm_fn(tmp_path, SimpleNamespace())
    assert len(calls) == 1


def test_parallel_display_serves_three_streams_concurrently() -> None:
    """Three stream views share one coordinator: bars + stages stay whole."""
    stream = io.StringIO()
    display = ParallelFinalizeDisplay(VoyageConsole(stream=stream))
    try:
        model_view = display.stream_view("model pass")
        music_view = display.stream_view("music takes")
        span_view = display.stream_view("model pass + music takes")
        errors: list[BaseException] = []

        def _run_bar(view: VoyageConsole, label: str) -> None:
            try:
                with view.bar(label, total=4) as tracker:
                    tracker.update(2)
                    tracker.update(2)
            except BaseException as exc:  # noqa: BLE001 - collected, asserted below
                errors.append(exc)

        model_thread = threading.Thread(target=_run_bar, args=(model_view, "upscale frames"))
        music_thread = threading.Thread(target=_run_bar, args=(music_view, "ace takes"))
        model_thread.start()
        music_thread.start()
        with span_view.stage("model pass + music takes"):
            pass
        model_thread.join()
        music_thread.join()
        assert errors == []
    finally:
        display.close()
    output = stream.getvalue()
    for label in ("upscale frames", "ace takes", "model pass + music takes"):
        assert label in output


def test_generation_display_aliases_parallel_coordinator() -> None:
    """Generation reuses the finalize coordinator (one implementation)."""
    assert GenerationDisplay is ParallelFinalizeDisplay


def test_model_pass_and_music_fork_runs_both_branches() -> None:
    """The fork-join runs model work and music work concurrently."""
    ran: list[str] = []
    run_model_pass_and_music_parallel(
        model_work=lambda: ran.append("model"),
        music_work=lambda: ran.append("music"),
    )
    assert sorted(ran) == ["model", "music"]


def test_model_pass_and_music_fork_joins_before_raising() -> None:
    """A branch error propagates only after the other branch joined."""
    music_done = threading.Event()

    def _slow_music() -> None:
        time.sleep(0.2)
        music_done.set()

    def _failing_model() -> None:
        raise RuntimeError("model leg failed")

    with pytest.raises(RuntimeError, match="model leg failed"):
        run_model_pass_and_music_parallel(
            model_work=_failing_model,
            music_work=_slow_music,
        )
    assert music_done.is_set()


def _live_supervisor(
    tmp_path: Path, stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> Supervisor:
    """Fake-backend supervisor with a recording progress and demanded bar."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="live", video_backend="fake")
    config = read_effective_config(run_dir)
    console = VoyageConsole(stream=stream)
    supervisor = Supervisor(run_dir, config, progress=RichSegmentProgress(console))
    monkeypatch.setattr(supervisor, "_model_pass_demanded", lambda: True)
    return supervisor


def test_prewarm_queue_drain_advances_model_pass_bar_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drained frame events advance the bar while the render blocks."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    try:
        supervisor._enqueue_prewarm_event(("upscale_frames", "000000", 6))
        supervisor._enqueue_prewarm_event(("interp_frames", "000000", 4))
        supervisor._enqueue_prewarm_event(("upscale_chunk", "000000", 0, 2))
        supervisor._drain_prewarm_queue(block=False)
        assert supervisor._pumped_upscale_frames == 6
        assert supervisor._pumped_interp_frames == 4
        assert supervisor._model_pass_tracker is not None
        assert supervisor._model_pass_tracker._done == 10
    finally:
        supervisor._close_model_pass_bar()


def test_post_commit_report_does_not_double_count_pumped_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ledger delta minus pumped frames: the bar moves once, note stays full."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    try:
        supervisor._enqueue_prewarm_event(("upscale_frames", "000000", 6))
        supervisor._enqueue_prewarm_event(("interp_frames", "000000", 4))
        supervisor._drain_prewarm_queue(block=False)
        assert supervisor._model_pass_tracker is not None
        assert supervisor._model_pass_tracker._done == 10
        supervisor._background = SimpleNamespace(
            ledgered_frames=lambda: (1, 0, 0, 6, 4, 1.0, 2.0),
            last_result=None,
        )
        supervisor._report_background_prewarm()
        assert supervisor._pumped_upscale_frames == 0
        assert supervisor._pumped_interp_frames == 0
        assert supervisor._model_pass_tracker._done == 10
        assert "pre-warm ledgered +6f upscale in 1.0s, +4f interp in 2.0s" in (stream.getvalue())
    finally:
        supervisor._close_model_pass_bar()


def test_segment_render_falls_back_to_direct_call_without_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No background driver means the direct render path (no pump thread)."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    supervisor._background = None
    sentinel = SimpleNamespace(frames=48)

    class _DirectAdapter:
        def generate_segment(self, request: Any, video_out: Any) -> SimpleNamespace:
            return sentinel

    try:
        assert (
            supervisor._generate_segment_with_live_prewarm(
                cast(Any, _DirectAdapter()), cast(Any, None), cast(Any, None)
            )
            is sentinel
        )
    finally:
        supervisor._close_model_pass_bar()


def test_segment_render_resurfaces_worker_errors_after_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker-thread failures re-raise on the main thread (joined first)."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    supervisor._background = SimpleNamespace(is_alive=lambda: True)

    class _FailingAdapter:
        def generate_segment(self, request: Any, video_out: Any) -> Any:
            raise RuntimeError("render failed")

    try:
        with pytest.raises(RuntimeError, match="render failed"):
            supervisor._generate_segment_with_live_prewarm(
                cast(Any, _FailingAdapter()), cast(Any, None), cast(Any, None)
            )
    finally:
        supervisor._close_model_pass_bar()


def test_segment_render_pumps_queued_events_during_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queued pre-warm events drain live while the worker thread renders."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    supervisor._background = SimpleNamespace(is_alive=lambda: True)
    supervisor._enqueue_prewarm_event(("upscale_frames", "000000", 3))

    class _SlowAdapter:
        def generate_segment(self, request: Any, video_out: Any) -> str:
            time.sleep(0.5)
            return "rendered"

    try:
        result = supervisor._generate_segment_with_live_prewarm(
            cast(Any, _SlowAdapter()), cast(Any, None), cast(Any, None)
        )
        assert result == "rendered"
        assert supervisor._pumped_upscale_frames == 3
    finally:
        supervisor._close_model_pass_bar()


def test_prewarm_once_include_interp_false_skips_interp_poller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation-time path: `include_interp=False` never runs interp."""
    calls: list[str] = []

    def _fake_upscale(_run_dir: Path, **kwargs: Any) -> SimpleNamespace:
        calls.append("upscale")
        return _poll_result(5)

    def _fake_interp(_run_dir: Path, **kwargs: Any) -> SimpleNamespace:
        calls.append("interp")
        return _poll_result(7)

    monkeypatch.setattr(
        "voyage.augment_background.resolve_background_plan",
        lambda run_dir, config: _background_plan(),
    )
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda device: None)
    result = prewarm_once(
        tmp_path,
        SimpleNamespace(),
        upscale_poll_fn=_fake_upscale,
        interp_poll_fn=_fake_interp,
        include_interp=False,
    )
    assert result is not None
    assert result.upscale_frames_done == 5
    assert result.interp_frames_done == 0
    assert result.skip_reason == ""
    assert calls == ["upscale"]


def test_prewarm_upscale_only_leg_at_low_vram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 GiB free: SRVGG upscale runs, FILM interp waits for finalize."""
    calls: list[str] = []

    def _fake_upscale(_run_dir: Path, **kwargs: Any) -> SimpleNamespace:
        calls.append("upscale")
        return _poll_result(5)

    def _fake_interp(_run_dir: Path, **kwargs: Any) -> SimpleNamespace:
        calls.append("interp")
        return _poll_result(7)

    monkeypatch.setattr(
        "voyage.augment_background.resolve_background_plan",
        lambda run_dir, config: _background_plan(),
    )
    # Resident llama sidecar leaves ~2.8 GiB: upscale fits, interp does not.
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda device: 2.0)
    result = prewarm_once(
        tmp_path,
        SimpleNamespace(),
        upscale_poll_fn=_fake_upscale,
        interp_poll_fn=_fake_interp,
    )
    assert result is not None
    assert result.upscale_frames_done == 5
    assert result.interp_frames_done == 0
    assert "interp skipped" in result.skip_reason
    assert calls == ["upscale"]


def test_background_loop_waits_for_idle_instead_of_dropping_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A notify landing during prefetch waits, then sweeps (no lost pass)."""
    monkeypatch.setattr("voyage.augment_background.PREWARM_IDLE_POLL_SECONDS", 0.01)
    calls: list[str] = []
    idle_calls = 0

    def _fake_prewarm(run_dir: Path, config: Any) -> PrewarmResult:
        calls.append("sweep")
        return PrewarmResult(0, 0, 0, 0, 0, 0)

    def _idle_soon() -> bool:
        nonlocal idle_calls
        idle_calls += 1
        return idle_calls >= 3

    driver = BackgroundPrewarm(tmp_path, object(), prewarm_fn=_fake_prewarm, idle_fn=_idle_soon)
    driver.start()
    try:
        deadline = time.monotonic() + 10.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls == ["sweep"]
        assert driver.ledgered_totals()[0] >= 1
    finally:
        driver.stop()


def test_background_loop_records_busy_past_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Director busy past the deadline: zero-count result, no sweep."""
    monkeypatch.setattr("voyage.augment_background.PREWARM_IDLE_WAIT_SECONDS", 0.05)
    monkeypatch.setattr("voyage.augment_background.PREWARM_IDLE_POLL_SECONDS", 0.01)
    calls: list[str] = []

    def _must_not_run(run_dir: Path, config: Any) -> PrewarmResult:
        calls.append("sweep")
        raise AssertionError("must not sweep while the director holds the device")

    driver = BackgroundPrewarm(tmp_path, object(), prewarm_fn=_must_not_run, idle_fn=lambda: False)
    driver.start()
    try:
        deadline = time.monotonic() + 10.0
        while driver.last_result is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert driver.last_result is not None
        assert "director busy" in driver.last_result.skip_reason
        assert calls == []
    finally:
        driver.stop()


def test_report_notes_held_back_skip_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A held-back pass surfaces one compact note; moot passes stay silent."""
    stream = io.StringIO()
    supervisor = _live_supervisor(tmp_path, stream, monkeypatch)
    supervisor._background = SimpleNamespace(
        ledgered_frames=lambda: (1, 0, 0, 0, 0, 0.0, 0.0),
        last_result=PrewarmResult(0, 0, 0, 0, 0, 0, skip_reason="interp skipped: low headroom"),
    )
    try:
        supervisor._report_background_prewarm()
        assert "pre-warm held back: interp skipped: low headroom" in stream.getvalue()
        supervisor._report_background_prewarm()
        assert stream.getvalue().count("pre-warm held back") == 1
    finally:
        supervisor._close_model_pass_bar()
