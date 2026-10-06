"""Console progress tests: prompts on screen, TTY fallback, hooks (CPU/fake).

The console layer is display-only: these tests assert words on stdout
(never log/artifact contents) plus that the supervisor calls the
progress sink exactly once per segment with non-empty prompts.
"""

from __future__ import annotations

import io
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage.cli import main
from voyage.console import RichSegmentProgress, VoyageConsole, rich_available
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "console") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _plan_info() -> dict[str, Any]:
    return {
        "number": 3,
        "segment_id": "000003",
        "destination": "neon reef at dusk",
        "phase": "ESTABLISH",
        "novelty_accepted": True,
        "drift_hold": False,
        "prefetch_hit": True,
        "director_backend": "qwen",
        "video_backend": "fake",
        "audio_backend": "fake",
        "geometry": "768x432",
        "fps": 24,
        "planned_frames": 48,
        "planned_duration": 2.0,
        "blocks": 1,
        "video_prompts": ["a neon reef at dusk, glowing polyps"],
        "video_seeds": [4242],
        "scene_cuts": [True],
        "transition_mechanism": "hybrid",
        "transition_stages": ["the reef brightens"],
        "audio_caption": "slow ambient electronic composition",
        "audio_energy": 0.52,
        "audio_bpm": 120.0,
        "audio_beats": 4,
        "audio_texture": "granular",
        "audio_environment": ["cave"],
        "notes": "novelty similarity 0.100 record 7",
    }


def _done_info() -> dict[str, Any]:
    return {
        "number": 3,
        "segment_id": "000003",
        "frames": 48,
        "duration": 2.0,
        "take_ids": ["take_000000"],
        "take_action": "render",
        "take_reason": "coverage low",
        "beats": 4,
        "bpm": 120.0,
        "video_backend": "fake",
        "overlap_fraction": 0.1,
        "overlap_cap_seconds": 0.5,
        "stage_seconds": {"video": 1.2, "audio": 0.4},
        "elapsed": 2.1,
        "prefetch_hit": True,
    }


def test_default_console_hides_prompts_behind_verbose() -> None:
    """Default output is a compact header; full prompts need --verbose."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    console.segment_start(3, "000003")
    console.segment_plan(_plan_info())
    console.segment_done(_done_info())
    out = stream.getvalue()
    assert "SEGMENT 000003" in out
    assert "neon reef at dusk" in out
    assert "120 BPM" in out
    assert "take_000000" in out
    assert "a neon reef at dusk, glowing polyps" not in out
    assert "slow ambient electronic composition" not in out
    verbose_stream = io.StringIO()
    VoyageConsole(verbose=True, stream=verbose_stream).segment_plan(_plan_info())
    verbose_out = verbose_stream.getvalue()
    assert "a neon reef at dusk, glowing polyps" in verbose_out
    assert "slow ambient electronic composition" in verbose_out


def test_deferred_done_shows_no_take_with_action() -> None:
    """All-deferred pin: empty take_ids renders as no-take + deferred action.

    New deferred commits carry no per-segment takes (finalize renders
    from `run/audio/takes.jsonl`); the console keeps showing the
    take action/reason so the deferral itself stays visible.
    """
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    console.segment_done(
        {
            "number": 3,
            "segment_id": "000003",
            "frames": 48,
            "duration": 2.0,
            "take_ids": [],
            "take_action": "deferred",
            "take_reason": "deferred-audio backend ltx25: silent stub committed",
            "beats": 4,
            "bpm": 120.0,
            "video_backend": "ltx25",
            "overlap_fraction": 0.1,
            "overlap_cap_seconds": 0.5,
            "stage_seconds": {"video": 1.2, "audio": 0.4},
            "elapsed": 2.1,
            "prefetch_hit": True,
        }
    )
    out = stream.getvalue()
    assert "no take" in out
    assert "deferred" in out
    verbose = io.StringIO()
    VoyageConsole(verbose=True, stream=verbose).segment_done(
        {
            "segment_id": "000003",
            "frames": 48,
            "duration": 2.0,
            "take_ids": [],
            "take_action": "deferred",
            "take_reason": "deferred-audio backend ltx25: silent stub committed",
            "beats": 4,
            "bpm": 120.0,
        }
    )
    assert "deferred-audio" in verbose.getvalue()


def test_verbose_console_adds_seeds_transitions_notes() -> None:
    """--verbose adds payload minutiae; compact mode omits them."""
    plain = io.StringIO()
    VoyageConsole(stream=plain).segment_plan(_plan_info())
    assert "[4242]" not in plain.getvalue()
    assert "the reef brightens" not in plain.getvalue()

    verbose = io.StringIO()
    VoyageConsole(verbose=True, stream=verbose).segment_plan(_plan_info())
    out = verbose.getvalue()
    assert "4242" in out
    assert "the reef brightens" in out
    assert "granular" in out
    assert "novelty similarity 0.100" in out


def test_no_color_disables_rich() -> None:
    """--no-color forces the plain path even on capable terminals."""
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    assert console.color_enabled is False
    console.ok("plain words")
    assert "plain words" in stream.getvalue()


def test_no_color_env_disables_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    """NO_COLOR=1 in the environment also forces the plain path."""
    monkeypatch.setenv("NO_COLOR", "1")
    console = VoyageConsole(stream=io.StringIO())
    assert console.color_enabled is False


def test_stage_context_reports_start_and_elapsed() -> None:
    """Stage blocks announce start and completion with timings."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    with console.stage("video", "fake 768x432"):
        pass
    out = stream.getvalue()
    assert "video" in out
    assert "fake 768x432" in out


def test_stage_context_reports_failure() -> None:
    """Stage failures are announced, then the error propagates."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    with pytest.raises(RuntimeError, match="boom"):
        with console.stage("audio"):
            raise RuntimeError("boom")
    assert "audio" in stream.getvalue()
    assert "failed" in stream.getvalue()


def test_quiet_suppresses_success_but_not_errors() -> None:
    """--quiet silences lines/ok/stages but failures still report."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream, quiet=True)
    console.line("plans")
    console.ok("done")
    with console.stage("video", "fake"):
        pass
    assert stream.getvalue() == ""
    console.error("boom")
    assert "boom" in stream.getvalue()
    with pytest.raises(RuntimeError, match="boom"):
        with console.stage("audio"):
            raise RuntimeError("boom")
    assert "audio" in stream.getvalue()


def test_bar_reports_boundaries_without_tty() -> None:
    """Non-TTY bar: start line + X/Y finish line, silent per-update."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    with console.bar("sfx windows", total=8) as tracker:
        for _ in range(8):
            tracker.update()
    out = stream.getvalue()
    assert "sfx windows" in out
    assert "8/8" in out


def test_bar_unknown_total_counts_completions() -> None:
    """total=None bar counts completions until set_total learns the total."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    with console.bar("ace takes") as tracker:
        tracker.update()
        tracker.update()
        tracker.set_total(2)
    out = stream.getvalue()
    assert "ace takes" in out
    assert "2/2" in out


def test_timing_table_names_slowest_and_total() -> None:
    """Finalize summary prints every stage + slowest + total."""
    stream = io.StringIO()
    VoyageConsole(stream=stream).timing_table("finalize", {"model": 10.0, "music": 2.0})
    out = stream.getvalue()
    assert "model 10.0s" in out
    assert "slowest model" in out
    assert "total 12.0s" in out


def test_timing_table_empty_is_silent() -> None:
    stream = io.StringIO()
    VoyageConsole(stream=stream).timing_table("finalize", {})
    assert stream.getvalue() == ""


def test_rich_available_returns_bool() -> None:
    assert isinstance(rich_available(), bool)


def _stub_ace_spawn(run_dir: Path, models_dir: object, device: str = "cuda:0") -> tuple[Any, Any]:
    """ACE spawn double: renders stdlib silence WAVs (no GPU, no models)."""

    def _render(payload: dict[str, Any], output_path: Path) -> None:
        import wave

        duration = float(payload["duration_seconds"])
        frames = max(1, int(48000 * duration))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(b"\0" * frames * 4)

    def _shutdown() -> None:
        return None

    del run_dir, models_dir, device
    return (_render, _shutdown)


def test_finalize_run_reports_stages_and_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finalize with a console shows the checklist + the timing table."""
    from voyage import audio_finalize
    from voyage.media import finalize_run

    monkeypatch.setattr(audio_finalize, "spawn_ace_render_fn", _stub_ace_spawn)
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = Supervisor(run_dir, read_effective_config(run_dir))
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    stream = io.StringIO()
    out = run_dir / "final.mp4"
    assert finalize_run(run_dir, out, progress=VoyageConsole(stream=stream)).exists()
    text = stream.getvalue()
    assert "triage segments" in text
    assert "mix final audio" in text
    assert "slowest" in text and "total" in text


def test_finalize_silent_without_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No progress sink → finalize prints nothing (library stays quiet)."""
    from voyage import audio_finalize
    from voyage.media import finalize_run

    monkeypatch.setattr(audio_finalize, "spawn_ace_render_fn", _stub_ace_spawn)
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = Supervisor(run_dir, read_effective_config(run_dir))
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert finalize_run(run_dir, run_dir / "final.mp4").exists()
    assert capsys.readouterr().out == ""


class _SilentSfxWorker:
    """SubprocessWorker double writing valid WAVs (no GPU, no models)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def start(self) -> None:
        return None

    def call(self, op: str, payload: dict[str, object]) -> dict[str, object]:
        import wave

        del op
        out = str(payload["output_path"])
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        duration = float(payload["duration_seconds"])  # type: ignore[arg-type]
        frames = max(1, int(48000 * duration))
        with wave.open(out, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(b"\0" * frames * 4)
        return {"artifacts": {"path": out}}

    def stop(self) -> None:
        return None


def test_render_sfx_bed_reports_window_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SFX bed counts windows X/Y on the console."""
    from voyage.sfx_finalize import render_sfx_bed

    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _SilentSfxWorker)
    run_dir = tmp_path / "run"
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    stream = io.StringIO()
    render_sfx_bed(
        run_dir,
        final_video,
        4.0,
        [(0.0, 4.0, "rain")],
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "small_44k",
        0,
        48000,
        2,
        1,
        progress=VoyageConsole(stream=stream),
    )
    text = stream.getvalue()
    assert "sfx windows" in text
    assert "1/1" in text


def test_rich_segment_progress_delegates() -> None:
    """The supervisor sink forwards all four calls to the console."""
    stream = io.StringIO()
    progress = RichSegmentProgress(VoyageConsole(verbose=True, stream=stream))
    progress.segment_start(3, "000003")
    stage: AbstractContextManager[Any] = progress.stage("video", "fake")
    with stage:
        pass
    progress.segment_plan(_plan_info())
    progress.segment_done(_done_info())
    out = stream.getvalue()
    assert "SEGMENT 000003" in out
    assert "a neon reef at dusk, glowing polyps" in out
    assert "slow ambient electronic composition" in out


class _RecordingBar:
    """Test bar tracker recording advance/total/extra."""

    def __init__(self) -> None:
        self.updates: list[int] = []
        self.totals: list[int] = []
        self.extra: str | None = None

    def update(self, advance: int = 1) -> None:
        self.updates.append(advance)

    def set_total(self, total: int) -> None:
        self.totals.append(total)

    def set_extra(self, extra: str) -> None:
        self.extra = extra


class _RecordingProgress:
    """Test sink recording every supervisor progress call."""

    def __init__(self) -> None:
        self.starts: list[tuple[int, str]] = []
        self.stages: list[str] = []
        self.plans: list[dict[str, Any]] = []
        self.dones: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.bars: list[tuple[str, _RecordingBar]] = []
        self.verbose = False

    def bar(self, label: str, total: int | None = None) -> AbstractContextManager[Any]:
        from contextlib import contextmanager

        del total
        tracker = _RecordingBar()
        self.bars.append((label, tracker))

        @contextmanager
        def _stay_open():  # type: ignore[no-untyped-def]
            yield tracker

        return _stay_open()

    def segment_start(self, number: int, segment_id: str) -> None:
        self.starts.append((number, segment_id))

    def stage(self, label: str, detail: str = "") -> AbstractContextManager[Any]:
        from contextlib import nullcontext

        self.stages.append(label)
        del detail
        return nullcontext()

    def segment_plan(self, info: dict[str, Any]) -> None:
        self.plans.append(info)

    def segment_done(self, info: dict[str, Any]) -> None:
        self.dones.append(info)

    def note(self, message: str) -> None:
        self.notes.append(message)


def test_supervisor_reports_each_segment_once(tmp_path: Path) -> None:
    """One commit → one start/plan/done with non-empty prompts.

    Fake backend is joint: it still renders per-segment takes, so
    `take_ids` is non-empty here. Deferred backends report empty
    `take_ids` with `take_action == "deferred"` (pinned by the display
    test above, not by a live GPU commit here).
    """
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    progress = _RecordingProgress()
    supervisor = Supervisor(run_dir, config, progress=progress)
    assert supervisor.run_segments(1) == ["000000"]
    assert progress.starts == [(0, "000000")]
    # Tail match: worker-startup stages may prefix the per-commit stages
    # (owned by the supervisor lifecycle track, not pinned here).
    assert progress.stages[-5:] == ["director", "video", "audio", "validate", "commit"]
    assert len(progress.plans) == 1
    plan = progress.plans[0]
    assert plan["video_prompts"] and all(plan["video_prompts"])
    assert plan["audio_caption"]
    assert len(progress.dones) == 1
    assert progress.dones[0]["frames"] > 0
    # Always-deferred: no per-segment takes — the done summary carries
    # the deferred action/reason instead (takes render at finalize from
    # `run/audio/takes.jsonl`).
    assert progress.dones[0]["take_ids"] == []
    assert progress.dones[0]["take_action"] == "deferred"


def test_supervisor_silent_by_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """No progress sink → no console output (tests stay quiet)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    assert capsys.readouterr().out == ""


def _configure_fake_console_run(name: str = "console", segments: str = "1") -> None:
    assert (
        main(
            [
                "configure",
                name,
                "--backend",
                "fake",
                "--segments",
                segments,
                "--style",
                "pastel neon line-art, peaceful",
                "--seed",
                "11",
            ]
        )
        == 0
    )


def test_run_cli_shows_segment_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`voyage generate` prints a compact segment header by default."""
    monkeypatch.chdir(tmp_path)
    _configure_fake_console_run()
    assert main(["generate", "console"]) == 0
    out = capsys.readouterr().out
    assert "SEGMENT 000000" in out
    assert "music:" not in out


def test_run_cli_accepts_verbose_and_no_color(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verbosity flags work on generate (050)."""
    monkeypatch.chdir(tmp_path)
    _configure_fake_console_run()
    assert main(["generate", "console", "--verbose"]) == 0
    verbose_out = capsys.readouterr().out
    assert "seeds:" in verbose_out
    assert "music:" in verbose_out
    _configure_fake_console_run(segments="2")
    assert main(["generate", "console", "--no-color"]) == 0
    assert "SEGMENT 000001" in capsys.readouterr().out


def test_note_reports_background_worker_line() -> None:
    """`note()` prints one ▸ line (prefetch submit/hit, pre-warm ledger)."""
    stream = io.StringIO()
    RichSegmentProgress(VoyageConsole(stream=stream)).note(
        "director prefetch hit (used as first candidate)"
    )
    assert "▸ director prefetch hit (used as first candidate)" in stream.getvalue()


def test_note_silent_when_quiet() -> None:
    """Quiet consoles suppress background-worker notes (failures only)."""
    stream = io.StringIO()
    RichSegmentProgress(VoyageConsole(stream=stream, quiet=True)).note(
        "pre-warm ledgered +2 upscale chunks"
    )
    assert stream.getvalue() == ""


def _prewarm_supervisor(tmp_path: Path, sink: _RecordingProgress) -> Supervisor:
    """Supervisor double with just enough state for the pre-warm report."""
    from types import SimpleNamespace

    supervisor = Supervisor.__new__(Supervisor)
    supervisor._background = None
    supervisor._progress = sink
    config: Any = SimpleNamespace(augment=SimpleNamespace(upscale=2, interpolate=2))
    supervisor._config = config
    supervisor._run_dir = tmp_path / "no-state-here"
    supervisor._reported_prewarm = (None, 0, 0, 0, 0, 0.0, 0.0)
    supervisor._model_pass_bar = None
    supervisor._model_pass_tracker = None
    return supervisor


def test_prewarm_report_announces_new_frames_with_seconds(tmp_path: Path) -> None:
    """Post-commit report: frame deltas with per-leg seconds, silent on idle."""
    from voyage.augment_background import PrewarmResult

    class _Driver:
        def __init__(self) -> None:
            self.ledger: tuple[int, int, int, int, int, float, float] = (
                0,
                0,
                0,
                0,
                0,
                0.0,
                0.0,
            )
            self.result: Any = None

        def ledgered_frames(self):  # type: ignore[no-untyped-def]
            return self.ledger

        @property
        def last_result(self):  # type: ignore[no-untyped-def]
            return self.result

    sink = _RecordingProgress()
    supervisor = _prewarm_supervisor(tmp_path, sink)
    # No driver yet: silent, no crash.
    supervisor._report_background_prewarm()
    assert sink.notes == [] and sink.bars == []
    driver = _Driver()
    supervisor._background = driver
    supervisor._report_background_prewarm()
    assert sink.notes == []
    driver.ledger = (1, 10, 8, 96, 64, 12.5, 9.0)
    supervisor._report_background_prewarm()
    assert sink.notes == [
        "pre-warm ledgered +96f upscale in 12.5s, +64f interp in 9.0s (total 96/64f)"
    ]
    # Persistent bar opened once; both legs advance through it (the pre-warm
    # interleaves upscale then interp per segment).
    assert [label for label, _tracker in sink.bars] == ["model-pass frames"]
    _label, tracker = sink.bars[0]
    assert tracker.updates == [160]
    # Nothing new since the last report: silent.
    supervisor._report_background_prewarm()
    assert len(sink.notes) == 1
    # A restarted driver resets its totals: baseline resets, full delta shown.
    replacement = _Driver()
    replacement.ledger = (1, 5, 0, 32, 0, 4.0, 0.0)
    supervisor._background = replacement
    supervisor._report_background_prewarm()
    assert sink.notes[-1] == "pre-warm ledgered +32f upscale in 4.0s (total 32/0f)"
    # Verbose adds the latest sweep breakdown (object-identity deduped).
    sink.verbose = True
    replacement.result = PrewarmResult(1, 5, 0, 3, 0, 0, 32, 24, 4.0, 6.0)
    replacement.ledger = (2, 8, 6, 64, 24, 9.0, 12.0)
    supervisor._report_background_prewarm()
    assert sink.notes[-1] == ("pre-warm sweep: 1 segments, upscale 32f in 4.0s, interp 24f in 6.0s")
    before = len(sink.notes)
    supervisor._report_background_prewarm()
    assert len(sink.notes) == before  # same result object: no repeat


def test_prewarm_report_tolerates_legacy_chunk_only_drivers(tmp_path: Path) -> None:
    """Drivers without frame accounting never break the report (stay silent)."""

    class _LegacyDriver:
        def __init__(self) -> None:
            self.totals: tuple[int, int, int] = (0, 0, 0)

        def ledgered_totals(self) -> tuple[int, int, int]:
            return self.totals

    sink = _RecordingProgress()
    supervisor = _prewarm_supervisor(tmp_path, sink)
    driver = _LegacyDriver()
    supervisor._background = driver
    supervisor._report_background_prewarm()
    assert sink.notes == []
    driver.totals = (1, 2, 3)
    supervisor._report_background_prewarm()
    assert sink.notes == []  # chunk counts alone no longer report
    assert sink.bars == []
