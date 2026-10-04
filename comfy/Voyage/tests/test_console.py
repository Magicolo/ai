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


def test_default_console_shows_video_and_audio_prompts() -> None:
    """Default output includes the full video + audio prompts per segment."""
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    console.segment_start(3, "000003")
    console.segment_plan(_plan_info())
    console.segment_done(_done_info())
    out = stream.getvalue()
    assert "SEGMENT 000003" in out
    assert "neon reef at dusk" in out
    assert "a neon reef at dusk, glowing polyps" in out
    assert "slow ambient electronic composition" in out
    assert "120 BPM" in out
    assert "take_000000" in out


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


def test_rich_available_returns_bool() -> None:
    assert isinstance(rich_available(), bool)


def test_rich_segment_progress_delegates() -> None:
    """The supervisor sink forwards all four calls to the console."""
    stream = io.StringIO()
    progress = RichSegmentProgress(VoyageConsole(stream=stream))
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


class _RecordingProgress:
    """Test sink recording every supervisor progress call."""

    def __init__(self) -> None:
        self.starts: list[tuple[int, str]] = []
        self.stages: list[str] = []
        self.plans: list[dict[str, Any]] = []
        self.dones: list[dict[str, Any]] = []

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


def test_supervisor_reports_each_segment_once(tmp_path: Path) -> None:
    """One commit → one start/plan/done with non-empty prompts."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config = read_effective_config(run_dir)
    progress = _RecordingProgress()
    supervisor = Supervisor(run_dir, config, progress=progress)
    assert supervisor.run_segments(1) == ["000000"]
    assert progress.starts == [(0, "000000")]
    assert progress.stages == ["director", "video", "audio", "validate", "commit"]
    assert len(progress.plans) == 1
    plan = progress.plans[0]
    assert plan["video_prompts"] and all(plan["video_prompts"])
    assert plan["audio_caption"]
    assert len(progress.dones) == 1
    assert progress.dones[0]["frames"] > 0
    assert progress.dones[0]["take_ids"]


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
    """`voyage generate` prints the segment header with prompts by default."""
    monkeypatch.chdir(tmp_path)
    _configure_fake_console_run()
    assert main(["generate", "console"]) == 0
    out = capsys.readouterr().out
    assert "SEGMENT 000000" in out
    assert "prompt" in out
    assert "music:" in out


def test_run_cli_accepts_verbose_and_no_color(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verbosity flags work on generate (050)."""
    monkeypatch.chdir(tmp_path)
    _configure_fake_console_run()
    assert main(["generate", "console", "--verbose"]) == 0
    assert "seeds:" in capsys.readouterr().out
    _configure_fake_console_run(segments="2")
    assert main(["generate", "console", "--no-color"]) == 0
    assert "SEGMENT 000001" in capsys.readouterr().out
