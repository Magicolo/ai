"""SFX caption rendered in both live surfaces (issue 161, TDD red-first).

Why this file exists: the supervisor ships ``audio_sfx_caption`` in the
plan dict, but neither ``console.segment_plan`` nor ``TuiProgress``
reads it — the third caption family's drift is invisible until the
finalize pass dubs stems. These tests pin one non-verbose SFX line per
sink (present/absent/empty), mirroring ``tests/test_console.py`` style.
The ``video_caption`` CLI pin is not in the plan dict (supervisor-owned,
out of scope) and stays a logged residual.
"""

from __future__ import annotations

import io
from typing import Any, cast


def _plan_info(sfx_caption: Any) -> dict[str, Any]:
    """Minimal plan dict with a configurable SFX caption value."""
    return {
        "number": 3,
        "segment_id": "000003",
        "destination": "neon reef at dusk",
        "phase": "ESTABLISH",
        "novelty_accepted": True,
        "drift_hold": False,
        "prefetch_hit": False,
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
        "scene_cuts": [False],
        "transition_mechanism": "hybrid",
        "transition_stages": [],
        "audio_caption": "slow ambient electronic composition",
        "audio_energy": 0.52,
        "audio_bpm": 120.0,
        "audio_beats": 4,
        "audio_texture": "",
        "audio_environment": [],
        "audio_sfx_caption": sfx_caption,
        "notes": "",
    }


def _tui_lines(plan: dict[str, Any]) -> list[str]:
    """Drive TuiProgress.segment_plan, return posted lines (sync fake)."""
    import pytest

    pytest.importorskip("textual")
    from voyage.tui import TuiProgress

    class _FakeApp:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def call_from_thread(self, callback: Any, *args: Any) -> Any:
            return callback(*args)

        def append_run_line(self, text: str) -> None:
            self.lines.append(text)

        def advance_run_bar(self, done: int, total: int) -> None:
            del done, total

    fake = _FakeApp()
    TuiProgress(cast(Any, fake), 1).segment_plan(plan)
    return list(fake.lines)


def test_console_plan_renders_sfx_caption() -> None:
    """Console prints the SFX caption alongside the music caption."""
    from voyage.console import VoyageConsole

    stream = io.StringIO()
    VoyageConsole(stream=stream).segment_plan(_plan_info("rain on canvas"))
    out = stream.getvalue()
    assert "rain on canvas" in out
    assert "slow ambient electronic composition" in out


def test_console_plan_marks_missing_sfx_caption() -> None:
    """Empty/missing SFX caption is itself visible (always-print rule)."""
    from voyage.console import VoyageConsole

    for missing in ("", None):
        plan = _plan_info(missing)
        if missing is None:
            del plan["audio_sfx_caption"]
        stream = io.StringIO()
        VoyageConsole(stream=stream).segment_plan(plan)
        assert "sfx" in stream.getvalue().lower()


def test_tui_plan_renders_sfx_caption() -> None:
    """TUI posts the SFX caption line after the music line."""
    blob = "\n".join(_tui_lines(_plan_info("rain on canvas")))
    assert "rain on canvas" in blob
    assert "slow ambient electronic composition" in blob


def test_tui_plan_marks_missing_sfx_caption() -> None:
    """Empty/missing SFX caption is itself visible in the TUI log."""
    for missing in ("", None):
        plan = _plan_info(missing)
        if missing is None:
            del plan["audio_sfx_caption"]
        blob = "\n".join(_tui_lines(plan))
        assert "sfx" in blob.lower()
