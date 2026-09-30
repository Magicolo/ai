"""TUI progress content parity with the console (issue 113, TDD red-first).

Why this file exists: ``TuiProgress`` docstring promises plain-text lines
mirror the console wording, but ``segment_plan``/``segment_done`` drop the
decision-relevant lines (backend labels, drift hold, video geometry,
energy, take ids + action, prefetch, elapsed total). These tests drive one
canned plan/done pair through both sinks and assert token parity on the
non-verbose subset. Verbose-gated detail (seeds/transitions/texture/notes)
stays console-only by contract. Caption-family lines (161) are out of
scope here — see ``tests/test_sfx_caption_render_161.py``.
"""

from __future__ import annotations

import io
from typing import Any, cast

from voyage.console import VoyageConsole


def _plan_info() -> dict[str, Any]:
    """One canned supervisor-style plan dict (all parity keys populated)."""
    return {
        "number": 3,
        "segment_id": "000003",
        "destination": "neon reef at dusk",
        "phase": "ESTABLISH",
        "novelty_accepted": True,
        "drift_hold": True,
        "prefetch_hit": True,
        "director_backend": "qwen",
        "video_backend": "fake",
        "audio_backend": "fake",
        "geometry": "768x432",
        "fps": 24,
        "planned_frames": 48,
        "planned_duration": 2.0,
        "blocks": 2,
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
    """One canned commit summary (take action + prefetch + elapsed set)."""
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


class _FakeApp:
    """Synchronous stand-in for VoyageApp (no event loop needed)."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.bars: list[tuple[int, int]] = []

    def call_from_thread(self, callback: Any, *args: Any) -> Any:
        return callback(*args)

    def append_run_line(self, text: str) -> None:
        self.lines.append(text)

    def advance_run_bar(self, done: int, total: int) -> None:
        self.bars.append((done, total))


def _tui_lines_for_plan() -> list[str]:
    """Drive TuiProgress.segment_plan on the canned dict, return history."""
    import pytest

    pytest.importorskip("textual")
    from voyage.tui import TuiProgress

    fake = _FakeApp()
    progress = TuiProgress(cast(Any, fake), 1)
    progress.segment_plan(_plan_info())
    return list(fake.lines)


def _tui_lines_for_done() -> list[str]:
    """Drive TuiProgress.segment_done on the canned dict, return history."""
    import pytest

    pytest.importorskip("textual")
    from voyage.tui import TuiProgress

    fake = _FakeApp()
    progress = TuiProgress(cast(Any, fake), 1)
    progress.segment_done(_done_info())
    return list(fake.lines)


def test_tui_plan_shows_backend_labels_and_drift_hold() -> None:
    """Drift line carries the director backend + hold flag like console."""
    lines = _tui_lines_for_plan()
    blob = "\n".join(lines)
    assert "qwen" in blob
    assert "drift hold" in blob


def test_tui_plan_shows_video_geometry_line() -> None:
    """Geometry/fps/frames/blocks/scene-cut reach the TUI run log."""
    lines = _tui_lines_for_plan()
    blob = "\n".join(lines)
    assert "768x432" in blob
    assert "24fps" in blob
    assert "48f" in blob
    assert "2 block(s)" in blob
    assert "scene-cut" in blob


def test_tui_plan_shows_audio_backend_and_energy() -> None:
    """Non-verbose audio line keeps backend + energy (not just beats)."""
    stream = io.StringIO()
    VoyageConsole(stream=stream).segment_plan(_plan_info())
    assert "energy 0.52" in stream.getvalue()
    blob = "\n".join(_tui_lines_for_plan())
    assert "fake" in blob
    assert "energy 0.52" in blob


def test_tui_done_shows_take_ids_and_action() -> None:
    """Commit line names the take + the keep/render/repaint judgment."""
    blob = "\n".join(_tui_lines_for_done())
    assert "take_000000" in blob
    assert "render" in blob


def test_tui_done_shows_prefetch_and_elapsed_total() -> None:
    """Prefetch-hit marker + elapsed total ride the timing line."""
    stream = io.StringIO()
    VoyageConsole(stream=stream).segment_done(_done_info())
    assert "prefetch hit" in stream.getvalue()
    blob = "\n".join(_tui_lines_for_done())
    assert "prefetch hit" in blob
    assert "2.1s" in blob
