"""SFX caption rendered in the live surface (issue 161, TDD red-first).

Why this file exists: the supervisor ships ``audio_sfx_caption`` in the
plan dict, and ``console.segment_plan`` must surface it — the third
caption family's drift is invisible until the finalize pass dubs stems.
These tests pin the verbose SFX line (present/absent/empty) plus its
absence from the compact default, mirroring ``tests/test_console.py`` style.
The ``video_caption`` CLI
pin is not in the plan dict (supervisor-owned, out of scope) and stays
a logged residual.
"""

from __future__ import annotations

import io
from typing import Any


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


def test_console_plan_renders_sfx_caption() -> None:
    """Verbose console prints the SFX caption alongside the music caption."""
    from voyage.console import VoyageConsole

    stream = io.StringIO()
    VoyageConsole(verbose=True, stream=stream).segment_plan(_plan_info("rain on canvas"))
    out = stream.getvalue()
    assert "rain on canvas" in out
    assert "slow ambient electronic composition" in out
    default_stream = io.StringIO()
    VoyageConsole(stream=default_stream).segment_plan(_plan_info("rain on canvas"))
    assert "rain on canvas" not in default_stream.getvalue()


def test_console_plan_marks_missing_sfx_caption() -> None:
    """Empty/missing SFX caption is itself visible in verbose mode ("-")."""
    from voyage.console import VoyageConsole

    for missing in ("", None):
        plan = _plan_info(missing)
        if missing is None:
            del plan["audio_sfx_caption"]
        stream = io.StringIO()
        VoyageConsole(verbose=True, stream=stream).segment_plan(plan)
        assert "sfx" in stream.getvalue().lower()
