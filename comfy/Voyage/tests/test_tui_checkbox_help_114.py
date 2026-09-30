"""TUI checkbox help coverage (issue 114, TDD red-first).

Why this file exists: the five flag checkboxes (draft/force/skip-bad/
verbose/no-color) have no tooltip, no FIELD_HELP entry, and no widget-id
mapping, so the focus-driven help panel falls back to the overview on
exactly the flags that need one sentence each. The batch-8 boxes
(no_download/no_sfx) already follow the help pattern and must keep
working — these tests pin both the new keys and the pre-existing ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep remembered TUI settings out of the real home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))


def _require_app() -> Any:
    """Import the Textual app (skip when the display extra is missing)."""
    pytest.importorskip("textual")
    from voyage.tui import VoyageApp

    return VoyageApp


_EXPECTED_HELP_KEYWORDS = {
    "draft": "640",
    "force": "non-empty",
    "skip_bad": "salvage",
    "verbose": "verbose",
    "no_color": "color",
}

_FLAG_WIDGET_IDS = {
    "draft": "flag-draft",
    "force": "flag-force",
    "skip_bad": "flag-skip-bad",
    "verbose": "flag-verbose",
    "no_color": "flag-no-color",
}


def test_field_help_covers_all_flag_checkboxes() -> None:
    """FIELD_HELP gains one entry per flag checkbox (no key collisions)."""
    from voyage.tui_state import FIELD_HELP

    for field in _EXPECTED_HELP_KEYWORDS:
        assert field in FIELD_HELP, f"FIELD_HELP missing {field!r}"
        assert _EXPECTED_HELP_KEYWORDS[field] in FIELD_HELP[field].lower()
    # Batch-8 keys stay intact (coordinate, don't collide).
    assert "no_download" in FIELD_HELP
    assert "no_sfx" in FIELD_HELP


def test_flag_checkboxes_carry_tooltips() -> None:
    """Each flag Checkbox passes its FIELD_HELP text as tooltip."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Checkbox

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for field, widget_id in _FLAG_WIDGET_IDS.items():
                box = app.query_one(f"#{widget_id}", Checkbox)
                tooltip = str(getattr(box, "tooltip", "") or "")
                assert tooltip, f"#{widget_id} has no tooltip"
                assert _EXPECTED_HELP_KEYWORDS[field] in tooltip.lower()

    asyncio.run(_run())


def test_help_panel_describes_focused_checkbox() -> None:
    """Focusing a flag checkbox shows its help, not the overview."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Checkbox, Static

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#help-body", Static)
            for field, widget_id in _FLAG_WIDGET_IDS.items():
                app.set_focus(app.query_one(f"#{widget_id}", Checkbox))
                await pilot.pause()
                body = str(panel.content).lower()
                assert "focus any field" not in body, f"#{widget_id} fell back to overview"
                assert _EXPECTED_HELP_KEYWORDS[field] in body, f"#{widget_id} help missing"

    asyncio.run(_run())
