"""Widget-to-state coverage for TUI flag checkboxes (issue 181, TDD red-first).

Why this file exists: ``test_tui_state.py`` covers state-to-namespace
(given ``force=True`` the namespace carries it) but nothing covers
widget-to-state (given the user checks "Force", ``_read_form().force``
becomes ``True``) — a transposed flag id in ``_read_form`` passes the
whole suite. These tests toggle each box through the real Pilot click
path (the issue-131 interaction precedent — never programmatic
``.value`` sets) and assert ``_read_form`` follows, plus a count/ids pin
so a deleted or renamed flag fails loudly.
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


_FLAG_FIELDS = (
    ("flag-draft", "draft"),
    ("flag-force", "force"),
    ("flag-skip-bad", "skip_bad"),
    ("flag-verbose", "verbose"),
    ("flag-no-color", "no_color"),
)

_EXPECTED_FLAG_IDS = frozenset(
    [
        "flag-draft",
        "flag-force",
        "flag-skip-bad",
        "flag-no-download",
        "flag-no-sfx",
        "flag-verbose",
        "flag-no-color",
    ]
)


async def _click_when_ready(pilot: Any, app: Any, selector: str) -> None:
    """Click once the target is really under the mouse (issue 093 pattern).

    ``pilot.click`` returns False instead of raising when the click lands
    on another widget — callers must honor that or a lost click looks
    like a hung toggle. OutOfBounds (target still off-screen while the
    scheduled scroll applies) retries the same way.
    """
    from textual.pilot import OutOfBounds
    from textual.widgets import Checkbox

    for _ in range(100):
        app.query_one(selector, Checkbox).scroll_visible()
        await pilot.pause(0.05)
        try:
            if await pilot.click(selector):
                return
        except OutOfBounds:
            continue
    raise AssertionError(f"checkbox {selector} never became clickable")


def test_flag_checkbox_id_set_is_pinned() -> None:
    """Exactly the seven known flag ids exist (renames/deletes fail)."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Checkbox

        app = VoyageApp()
        # Tall viewport (verified live 2026-09-30): at (120, 40) the last
        # flag scrolls under the fold and repeated scroll_visible + click
        # sequences land on other widgets, so the toggle test below mounts
        # at (120, 60) where every flag is clickable without scrolling.
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            found = {box.id for box in app.query(Checkbox)}
            assert found == _EXPECTED_FLAG_IDS

    asyncio.run(_run())


def test_flag_toggles_propagate_to_read_form() -> None:
    """Real clicks toggle each flag and ``_read_form`` follows both ways."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        app = VoyageApp()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            for widget_id, field in _FLAG_FIELDS:
                assert getattr(app._read_form(), field) is False
                await _click_when_ready(pilot, app, f"#{widget_id}")
                await pilot.pause()
                assert getattr(app._read_form(), field) is True, widget_id
                await _click_when_ready(pilot, app, f"#{widget_id}")
                await pilot.pause()
                assert getattr(app._read_form(), field) is False, widget_id

    asyncio.run(_run())
