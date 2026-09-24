"""Launcher app behavior tests: focus, dropdown, help panel, keys (headless Pilot).

Why this module exists: users reported the form felt uneditable (nothing
took focus on mount so typing went nowhere), the backend Select was not
perceived as a dropdown, there were no visible field descriptions, the
layout was airy, and keyboard-only operation was not real (plus ctrl+s
is terminal XOFF, and a crashed run froze the UI). These tests drive the
real ``voyage.tui.VoyageApp`` via Textual's ``run_test()`` pilot and pin
the fixed behavior: autofocused multiline style editor, obvious dropdown
affordances, a focus-driven help panel, red invalid borders, a real key
map (ctrl+g / ctrl+x / b / ctrl+q), live validation, worker-failure
recovery, and confirm-before-quit. The pure state helpers stay covered in
``tests/test_tui.py``; this file covers the app shell only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep remembered TUI settings out of the real home directory.

    VoyageApp prefills from / saves to ~/.config/voyage/tui-last.toml;
    without isolation one test's save prefills the next test's form.
    """
    monkeypatch.setenv("HOME", str(tmp_path))


def _require_app() -> Any:
    """Import the Textual app (skip when the display extra is missing)."""
    pytest.importorskip("textual")
    from voyage.tui import VoyageApp

    return VoyageApp


def test_auto_focus_targets_style_field() -> None:
    VoyageApp = _require_app()
    assert VoyageApp.AUTO_FOCUS == "#field-style"


def test_style_field_has_focus_on_mount() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)):
            await asyncio.sleep(0)
            focused = app.focused
            assert isinstance(focused, TextArea)
            assert focused.id == "field-style"

    asyncio.run(_run())


def test_typing_lands_in_style_with_no_click() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            style_box = app.query_one("#field-style", TextArea)
            before = style_box.text
            await pilot.press(*"quiet harbor")
            await pilot.pause()
            assert style_box.text == before + "quiet harbor"

    asyncio.run(_run())


def test_style_editor_is_compact_multiline_textarea() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            style_box = app.query_one("#field-style", TextArea)
            assert style_box.soft_wrap is True
            assert style_box.outer_size.height == 3

    asyncio.run(_run())


def test_backend_select_is_visible_with_default_and_affordance() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Select, Static

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            backend = app.query_one("#field-backend", Select)
            assert backend.display
            assert str(backend.value) == app.initial_state.backend
            captions = [str(widget.content) for widget in app.query(Static)]
            assert any("Backend" in caption and "▾" in caption for caption in captions)

    asyncio.run(_run())


def test_backend_select_is_operable() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Select, Static

        import voyage.tui_state as tui_state_module

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#field-backend", Select).value = "fake"
            await pilot.pause()
            assert app._read_form().backend == "fake"
            reporter = getattr(tui_state_module, "gpu_warning", None)
            expected = str(reporter("fake")) if callable(reporter) else ""
            assert str(app.query_one("#gpu-warning", Static).content) == expected

    asyncio.run(_run())


def test_help_panel_describes_focused_field() -> None:
    """The right-side help panel replaces card descriptions: it always shows
    help for the currently focused field plus any validation error."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Input, Static

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.query_one("#help-body", Static)
            assert "style" in str(panel.content).lower()
            app.set_focus(app.query_one("#field-duration", Input))
            await pilot.pause()
            assert "5s" in str(panel.content)
            app.query_one("#field-duration", Input).value = "soon"
            await pilot.pause()
            assert "⚠" in str(panel.content)

    asyncio.run(_run())


def test_narrow_layout_stacks_help_below_form() -> None:
    _require_app()
    import voyage.tui as tui_module

    assert tui_module.NARROW_WIDTH == 100
    assert "#form-columns.narrow" in tui_module.VoyageApp.CSS


def test_key_hint_bar_lists_shortcuts() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Static

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            hints = str(app.query_one("#key-hints", Static).content)
            for key in ("ctrl+g", "ctrl+x", "ctrl+q", "back"):
                assert key in hints

    asyncio.run(_run())


def test_bindings_cover_generate_stop_back_quit() -> None:
    VoyageApp = _require_app()
    bound = {key: action for key, action, _ in VoyageApp.BINDINGS}
    assert bound.get("ctrl+g") == "generate"
    assert bound.get("ctrl+x") == "stop"
    assert bound.get("b") == "go_back"
    assert bound.get("ctrl+q") == "quit_app"


def test_focus_styling_in_css() -> None:
    VoyageApp = _require_app()
    css = VoyageApp.CSS
    assert "Input:focus" in css
    assert "TextArea:focus" in css
    assert "Select:focus" in css
    assert "Button:focus" in css


def test_live_validation_renders_errors_continuously() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Input, Static

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            errors_line = app.query_one("#errors-line", Static)
            assert "style" in str(errors_line.content)
            await pilot.press(*"quiet harbor")
            await pilot.pause()
            app.query_one("#field-duration", Input).value = "soon"
            await pilot.pause()
            assert "5s" in str(errors_line.content)
            app.query_one("#field-duration", Input).value = "5s"
            await pilot.pause()
            assert str(errors_line.content) == ""

    asyncio.run(_run())


def test_invalid_fields_get_red_borders() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Input

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            duration = app.query_one("#field-duration", Input)
            assert duration.has_class("field-invalid") is False
            duration.value = "soon"
            await pilot.pause()
            assert duration.has_class("field-invalid") is True
            duration.value = "5s"
            await pilot.pause()
            assert duration.has_class("field-invalid") is False

    asyncio.run(_run())


def test_generate_gates_on_errors() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.containers import ScrollableContainer
        from textual.widgets import Static

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_generate()
            await pilot.pause()
            assert "style" in str(app.query_one("#errors-line", Static).content)
            assert app.query_one("#form-view", ScrollableContainer).display

    asyncio.run(_run())


def test_keys_alone_full_flow() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.containers import ScrollableContainer

        app = VoyageApp()
        started: list[bool] = []

        def _fake_worker(_work: object, **_kwargs: object) -> None:
            started.append(True)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press(*"quiet harbor")
            await pilot.pause()
            app.run_worker = _fake_worker
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert started == [True]
            assert app._running is True
            assert not app.query_one("#form-view", ScrollableContainer).display
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert started == [True]
            await pilot.press("ctrl+x")
            await pilot.pause()
            assert any("nothing running" in line for line in app.run_history)
            app._finish_generation("done")
            await pilot.pause()
            await pilot.press("b")
            await pilot.pause()
            assert app.query_one("#form-view", ScrollableContainer).display

    asyncio.run(_run())


def test_worker_failure_restores_form_with_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze regression: a crashing generation run must return to the form
    with a visible error instead of leaving the UI stuck."""
    import asyncio

    VoyageApp = _require_app()

    def _boom(_args: object) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr("voyage.cli.cmd_generate", _boom)

    async def _run() -> None:
        from textual.containers import ScrollableContainer
        from textual.widgets import Button, Static, TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#field-style", TextArea).text = "quiet harbor"
            await pilot.pause()
            app.query_one("#button-generate", Button).scroll_visible()
            await pilot.pause()
            await pilot.click("#button-generate")
            for _ in range(60):
                await pilot.pause(0.5)
                if not app._running:
                    break
            assert not app._running
            assert app.query_one("#form-view", ScrollableContainer).display
            assert "boom" in str(app.query_one("#errors-line", Static).content)

    asyncio.run(_run())


def test_confirm_before_quit_arms_then_exits() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Static

        app = VoyageApp()
        exits: list[bool] = []

        def _record_exit(*_args: object, **_kwargs: object) -> None:
            exits.append(True)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.exit = _record_exit
            app._running = True
            app.action_quit_app()
            await pilot.pause()
            assert exits == []
            result = str(app.query_one("#run-result", Static).content)
            assert "press Quit again" in result
            app.action_quit_app()
            await pilot.pause()
            assert exits == [True]
            app._running = False
            app._quit_armed = False
            app.action_quit_app()
            assert exits == [True, True]

    asyncio.run(_run())


def test_initial_state_prefills_widgets() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Input, Select, TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.query_one("#field-style", TextArea).text == app.initial_state.style
            assert app.query_one("#field-name", Input).value == app.initial_state.name
            assert str(app.query_one("#field-backend", Select).value) == app.initial_state.backend

    asyncio.run(_run())


def test_gpu_warning_line_present() -> None:
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Static

        import voyage.tui_state as tui_state_module

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            reporter = getattr(tui_state_module, "gpu_warning", None)
            if callable(reporter):
                expected = str(reporter(app.initial_state.backend))
            else:
                expected = ""
            assert str(app.query_one("#gpu-warning", Static).content) == expected

    asyncio.run(_run())
