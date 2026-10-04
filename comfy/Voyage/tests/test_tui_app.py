"""Launcher app behavior tests: focus, dropdown, help panel, keys (headless Pilot).

Why this module exists: users reported the form felt uneditable (nothing
took focus on mount so typing went nowhere), the backend Select was not
perceived as a dropdown, there were no visible field descriptions, the
layout was airy, and keyboard-only operation was not real (plus ctrl+s
is terminal XOFF, and a crashed run froze the UI). These tests drive the
real ``voyage.tui.VoyageApp`` via Textual's ``run_test()`` pilot and pin
the fixed behavior: autofocused multiline style editor, obvious dropdown
affordances, a focus-driven help panel, invalid highlighting, a real key
map (ctrl+g / ctrl+x / b / ctrl+q), live validation, worker-failure
recovery, and confirm-before-quit. The pure state helpers stay covered in
``tests/test_tui.py``; this file covers the app shell only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow
"""Slow tail (issue 089): 33 Pilot tests (~34 s) — deselect with `-m "not slow"`."""


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


async def _click_generate_when_ready(pilot: Any, app: Any) -> None:
    """Click Generate once it is really under the mouse (issue 093).

    `scroll_visible` only schedules scrolling and a single `pause()` is
    not enough under load; `pilot.click` returns False (instead of
    raising) when the click lands on another widget, and every caller
    used to ignore that — a lost click looks exactly like a hung run.
    Poll until the click lands. OutOfBounds (target still off-screen
    while the scheduled scroll applies — e.g. after form-growth changes
    in tui.py) retries the same way instead of failing the first poll.
    """
    from textual.pilot import OutOfBounds
    from textual.widgets import Button

    for _ in range(100):
        app.query_one("#button-generate", Button).scroll_visible()
        await pilot.pause(0.05)
        try:
            if await pilot.click("#button-generate"):
                return
        except OutOfBounds:
            continue
    raise AssertionError("generate button never became clickable")


_STARTUP_WAIT_ITERATIONS = 600
"""Polls for worker-startup gates (200 × 0.05 s = 10 s was too tight when
the box is shared: thread/event-loop scheduling under load starves the
10 s budget with no code fault — issue 093. 30 s still fails loudly on a
truly dead worker. Finish-waits keep their own budgets below.)"""


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
    """The backend dropdown operates via the keyboard overlay (issue 131).

    Focus + enter opens the Select overlay, down + enter commits the next
    option, and the form reader plus the GPU-warning line follow the
    committed value — the overlay open/commit path the old programmatic
    `.value` assignment wrote past.
    """
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Select, Static

        import voyage.tui_state as tui_state_module

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            backend = app.query_one("#field-backend", Select)
            initial = str(backend.value)
            app.set_focus(backend)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert backend.expanded
            await pilot.press("down")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert not backend.expanded
            committed = str(backend.value)
            assert committed != initial
            assert app._read_form().backend == committed
            reporter = getattr(tui_state_module, "gpu_warning", None)
            expected = str(reporter(committed)) if callable(reporter) else ""
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


def test_invalid_fields_get_flagged() -> None:
    """Invalid fields carry the field-invalid class (red-tinted background)."""
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
            assert app._generation_running is True
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

    monkeypatch.setattr("voyage.cli.cmd_configure", lambda _args: 0)
    monkeypatch.setattr("voyage.cli.cmd_generate", _boom)

    async def _run() -> None:
        from textual.containers import ScrollableContainer
        from textual.widgets import Static, TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#field-style", TextArea).text = "quiet harbor"
            await pilot.pause()
            await _click_generate_when_ready(pilot, app)
            errors = app.query_one("#errors-line", Static)
            for _ in range(_STARTUP_WAIT_ITERATIONS):
                if app._generation_running or "boom" in str(errors.content):
                    break
                await pilot.pause(0.05)
            assert app._generation_running or "boom" in str(errors.content)
            for _ in range(60):
                await pilot.pause(0.5)
                if not app._generation_running:
                    break
            assert not app._generation_running
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
            app._generation_running = True
            app.action_quit_app()
            await pilot.pause()
            assert exits == []
            result = str(app.query_one("#run-result", Static).content)
            assert "press Quit again" in result
            app.action_quit_app()
            await pilot.pause()
            assert exits == [True]
            app._generation_running = False
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


def test_app_title_heads_the_form() -> None:
    """Hero title: massive block-letter VOYAGE, full-width, italic subtitle."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Static

        from voyage.tui import TITLE_ART

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            title = app.query_one("#app-title", Static)
            assert str(title.render()).splitlines()[0] == TITLE_ART.splitlines()[0]
            # Full-width above the form (a screen child before #form-view,
            # not nested inside a column).
            assert title.parent is app.screen
            form_view = app.query_one("#form-view")
            assert list(app.screen.children).index(title) < list(app.screen.children).index(
                form_view
            )
            assert title.styles.text_align == "center"
            subtitle = app.query_one("#app-subtitle", Static)
            assert subtitle.styles.text_style.italic is True

    asyncio.run(_run())


def test_field_rows_are_compact_and_constant() -> None:
    """Compact form: text rows are 1 line; the style editor and the
    bordered dropdowns are 3 lines (a dropdown cannot be 1 line — an
    explicit height blanks its value line on Textual 8, and its closed
    value line needs its border chrome)."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from voyage.tui import FIELD_WIDGET_IDS

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for field, widget_id in FIELD_WIDGET_IDS.items():
                widget = app.query_one(f"#{widget_id}")
                height = widget.outer_size.height
                if field in ("style", "backend", "director", "quantization"):
                    assert height == 3, (field, height)
                else:
                    assert height == 1, (field, height)

    asyncio.run(_run())


def test_focus_and_invalid_keep_geometry_constant() -> None:
    """Focusing a field (or flagging it invalid) must not move any row.

    Users reported fields like backend changing the vertical spacing on
    focus; rows and widgets keep identical outer heights in every state.
    """
    import asyncio

    VoyageApp = _require_app()

    async def _snapshot(app: Any) -> dict[str, int]:
        rows = app.query(".field-row")
        sizes = {}
        for index, row in enumerate(rows):
            widget = row.query("*").last()
            sizes[f"row:{index}"] = row.outer_size.height
            sizes[f"widget:{widget.id or index}"] = widget.outer_size.height
        return sizes

    async def _run() -> None:
        from voyage.tui import FIELD_WIDGET_IDS

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            baseline = await _snapshot(app)
            focusables = [f"#{wid}" for wid in FIELD_WIDGET_IDS.values()]
            focusables += ["#button-generate", "#button-quit"]
            for selector in focusables:
                app.set_focus(app.query_one(selector))
                await pilot.pause()
                assert await _snapshot(app) == baseline, selector
            for widget_id in FIELD_WIDGET_IDS.values():
                widget = app.query_one(f"#{widget_id}")
                widget.add_class("field-invalid")
                await pilot.pause()
                assert await _snapshot(app) == baseline, widget_id
                widget.remove_class("field-invalid")
            await pilot.pause()
            assert await _snapshot(app) == baseline

    asyncio.run(_run())


def test_empty_notice_lines_take_no_space() -> None:
    """Empty #errors-line / #gpu-warning are hidden (display False)."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Select, Static, TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            errors = app.query_one("#errors-line", Static)
            assert errors.display is True  # style is required and empty by default
            app.query_one("#field-style", TextArea).text = "calm neon harbor"
            app._refresh_plan_and_errors()
            await pilot.pause()
            assert str(errors.content) == ""
            assert errors.display is False
            warning = app.query_one("#gpu-warning", Static)
            app.query_one("#field-backend", Select).value = "fake"
            app._refresh_gpu_warning()
            await pilot.pause()
            assert str(warning.content) == ""
            assert warning.display is False

    asyncio.run(_run())


def test_title_reserves_full_art_height() -> None:
    """Title clipping: #app-title must fit all 5 art rows plus its top pad."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Static

        from voyage.tui import TITLE_ART

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            title = app.query_one("#app-title", Static)
            assert title.region.height >= len(TITLE_ART.splitlines()) + 1

    asyncio.run(_run())


def test_form_view_constrains_height_and_scrolls() -> None:
    """Clipped fields: on a short terminal the form must scroll so every
    field (e.g. backend) can be brought fully into view."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.containers import ScrollableContainer
        from textual.widgets import Select

        app = VoyageApp()
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.pause()
            form_view = app.query_one("#form-view", ScrollableContainer)
            virtual = form_view.virtual_size.height
            visible = form_view.scrollable_content_region.height
            assert visible < virtual  # constrained: scrolling has somewhere to go
            backend = app.query_one("#field-backend", Select)
            backend.scroll_visible()
            await pilot.pause()
            outer = form_view.region
            inner = backend.region
            assert outer.x <= inner.x and outer.y <= inner.y
            assert outer.right >= inner.right and outer.bottom >= inner.bottom

    asyncio.run(_run())


def test_generate_button_runs_real_fake_backend_to_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Generate-button freeze: clicking Generate with a real (fake-backend)
    run must reach the run view and finish with a success line."""
    import asyncio

    VoyageApp = _require_app()
    monkeypatch.chdir(tmp_path)

    async def _run() -> None:
        from textual.containers import ScrollableContainer
        from textual.widgets import Select, Static, TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#field-style", TextArea).text = "calm neon harbor"
            app.query_one("#field-backend", Select).value = "fake"
            await pilot.pause()
            await _click_generate_when_ready(pilot, app)
            for _ in range(120):
                await pilot.pause(0.5)
                if not app._generation_running:
                    break
            assert not app._generation_running
            assert app.query_one("#run-view").display
            result = str(app.query_one("#run-result", Static).content)
            assert result.startswith("✓ generated"), result
            assert app.query_one("#form-view", ScrollableContainer).display is False

    asyncio.run(_run())


def test_base_exception_in_worker_restores_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker robustness: even a BaseException (e.g. SystemExit from a CLI
    fast-fail) must return to the form with an error, never a stuck view."""
    import asyncio

    VoyageApp = _require_app()

    def _quit(_args: object) -> int:
        raise SystemExit(2)

    monkeypatch.setattr("voyage.cli.cmd_configure", lambda _args: 0)
    monkeypatch.setattr("voyage.cli.cmd_generate", _quit)

    async def _run() -> None:
        from textual.containers import ScrollableContainer
        from textual.widgets import Static, TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#field-style", TextArea).text = "calm neon harbor"
            await pilot.pause()
            await _click_generate_when_ready(pilot, app)
            for _ in range(60):
                await pilot.pause(0.5)
                if not app._generation_running:
                    break
            assert not app._generation_running
            assert app.query_one("#form-view", ScrollableContainer).display
            assert "2" in str(app.query_one("#errors-line", Static).content)

    asyncio.run(_run())


def _form_svg_rows(app: Any, tmp_path: Path) -> list[str]:
    """Render the running app to SVG and return its text rows (stripped)."""
    import re

    path = tmp_path / "form.svg"
    app.save_screenshot(str(path))
    rows: list[str] = []
    for match in re.findall(r"<text[^>]*>(.*?)</text>", path.read_text()):
        rows.append(re.sub(r"<[^>]+>", "", match).replace("&#160;", " "))
    return rows


def test_select_values_render_in_form_text(tmp_path: Path) -> None:
    """Dropdown values must be visible text, not blank clipped boxes."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        app = VoyageApp()
        async with app.run_test(size=(120, 60)) as pilot:
            # Poll for the render (issue 130): a single 0.3 s pause is not
            # enough under load — the three Select widgets may not have
            # finished their first paint. Same poll-until-landed shape as
            # _click_generate_when_ready above; the final asserts still fail
            # loudly on a truly blank render.
            text = ""
            for _ in range(25):
                await pilot.pause(0.2)
                text = "\n".join(_form_svg_rows(app, tmp_path))
                if "ltx25" in text and "llama" in text and "fp8" in text:
                    break
            assert "ltx25" in text
            assert "llama" in text
            assert "fp8" in text

    asyncio.run(_run())


def test_select_rows_reserve_full_widget_height() -> None:
    """Dropdown rows claim the full bordered-widget height (no vertical clip)."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.widgets import Select

        app = VoyageApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            for wid in ("#field-backend", "#field-director", "#field-quantization"):
                assert app.query_one(wid, Select).region.height >= 3

    asyncio.run(_run())


def test_slow_run_stays_responsive_and_reaches_monitoring_view(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Generate with a slow backend: the run view appears and the event loop
    keeps processing input mid-run (Tab moves focus), then completion lands.
    """
    import asyncio
    import threading

    VoyageApp = _require_app()
    monkeypatch.chdir(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def _slow_generate(_namespace: object) -> int:
        started.set()
        release.wait(30)
        return 0

    monkeypatch.setattr("voyage.cli.cmd_configure", lambda _args: 0)
    monkeypatch.setattr("voyage.cli.cmd_generate", _slow_generate)

    async def _run() -> None:
        from textual.containers import ScrollableContainer, Vertical
        from textual.widgets import Static, TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#field-style", TextArea).text = "slow harbor"
            before = app.focused
            await _click_generate_when_ready(pilot, app)
            for _ in range(_STARTUP_WAIT_ITERATIONS):
                if started.is_set():
                    break
                await pilot.pause(0.05)
            assert started.is_set()
            assert app.query_one("#run-view", Vertical).display
            assert not app.query_one("#form-view", ScrollableContainer).display
            await pilot.press("tab")
            await pilot.pause(0.2)
            assert app.focused is not before
            release.set()
            for _ in range(200):
                if not app._generation_running:
                    break
                await pilot.pause(0.05)
            assert not app._generation_running
            assert "generated" in str(app.query_one("#run-result", Static).content)

    asyncio.run(_run())


def test_mid_run_progress_reaches_log_before_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Progress posted mid-run (segment lines) lands in the run history
    while the worker is still busy — the log is live, not end-flushed."""
    import asyncio
    import threading

    VoyageApp = _require_app()
    monkeypatch.chdir(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def _posting_generate(namespace: object) -> int:
        sink = getattr(namespace, "progress_sink", None)
        entered.set()
        if sink is not None:
            sink.segment_start(1, "000001")
        release.wait(30)
        return 0

    monkeypatch.setattr("voyage.cli.cmd_configure", lambda _args: 0)
    monkeypatch.setattr("voyage.cli.cmd_generate", _posting_generate)

    async def _run() -> None:
        from textual.widgets import TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#field-style", TextArea).text = "posting harbor"
            await _click_generate_when_ready(pilot, app)
            for _ in range(_STARTUP_WAIT_ITERATIONS):
                if entered.is_set():
                    break
                await pilot.pause(0.05)
            assert entered.is_set()
            for _ in range(200):
                if any("000001" in line for line in app.run_history):
                    break
                await pilot.pause(0.05)
            assert any("000001" in line for line in app.run_history)
            release.set()
            for _ in range(200):
                if not app._generation_running:
                    break
                await pilot.pause(0.05)
            assert not app._generation_running

    asyncio.run(_run())


def test_cuda_fast_fail_returns_to_form_with_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default CUDA backend without torch: fast visible error on the form,
    never a stuck run view."""
    import asyncio

    VoyageApp = _require_app()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("voyage.cli._torch_available", lambda: False)
    monkeypatch.setattr("voyage.models_ensure.ensure_models", lambda *a, **k: 0)

    async def _run() -> None:
        from textual.containers import ScrollableContainer
        from textual.widgets import Static, TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#field-style", TextArea).text = "nucuda harbor"
            await _click_generate_when_ready(pilot, app)
            errors = app.query_one("#errors-line", Static)
            for _ in range(_STARTUP_WAIT_ITERATIONS):
                if app._generation_running or errors.display:
                    break
                await pilot.pause(0.05)
            assert app._generation_running or errors.display
            for _ in range(200):
                if not app._generation_running:
                    break
                await pilot.pause(0.05)
            assert not app._generation_running
            assert app.query_one("#form-view", ScrollableContainer).display
            assert "CUDA" in str(app.query_one("#errors-line", Static).content)

    asyncio.run(_run())


def test_run_head_ticks_elapsed_while_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The run head shows a synchronous starting line plus a ticking
    elapsed timer — visible proof the UI is alive during silent stretches."""
    import asyncio
    import threading

    VoyageApp = _require_app()
    monkeypatch.chdir(tmp_path)
    release = threading.Event()

    def _slow_generate(_namespace: object) -> int:
        release.wait(30)
        return 0

    monkeypatch.setattr("voyage.cli.cmd_configure", lambda _args: 0)
    monkeypatch.setattr("voyage.cli.cmd_generate", _slow_generate)

    async def _run() -> None:
        from textual.widgets import Static, TextArea

        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#field-style", TextArea).text = "ticking harbor"
            await _click_generate_when_ready(pilot, app)
            for _ in range(_STARTUP_WAIT_ITERATIONS):
                if app._generation_running:
                    break
                await pilot.pause(0.05)
            assert app._generation_running
            assert any("starting" in line for line in app.run_history)
            # Poll for the heartbeat tick (issue 130): the head ticks on a
            # 1.0 s set_interval, so one 2.5 s pause allows barely two ticks
            # with zero scheduling slack — a starved interval under load
            # fails the test with no retry. Poll up to ~10 s instead; the
            # final assert still fails loudly when the ticker is dead.
            head = ""
            for _ in range(20):
                await pilot.pause(0.5)
                head = str(app.query_one("#run-head", Static).content)
                if "elapsed" in head:
                    break
            assert "elapsed" in head
            release.set()
            for _ in range(200):
                if not app._generation_running:
                    break
                await pilot.pause(0.05)
            assert not app._generation_running
            assert app._heartbeat is None

    asyncio.run(_run())
