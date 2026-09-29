"""Launcher TUI tests: form state/validation/namespace/plan (CPU-only).

Covers the pure ``voyage.tui_state`` helpers plus the bare-command
wiring in ``voyage.cli`` (launcher monkeypatched — never starts Textual)
and the Textual app structure (skipped when Textual is not installed, so
CPU-only images without the display extra still gate green).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import cli
from voyage.config import Unset
from voyage.tui_state import (
    GenerateFormState,
    plan_summary,
    to_generate_namespace,
    validate,
)


def _valid_state() -> GenerateFormState:
    return GenerateFormState(style="pastel neon line-art, peaceful")


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the e2e run's remembered settings out of the real home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))


def test_defaults_match_generate_effective_settings() -> None:
    state = GenerateFormState(style="x")
    assert state.backend == "ltxv"
    assert state.director == "qwen"
    assert state.quantization == "fp8"
    assert state.name == "voyage"
    assert state.duration == "5s"


def test_empty_style_is_invalid() -> None:
    assert any("style" in error for error in validate(GenerateFormState()))


def test_bad_duration_reports_examples() -> None:
    errors = validate(GenerateFormState(style="x", duration="soon"))
    assert any("5s" in error for error in errors)


def test_bad_numbers_report_field_names() -> None:
    state = GenerateFormState(style="x", blocks="0", seed="abc", drift_every_n="-2")
    errors = validate(state)
    assert any("blocks" in error for error in errors)
    assert any("seed" in error for error in errors)
    assert any("drift-every-n" in error for error in errors)


def test_unknown_choices_rejected() -> None:
    state = GenerateFormState(style="x", backend="nope", director="nope")
    errors = validate(state)
    assert any("backend" in error for error in errors)
    assert any("director" in error for error in errors)


def test_namespace_maps_empty_optionals_to_unset() -> None:
    """Blank TUI optionals emit Unset (issue 045), never None."""
    namespace = to_generate_namespace(_valid_state())
    assert namespace.duration == pytest.approx(5.0)
    assert namespace.run_id == "voyage"
    assert namespace.blocks is Unset
    assert namespace.take_seconds is Unset
    assert namespace.beats_per_segment is Unset
    assert namespace.drift_every_n is Unset
    assert namespace.seed == 0


def test_namespace_derives_output_from_name() -> None:
    namespace = to_generate_namespace(_valid_state())
    assert namespace.output == "output/voyage"
    assert namespace.final_video == "output/voyage/final.mp4"


def test_namespace_carries_explicit_values() -> None:
    state = _valid_state()
    state.blocks = "2"
    state.take_seconds = "30"
    state.name = "my-run"
    namespace = to_generate_namespace(state)
    assert namespace.blocks == 2
    assert namespace.take_seconds == pytest.approx(30.0)
    assert namespace.run_id == "my-run"
    assert namespace.output == "output/my-run"


def test_namespace_raises_on_invalid() -> None:
    with pytest.raises(ValueError, match="style"):
        to_generate_namespace(GenerateFormState())


def test_causvid_backend_is_accepted() -> None:
    state = _valid_state()
    state.backend = "causvid"
    assert validate(state) == []


def test_plan_summary_causvid_reports_72f_at_16fps() -> None:
    state = _valid_state()
    state.backend = "causvid"
    summary = plan_summary(state)
    assert "causvid" in summary
    assert "72f/segment" in summary
    assert "16fps" in summary


def test_plan_summary_default_is_five_seconds() -> None:
    summary = plan_summary(_valid_state())
    assert "2 segment(s)" in summary
    assert "192 frames" in summary
    assert "96f/segment" in summary
    assert "ltxv" in summary


def test_plan_summary_reports_block_math() -> None:
    state = _valid_state()
    state.blocks = "2"
    assert "192 frames" in plan_summary(state)
    state.blocks = "1"
    assert "192 frames" in plan_summary(state)


def test_plan_summary_longlive2_reports_29f_at_24fps() -> None:
    state = _valid_state()
    state.backend = "longlive2"
    summary = plan_summary(state)
    assert "longlive2" in summary
    assert "29f/segment" in summary
    assert "24fps" in summary


def test_plan_summary_reports_bad_input() -> None:
    state = GenerateFormState(style="x", duration="soon")
    assert plan_summary(state).startswith("cannot plan:")


def test_bare_command_launches_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []

    def _fake_launch() -> int:
        calls.append(None)
        return 0

    monkeypatch.setattr(cli, "launch_tui", _fake_launch)
    assert cli.main([]) == 0
    assert len(calls) == 1


def test_parser_has_no_required_command() -> None:
    args = cli.build_parser().parse_args([])
    assert args.command is None


def test_generate_end_to_end_fake_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Headless TUI run: fake backend, 1s video, deterministic director.

    Drives the real app (Pilot): fills the form, presses Generate, waits
    for the worker, then asserts the run log shows the segment lifecycle
    and the result line names the final video. CPU-only; the run lands in
    an isolated working directory (Name is a flat folder name, so the
    export root resolves under the cwd), never the repo tree.
    """
    pytest.importorskip("textual")
    import asyncio

    from textual.widgets import Button, Input, Select, Static, TextArea

    from voyage.tui import VoyageApp

    monkeypatch.chdir(tmp_path)

    async def _run() -> None:
        app = VoyageApp()
        async with app.run_test(size=(120, 60)) as pilot:
            app.query_one("#field-style", TextArea).text = "pastel neon line-art, peaceful"
            app.query_one("#field-duration", Input).value = "1s"
            app.query_one("#field-name", Input).value = "tui-e2e"
            app.query_one("#field-backend", Select).value = "fake"
            app.query_one("#field-director", Select).value = "deterministic"
            await pilot.pause()
            app.query_one("#button-generate", Button).scroll_visible()
            await pilot.pause()
            await pilot.click("#button-generate")
            for _ in range(240):
                await pilot.pause(1.0)
                if not app._generation_running:
                    break
            assert not app._generation_running, "generation worker did not finish in time"
            assert len(app.run_history) >= 3
            assert any("SEGMENT" in line for line in app.run_history)
            result = app.query_one("#run-result", Static)
            assert "generated" in str(result.content)

    asyncio.run(_run())
    assert (tmp_path / "output" / "tui-e2e" / "final.mp4").exists()


def test_app_structure_matches_form_fields() -> None:
    pytest.importorskip("textual")
    import asyncio

    from voyage.tui import VoyageApp

    async def _check() -> None:
        app = VoyageApp()
        async with app.run_test(size=(120, 40)):
            for field_id in (
                "field-backend",
                "field-duration",
                "field-style",
                "field-name",
                "field-seed",
                "field-director",
                "field-quantization",
                "button-generate",
                "help-panel",
                "run-log",
            ):
                assert app.query_one(f"#{field_id}") is not None
            assert isinstance(app.initial_state, GenerateFormState)

    asyncio.run(_check())


def test_stop_with_corrupt_state_reports_feedback(tmp_path: Path) -> None:
    """Stop-button guard: a corrupt state.json surfaces a feedback line.

    Drives the real app (Pilot): with a run marked running but its
    state.json holding garbage, _request_stop must append a "cannot
    stop" line instead of raising out of the handler (issue 078).
    """
    pytest.importorskip("textual")
    import asyncio

    from voyage.tui import VoyageApp

    async def _run() -> None:
        app = VoyageApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            run_dir = tmp_path / "output" / "stuck-run"
            run_dir.mkdir(parents=True)
            (run_dir / "state.json").write_text("{bad json", encoding="utf-8")
            app._run_dir = run_dir
            app._generation_running = True
            app._request_stop()
            await pilot.pause()
            assert any("cannot stop" in line for line in app.run_history)
            assert app._generation_running

    asyncio.run(_run())
