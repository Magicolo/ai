"""Launcher TUI tests: form state/validation/namespace/plan (CPU-only).

Covers the pure ``voyage.tui_state`` helpers plus the bare-command
wiring in ``voyage.cli`` (launcher monkeypatched — never starts Textual)
and the Textual app structure (skipped when Textual is not installed, so
CPU-only images without the display extra still gate green).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import pytest

from voyage import cli
from voyage.config import AugmentConfig, ProjectConfig, Unset, VideoConfig, resolve_config
from voyage.console import VoyageConsole
from voyage.tui_state import (
    GenerateFormState,
    _default_settings_path,
    _flat_folder_name,
    field_errors,
    gpu_warning,
    load_last_settings,
    plan_counts,
    plan_summary,
    save_last_settings,
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
    assert state.director == "llama"
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


@pytest.mark.parametrize("backend", ["ltx25", "ltx23"])
def test_ltx_backend_is_accepted(backend: str) -> None:
    state = _valid_state()
    state.backend = backend
    assert validate(state) == []


@pytest.mark.parametrize("backend", ["ltx25", "ltx23"])
def test_plan_summary_ltx_reports_96f_at_24fps(backend: str) -> None:
    state = _valid_state()
    state.backend = backend
    summary = plan_summary(state)
    assert backend in summary
    assert "96f/segment" in summary
    assert "24fps" in summary


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


@pytest.mark.slow
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


# --- 088 fold: tests/test_tui_absent_defaults_023.py (4 tests) ---
# """TUI untouched-form inheritance (issue 023).
#
# The CLI spells absence as ``None`` (stored TOML wins) but the TUI form
# prefills ``quantization``/``min_fps``/``min_resolution`` with concrete
# defaults, so pressing Generate on an untouched form silently reverted a
# run customized to ``bf16``/``60``/``1920x1080``. These tests pin the
# absent-encoding: untouched-at-default fields emit ``Unset`` and resolve
# to the stored config unchanged, while explicit non-defaults still win.
# """
# NOTE: the source file carries no home fixture; the moved tests now run
# under this file's autouse `_isolated_home` (HOME→tmp_path) — a no-op for
# them (pure namespace/resolve transforms, never read HOME).


def _customized_base() -> ProjectConfig:
    """Stored config differing from every TUI form default."""
    base = ProjectConfig(style="inherit-probe")
    video = VideoConfig(**{**base.video.model_dump(), "quantization": "bf16"})
    augment = AugmentConfig(min_fps=60, min_width=1920, min_height=1080)
    return base.model_copy(update={"video": video, "augment": augment})


def test_untouched_form_emits_unset_for_quantization_and_floors() -> None:
    """Default-valued TUI fields encode absence, never explicit values."""
    namespace = to_generate_namespace(GenerateFormState(style="x", name="probe"))
    assert namespace.quantization is Unset
    assert namespace.min_fps is Unset
    assert namespace.min_resolution is Unset


def test_untouched_form_preserves_stored_quantization_and_floors() -> None:
    """An untouched form resolves to the stored config unchanged (023)."""
    namespace = to_generate_namespace(GenerateFormState(style="x", name="probe"))
    resolved = resolve_config(
        _customized_base(),
        quantization=namespace.quantization,
        min_fps=namespace.min_fps,
        min_resolution=namespace.min_resolution,
    )
    assert resolved.video.quantization == "bf16"
    assert (resolved.augment.min_fps, resolved.augment.min_width, resolved.augment.min_height) == (
        60,
        1920,
        1080,
    )


def test_explicit_non_default_tui_values_still_win() -> None:
    """Deliberate TUI choices override the stored config as before."""
    state = GenerateFormState(
        style="x",
        name="probe",
        quantization="bf16",
        min_fps="60",
        min_resolution="1920x1080",
    )
    namespace = to_generate_namespace(state)
    assert namespace.quantization == "bf16"
    assert namespace.min_fps == 60
    assert namespace.min_resolution == "1920x1080"
    resolved = resolve_config(
        ProjectConfig(style="probe"),
        quantization=namespace.quantization,
        min_fps=namespace.min_fps,
        min_resolution=namespace.min_resolution,
    )
    assert resolved.video.quantization == "bf16"
    assert (resolved.augment.min_fps, resolved.augment.min_width, resolved.augment.min_height) == (
        60,
        1920,
        1080,
    )


def test_blank_tui_floors_still_emit_unset() -> None:
    """Actively cleared floor fields keep the pre-existing blank→Unset path."""
    namespace = to_generate_namespace(
        GenerateFormState(style="x", name="probe", min_fps="", min_resolution="")
    )
    assert namespace.min_fps is Unset
    assert namespace.min_resolution is Unset


# --- 088 fold: tests/test_tui_checkbox_help_114.py (3 tests) ---
# """TUI checkbox help coverage (issue 114, TDD red-first).
#
# Why this file exists: the five flag checkboxes (draft/force/skip-bad/
# verbose/no-color) have no tooltip, no FIELD_HELP entry, and no widget-id
# mapping, so the focus-driven help panel falls back to the overview on
# exactly the flags that need one sentence each. The batch-8 boxes
# (no_download/no_sfx) already follow the help pattern and must keep
# working — these tests pin both the new keys and the pre-existing ones.
# """
# NOTE: source `_isolated_home` (docstring "remembered TUI settings") is
# behavior-identical to this file's fixture — reused, not duplicated;
# `_require_app` is byte-identical to the 181 file's — kept once here.


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


# --- 088 fold: tests/test_tui_checkbox_toggle_181.py (2 tests) ---
# """Widget-to-state coverage for TUI flag checkboxes (issue 181, TDD red-first).
#
# Why this file exists: ``test_tui_state.py`` covers state-to-namespace
# (given ``force=True`` the namespace carries it) but nothing covers
# widget-to-state (given the user checks "Force", ``_read_form().force``
# becomes ``True``) — a transposed flag id in ``_read_form`` passes the
# whole suite. These tests toggle each box through the real Pilot click
# path (the issue-131 interaction precedent — never programmatic
# ``.value`` sets) and assert ``_read_form`` follows, plus a count/ids pin
# so a deleted or renamed flag fails loudly.
# """
# NOTE: `_require_app` kept once in the 114 section above; this section's
# `_isolated_home` is behavior-identical to this file's fixture — reused.


_FLAG_FIELDS = (
    ("flag-draft", "draft"),
    ("flag-force", "force"),
    ("flag-skip-bad", "skip_bad"),
    ("flag-use-model-pass", "use_model_pass"),
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
        "flag-use-model-pass",
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
    """Exactly the eight known flag ids exist (renames/deletes fail)."""
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
        defaults = GenerateFormState()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            for widget_id, field in _FLAG_FIELDS:
                # Initial state follows the form defaults (use_model_pass
                # defaults on per DESIGN §140 GPU defaults, the rest off).
                initial = getattr(defaults, field)
                assert getattr(app._read_form(), field) is initial, widget_id
                await _click_when_ready(pilot, app, f"#{widget_id}")
                await pilot.pause()
                assert getattr(app._read_form(), field) is (not initial), widget_id
                await _click_when_ready(pilot, app, f"#{widget_id}")
                await pilot.pause()
                assert getattr(app._read_form(), field) is initial, widget_id

    asyncio.run(_run())


# --- 088 fold: tests/test_tui_progress_parity_113.py (5 tests) ---
# """TUI progress content parity with the console (issue 113, TDD red-first).
#
# Why this file exists: ``TuiProgress`` docstring promises plain-text lines
# mirror the console wording, but ``segment_plan``/``segment_done`` drop the
# decision-relevant lines (backend labels, drift hold, video geometry,
# energy, take ids + action, prefetch, elapsed total). These tests drive one
# canned plan/done pair through both sinks and assert token parity on the
# non-verbose subset. Verbose-gated detail (seeds/transitions/texture/notes)
# stays console-only by contract. Caption-family lines (161) are out of
# scope here — see ``tests/test_sfx_caption_render_161.py``.
# """


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


# --- 088 fold: tests/test_tui_state.py (26 tests, verbatim) ---
# Original module docstring (banner, issue ID stays greppable):
# """Last-settings persistence + GPU warning tests (pure, CPU-only, no Textual).
#
# Covers the ``voyage.tui_state`` prefill contract Stream B consumes:
# save/load round-trip, missing/corrupt files, per-field fallback, silent
# save failures, and the per-backend GPU notice.
# """
# NOTE: the target's autouse `_isolated_home` fixture applies to the moved
# tests as a no-op for HOME-independent legs and as the intended isolation
# for last-settings legs (they never touch the real home directory).


def _filled_state() -> GenerateFormState:
    return GenerateFormState(
        backend="fake",
        duration="1m30s",
        style='neon "glow"\nline art',
        name="test-run",
        seed="42",
        force=True,
        skip_bad=True,
        draft=True,
        director="deterministic",
        blocks="3",
        take_seconds="30",
        quantization="bf16",
        beats_per_segment="8",
        drift_every_n="2",
        verbose=True,
        no_color=True,
    )


def test_last_settings_path_default() -> None:
    assert _default_settings_path() == Path.home() / ".config" / "voyage" / "tui-last.toml"


def test_save_load_round_trip(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    state = _filled_state()
    save_last_settings(state, settings_file)
    assert load_last_settings(settings_file) == state


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    settings_file = tmp_path / "nested" / "dir" / "tui-last.toml"
    save_last_settings(_filled_state(), settings_file)
    assert settings_file.exists()
    assert load_last_settings(settings_file) == _filled_state()


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert load_last_settings(tmp_path / "absent.toml") == GenerateFormState()


def test_load_corrupt_toml_returns_defaults(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text("backend = [unclosed\n", encoding="utf-8")
    assert load_last_settings(settings_file) == GenerateFormState()


def test_load_wrong_types_fall_back_while_valid_fields_survive(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text(
        'backend = 123\nforce = "yes"\nstyle = "kept style"\nname = "kept-run"\n',
        encoding="utf-8",
    )
    loaded = load_last_settings(settings_file)
    defaults = GenerateFormState()
    assert loaded.backend == defaults.backend
    assert loaded.force is False
    assert loaded.style == "kept style"
    assert loaded.name == "kept-run"


def test_load_legacy_run_id_migrates_to_name(tmp_path: Path) -> None:
    """Pre-rework settings files stored run_id; they prefill Name now."""
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text('run_id = "old-run"\nstyle = "kept style"\n', encoding="utf-8")
    loaded = load_last_settings(settings_file)
    assert loaded.name == "old-run"
    settings_file.write_text('run_id = "old-run"\nname = "new-name"\n', encoding="utf-8")
    assert load_last_settings(settings_file).name == "new-name"


def test_load_invalid_choices_fall_back_to_defaults(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text(
        'backend = "nope"\ndirector = "nope"\nquantization = "int8"\nduration = "9s"\n',
        encoding="utf-8",
    )
    loaded = load_last_settings(settings_file)
    defaults = GenerateFormState()
    assert loaded.backend == defaults.backend
    assert loaded.director == defaults.director
    assert loaded.quantization == defaults.quantization
    assert loaded.duration == "9s"


def test_load_ignores_unknown_keys(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text('future_option = "x"\nstyle = "kept style"\n', encoding="utf-8")
    loaded = load_last_settings(settings_file)
    assert loaded.style == "kept style"
    assert loaded == GenerateFormState(style="kept style")


def test_save_failure_never_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    save_last_settings(GenerateFormState(style="x"), blocker / "tui-last.toml")


def test_gpu_warning_empty_for_fake_and_unknown() -> None:
    assert gpu_warning("fake") == ""
    assert gpu_warning("nope") == ""


@pytest.mark.parametrize("backend", ["ltxv", "causvid"])
def test_gpu_warning_names_image_and_gpu_flag(backend: str) -> None:
    warning = gpu_warning(backend)
    assert warning
    assert "\n" not in warning
    assert "VOYAGE_IMAGE" in warning
    assert "voyage-video" in warning
    assert "--gpus" in warning


@pytest.mark.parametrize("backend", ["ltx25", "ltx23"])
def test_gpu_warning_names_ltx_image(backend: str) -> None:
    warning = gpu_warning(backend)
    assert warning
    assert "\n" not in warning
    assert "voyage-ltx" in warning
    assert "--gpus" in warning


def test_plan_counts_causvid_uses_72_novel_per_block_at_16fps() -> None:
    state = GenerateFormState(style="x", backend="causvid", duration="5s")
    assert plan_counts(state) == (2, 144, pytest.approx(9.0))  # type: ignore[comparison-overlap]


def test_plan_counts_causvid_scales_with_blocks() -> None:
    state = GenerateFormState(style="x", backend="causvid", duration="5s", blocks="2")
    assert plan_counts(state) == (1, 144, pytest.approx(9.0))  # type: ignore[comparison-overlap]


@pytest.mark.parametrize("backend", ["ltx25", "ltx23"])
def test_plan_counts_ltx_uses_96_novel_per_block_at_24fps(backend: str) -> None:
    state = GenerateFormState(style="x", backend=backend, duration="5s")
    assert plan_counts(state) == (2, 192, pytest.approx(8.0))  # type: ignore[comparison-overlap]


@pytest.mark.parametrize("backend", ["ltx25", "ltx23"])
def test_plan_counts_ltx_scales_with_blocks(backend: str) -> None:
    state = GenerateFormState(style="x", backend=backend, duration="5s", blocks="2")
    assert plan_counts(state) == (1, 192, pytest.approx(8.0))  # type: ignore[comparison-overlap]


def test_field_errors_keyed_by_field() -> None:
    errors = field_errors(GenerateFormState())
    assert set(errors) == {"style"}
    assert "style" in errors["style"]
    errors = field_errors(GenerateFormState(style="x", duration="soon", name="a/b"))
    assert "duration" in errors
    assert "name" in errors
    assert "5s" in errors["duration"]


def test_name_must_be_flat() -> None:
    assert "name" in field_errors(GenerateFormState(style="x", name=""))
    assert "name" in field_errors(GenerateFormState(style="x", name="a/b"))
    assert "name" not in field_errors(GenerateFormState(style="x", name="my-run_01"))


def test_plan_counts_match_cli_truth_all_backends_and_blocks() -> None:
    """Single-source planning (issue 024): TUI == cli._frames_per_segment math.

    Cross-test over every backend × blocks 1..3 against a planning-only
    ProjectConfig with preset fps/segment_frames — the same inputs
    cmd_generate plans with. Any drift on either side fails here.
    """
    from voyage.cli import _frames_per_segment, segments_for_duration
    from voyage.config import ProjectConfig, VideoConfig, _video_preset

    for backend in ("ltxv", "causvid", "fake"):
        preset = _video_preset(backend)
        raw_fps = preset.get("fps", 24)
        assert isinstance(raw_fps, int)
        raw_frames = preset.get("segment_frames", 48)
        assert isinstance(raw_frames, int)
        for blocks in (1, 2, 3):
            state = GenerateFormState(style="x", backend=backend, duration="5s", blocks=str(blocks))
            config = ProjectConfig(
                style="planning",
                video=VideoConfig(
                    backend=backend,
                    blocks_per_segment=blocks,
                    fps=raw_fps,
                    segment_frames=raw_frames,
                ),
            )
            frames_per_segment = _frames_per_segment(config)
            segments = segments_for_duration(5.0, raw_fps, frames_per_segment)
            assert plan_counts(state) == (  # type: ignore[comparison-overlap]
                segments,
                segments * frames_per_segment,
                pytest.approx(segments * frames_per_segment / raw_fps),
            )


def test_plan_summary_uses_single_source_struct() -> None:
    """plan_summary formats plan_counts (no independent frame math)."""
    for backend, expected_fragment in (
        ("ltxv", "96f/segment @ 24fps"),
        ("causvid", "72f/segment @ 16fps"),
        ("fake", "48f/segment @ 24fps"),
    ):
        state = GenerateFormState(style="x", backend=backend, duration="5s")
        counts = plan_counts(state)
        assert counts is not None
        segments, planned_frames, _seconds = counts
        summary = plan_summary(state)
        assert expected_fragment in summary
        assert f"{segments} segment(s)" in summary
        assert f"{planned_frames} frames" in summary


def test_gpu_warning_derives_from_shared_cuda_set() -> None:
    """gpu_warning agrees with cli._CUDA_BACKENDS (issue 024)."""
    from voyage.cli import _CUDA_BACKENDS

    for backend in ("ltxv", "causvid", "fake", "nope"):
        assert bool(gpu_warning(backend)) == (backend in _CUDA_BACKENDS)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NAN", "Infinity"])
def test_take_seconds_rejects_non_finite(raw: str) -> None:
    """Non-finite take lengths never reach planning or the ACE payload."""
    assert "take_seconds" in field_errors(GenerateFormState(style="x", take_seconds=raw))
    with pytest.raises(ValueError, match="take-seconds"):
        to_generate_namespace(GenerateFormState(style="x", take_seconds=raw))


@pytest.mark.parametrize("raw", ["-5", "0", "-0.5"])
def test_take_seconds_rejects_non_positive(raw: str) -> None:
    assert "take_seconds" in field_errors(GenerateFormState(style="x", take_seconds=raw))


def test_take_seconds_accepts_positive_finite() -> None:
    assert "take_seconds" not in field_errors(GenerateFormState(style="x", take_seconds="30"))
    assert to_generate_namespace(
        GenerateFormState(style="x", take_seconds="30")
    ).take_seconds == pytest.approx(30.0)


@pytest.mark.parametrize(
    "raw",
    [
        ".",
        "con",
        "CON",
        "prn",
        "aux",
        "nul",
        "NUL",
        "com1",
        "COM9",
        "lpt1",
        "LPT9",
        "con.txt",
        "a/b",
        "..",
        "",
    ],
)
def test_flat_folder_name_rejects_dot_and_reserved(raw: str) -> None:
    """Single dot targets shared output/; reserved names are never folders."""
    assert _flat_folder_name(raw) is False
    assert "name" in field_errors(GenerateFormState(style="x", name=raw))


@pytest.mark.parametrize("raw", ["my-run_01", "voyage", "run 2", "a.b", "comet", "null"])
def test_flat_folder_name_accepts_ordinary_names(raw: str) -> None:
    assert _flat_folder_name(raw) is True
    assert "name" not in field_errors(GenerateFormState(style="x", name=raw))


@pytest.mark.parametrize("raw", ["a\x01b", "hello\x00world", "x\x0by\x0cz", "tab\there", "q\x1f"])
def test_last_settings_round_trip_control_characters(raw: str, tmp_path: Path) -> None:
    """Every C0 control survives save/load; the file holds no raw control."""
    settings_file = tmp_path / "tui-last.toml"
    state = GenerateFormState(style=raw, backend="fake", name="myrun")
    save_last_settings(state, settings_file)
    file_bytes = settings_file.read_bytes()
    assert all(byte >= 0x20 or byte == 0x0A for byte in file_bytes)
    assert load_last_settings(settings_file) == state


def test_generate_namespace_carries_finalize_sfx_attrs() -> None:
    """Issue 097: the TUI namespace must satisfy cmd_generate's finalize block.

    cmd_generate forwards sfx_*/no_sfx/skip_bad into cmd_finalize; a missing
    attribute is an AttributeError after segments commit. Parser defaults are
    the contract (None/None/None/None/1 + no_sfx False).
    """
    ns = to_generate_namespace(GenerateFormState(style="x", backend="fake", name="sfxns"))
    assert ns.skip_bad is False
    assert ns.no_sfx is False
    assert ns.sfx_backend is None
    assert ns.sfx_caption is None
    assert ns.sfx_device is None
    assert ns.sfx_model_size is None
    assert ns.sfx_workers == 1
