"""Interactive launcher TUI: bare `voyage` configures `generate` in-form.

Why this module exists: the bare ``voyage`` command (no verb) launches
this Textual app, which shows every ``generate`` setting with its
default, validates live, derives the segment/frame plan, and runs the
generation with per-segment progress routed into a RichLog — the same
video + audio prompts the console layer prints, plus a progress bar and
a Stop button. Generation reuses :func:`voyage.cli.cmd_generate`
unchanged (same init → run → validate → finalize), with the progress
sink swapped from stdout to the TUI via the ``progress_sink`` namespace
slot (see ``cmd_run``/``cmd_generate``).

Form behavior: a single compact field list (required first), one row
per setting with a fixed-width label: single-line borderless wells
(1 line each; the Style editor is 3), empty notice lines hidden so they
take no space. Focus/invalid recolor the background only, so row
geometry is identical in every state. Focusing any field shows its
description in the side help panel (stacked below the form on narrow
terminals). Every change re-validates per field: invalid widgets get a
red-tinted background and the panel shows the message. The Style editor takes
focus on mount so typing lands immediately. ``tui_state`` persistence /
GPU-warning helpers are consumed defensively via ``getattr`` so the app
mounts and generates either way. Quit while a run is active arms a
two-press confirm instead of exiting at once. Keyboard alone drives the
whole flow: ctrl+g generates, ctrl+x stops from the run view, ``b``
goes back, ctrl+q quits (ctrl+s is deliberately unbound — it is
terminal XOFF flow control and freezes output).

Threading: generation runs in a Textual worker thread; every widget
touch from that thread goes through ``App.call_from_thread``. The
view-switch path is exception-guarded so a failure renders as an error
line instead of a frozen form, and the worker's stdout/stderr is
redirected into the run log so ``cmd_generate`` prints can never fight
Textual's alternate screen. Stop writes STOP_REQUESTED to the run-dir
state file — the same control plane as ``voyage stop`` — so the run
loop exits at the next segment boundary and still validates +
finalizes the partial run.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.timer import Timer
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    ProgressBar,
    RichLog,
    Select,
    Static,
    TextArea,
)

from voyage import tui_state as tui_state_module
from voyage.console import SegmentProgress
from voyage.tui_state import (
    BACKENDS,
    DIRECTORS,
    FIELD_HELP,
    QUANTIZATIONS,
    GenerateFormState,
    field_errors,
    plan_counts,
    plan_summary,
    to_generate_namespace,
)

# Form field name -> widget id (single source for error styling, the
# help panel, and the tests).
FIELD_WIDGET_IDS = {
    "style": "field-style",
    "name": "field-name",
    "duration": "field-duration",
    "backend": "field-backend",
    "director": "field-director",
    "quantization": "field-quantization",
    "blocks": "field-blocks",
    "take_seconds": "field-take-seconds",
    "beats_per_segment": "field-beats",
    "drift_every_n": "field-drift",
    "min_fps": "field-min-fps",
    "min_resolution": "field-min-resolution",
    "seed": "field-seed",
}
WIDGET_FIELD_NAMES = {widget_id: field for field, widget_id in FIELD_WIDGET_IDS.items()}

_BUTTON_HELP = {
    "button-generate": "Generate: validate the form and start the run (same as ctrl+g).",
    "button-quit": "Quit the TUI (same as ctrl+q).",
    "button-stop": "Stop: finish the current segment, then validate + finalize (same as ctrl+x).",
    "button-back": "Back to the form (same as b).",
    "button-quit-run": "Quit the TUI (same as ctrl+q).",
}

_HELP_OVERVIEW = (
    "Focus any field to see what it does.\n\n"
    "Required first: style, then the run name. Everything else has a "
    "working default — Generate is one style string away."
)

# Below this width the help panel stacks under the form instead of
# sitting beside it.
NARROW_WIDTH = 100

# Massive block-letter hero art (5 rows x 35 cols, centered by CSS).
# Textual cannot scale fonts, so presence comes from size + accent color.
TITLE_ART = "\n".join(
    [
        "█   █  ███  █   █   █    ████ █████",
        "█   █ █   █ █   █  █ █  █     █    ",
        "█   █ █   █  █ █  █████ █ ███ ████ ",
        " █ █  █   █   █   █   █ █   █ █    ",
        "  █    ███    █   █   █  ███  █████",
    ]
)


def run_tui() -> int:
    """Launch the launcher TUI (TTY + Textual required, else guidance)."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "voyage with no arguments launches the interactive TUI, which needs a terminal.",
            file=sys.stderr,
        )
        print(
            'hint: voyage generate --duration 5s --style "..." (see voyage generate --help)',
            file=sys.stderr,
        )
        return 2
    try:
        import textual  # noqa: F401
    except ImportError:
        print(
            "the voyage TUI needs the 'textual' package (pip install \"textual>=8.0\").",
            file=sys.stderr,
        )
        print(
            'hint: voyage generate --duration 5s --style "..." (see voyage generate --help)',
            file=sys.stderr,
        )
        return 2
    del textual
    VoyageApp().run()
    return 0


def _load_initial_state() -> GenerateFormState:
    """Best-effort restore of the last settings (Stream A may not exist yet)."""
    loader = getattr(tui_state_module, "load_last_settings", None)
    if not callable(loader):
        return GenerateFormState()
    try:
        loaded = loader()
    except Exception:
        return GenerateFormState()
    if not isinstance(loaded, GenerateFormState):
        return GenerateFormState()
    fresh = GenerateFormState()
    if loaded.backend not in BACKENDS:
        loaded.backend = fresh.backend
    if loaded.director not in DIRECTORS:
        loaded.director = fresh.director
    if loaded.quantization not in QUANTIZATIONS:
        loaded.quantization = fresh.quantization
    return loaded


def _backend_gpu_warning(backend: str) -> str:
    """Per-backend GPU hint, or "" when Stream A has not landed / no warning."""
    reporter = getattr(tui_state_module, "gpu_warning", None)
    if not callable(reporter):
        return ""
    try:
        warning = reporter(backend)
    except Exception:
        return ""
    return warning if isinstance(warning, str) else ""


def _save_last_settings(state: GenerateFormState) -> None:
    """Best-effort persist of the validated form (never raises)."""
    saver = getattr(tui_state_module, "save_last_settings", None)
    if not callable(saver):
        return
    try:
        saver(state)
    except Exception:
        pass


def _buffer_tail(buffer: io.StringIO) -> str:
    """Last non-empty captured worker line (the reason behind an exit code)."""
    lines = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]
    return lines[-1] if lines else ""


class TuiProgress(SegmentProgress):
    """Supervisor progress sink that posts into the TUI run view.

    Plain-text lines mirror the console wording (SEGMENT headers, full
    video prompts, music caption with the beat grid, commit summaries)
    so behavior stays recognizable across both displays.
    """

    def __init__(self, app: VoyageApp, total_segments: int) -> None:
        self._app = app
        self._total = total_segments
        self._done = 0

    def _post(self, text: str) -> None:
        self._app.call_from_thread(self._app.append_run_line, text)

    def segment_start(self, number: int, segment_id: str) -> None:
        self._post(f"▶ SEGMENT {segment_id} (#{number})")

    @contextmanager
    def stage(self, label: str, detail: str = "") -> Iterator[None]:
        started = time.monotonic()
        head = f"{label} … {detail}".rstrip(" …")
        self._post(f"▸ {head} ...")
        try:
            yield
        except Exception:
            self._post(f"✗ {label} failed")
            raise
        elapsed = time.monotonic() - started
        self._post(f"✓ {label} in {elapsed:.1f}s")

    def segment_plan(self, info: dict[str, Any]) -> None:
        destination = str(info.get("destination", ""))
        phase = str(info.get("phase", ""))
        novel = "novel ✓" if info.get("novelty_accepted") else "hold"
        self._post(f"◆ drift → {destination} · {phase} · {novel}")
        prompts = info.get("video_prompts", [])
        if isinstance(prompts, list):
            for index, prompt in enumerate(prompts):
                tag = f"prompt [{index + 1}/{len(prompts)}]" if len(prompts) > 1 else "prompt"
                self._post(f"  🎬 {tag}: {prompt}")
        beats = info.get("audio_beats", 0)
        bpm = info.get("audio_bpm", 0.0)
        caption = str(info.get("audio_caption", ""))
        self._post(f"  🎵 music: {caption} ({beats} beats @ {bpm:.0f} BPM)")

    def segment_done(self, info: dict[str, Any]) -> None:
        segment_id = str(info.get("segment_id", ""))
        frames = info.get("frames", 0)
        duration = info.get("duration", 0.0)
        beats = info.get("beats", 0)
        bpm = info.get("bpm", 0.0)
        self._post(
            f"✓ SEGMENT {segment_id} committed · {frames}f ≈ {duration:.2f}s · "
            f"{beats} beats @ {bpm:.0f} BPM"
        )
        stages = info.get("stage_seconds", {})
        if isinstance(stages, dict) and stages:
            cells = " · ".join(f"{name} {seconds:.1f}s" for name, seconds in stages.items())
            self._post(f"  ⏱ {cells}")
        self._done += 1
        done = self._done
        total = self._total
        self._app.call_from_thread(self._app.advance_run_bar, done, total)


def _read_text_field(app: VoyageApp, widget_id: str) -> str:
    """Raw text of a single-line Input (empty means the preset default)."""
    return app.query_one(widget_id, Input).value


def _read_choice_field(app: VoyageApp, widget_id: str) -> str:
    """Current value of a dropdown Select as a plain string."""
    return str(app.query_one(widget_id, Select).value)


def _read_flag_field(app: VoyageApp, widget_id: str) -> bool:
    """Checked state of a form Checkbox."""
    return app.query_one(widget_id, Checkbox).value


class VoyageApp(App[None]):
    """Configure `generate`, then watch it run."""

    TITLE = "Voyage"
    SUB_TITLE = "configure → generate"
    AUTO_FOCUS = "#field-style"

    CSS = """
    #form-columns {
        layout: horizontal;
        height: auto;
    }
    #form-columns.narrow {
        layout: vertical;
    }
    #form-col {
        width: 1fr;
        height: auto;
    }
    #help-panel {
        width: 38;
        height: auto;
        border: tall $primary;
        padding: 0 1;
        margin-left: 1;
    }
    #form-columns.narrow #help-panel {
        width: 1fr;
        margin-left: 0;
        margin-top: 1;
    }
    #help-title {
        text-style: bold;
        color: $primary;
    }
    /* Full-width hero title above the form/help columns: centered,
    massive block-letter art. Height is automatic so the art rows plus
    the top pad always fit — a fixed height clipped the bottom row. */
    #app-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        padding-top: 1;
        height: auto;
    }
    #app-subtitle {
        text-align: center;
        text-style: italic;
        color: $text-muted;
        margin-bottom: 1;
        height: 1;
    }
    /* The two views each claim the leftover screen height and scroll
    internally — content-height views overflowed small terminals and
    clipped fields (e.g. backend) with nowhere to scroll to. */
    #form-view {
        height: 1fr;
    }
    .field-row {
        layout: horizontal;
        height: 1;
        margin: 0 1;
    }
    .field-row-tall {
        height: 3;
    }
    .field-row > .field-label {
        width: 17;
        color: $text-muted;
        padding-right: 1;
    }
    /* Inputs stay borderless single-line wells. Selects keep their
    bordered chrome at height 3: stripping it (border:none + height:1)
    rendered the value line blank — blank focusable boxes — while the
    border color alone carries focus/invalid, so geometry never shifts. */
    .field-row Input, #field-style {
        border: none;
        background: $surface;
        padding: 0 1;
    }
    .field-row Input {
        height: 1;
    }
    /* NOTE: height must stay auto — an explicit height (even the same
    3 rows the border computes to) collapses the value line to blank on
    Textual 8.2.8. The tall border fixes the geometry at 3 rows; focus
    and invalid only recolor it, so spacing is constant. */
    .field-row Select {
        height: auto;
        border: tall $surface;
        background: $surface;
        padding: 0 1;
    }
    /* The closed value line carries its own tall border (2 rows) on top
    of the Select border. Drop the inner border so the composed widget
    totals exactly 3 rows; never force an explicit height (that blanks
    the value line on Textual 8). Recoloring stays on the outer border. */
    .field-row SelectCurrent {
        border: none;
        padding: 0 1;
        height: auto;
    }
    #field-style {
        height: 3;
    }
    Input:focus, TextArea:focus {
        background: $surface-lighten-1;
    }
    Select:focus {
        border: tall $accent;
        background: $surface-lighten-1;
    }
    Input.field-invalid, TextArea.field-invalid {
        background: $error-muted;
    }
    Select.field-invalid {
        border: tall $error;
    }
    #form-col Checkbox {
        height: 1;
        margin: 0 1;
    }
    #gpu-warning {
        color: $warning;
        display: none;
        margin: 0 1;
    }
    #plan-line {
        color: $success;
        margin: 0 1;
    }
    #errors-line {
        color: $error;
        display: none;
        margin: 0 1;
    }
    #button-row {
        align: center middle;
        height: auto;
        margin: 1 1 0 1;
    }
    #button-generate {
        margin-right: 2;
    }
    #key-hints, #run-keys {
        text-align: center;
        color: $text-muted;
        margin: 0 1;
    }
    #run-view {
        height: 1fr;
        margin: 0 1;
    }
    #run-head {
        text-style: bold;
        color: $accent;
    }
    #run-bar {
        margin-bottom: 1;
    }
    #run-log {
        height: 1fr;
        border: tall $primary;
        padding: 0 1;
    }
    #run-result {
        margin-top: 1;
    }
    #run-buttons {
        height: auto;
        margin-top: 1;
    }
    #run-buttons Button {
        margin-right: 2;
    }
    Button:focus {
        text-style: bold reverse;
    }
    """

    BINDINGS = [
        ("ctrl+g", "generate", "Generate"),
        ("ctrl+x", "stop", "Stop"),
        ("b", "go_back", "Back"),
        ("ctrl+q", "quit_app", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.initial_state = _load_initial_state()
        self._run_dir: Path | None = None
        # Generation flag (issue 093): NEVER name this `_running` — Textual's
        # own `App._running` (True from mount) would collide and mask it.
        self._generation_running = False
        self._quit_armed = False
        # Elapsed-time heartbeat for the run view (ticks iff the event
        # loop is alive — the visible answer to "is it frozen?").
        self._run_head_base = ""
        self._run_t0 = 0.0
        self._heartbeat: Timer | None = None
        # Plain-text record of every run-view line (RichLog exposes no
        # public line count; this keeps the progress assertable).
        self.run_history: list[str] = []

    def _field_row(
        self,
        label: str,
        field: str,
        widget: Input | Select[str] | TextArea,
        tall: bool = False,
    ) -> Horizontal:
        """One compact label + widget row (widget id must match FIELD_WIDGET_IDS)."""
        del field  # documented by the widget id; kept for call-site readability
        classes = "field-row field-row-tall" if tall else "field-row"
        return Horizontal(
            Static(label, classes="field-label"),
            widget,
            classes=classes,
        )

    def _set_line(self, selector: str, text: str) -> None:
        """Update a notice line, hiding it entirely when empty (no dead rows)."""
        line = self.query_one(selector, Static)
        line.update(text)
        line.display = bool(text)

    def _form_fields(self) -> ComposeResult:
        """Form-column widgets, in order (issue 020: form-vs-run seam).

        Yields exactly the widgets `compose` used to inline inside
        `#form-col` — field rows, flags, plan/errors lines, buttons, key
        hints. The `with` container keeps compose-stack semantics: building
        the button row with explicit children instead mounts an identical
        DOM that silently drops button clicks (verified live 2026-09-25),
        so the original form stays.
        """
        yield self._field_row(
            "Style *",
            "style",
            TextArea(
                text=self.initial_state.style,
                soft_wrap=True,
                id="field-style",
                tooltip=FIELD_HELP["style"],
            ),
            tall=True,
        )
        yield self._field_row(
            "Name *",
            "name",
            Input(
                value=self.initial_state.name,
                id="field-name",
                tooltip=FIELD_HELP["name"],
            ),
        )
        yield self._field_row(
            "Duration",
            "duration",
            Input(
                value=self.initial_state.duration,
                id="field-duration",
                tooltip=FIELD_HELP["duration"],
            ),
        )
        yield self._field_row(
            "Backend ▾",
            "backend",
            Select(
                [(name, name) for name in BACKENDS],
                value=self.initial_state.backend,
                allow_blank=False,
                prompt="▾ pick a backend",
                id="field-backend",
                tooltip=FIELD_HELP["backend"],
            ),
            tall=True,
        )
        yield Static("", id="gpu-warning")
        yield self._field_row(
            "Director ▾",
            "director",
            Select(
                [(name, name) for name in DIRECTORS],
                value=self.initial_state.director,
                allow_blank=False,
                prompt="▾ pick a director",
                id="field-director",
                tooltip=FIELD_HELP["director"],
            ),
            tall=True,
        )
        yield self._field_row(
            "Quantization ▾",
            "quantization",
            Select(
                [(name, name) for name in QUANTIZATIONS],
                value=self.initial_state.quantization,
                allow_blank=False,
                prompt="▾ pick a precision",
                id="field-quantization",
                tooltip=FIELD_HELP["quantization"],
            ),
            tall=True,
        )
        yield self._field_row(
            "Blocks",
            "blocks",
            Input(
                value=self.initial_state.blocks,
                placeholder="1",
                id="field-blocks",
                tooltip=FIELD_HELP["blocks"],
            ),
        )
        yield self._field_row(
            "Take seconds",
            "take_seconds",
            Input(
                value=self.initial_state.take_seconds,
                placeholder="45.0",
                id="field-take-seconds",
                tooltip=FIELD_HELP["take_seconds"],
            ),
        )
        yield self._field_row(
            "Beats/segment",
            "beats_per_segment",
            Input(
                value=self.initial_state.beats_per_segment,
                placeholder="4",
                id="field-beats",
                tooltip=FIELD_HELP["beats_per_segment"],
            ),
        )
        yield self._field_row(
            "Drift every N",
            "drift_every_n",
            Input(
                value=self.initial_state.drift_every_n,
                placeholder="1",
                id="field-drift",
                tooltip=FIELD_HELP["drift_every_n"],
            ),
        )
        yield self._field_row(
            "Min fps",
            "min_fps",
            Input(
                value=self.initial_state.min_fps,
                placeholder="32",
                id="field-min-fps",
                tooltip=FIELD_HELP["min_fps"],
            ),
        )
        yield self._field_row(
            "Min resolution",
            "min_resolution",
            Input(
                value=self.initial_state.min_resolution,
                placeholder="1280x720",
                id="field-min-resolution",
                tooltip=FIELD_HELP["min_resolution"],
            ),
        )
        yield self._field_row(
            "Seed",
            "seed",
            Input(
                value=self.initial_state.seed,
                id="field-seed",
                tooltip=FIELD_HELP["seed"],
            ),
        )
        yield Checkbox(
            "Draft profile (fast low-res iteration)",
            value=self.initial_state.draft,
            id="flag-draft",
        )
        yield Checkbox(
            "Force (init into a non-empty directory)",
            value=self.initial_state.force,
            id="flag-force",
        )
        yield Checkbox(
            "Skip bad segments at finalize",
            value=self.initial_state.skip_bad,
            id="flag-skip-bad",
        )
        yield Checkbox(
            "Verbose console lines behind the TUI",
            value=self.initial_state.verbose,
            id="flag-verbose",
        )
        yield Checkbox(
            "No color (plain output)",
            value=self.initial_state.no_color,
            id="flag-no-color",
        )
        yield Static("", id="plan-line")
        yield Static("", id="errors-line")
        with Horizontal(id="button-row"):
            yield Button("Generate ▶", variant="primary", id="button-generate")
            yield Button("Quit", id="button-quit")
        yield Static(
            "keys: ctrl+g generate · ctrl+x stop · b back · ctrl+q quit",
            id="key-hints",
        )

    def _run_view_widgets(self) -> ComposeResult:
        """Run-view widgets, in order (issue 020: form-vs-run seam).

        Yields exactly the widgets `compose` used to inline inside
        `#run-view` — head, progress bar, log, result, buttons, key hints.
        Same `with`-container caveat as `_form_fields`: explicit children
        mount a click-dead copy, so the original form stays.
        """
        yield Static("", id="run-head")
        yield ProgressBar(id="run-bar")
        yield RichLog(id="run-log", highlight=False)
        yield Static("", id="run-result")
        with Horizontal(id="run-buttons"):
            yield Button("Stop ■", variant="error", id="button-stop")
            yield Button("← Back", id="button-back")
            yield Button("Quit", id="button-quit-run")
        yield Static(
            "keys: ctrl+x stop · b back · ctrl+q quit",
            id="run-keys",
        )

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(TITLE_ART, id="app-title")
        yield Static("one-shot music-video generator", id="app-subtitle")
        with ScrollableContainer(id="form-view"):
            with Horizontal(id="form-columns"):
                with Vertical(id="form-col"):
                    yield from self._form_fields()
                with Vertical(id="help-panel"):
                    yield Static("❓ Field help", id="help-title")
                    yield Static(_HELP_OVERVIEW, id="help-body")
        with Vertical(id="run-view"):
            yield from self._run_view_widgets()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#run-view", Vertical).display = False
        self._apply_narrow()
        self._refresh_plan_and_errors()
        self._refresh_gpu_warning()
        self._refresh_help()
        try:
            self.set_focus(self.query_one("#field-style", TextArea))
        except Exception:
            pass

    def on_resize(self) -> None:
        self._apply_narrow()

    def _apply_narrow(self) -> None:
        """Stack the help panel below the form on narrow terminals."""
        try:
            columns = self.query_one("#form-columns", Horizontal)
        except Exception:
            return
        columns.set_class(self.size.width < NARROW_WIDTH, "narrow")

    def _read_form(self) -> GenerateFormState:
        return GenerateFormState(
            backend=_read_choice_field(self, "#field-backend"),
            duration=_read_text_field(self, "#field-duration"),
            style=self.query_one("#field-style", TextArea).text,
            name=_read_text_field(self, "#field-name"),
            seed=_read_text_field(self, "#field-seed"),
            force=_read_flag_field(self, "#flag-force"),
            skip_bad=_read_flag_field(self, "#flag-skip-bad"),
            draft=_read_flag_field(self, "#flag-draft"),
            director=_read_choice_field(self, "#field-director"),
            blocks=_read_text_field(self, "#field-blocks"),
            take_seconds=_read_text_field(self, "#field-take-seconds"),
            quantization=_read_choice_field(self, "#field-quantization"),
            beats_per_segment=_read_text_field(self, "#field-beats"),
            drift_every_n=_read_text_field(self, "#field-drift"),
            min_fps=_read_text_field(self, "#field-min-fps"),
            min_resolution=_read_text_field(self, "#field-min-resolution"),
            verbose=_read_flag_field(self, "#flag-verbose"),
            no_color=_read_flag_field(self, "#flag-no-color"),
        )

    def _apply_field_errors(self, errors: dict[str, str]) -> None:
        """Red borders on invalid fields, cleared everywhere else."""
        for field, widget_id in FIELD_WIDGET_IDS.items():
            try:
                widget = self.query_one(f"#{widget_id}")
            except Exception:
                continue
            widget.set_class(field in errors, "field-invalid")

    def _refresh_plan_and_errors(self) -> None:
        try:
            state = self._read_form()
        except Exception:
            return
        self._set_line("#plan-line", f"plan: {plan_summary(state)}")
        errors = field_errors(state)
        self._apply_field_errors(errors)
        self._set_line("#errors-line", " · ".join(errors.values()))
        self._refresh_help(errors)

    def _refresh_help(self, errors: dict[str, str] | None = None) -> None:
        """Show the focused field's description (plus its error, if any)."""
        try:
            body = self.query_one("#help-body", Static)
        except Exception:
            return
        focused = self.focused
        widget_id = focused.id if focused is not None else None
        if widget_id in _BUTTON_HELP:
            body.update(_BUTTON_HELP[widget_id])
            return
        field = WIDGET_FIELD_NAMES.get(widget_id or "")
        if field is None or field not in FIELD_HELP:
            body.update(_HELP_OVERVIEW)
            return
        text = FIELD_HELP[field]
        if errors is not None and field in errors:
            text = f"{text}\n\n⚠ {errors[field]}"
        body.update(text)

    def on_descendant_focus(self, _event: object) -> None:
        try:
            errors = field_errors(self._read_form())
        except Exception:
            errors = {}
        self._refresh_help(errors)

    def _refresh_gpu_warning(self) -> None:
        try:
            backend = str(self.query_one("#field-backend", Select).value)
        except Exception:
            backend = self.initial_state.backend
        self._set_line("#gpu-warning", _backend_gpu_warning(backend))

    def on_input_changed(self, _event: Input.Changed) -> None:
        self._refresh_plan_and_errors()

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        self._refresh_plan_and_errors()

    def on_select_changed(self, _event: Select.Changed) -> None:
        self._refresh_plan_and_errors()
        self._refresh_gpu_warning()

    def action_generate(self) -> None:
        if not self.query_one("#form-view", ScrollableContainer).display:
            return  # already running — the Stop button owns control
        self._start_generation()

    def action_stop(self) -> None:
        if self.query_one("#form-view", ScrollableContainer).display:
            return  # form view owns no stop target — stop lives in the run view
        self._request_stop()

    def action_go_back(self) -> None:
        if self.query_one("#form-view", ScrollableContainer).display:
            return  # already on the form — nothing to go back to
        self._show_form()

    def action_quit_app(self) -> None:
        self._confirm_or_quit()

    def _confirm_or_quit(self) -> None:
        """Two-press quit while a run is active; immediate quit otherwise."""
        if self._generation_running and not self._quit_armed:
            self._quit_armed = True
            self.query_one("#run-result", Static).update(
                "generation is running — press Quit again (ctrl+q) to confirm"
            )
            return
        self.exit()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        pressed = event.button.id
        if pressed == "button-generate":
            self.action_generate()
        elif pressed in ("button-quit", "button-quit-run"):
            self._confirm_or_quit()
        elif pressed == "button-stop":
            self._request_stop()
        elif pressed == "button-back":
            self._show_form()

    def _start_generation(self) -> None:
        try:
            state = self._read_form()
        except Exception as exc:
            self._set_line("#errors-line", f"cannot read form: {exc}")
            return
        errors = field_errors(state)
        self._apply_field_errors(errors)
        if errors:
            self._set_line("#errors-line", " · ".join(errors.values()))
            return
        self._set_line("#errors-line", "")
        _save_last_settings(state)
        self._quit_armed = False
        counts = plan_counts(state)
        total = counts[0] if counts is not None else 0
        try:
            namespace = to_generate_namespace(state)
        except ValueError as exc:
            self._set_line("#errors-line", f"invalid settings: {exc}")
            return
        run_dir = self._run_dir_for(namespace)
        head = f"▶ generating · {namespace.backend} · {total} segment(s) → {run_dir}"
        try:
            self.query_one("#form-view", ScrollableContainer).display = False
            run_view = self.query_one("#run-view", Vertical)
            run_view.display = True
            bar = self.query_one("#run-bar", ProgressBar)
            bar.update(total=total, progress=0)
            log = self.query_one("#run-log", RichLog)
            log.clear()
            self.run_history.clear()
            self.query_one("#run-result", Static).update("")
            self.query_one("#button-stop", Button).disabled = False
            self._generation_running = True
            # Synchronous first paint: the run view already says what is
            # starting before the worker thread boots (no silent gap).
            self._run_head_base = head
            self._run_t0 = time.monotonic()
            self.query_one("#run-head", Static).update(head)
            self.append_run_line("▶ starting — workers are warming up …")
            self._start_heartbeat()
            self.set_focus(self.query_one("#button-stop", Button))
            self.refresh(layout=True)
        except Exception as exc:
            self.query_one("#form-view", ScrollableContainer).display = True
            self.query_one("#run-view", Vertical).display = False
            self._generation_running = False
            self._stop_heartbeat()
            self._set_line("#errors-line", f"cannot open run view: {exc}")
            return
        self.run_worker(
            lambda: self._generate_in_thread(namespace, run_dir, total),
            thread=True,
            exclusive=True,
            description="voyage generate",
        )

    def _start_heartbeat(self) -> None:
        """Tick the run-head elapsed timer (stops with the run)."""
        self._stop_heartbeat()
        try:
            self._heartbeat = self.set_interval(1.0, self._tick_run_head)
        except Exception:
            self._heartbeat = None

    def _stop_heartbeat(self) -> None:
        heartbeat, self._heartbeat = self._heartbeat, None
        if heartbeat is not None:
            try:
                heartbeat.stop()
            except Exception:
                pass

    def _tick_run_head(self) -> None:
        """Elapsed-time suffix — advances only while the loop is alive."""
        if not self._generation_running or not self._run_head_base:
            return
        try:
            elapsed = time.monotonic() - self._run_t0
            self.query_one("#run-head", Static).update(
                f"{self._run_head_base} · {elapsed:.0f}s elapsed"
            )
        except Exception:
            pass

    def _run_dir_for(self, args: argparse.Namespace) -> Path:
        output = args.output or str(Path("output") / args.run_id)
        return Path(output).resolve()

    def _generate_in_thread(self, namespace: argparse.Namespace, run_dir: Path, total: int) -> None:
        from voyage.cli import cmd_generate
        from voyage.errors import VoyageError

        self._run_dir = run_dir
        namespace.progress_sink = TuiProgress(self, total)
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                code = cmd_generate(namespace)
        except VoyageError as exc:
            self._flush_captured_output(buffer)
            self.call_from_thread(self._finish_generation, f"✗ generation failed: {exc}")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # defensive: never trap the worker silently
            self._flush_captured_output(buffer)
            self.call_from_thread(self._finish_generation, f"✗ unexpected error: {exc!r}")
        else:
            tail = _buffer_tail(buffer)
            self._flush_captured_output(buffer)
            if code == 0:
                final = namespace.final_video or str(run_dir / "final.mp4")
                self.call_from_thread(self._finish_generation, f"✓ generated {final}")
            elif tail:
                self.call_from_thread(
                    self._finish_generation, f"✗ generation exited with code {code}: {tail}"
                )
            else:
                self.call_from_thread(
                    self._finish_generation, f"✗ generation exited with code {code}"
                )

    def _flush_captured_output(self, buffer: io.StringIO) -> None:
        """Route worker-thread prints into the run log (never real stdout)."""
        for line in buffer.getvalue().splitlines():
            text = line.strip()
            if text:
                self.call_from_thread(self.append_run_line, text)

    def _open_run_view(self, namespace: argparse.Namespace, run_dir: Path, total: int) -> None:
        """Legacy worker-side run-head paint (kept for direct callers).

        The normal path sets the run head synchronously in
        ``_start_generation`` before the worker boots, so the run view is
        never blank; this re-applies the same headline if invoked late.
        """
        self._run_head_base = f"▶ generating · {namespace.backend} · {total} segment(s) → {run_dir}"
        self.query_one("#run-head", Static).update(self._run_head_base)

    def append_run_line(self, text: str) -> None:
        self.run_history.append(text)
        self.query_one("#run-log", RichLog).write(text)

    def advance_run_bar(self, done: int, total: int) -> None:
        self.query_one("#run-bar", ProgressBar).update(total=total, progress=done)

    def _finish_generation(self, result: str) -> None:
        self._generation_running = False
        self._quit_armed = False
        self._stop_heartbeat()
        self.query_one("#button-stop", Button).disabled = True
        if result.startswith("✗"):
            # Failure: back to the form with the error visible (never a stuck view).
            self._show_form()
            self._set_line("#errors-line", result)
            return
        self.query_one("#run-result", Static).update(result)

    def _request_stop(self) -> None:
        if not self._generation_running or self._run_dir is None:
            self.append_run_line("■ nothing running")
            return
        state_path = self._run_dir / "state.json"
        if not state_path.exists():
            self.append_run_line("■ run has not initialized yet — stopping is a no-op")
            return
        from voyage.errors import StateError
        from voyage.persistence import read_state, write_state

        # A corrupt state.json must surface as a feedback line, not an
        # exception out of the button/key handler (issue 078) — Stop is
        # the control-plane action reached for when things already go wrong.
        try:
            state = read_state(self._run_dir)
        except (StateError, OSError) as exc:
            self.append_run_line(f"■ cannot stop: state unreadable ({exc})")
            return
        state.status = "STOP_REQUESTED"
        try:
            write_state(self._run_dir, state)
        except OSError as exc:
            self.append_run_line(f"■ cannot stop: state write failed ({exc})")
            return
        self.append_run_line("■ stop requested — finishing the current segment …")

    def _show_form(self) -> None:
        if self._generation_running:
            return
        self._stop_heartbeat()
        self.query_one("#run-view", Vertical).display = False
        self.query_one("#form-view", ScrollableContainer).display = True
        self._refresh_plan_and_errors()
        self.set_focus(self.query_one("#field-style", TextArea))
