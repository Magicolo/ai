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
per setting with a fixed-width label; focusing any field shows its
description in the side help panel (stacked below the form on narrow
terminals). Every change re-validates per field: invalid widgets get a
red border and the panel shows the message. The Style editor takes
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
import io
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
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
    .field-row {
        layout: horizontal;
        height: auto;
        margin: 0 1;
    }
    .field-row > .field-label {
        width: 17;
        color: $text-muted;
        padding-right: 1;
    }
    #field-style {
        height: 3;
    }
    Input.field-invalid, TextArea.field-invalid, Select.field-invalid {
        border: tall $error;
    }
    #gpu-warning {
        color: $warning;
        margin: 0 1;
    }
    #plan-line {
        color: $success;
        margin: 1 1 0 1;
    }
    #errors-line {
        color: $error;
        margin: 0 1;
    }
    #button-row {
        align: center middle;
        height: auto;
        margin: 1 1;
    }
    #button-generate {
        margin-right: 2;
    }
    #key-hints, #run-keys {
        text-align: center;
        color: $text-muted;
        margin: 0 1 1 1;
    }
    #run-view {
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
    Input:focus, TextArea:focus, Select:focus {
        border: tall $accent;
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
        self._running = False
        self._quit_armed = False
        # Plain-text record of every run-view line (RichLog exposes no
        # public line count; this keeps the progress assertable).
        self.run_history: list[str] = []

    def _field_row(
        self,
        label: str,
        field: str,
        widget: Input | Select[str] | TextArea,
    ) -> Horizontal:
        """One compact label + widget row (widget id must match FIELD_WIDGET_IDS)."""
        del field  # documented by the widget id; kept for call-site readability
        return Horizontal(
            Static(label, classes="field-label"),
            widget,
            classes="field-row",
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="form-view"):
            with Horizontal(id="form-columns"):
                with Vertical(id="form-col"):
                    yield self._field_row(
                        "Style *",
                        "style",
                        TextArea(
                            text=self.initial_state.style,
                            soft_wrap=True,
                            id="field-style",
                            tooltip=FIELD_HELP["style"],
                        ),
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
                with Vertical(id="help-panel"):
                    yield Static("❓ Field help", id="help-title")
                    yield Static(_HELP_OVERVIEW, id="help-body")
        with Vertical(id="run-view"):
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
        def text(widget_id: str) -> str:
            return self.query_one(widget_id, Input).value

        def choice(widget_id: str) -> str:
            return str(self.query_one(widget_id, Select).value)

        def flag(widget_id: str) -> bool:
            return self.query_one(widget_id, Checkbox).value

        return GenerateFormState(
            backend=choice("#field-backend"),
            duration=text("#field-duration"),
            style=self.query_one("#field-style", TextArea).text,
            name=text("#field-name"),
            seed=text("#field-seed"),
            force=flag("#flag-force"),
            skip_bad=flag("#flag-skip-bad"),
            draft=flag("#flag-draft"),
            director=choice("#field-director"),
            blocks=text("#field-blocks"),
            take_seconds=text("#field-take-seconds"),
            quantization=choice("#field-quantization"),
            beats_per_segment=text("#field-beats"),
            drift_every_n=text("#field-drift"),
            verbose=flag("#flag-verbose"),
            no_color=flag("#flag-no-color"),
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
        self.query_one("#plan-line", Static).update(f"plan: {plan_summary(state)}")
        errors = field_errors(state)
        self._apply_field_errors(errors)
        self.query_one("#errors-line", Static).update(" · ".join(errors.values()))
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
        self.query_one("#gpu-warning", Static).update(_backend_gpu_warning(backend))

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
        if self._running and not self._quit_armed:
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
            self.query_one("#errors-line", Static).update(f"cannot read form: {exc}")
            return
        errors = field_errors(state)
        self._apply_field_errors(errors)
        if errors:
            self.query_one("#errors-line", Static).update(" · ".join(errors.values()))
            return
        self.query_one("#errors-line", Static).update("")
        _save_last_settings(state)
        self._quit_armed = False
        counts = plan_counts(state)
        total = counts[0] if counts is not None else 0
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
            self._running = True
            self.set_focus(self.query_one("#button-stop", Button))
            self.refresh(layout=True)
        except Exception as exc:
            self.query_one("#form-view", ScrollableContainer).display = True
            self.query_one("#run-view", Vertical).display = False
            self._running = False
            self.query_one("#errors-line", Static).update(f"cannot open run view: {exc}")
            return
        self.run_worker(
            lambda: self._generate_in_thread(state, total),
            thread=True,
            exclusive=True,
            description="voyage generate",
        )

    def _run_dir_for(self, args: argparse.Namespace) -> Path:
        output = args.output or str(Path("output") / args.run_id)
        return Path(output).resolve()

    def _generate_in_thread(self, state: GenerateFormState, total: int) -> None:
        from voyage.cli import cmd_generate
        from voyage.errors import VoyageError

        try:
            namespace = to_generate_namespace(state)
        except ValueError as exc:
            self.call_from_thread(self._finish_generation, f"✗ invalid settings: {exc}")
            return
        run_dir = self._run_dir_for(namespace)
        self._run_dir = run_dir
        self.call_from_thread(self._open_run_view, namespace, run_dir, total)
        namespace.progress_sink = TuiProgress(self, total)
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                code = cmd_generate(namespace)
        except VoyageError as exc:
            self._flush_captured_output(buffer)
            self.call_from_thread(self._finish_generation, f"✗ generation failed: {exc}")
        except Exception as exc:  # defensive: never trap the worker silently
            self._flush_captured_output(buffer)
            self.call_from_thread(self._finish_generation, f"✗ unexpected error: {exc!r}")
        else:
            self._flush_captured_output(buffer)
            if code == 0:
                final = namespace.final_video or str(run_dir / "final.mp4")
                self.call_from_thread(self._finish_generation, f"✓ generated {final}")
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
        self.query_one("#run-head", Static).update(
            f"▶ generating · {namespace.backend} · {total} segment(s) → {run_dir}"
        )

    def append_run_line(self, text: str) -> None:
        self.run_history.append(text)
        self.query_one("#run-log", RichLog).write(text)

    def advance_run_bar(self, done: int, total: int) -> None:
        self.query_one("#run-bar", ProgressBar).update(total=total, progress=done)

    def _finish_generation(self, result: str) -> None:
        self._running = False
        self._quit_armed = False
        self.query_one("#button-stop", Button).disabled = True
        if result.startswith("✗"):
            # Failure: back to the form with the error visible (never a stuck view).
            self._show_form()
            self.query_one("#errors-line", Static).update(result)
            return
        self.query_one("#run-result", Static).update(result)

    def _request_stop(self) -> None:
        if not self._running or self._run_dir is None:
            self.append_run_line("■ nothing running")
            return
        state_path = self._run_dir / "state.json"
        if not state_path.exists():
            self.append_run_line("■ run has not initialized yet — stopping is a no-op")
            return
        from voyage.persistence import read_state, write_state

        state = read_state(self._run_dir)
        state.status = "STOP_REQUESTED"
        write_state(self._run_dir, state)
        self.append_run_line("■ stop requested — finishing the current segment …")

    def _show_form(self) -> None:
        if self._running:
            return
        self.query_one("#run-view", Vertical).display = False
        self.query_one("#form-view", ScrollableContainer).display = True
        self._refresh_plan_and_errors()
        self.set_focus(self.query_one("#field-style", TextArea))
