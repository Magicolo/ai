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

Threading: generation runs in a Textual worker thread; every widget
touch from that thread goes through ``App.call_from_thread``. Stop
writes STOP_REQUESTED to the run-dir state file — the same control
plane as ``voyage stop`` — so the run loop exits at the next segment
boundary and still validates + finalizes the partial run.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Footer,
    Header,
    Input,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from voyage.console import SegmentProgress
from voyage.tui_state import (
    BACKENDS,
    DIRECTORS,
    FIELD_HELP,
    QUANTIZATIONS,
    GenerateFormState,
    plan_counts,
    plan_summary,
    to_generate_namespace,
    validate,
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

    CSS = """
    #title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }
    #subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }
    .card {
        border: tall $primary;
        padding: 1 2;
        margin: 1 2;
    }
    .card-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    .field-label {
        color: $text-muted;
        margin-top: 1;
    }
    #plan-line {
        text-align: center;
        color: $success;
        margin: 1 2;
    }
    #errors-line {
        text-align: center;
        color: $error;
        margin: 0 2 1 2;
    }
    #button-row {
        align: center middle;
        height: auto;
        margin: 1 2 2 2;
    }
    #button-generate {
        margin-right: 2;
    }
    #run-view {
        margin: 1 2;
    }
    #run-head {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
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
    """

    BINDINGS = [
        ("ctrl+g", "generate", "Generate"),
        ("ctrl+q", "quit_app", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.initial_state = GenerateFormState()
        self._run_dir: Path | None = None
        self._running = False
        # Plain-text record of every run-view line (RichLog exposes no
        # public line count; this keeps the progress assertable).
        self.run_history: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="form-view"):
            yield Static("🎬  VOYAGE", id="title")
            yield Static(
                "Autonomous infinite audiovisual voyage — edit the settings, "
                "then Generate. (Ctrl+G generates, Ctrl+Q quits.)",
                id="subtitle",
            )
            with Vertical(classes="card"):
                yield Static("1 · Story  —  required", classes="card-title")
                yield Static("Style", classes="field-label")
                yield Input(
                    placeholder="pastel neon line-art, peaceful",
                    id="field-style",
                    tooltip=FIELD_HELP["style"],
                )
                yield Static("Run id", classes="field-label")
                yield Input(
                    value=self.initial_state.run_id,
                    id="field-run-id",
                    tooltip=FIELD_HELP["run_id"],
                )
                yield Static("Seed", classes="field-label")
                yield Input(
                    value=self.initial_state.seed,
                    id="field-seed",
                    tooltip=FIELD_HELP["seed"],
                )
            with Vertical(classes="card"):
                yield Static("2 · Video & audio", classes="card-title")
                yield Static("Backend", classes="field-label")
                yield Select(
                    [(name, name) for name in BACKENDS],
                    value=self.initial_state.backend,
                    allow_blank=False,
                    id="field-backend",
                    tooltip=FIELD_HELP["backend"],
                )
                yield Static("Duration", classes="field-label")
                yield Input(
                    value=self.initial_state.duration,
                    id="field-duration",
                    tooltip=FIELD_HELP["duration"],
                )
                yield Static("Director", classes="field-label")
                yield Select(
                    [(name, name) for name in DIRECTORS],
                    value=self.initial_state.director,
                    allow_blank=False,
                    id="field-director",
                    tooltip=FIELD_HELP["director"],
                )
                yield Static("Quantization", classes="field-label")
                yield Select(
                    [(name, name) for name in QUANTIZATIONS],
                    value=self.initial_state.quantization,
                    allow_blank=False,
                    id="field-quantization",
                    tooltip=FIELD_HELP["quantization"],
                )
            with Collapsible(title="Advanced …", collapsed=True):
                yield Static("Blocks per segment", classes="field-label")
                yield Input(
                    placeholder="1",
                    id="field-blocks",
                    tooltip=FIELD_HELP["blocks"],
                )
                yield Static("Take seconds", classes="field-label")
                yield Input(
                    placeholder="45.0",
                    id="field-take-seconds",
                    tooltip=FIELD_HELP["take_seconds"],
                )
                yield Static("Beats per segment", classes="field-label")
                yield Input(
                    placeholder="4",
                    id="field-beats",
                    tooltip=FIELD_HELP["beats_per_segment"],
                )
                yield Static("Drift every N segments", classes="field-label")
                yield Input(
                    placeholder="1",
                    id="field-drift",
                    tooltip=FIELD_HELP["drift_every_n"],
                )
                yield Static("Output dir (empty = output/<run-id>)", classes="field-label")
                yield Input(
                    placeholder="output/voyage",
                    id="field-output",
                    tooltip=FIELD_HELP["output"],
                )
                yield Static("Final video path (empty = <run>/final.mp4)", classes="field-label")
                yield Input(
                    placeholder="<run>/final.mp4",
                    id="field-final-video",
                    tooltip=FIELD_HELP["final_video"],
                )
                yield Checkbox("Draft profile (fast low-res iteration)", id="flag-draft")
                yield Checkbox("Force (init into a non-empty directory)", id="flag-force")
                yield Checkbox("Skip bad segments at finalize", id="flag-skip-bad")
                yield Checkbox("Verbose console lines behind the TUI", id="flag-verbose")
                yield Checkbox("No color (plain output)", id="flag-no-color")
            yield Static("", id="plan-line")
            yield Static("", id="errors-line")
            with Horizontal(id="button-row"):
                yield Button("Generate ▶", variant="primary", id="button-generate")
                yield Button("Quit", id="button-quit")
        with Vertical(id="run-view"):
            yield Static("", id="run-head")
            yield ProgressBar(id="run-bar")
            yield RichLog(id="run-log", highlight=False)
            yield Static("", id="run-result")
            with Horizontal(id="run-buttons"):
                yield Button("Stop ■", variant="error", id="button-stop")
                yield Button("← Back", id="button-back")
                yield Button("Quit", id="button-quit-run")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#run-view", Vertical).display = False
        self._refresh_plan()

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
            style=text("#field-style"),
            run_id=text("#field-run-id"),
            output=text("#field-output"),
            seed=text("#field-seed"),
            final_video=text("#field-final-video"),
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

    def _refresh_plan(self) -> None:
        state = self._read_form()
        self.query_one("#plan-line", Static).update(f"plan: {plan_summary(state)}")

    def on_input_changed(self, _event: Input.Changed) -> None:
        self._refresh_plan()

    def on_select_changed(self, _event: Select.Changed) -> None:
        self._refresh_plan()

    def action_generate(self) -> None:
        if not self.query_one("#form-view", ScrollableContainer).display:
            return  # already running — the Stop button owns control
        self._start_generation()

    def action_quit_app(self) -> None:
        self.exit()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        pressed = event.button.id
        if pressed == "button-generate":
            self.action_generate()
        elif pressed in ("button-quit", "button-quit-run"):
            self.exit()
        elif pressed == "button-stop":
            self._request_stop()
        elif pressed == "button-back":
            self._show_form()

    def _start_generation(self) -> None:
        state = self._read_form()
        errors = validate(state)
        if errors:
            self.query_one("#errors-line", Static).update(" · ".join(errors))
            return
        self.query_one("#errors-line", Static).update("")
        counts = plan_counts(state)
        total = counts[0] if counts is not None else 0
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
        try:
            code = cmd_generate(namespace)
        except VoyageError as exc:
            self.call_from_thread(self._finish_generation, f"✗ generation failed: {exc}")
        except Exception as exc:  # defensive: never trap the worker silently
            self.call_from_thread(self._finish_generation, f"✗ unexpected error: {exc!r}")
        else:
            if code == 0:
                final = namespace.final_video or str(run_dir / "final.mp4")
                self.call_from_thread(self._finish_generation, f"✓ generated {final}")
            else:
                self.call_from_thread(
                    self._finish_generation, f"✗ generation exited with code {code}"
                )

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
        self.query_one("#run-result", Static).update(result)
        self.query_one("#button-stop", Button).disabled = True

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
        self._refresh_plan()
