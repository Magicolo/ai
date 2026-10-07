"""Beautiful console progress for runs (console-only layer, DESIGN §59).

Why this module exists: long GPU stages (video render, ACE-Step takes,
Qwen director calls) used to run silent for minutes — the only feedback
was a bare ``committed segment`` line at the end. This module adds
per-segment headers with the full video + audio prompts, animated
spinners with live elapsed timers per stage, and a timing summary per
commit, so progress and stalls are visible at a glance.

Console-only by contract: styling never touches ``logs/metrics.jsonl``
or any artifact — those stay machine-readable plain text. ``rich`` is
an optional display dependency: when it is missing, the stream is not
a TTY (tests, pipes), or ``--no-color``/``NO_COLOR`` is set, every
method degrades to plain ``print`` lines with the same words, so
substring assertions in tests keep passing either way. Non-TTY prints
stage boundaries only (start + finish — per-unit updates would spam
logs); ``quiet=True`` silences everything except failures plus the
final paths.

Two complementary arms: ``stage()`` marks one checklist step with a
spinner + elapsed timer (unknown duration), while ``bar()`` counts X/Y
inside it (SFX windows, ACE takes, drained segments) with % + elapsed
+ ETA when the total is known. ``timing_table()`` closes a phase with
per-stage seconds + slowest + total. Library layers (media/augment/
audio/sfx) never print — they take an optional ``progress`` sink
(``VoyageConsole`` satisfies it; ``None`` = silent) and report upward.

Stream contract (063/028): every method — including ``error()`` — writes
to the injected ``stream`` (default ``sys.stdout``), never to the real
``sys.stderr`` directly, so embeds and tests capturing the stream see
the same words either way.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from typing import Any, Protocol, TextIO

from voyage.motion_sense import LOW_MOTION_FLOOR

_SPINNER_TICK_SECONDS = 0.2
"""Live-elapsed refresh interval for the rich spinner (fast enough to feel
alive, slow enough to never fight the render thread)."""


def rich_available() -> bool:
    """Whether the ``rich`` display dependency can be imported."""
    import importlib.util

    return importlib.util.find_spec("rich") is not None


class SegmentProgress(Protocol):
    """Progress sink the supervisor reports each commit to (DESIGN §59).

    The supervisor holds an optional ``SegmentProgress`` (``None`` =
    silent, the default for tests and library use). ``RichSegmentProgress``
    below is the console implementation. Plain dicts cross the boundary
    so the supervisor never imports display code paths.
    """

    def segment_start(self, number: int, segment_id: str) -> None:
        """A new segment commit began (number + zero-padded id)."""
        ...

    def note(self, message: str) -> None:
        """One background-worker line (prefetch submit/hit, pre-warm ledger)."""
        ...

    @property
    def verbose(self) -> bool:
        """Whether --verbose detail lines are wanted (sweep breakdowns)."""
        ...

    def bar(self, label: str, total: int | None = None) -> AbstractContextManager[Any]:
        """Determinate bar (known total) or spinner (total=None, unknown).

        The owner advances via the yielded tracker (`update()` /
        `set_total()`); exactly one bar is ever live at a time (never
        two concurrent rich Live displays).
        """
        ...

    def stage(self, label: str, detail: str = "") -> AbstractContextManager[Any]:
        """Spinner + elapsed timer around one commit stage."""
        ...

    def segment_plan(self, info: dict[str, Any]) -> None:
        """Director decision + prompts, shown before the long render.

        Keys: number, segment_id, destination, phase, novelty_accepted,
        drift_hold, prefetch_hit, director_backend, video_backend,
        audio_backend, geometry, fps, planned_frames, planned_duration,
        blocks, video_prompts, video_seeds, scene_cuts,
        transition_mechanism, transition_stages, audio_caption,
        audio_energy, audio_bpm, audio_beats, audio_texture,
        audio_environment, audio_sfx_caption, notes.
        """
        ...

    def segment_done(self, info: dict[str, Any]) -> None:
        """Commit summary after state advance.

        Keys: number, segment_id, frames, duration, take_ids,
        take_action, take_reason, beats, bpm, video_backend,
        overlap_fraction, overlap_cap_seconds, stage_seconds, elapsed.
        """
        ...


class VoyageConsole:
    """Single styling touchpoint for all ``voyage`` console output.

    ``verbose=False`` prints the compact default (segment header with
    destinations, geometry, beat grid and timings — no per-segment prompts).
    ``verbose=True`` adds the full video + audio prompts per segment plus
    prefetch reasons and payload minutiae. Colors/animation engage only
    on a real TTY with ``rich`` installed and color allowed.
    ``quiet=True`` silences everything except failures (``error()`` and
    the failure arm of ``stage()``/``bar()`` still report) — for
    scripts that only care about the exit code plus the final paths.
    """

    def __init__(
        self,
        verbose: bool = False,
        no_color: bool = False,
        stream: TextIO | None = None,
        quiet: bool = False,
    ) -> None:
        self._verbose = verbose
        self._no_color = no_color or bool(os.environ.get("NO_COLOR"))
        self._stream = stream if stream is not None else sys.stdout
        self._quiet = quiet
        self._rich: Any = None
        if not self._no_color and self._is_tty and rich_available():
            from rich.console import Console

            self._rich = Console(file=self._stream, color_system="auto", highlight=False)

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def quiet(self) -> bool:
        return self._quiet

    @property
    def color_enabled(self) -> bool:
        return self._rich is not None

    @property
    def _is_tty(self) -> bool:
        isatty = getattr(self._stream, "isatty", None)
        try:
            return bool(isatty()) if callable(isatty) else False
        except ValueError:
            return False

    def line(self, text: str = "") -> None:
        """One plain line (wraps long prompts when rich owns the stream)."""
        if self._quiet:
            return
        if self._rich is not None:
            self._rich.print(text, markup=False, highlight=False)
        else:
            print(text, file=self._stream)

    def styled(self, icon: str, text: str, style: str) -> None:
        """Icon + text, colored on a color terminal, plain otherwise."""
        if self._quiet:
            return
        if self._rich is not None:
            from rich.text import Text

            rendered = Text()
            rendered.append(f"{icon} ", style=style)
            rendered.append(text)
            self._rich.print(rendered)
        else:
            print(f"{icon} {text}", file=self._stream)

    def info(self, message: str) -> None:
        self.styled("▸", message, "cyan")

    def ok(self, message: str) -> None:
        self.styled("✓", message, "green")

    def warn(self, message: str) -> None:
        self.styled("⚠", message, "yellow")

    def error(self, message: str) -> None:
        print(f"✗ {message}", file=self._stream)

    def rule(self, title: str) -> None:
        """Section header (rich rule on a TTY, plain dashes otherwise)."""
        if self._quiet:
            return
        if self._rich is not None and self._is_tty:
            self._rich.rule(title)
        else:
            self.line(f"── {title} ──")

    @contextmanager
    def stage(self, label: str, detail: str = "") -> Iterator[None]:
        """Spinner with a live elapsed timer; always reports the total."""
        started = time.monotonic()
        head = f"{label} … {detail}".rstrip(" …")
        if self._quiet:
            try:
                yield
            except BaseException:
                elapsed = time.monotonic() - started
                self.error(f"{label} failed after {elapsed:.1f}s")
                raise
            return
        if self._rich is not None and self._is_tty:
            from rich.status import Status

            status: Status = self._rich.status(f"◌ {head}", spinner="dots")
            stop = threading.Event()

            def _tick() -> None:
                while not stop.wait(_SPINNER_TICK_SECONDS):
                    elapsed = time.monotonic() - started
                    status.update(f"◌ {head} ({elapsed:.1f}s)")

            ticker = threading.Thread(target=_tick, daemon=True)
            status.start()
            ticker.start()
            try:
                yield
            except BaseException:
                stop.set()
                status.stop()
                elapsed = time.monotonic() - started
                self.styled("✗", f"{label} failed after {elapsed:.1f}s", "red")
                raise
            stop.set()
            status.stop()
            elapsed = time.monotonic() - started
            self.styled("✓", f"{label} ({elapsed:.1f}s)", "green")
        else:
            self.line(f"▸ {head} ...")
            try:
                yield
            except BaseException:
                elapsed = time.monotonic() - started
                self.line(f"✗ {label} failed after {elapsed:.1f}s")
                raise
            elapsed = time.monotonic() - started
            self.line(f"✓ {label} ({elapsed:.1f}s)")

    @contextmanager
    def bar(self, label: str, total: int | None = None) -> Iterator[BarTracker]:
        """Determinate bar (known total) or spinner (total=None, unknown).

        Hybrid checklist arm next to ``stage()``: ``stage()`` marks one
        checklist step, ``bar()`` counts X/Y inside it (SFX windows,
        ACE takes, model-pass chunks) with % + elapsed + ETA when the
        total is known. Non-TTY prints stage boundaries only (start +
        finish lines — per-update lines would spam logs); quiet prints
        nothing on success and the failure line on error.
        """
        tracker = BarTracker(self, label, total)
        tracker._begin()
        try:
            yield tracker
        except BaseException:
            tracker._fail()
            raise
        else:
            tracker._finish()

    def timing_table(self, title: str, timings: Mapping[str, float]) -> None:
        """Per-stage seconds + slowest stage + total (finalize summary).

        Plain words (``⏱`` + ``name Ns`` cells) so substring assertions
        hold with or without rich — the same degradation contract as
        every other method here.
        """
        if self._quiet or not timings:
            return
        cells = " · ".join(f"{name} {seconds:.1f}s" for name, seconds in timings.items())
        total = sum(timings.values())
        slowest = max(timings.items(), key=lambda item: item[1])
        self.line(
            f"     ⏱ {title}: {cells} · slowest {slowest[0]} ({slowest[1]:.1f}s)"
            f" · total {total:.1f}s"
        )

    def segment_start(self, number: int, segment_id: str) -> None:
        self.line("")
        self.styled("▶", f"SEGMENT {segment_id} (#{number})", "bold cyan")

    def segment_plan(self, info: dict[str, Any]) -> None:
        """Decision detail: compact header by default, full prompts verbose."""
        destination = str(info.get("destination", ""))
        phase = str(info.get("phase", ""))
        novel = "novel ✓" if info.get("novelty_accepted") else "hold"
        drift = " · drift hold" if info.get("drift_hold") else ""
        backend = str(info.get("director_backend", ""))
        self.styled("◆", f"drift → {destination} · {phase} · {novel}{drift} ({backend})", "magenta")

        geometry = str(info.get("geometry", ""))
        fps = info.get("fps", 0)
        frames = info.get("planned_frames", 0)
        duration = info.get("planned_duration", 0.0)
        blocks = info.get("blocks", 1)
        cuts = info.get("scene_cuts", [])
        cut_flag = " · scene-cut" if any(bool(cut) for cut in cuts) else ""
        self.styled(
            "🎬",
            f"video · {info.get('video_backend')} · {geometry} @{fps}fps · "
            f"{frames}f ≈ {duration:.2f}s · {blocks} block(s){cut_flag}",
            "blue",
        )
        prompts = info.get("video_prompts", [])
        if self._verbose and isinstance(prompts, list):
            for index, prompt in enumerate(prompts):
                tag = f"prompt [{index + 1}/{len(prompts)}]" if len(prompts) > 1 else "prompt"
                self.line(f"     {tag}: {prompt}")
        if self._verbose:
            seeds = info.get("video_seeds", [])
            self.line(f"     seeds: {seeds} · scene_cuts: {cuts}")
            mechanism = str(info.get("transition_mechanism", ""))
            for stage_text in info.get("transition_stages", []):
                self.line(f"     transition ({mechanism}): {stage_text}")

        if self._verbose:
            caption = str(info.get("audio_caption", ""))
            energy = info.get("audio_energy", 0.0)
            beats = info.get("audio_beats", 0)
            bpm = info.get("audio_bpm", 0.0)
            self.styled(
                "🎵",
                f"audio · {info.get('audio_backend')} · {beats} beats @ {bpm:.0f} BPM · "
                f"energy {energy:.2f}",
                "yellow",
            )
            self.line(f"     music: {caption}")
        # Third caption family (issue 161): the SFX caption the director
        # computed is the only pre-finalize signal for what the MMAudio
        # pass will condition on. Verbose-gated with the other prompts so
        # the default generate output stays a compact header; a missing
        # caption on pre-SFX runs is itself visible as "-".
        if self._verbose:
            sfx_caption = str(info.get("audio_sfx_caption", "") or "-")
            self.line(f"     sfx: {sfx_caption}")
        if self._verbose:
            texture = str(info.get("audio_texture", ""))
            environment = info.get("audio_environment", [])
            if texture:
                self.line(f"     texture: {texture}")
            if environment:
                self.line(f"     environment: {environment}")
        notes = str(info.get("notes", ""))
        if notes and self._verbose:
            self.line(f"     notes: {notes}")

    def segment_done(self, info: dict[str, Any]) -> None:
        """Commit summary: frames, take, beat grid, per-stage timings."""
        segment_id = str(info.get("segment_id", ""))
        frames = info.get("frames", 0)
        duration = info.get("duration", 0.0)
        summary = f"SEGMENT {segment_id} committed · {frames}f ≈ {duration:.2f}s"
        if self._verbose:
            takes = info.get("take_ids", [])
            take_action = str(info.get("take_action", "keep"))
            takes_text = ", ".join(str(take) for take in takes) if takes else "no take"
            beats = info.get("beats", 0)
            bpm = info.get("bpm", 0.0)
            summary += f" · {takes_text} ({take_action}) · {beats} beats @ {bpm:.0f} BPM"
        self.ok(summary)
        motion = info.get("motion_energy")
        sense_seconds = info.get("motion_seconds")
        if isinstance(motion, (int, float)) and not isinstance(motion, bool):
            flag = ""
            if math.isfinite(float(motion)) and float(motion) < LOW_MOTION_FLOOR:
                flag = " · LOW-MOTION — next prompt steers harder"
            seconds_text = (
                f" ({float(sense_seconds):.2f}s sense)"
                if isinstance(sense_seconds, (int, float)) and not isinstance(sense_seconds, bool)
                else ""
            )
            self.line(f"     motion energy {float(motion):.3f}{seconds_text}{flag}")
        elif "motion_energy" in info:
            self.line("     motion energy unknown (sense skipped)")
        if self._verbose:
            reason = str(info.get("take_reason", ""))
            if reason:
                self.line(f"     take: {str(info.get('take_action', 'keep'))} — {reason}")
            fraction = info.get("overlap_fraction", 0.0)
            cap = info.get("overlap_cap_seconds", 0.0)
            self.line(f"     finalize blend: {fraction} of shortest neighbor, cap {cap}s")
        stages = info.get("stage_seconds", {})
        prefetch = " · prefetch hit" if info.get("prefetch_hit") else ""
        if isinstance(stages, dict) and stages:
            if not self._verbose:
                stages = {name: seconds for name, seconds in stages.items() if name != "audio"}
            if stages:
                cells = " · ".join(f"{name} {seconds:.1f}s" for name, seconds in stages.items())
                self.line(f"     ⏱ {cells}{prefetch} · total {info.get('elapsed', 0.0):.1f}s")
            elif prefetch:
                self.line(f"     ⏱{prefetch}")
        elif prefetch:
            self.line(f"     ⏱{prefetch}")

    @contextmanager
    def parallel_downloads(self, labels: Sequence[str]) -> Iterator[ParallelDownloadTracker]:
        """Per-model spinners for parallel downloads (rich TTY) or plain lines.

        Same degradation contract as every other method here: non-TTY,
        missing rich, or `--no-color`/`NO_COLOR` prints one plain line per
        model instead of animated spinners, with the same words.
        """
        tracker = ParallelDownloadTracker(self, list(labels))
        tracker._begin()
        try:
            yield tracker
        finally:
            tracker._finish()


class RichSegmentProgress:
    """Console implementation of the supervisor's SegmentProgress sink."""

    def __init__(self, console: VoyageConsole) -> None:
        self._console = console

    def segment_start(self, number: int, segment_id: str) -> None:
        self._console.segment_start(number, segment_id)

    def note(self, message: str) -> None:
        self._console.info(message)

    @property
    def verbose(self) -> bool:
        return self._console.verbose

    def stage(self, label: str, detail: str = "") -> AbstractContextManager[Any]:
        return self._console.stage(label, detail)

    def segment_plan(self, info: dict[str, Any]) -> None:
        self._console.segment_plan(info)

    def segment_done(self, info: dict[str, Any]) -> None:
        self._console.segment_done(info)

    @contextmanager
    def bar(self, label: str, total: int | None = None) -> Iterator[BarTracker]:
        with self._console.bar(label, total) as tracker:
            yield tracker

    def timing_table(self, title: str, timings: Mapping[str, float]) -> None:
        self._console.timing_table(title, timings)


class ParallelDownloadTracker:
    """Per-model finish reporter for `VoyageConsole.parallel_downloads`.

    Why a separate object: hub fetches run on worker threads but `rich`
    is not thread-safe for concurrent task updates — the `ensure_models`
    driver reports each completion from its main-thread `as_completed`
    loop, so all progress writes stay on one thread. Main thread only.
    """

    def __init__(self, console: VoyageConsole, labels: list[str]) -> None:
        self._console = console
        self._labels = labels
        self._started: dict[str, float] = {}
        self._progress: Any = None
        self._tasks: dict[str, Any] = {}

    def _begin(self) -> None:
        now = time.monotonic()
        for label in self._labels:
            self._started[label] = now
        if self._console._rich is not None and self._console._is_tty:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                TimeElapsedColumn(),
                console=self._console._rich,
                transient=False,
            )
            self._progress.start()
            for label in self._labels:
                self._tasks[label] = self._progress.add_task(label, total=1, completed=0)
        else:
            for label in self._labels:
                self._console.line(f"▸ downloading {label} ...")

    def succeed(self, label: str) -> None:
        """Mark one model fetched + verified (prints its elapsed time)."""
        elapsed = time.monotonic() - self._started.get(label, time.monotonic())
        if self._progress is not None:
            task_id = self._tasks.get(label)
            if task_id is None:
                self._console.line(f"⚠ unknown download label: {label}")
            else:
                self._progress.update(task_id, completed=1)
        elif label not in self._started:
            self._console.line(f"⚠ unknown download label: {label}")
        self._console.styled("✓", f"{label} ready ({elapsed:.1f}s)", "green")

    def fail(self, label: str, detail: str = "") -> None:
        """Mark one model failed (stays on stdout so TUI capture keeps it)."""
        suffix = f": {detail}" if detail else ""
        if self._progress is not None:
            task_id = self._tasks.get(label)
            if task_id is None:
                self._console.line(f"⚠ unknown download label: {label}")
            else:
                self._progress.update(task_id, completed=1)
        elif label not in self._started:
            self._console.line(f"⚠ unknown download label: {label}")
        self._console.styled("✗", f"{label} failed{suffix}", "red")

    def _finish(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None


class BarTracker:
    """X/Y counter for one ``VoyageConsole.bar`` step (main thread only).

    Why a separate object: same reason as ``ParallelDownloadTracker`` —
    all progress writes stay on the thread that entered the ``bar()``
    context (rich ``Progress`` updates are lock-guarded, but one owner
    keeps the finish accounting exact). Callers advance after each
    completed unit; ``total=None`` (unknown upfront, e.g. ACE takes)
    shows a spinner + count until ``set_total`` learns the total.
    """

    def __init__(self, console: VoyageConsole, label: str, total: int | None) -> None:
        self._console = console
        self._label = label
        self._total = total
        self._done = 0
        self._started = time.monotonic()
        self._progress: Any = None
        self._task: Any = None
        self._extra: str | None = None

    @property
    def done(self) -> int:
        return self._done

    def _begin(self) -> None:
        if self._console._quiet:
            return
        if self._console._rich is not None and self._console._is_tty:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            # One determinate layout whether or not the total is known yet:
            # rich renders a total=None task without X/Y or ETA, and the
            # same columns pick up the bar + ETA live once set_total()
            # learns it — a spinner-only layout picked here could never
            # upgrade (the observed "⠧ interp frames 0:01:54" with no
            # X/Y or ETA for the whole pass).
            self._progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console._rich,
                transient=False,
            )
            self._progress.start()
            self._task = self._progress.add_task(self._label, total=self._total)
        else:
            suffix = f" (0/{self._total})" if self._total is not None else ""
            self._console.line(f"▸ {self._label} ...{suffix}")

    def update(self, advance: int = 1) -> None:
        """Mark units complete (finish accounting reads ``done``)."""
        self._done += advance
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, advance=advance)

    def set_total(self, total: int) -> None:
        """Learn the total mid-step (X/Y + ETA appear live on a TTY)."""
        self._total = total
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, total=total)

    def set_extra(self, extra: str) -> None:
        """Extra detail appended to the finish line (e.g. a frames/s rate).

        Opt-in per bar — unset bars keep the historical `X/Y, Ns` format
        byte-identical, so existing pins never move.
        """
        self._extra = extra

    def _finish(self) -> None:
        elapsed = time.monotonic() - self._started
        if self._progress is not None:
            if self._task is not None:
                self._progress.update(self._task, completed=self._done)
            self._progress.stop()
            self._progress = None
        extra = f", {self._extra}" if self._extra else ""
        if not self._console._quiet:
            if self._total is not None:
                self._console.styled(
                    "✓",
                    f"{self._label} ({self._done}/{self._total}, {elapsed:.1f}s{extra})",
                    "green",
                )
            else:
                self._console.styled(
                    "✓", f"{self._label} ({self._done} done, {elapsed:.1f}s{extra})", "green"
                )

    def _fail(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        elapsed = time.monotonic() - self._started
        if self._console._quiet:
            self._console.error(f"{self._label} failed after {elapsed:.1f}s")
        else:
            self._console.styled("✗", f"{self._label} failed after {elapsed:.1f}s", "red")


@contextmanager
def optional_stage(progress: VoyageConsole | None, label: str, detail: str = "") -> Iterator[None]:
    """Spinner around one stage, silent no-op when ``progress`` is None.

    Library layers (media/augment/audio/sfx) report through this so
    every ``progress`` parameter stays a one-line ``with`` — no
    ``nullcontext`` ternaries at the call sites.
    """
    if progress is None:
        with nullcontext():
            yield
    else:
        with progress.stage(label, detail):
            yield


@contextmanager
def optional_bar(
    progress: VoyageConsole | None, label: str, total: int | None = None
) -> Iterator[BarTracker | None]:
    """X/Y counter around one step, silent no-op when ``progress`` is None.

    Yields the tracker (or ``None`` when silent — guard ``update()``
    calls with ``if tracker is not None``).
    """
    if progress is None:
        yield None
    else:
        with progress.bar(label, total) as tracker:
            yield tracker


class ParallelFinalizeDisplay:
    """One shared live display for N concurrent streams (parallel work).

    Why: each ``BarTracker`` owns its own ``Progress`` + ``Live`` — two
    concurrent bars would run two Live displays on the same terminal and
    garble. The coordinator owns ONE ``Progress`` (on the main console's
    rich console, the same object sequential bars already share) plus a
    lock; per-stream ``ParallelStreamDisplay`` views route their bars to
    shared tasks and their lines through the lock. Text and bars stay
    whole; the non-TTY path degrades to lock-guarded boundary lines and
    quiet stays silent, exactly like the sequential display.

    First used for the two-stream finalize forks (model∥music, sfx∥
    publish); generation reuses the same coordinator for the video∥
    model-pass fork (the video stream's stage degrades to plain lines,
    so the model-pass bar owns the only Live).
    """

    def __init__(self, console: VoyageConsole) -> None:
        self._console = console
        self._lock = threading.Lock()
        self._progress: Any = None

    def _live(self) -> bool:
        return self._console._rich is not None and self._console._is_tty

    def _ensure_progress(self) -> Any:
        if self._progress is None and self._live() and not self._console._quiet:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console._rich,
                transient=False,
            )
            self._progress.start()
        return self._progress

    def add_task(self, label: str, total: int | None) -> Any:
        """Register one bar; returns the task id (None when silent)."""
        with self._lock:
            if self._console._quiet:
                return None
            progress = self._ensure_progress()
            if progress is None:
                return None
            return progress.add_task(label, total=total)

    def update_task(self, task: Any, advance: int = 1) -> None:
        with self._lock:
            if self._progress is not None and task is not None:
                self._progress.update(task, advance=advance)

    def set_task_total(self, task: Any, total: int) -> None:
        with self._lock:
            if self._progress is not None and task is not None:
                self._progress.update(task, total=total)

    def complete_task(self, task: Any, done: int) -> None:
        with self._lock:
            if self._progress is not None and task is not None:
                with suppress(Exception):
                    self._progress.update(task, completed=done)

    def stream_view(self, label: str) -> VoyageConsole:
        """One per-stream view (a VoyageConsole, so signatures never churn)."""
        return ParallelStreamDisplay(self, label)

    def close(self) -> None:
        """Stop the shared display (bars already finished via their contexts)."""
        with self._lock:
            if self._progress is not None:
                with suppress(Exception):
                    self._progress.stop()
                self._progress = None


class ParallelStreamDisplay(VoyageConsole):
    """One stream's view of a ``ParallelFinalizeDisplay`` (branch threads).

    A real VoyageConsole sharing the coordinator's rich console object,
    so branch ``optional_bar``/``optional_stage`` calls behave exactly
    like sequential ones. Bars route to shared tasks; stages degrade to
    lock-guarded plain lines (two spinners would fight the shared Live —
    same reason the model∥music fork silences its branches); every text
    emit holds the coordinator lock so concurrent lines stay whole.
    """

    def __init__(self, display: ParallelFinalizeDisplay, label: str) -> None:
        source = display._console
        self._verbose = source._verbose
        self._no_color = source._no_color
        self._stream = source._stream
        self._quiet = source._quiet
        self._rich = source._rich
        self._display = display
        self._stream_label = label

    @contextmanager
    def stage(self, label: str, detail: str = "") -> Iterator[None]:
        """Lock-guarded plain lines (no spinner — shared Live owns the screen)."""
        started = time.monotonic()
        head = f"{label} … {detail}".rstrip(" …")
        if self._quiet:
            try:
                yield
            except BaseException:
                elapsed = time.monotonic() - started
                self.error(f"{label} failed after {elapsed:.1f}s")
                raise
            return
        with self._display._lock:
            VoyageConsole.line(self, f"▸ {head} ...")
        try:
            yield
        except BaseException:
            elapsed = time.monotonic() - started
            with self._display._lock:
                VoyageConsole.line(self, f"✗ {label} failed after {elapsed:.1f}s")
            raise
        elapsed = time.monotonic() - started
        with self._display._lock:
            VoyageConsole.line(self, f"✓ {label} ({elapsed:.1f}s)")

    @contextmanager
    def bar(self, label: str, total: int | None = None) -> Iterator[BarTracker]:
        """Route the bar to a shared task (boundary lines when not live)."""
        tracker = SharedBarTracker(self, label, total, self._display)
        tracker._begin()
        try:
            yield tracker
        except BaseException:
            tracker._fail()
            raise
        else:
            tracker._finish()

    def line(self, text: str = "") -> None:
        with self._display._lock:
            VoyageConsole.line(self, text)

    def styled(self, icon: str, text: str, style: str) -> None:
        with self._display._lock:
            VoyageConsole.styled(self, icon, text, style)

    def error(self, message: str) -> None:
        with self._display._lock:
            VoyageConsole.error(self, message)

    # NOTE (no-deadlock rule): only leaf emits are overridden here. The
    # base `info`/`ok` fan out through `self.styled`, so they are
    # inherited untouched — overriding them with their own lock (and
    # calling super) would re-acquire this non-reentrant lock on the
    # same thread and wedge the caller forever.


class SharedBarTracker(BarTracker):
    """A ``BarTracker`` routed to the shared parallel display (branch use).

    Same finish-line contract as ``BarTracker`` (opt-in ``set_extra``,
    historical format otherwise); only the live task is shared instead
    of owned, so ``_finish`` completes the task without stopping the
    display — the coordinator stops it after the join.
    """

    def __init__(
        self,
        console: ParallelStreamDisplay,
        label: str,
        total: int | None,
        display: ParallelFinalizeDisplay,
    ) -> None:
        super().__init__(console, label, total)
        self._display = display

    def _begin(self) -> None:
        if self._console._quiet:
            return
        self._task = self._display.add_task(self._label, self._total)
        if self._task is None:
            suffix = f" (0/{self._total})" if self._total is not None else ""
            with self._display._lock:
                VoyageConsole.line(self._console, f"▸ {self._label} ...{suffix}")

    def update(self, advance: int = 1) -> None:
        """Mark units complete (finish accounting reads ``done``)."""
        self._done += advance
        self._display.update_task(self._task, advance)

    def set_total(self, total: int) -> None:
        """Learn the total mid-step (X/Y + ETA appear live on a TTY)."""
        self._total = total
        self._display.set_task_total(self._task, total)

    def _finish(self) -> None:
        self._display.complete_task(self._task, self._done)
        elapsed = time.monotonic() - self._started
        extra = f", {self._extra}" if self._extra else ""
        if not self._console._quiet:
            with self._display._lock:
                if self._total is not None:
                    VoyageConsole.styled(
                        self._console,
                        "✓",
                        f"{self._label} ({self._done}/{self._total}, {elapsed:.1f}s{extra})",
                        "green",
                    )
                else:
                    VoyageConsole.styled(
                        self._console,
                        "✓",
                        f"{self._label} ({self._done} done, {elapsed:.1f}s{extra})",
                        "green",
                    )

    def _fail(self) -> None:
        self._display.complete_task(self._task, self._done)
        elapsed = time.monotonic() - self._started
        if self._console._quiet:
            self._console.error(f"{self._label} failed after {elapsed:.1f}s")
        else:
            with self._display._lock:
                VoyageConsole.styled(
                    self._console, "✗", f"{self._label} failed after {elapsed:.1f}s", "red"
                )


#: Generation-side name for the shared live display (same coordinator as
#: the finalize forks — stream-count agnostic; the generation fork is
#: video ∥ model-pass). A plain alias so call sites read in their own
#: domain without a second implementation to drift.
GenerationDisplay = ParallelFinalizeDisplay
