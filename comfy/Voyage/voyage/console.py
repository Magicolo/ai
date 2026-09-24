"""Beautiful console progress for runs (console-only layer).

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
substring assertions in tests keep passing either way.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, TextIO


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
        audio_environment, notes.
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
    full video + audio prompts, spinners, timing summary).
    ``verbose=True`` adds retry feedback, beat math, take decisions,
    prefetch reasons and payload minutiae. Colors/animation engage only
    on a real TTY with ``rich`` installed and color allowed.
    """

    def __init__(
        self,
        verbose: bool = False,
        no_color: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self._verbose = verbose
        self._no_color = no_color or bool(os.environ.get("NO_COLOR"))
        self._stream = stream if stream is not None else sys.stdout
        self._rich: Any = None
        if not self._no_color and self._is_tty and rich_available():
            from rich.console import Console

            self._rich = Console(file=self._stream, color_system="auto", highlight=False)

    @property
    def verbose(self) -> bool:
        return self._verbose

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
        if self._rich is not None:
            self._rich.print(text, markup=False, highlight=False)
        else:
            print(text, file=self._stream)

    def styled(self, icon: str, text: str, style: str) -> None:
        """Icon + text, colored on a color terminal, plain otherwise."""
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
        print(f"✗ {message}", file=sys.stderr)

    def rule(self, title: str) -> None:
        """Section header (rich rule on a TTY, plain dashes otherwise)."""
        if self._rich is not None and self._is_tty:
            self._rich.rule(title)
        else:
            self.line(f"── {title} ──")

    @contextmanager
    def stage(self, label: str, detail: str = "") -> Iterator[None]:
        """Spinner with a live elapsed timer; always reports the total."""
        started = time.monotonic()
        head = f"{label} … {detail}".rstrip(" …")
        if self._rich is not None and self._is_tty:
            from rich.status import Status

            status: Status = self._rich.status(f"◌ {head}", spinner="dots")
            stop = threading.Event()

            def _tick() -> None:
                while not stop.wait(0.2):
                    elapsed = time.monotonic() - started
                    status.update(f"◌ {head} ({elapsed:.1f}s)")

            ticker = threading.Thread(target=_tick, daemon=True)
            status.start()
            ticker.start()
            try:
                yield
            except Exception:
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
            except Exception:
                self.line(f"✗ {label} failed")
                raise
            elapsed = time.monotonic() - started
            self.line(f"✓ {label} in {elapsed:.1f}s")

    def segment_start(self, number: int, segment_id: str) -> None:
        self.line("")
        self.styled("▶", f"SEGMENT {segment_id} (#{number})", "bold cyan")

    def segment_plan(self, info: dict[str, Any]) -> None:
        """Decision detail: destination, full video + audio prompts."""
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
        if isinstance(prompts, list):
            for index, prompt in enumerate(prompts):
                tag = f"prompt [{index + 1}/{len(prompts)}]" if len(prompts) > 1 else "prompt"
                self.line(f"     {tag}: {prompt}")
        if self._verbose:
            seeds = info.get("video_seeds", [])
            self.line(f"     seeds: {seeds} · scene_cuts: {cuts}")
            mechanism = str(info.get("transition_mechanism", ""))
            for stage_text in info.get("transition_stages", []):
                self.line(f"     transition ({mechanism}): {stage_text}")

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
        takes = info.get("take_ids", [])
        take_action = str(info.get("take_action", "keep"))
        takes_text = ", ".join(str(take) for take in takes) if takes else "no take"
        beats = info.get("beats", 0)
        bpm = info.get("bpm", 0.0)
        self.ok(
            f"SEGMENT {segment_id} committed · {frames}f ≈ {duration:.2f}s · "
            f"{takes_text} ({take_action}) · {beats} beats @ {bpm:.0f} BPM"
        )
        if self._verbose:
            reason = str(info.get("take_reason", ""))
            if reason:
                self.line(f"     take: {take_action} — {reason}")
            fraction = info.get("overlap_fraction", 0.0)
            cap = info.get("overlap_cap_seconds", 0.0)
            self.line(f"     finalize blend: {fraction} of shortest neighbor, cap {cap}s")
        stages = info.get("stage_seconds", {})
        prefetch = " · prefetch hit" if info.get("prefetch_hit") else ""
        if isinstance(stages, dict) and stages:
            cells = " · ".join(f"{name} {seconds:.1f}s" for name, seconds in stages.items())
            self.line(f"     ⏱ {cells}{prefetch} · total {info.get('elapsed', 0.0):.1f}s")
        elif prefetch:
            self.line(f"     ⏱{prefetch}")


class RichSegmentProgress:
    """Console implementation of the supervisor's SegmentProgress sink."""

    def __init__(self, console: VoyageConsole) -> None:
        self._console = console

    def segment_start(self, number: int, segment_id: str) -> None:
        self._console.segment_start(number, segment_id)

    def stage(self, label: str, detail: str = "") -> AbstractContextManager[Any]:
        return self._console.stage(label, detail)

    def segment_plan(self, info: dict[str, Any]) -> None:
        self._console.segment_plan(info)

    def segment_done(self, info: dict[str, Any]) -> None:
        self._console.segment_done(info)
