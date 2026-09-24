"""Launcher form state for the bare-command Textual TUI (no Textual import).

Why this module exists: the TUI form (``voyage/tui.py``) needs
validation, ``generate``-namespace building, and plan math that is
unit-testable without a terminal. Everything here is pure stdlib, so
``tests/test_tui.py`` covers it without Textual running. The Textual app
only reads widgets into :class:`GenerateFormState` and calls these
helpers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

BACKENDS = ("ltxv", "longlive2", "fake")
DIRECTORS = ("qwen", "deterministic")
QUANTIZATIONS = ("fp8", "bf16")

# Frames each committed segment carries per backend (mirrors
# voyage.cli._frames_per_segment: LTXV block 0 renders 25 native frames,
# each extension block adds 24 new ones; other backends use segment_frames).
_LTXV_FIRST_BLOCK_FRAMES = 25
_DEFAULT_SEGMENT_FRAMES = 48
_FPS = 24

FIELD_HELP = {
    "backend": "Video backend preset (geometry + device + audio pairing). "
    "ltxv/longlive2 need the CUDA worker image + a GPU.",
    "duration": "Target length, e.g. '5s', '90', '1m30s', '2m'. Rounds up to whole segments.",
    "style": "Human-owned style string. Required — baked into the run config and every prompt.",
    "run_id": "Run name; also the default output folder name.",
    "seed": "Base seed (integer). Video/audio takes derive deterministically from it.",
    "output": "Run directory. Empty = output/<run-id> under the current directory.",
    "final_video": "Final mp4 path. Empty = <run>/final.mp4.",
    "director": "Director backend. qwen drifts the story every Nth segment; "
    "deterministic holds the style.",
    "blocks": "Video blocks per segment (empty = preset default 1). More blocks = longer segments.",
    "take_seconds": "Music take length in seconds (empty = 45). Must stay above the "
    "20s audio-ahead window.",
    "quantization": "DiT weight precision. bf16 keeps highlights clean at +~4.5GB VRAM.",
    "beats_per_segment": "Beats per segment for the rhythm grid (empty = 4, "
    "doubles to hold >=60 BPM).",
    "drift_every_n": "Director drifts every Nth segment (empty = 1); other segments hold.",
}


@dataclass
class GenerateFormState:
    """Raw TUI field values (strings as typed, flags as checked).

    Numeric fields stay strings so empty means "preset default" and the
    app can show the field-specific error without losing what the user
    typed. :func:`validate` + :func:`to_generate_namespace` parse them.
    """

    backend: str = "ltxv"
    duration: str = "5s"
    style: str = ""
    run_id: str = "voyage"
    output: str = ""
    seed: str = "0"
    final_video: str = ""
    force: bool = False
    skip_bad: bool = False
    draft: bool = False
    director: str = "qwen"
    blocks: str = ""
    take_seconds: str = ""
    quantization: str = "fp8"
    beats_per_segment: str = ""
    drift_every_n: str = ""
    verbose: bool = False
    no_color: bool = False


def textual_available() -> bool:
    """Whether the Textual display dependency can be imported."""
    import importlib.util

    return importlib.util.find_spec("textual") is not None


def _positive_int(raw: str, field_name: str, errors: list[str]) -> int | None:
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{field_name} must be a positive integer, got {raw!r}")
        return None
    if value <= 0:
        errors.append(f"{field_name} must be positive, got {raw!r}")
        return None
    return value


def validate(state: GenerateFormState) -> list[str]:
    """Field errors in form order (empty = ready to generate)."""
    from voyage.cli import parse_duration

    errors: list[str] = []
    if state.backend not in BACKENDS:
        errors.append(f"backend must be one of {', '.join(BACKENDS)}, got {state.backend!r}")
    try:
        parse_duration(state.duration)
    except ValueError as exc:
        errors.append(str(exc))
    if not state.style.strip():
        errors.append("style must be a non-empty human-owned style string")
    if not state.run_id.strip():
        errors.append("run-id must be non-empty")
    try:
        int(state.seed)
    except ValueError:
        errors.append(f"seed must be an integer, got {state.seed!r}")
    if state.director not in DIRECTORS:
        errors.append(f"director must be one of {', '.join(DIRECTORS)}, got {state.director!r}")
    if state.blocks.strip():
        _positive_int(state.blocks.strip(), "blocks", errors)
    if state.take_seconds.strip():
        try:
            take = float(state.take_seconds.strip())
        except ValueError:
            errors.append(f"take-seconds must be a positive number, got {state.take_seconds!r}")
        else:
            if take <= 0:
                errors.append(f"take-seconds must be positive, got {state.take_seconds!r}")
    if state.quantization not in QUANTIZATIONS:
        errors.append(
            f"quantization must be one of {', '.join(QUANTIZATIONS)}, got {state.quantization!r}"
        )
    if state.beats_per_segment.strip():
        _positive_int(state.beats_per_segment.strip(), "beats-per-segment", errors)
    if state.drift_every_n.strip():
        _positive_int(state.drift_every_n.strip(), "drift-every-n", errors)
    return errors


def to_generate_namespace(state: GenerateFormState) -> argparse.Namespace:
    """Build the ``cmd_generate`` namespace (raises ValueError if invalid)."""
    from voyage.cli import parse_duration

    errors = validate(state)
    if errors:
        raise ValueError("; ".join(errors))

    def optional_int(raw: str) -> int | None:
        return int(raw.strip()) if raw.strip() else None

    take_raw = state.take_seconds.strip()
    return argparse.Namespace(
        backend=state.backend,
        duration=parse_duration(state.duration),
        style=state.style.strip(),
        run_id=state.run_id.strip(),
        output=state.output.strip() or None,
        seed=int(state.seed),
        force=state.force,
        final_video=state.final_video.strip() or None,
        skip_bad=state.skip_bad,
        draft=state.draft,
        director=state.director,
        blocks=optional_int(state.blocks),
        take_seconds=float(take_raw) if take_raw else None,
        quantization=state.quantization,
        beats_per_segment=optional_int(state.beats_per_segment),
        drift_every_n=optional_int(state.drift_every_n),
        verbose=state.verbose,
        no_color=state.no_color,
    )


def plan_counts(state: GenerateFormState) -> tuple[int, int, float] | None:
    """(segments, frames, seconds) for the current form, or None if invalid."""
    from voyage.cli import parse_duration, segments_for_duration

    try:
        duration_seconds = parse_duration(state.duration)
    except ValueError:
        return None
    blocks = 1
    if state.blocks.strip():
        try:
            blocks = int(state.blocks.strip())
        except ValueError:
            return None
        if blocks <= 0:
            return None
    if state.backend == "ltxv":
        frames_per_segment = _LTXV_FIRST_BLOCK_FRAMES + (blocks - 1) * (
            _LTXV_FIRST_BLOCK_FRAMES - 1
        )
    else:
        frames_per_segment = _DEFAULT_SEGMENT_FRAMES
    segments = segments_for_duration(duration_seconds, _FPS, frames_per_segment)
    planned_frames = segments * frames_per_segment
    return segments, planned_frames, planned_frames / _FPS


def plan_summary(state: GenerateFormState) -> str:
    """One-line derived plan (segments/frames/seconds) or the first error."""
    from voyage.cli import parse_duration

    try:
        parse_duration(state.duration)
    except ValueError as exc:
        return f"cannot plan: {exc}"
    if state.blocks.strip():
        try:
            blocks = int(state.blocks.strip())
        except ValueError:
            return f"cannot plan: blocks must be a positive integer, got {state.blocks!r}"
        if blocks <= 0:
            return f"cannot plan: blocks must be positive, got {state.blocks!r}"
    counts = plan_counts(state)
    if counts is None:  # defensive: inputs above already parsed cleanly
        return "cannot plan: check duration/blocks"
    segments, planned_frames, seconds = counts
    if state.backend == "ltxv":
        blocks = int(state.blocks.strip()) if state.blocks.strip() else 1
        frames_per_segment = _LTXV_FIRST_BLOCK_FRAMES + (blocks - 1) * (
            _LTXV_FIRST_BLOCK_FRAMES - 1
        )
    else:
        frames_per_segment = _DEFAULT_SEGMENT_FRAMES
    return (
        f"≈{seconds:.1f}s · {segments} segment(s) · {planned_frames} frames "
        f"· {state.backend} {frames_per_segment}f/segment @ {_FPS}fps"
    )
