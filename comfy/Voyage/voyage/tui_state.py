"""Launcher form state for the bare-command Textual TUI (no Textual import).

Why this module exists: the TUI form (``voyage/tui.py``) needs
validation, ``generate``-namespace building, and plan math that is
unit-testable without a terminal. Everything here is pure stdlib, so
``tests/test_tui.py`` covers it without Textual running. The Textual app
only reads widgets into :class:`GenerateFormState` and calls these
helpers. Last-settings persistence lives here for the same reason: the
next launch prefills the form from ``LAST_SETTINGS_PATH`` via
:func:`load_last_settings` (Stream B), and every Generate stores the
submitted form via :func:`save_last_settings`. Both directions stay
silent on I/O failure so a bad home directory can never break the UI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # Python 3.10 worker image (upstream env)
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

BACKENDS = ("ltxv", "longlive2", "fake")
DIRECTORS = ("qwen", "deterministic")
QUANTIZATIONS = ("fp8", "bf16")

# Home for the last submitted TUI form (TOML). Stream B loads it at launch
# to prefill the form and saves it on every Generate.
LAST_SETTINGS_PATH: Path = Path.home() / ".config" / "voyage" / "tui-last.toml"


def _default_settings_path() -> Path:
    """Home-relative settings path, resolved at call time (not import).

    Call-time resolution keeps HOME redirection working (tests isolate the
    home directory per case); the module constant above is the default
    value for documentation and explicit passing.
    """
    return Path.home() / ".config" / "voyage" / "tui-last.toml"


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
    "name": "Run + folder name. Required — the run lands in output/<name>/ with final.mp4 inside.",
    "seed": "Base seed (integer). Video/audio takes derive deterministically from it.",
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
    name: str = "voyage"
    seed: str = "0"
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


def _flat_folder_name(raw: str) -> bool:
    """Whether the value is usable as a single output folder name."""
    text = raw.strip()
    return bool(text) and "/" not in text and "\\" not in text and ".." not in text


def field_errors(state: GenerateFormState) -> dict[str, str]:
    """Per-field error messages keyed by form field name (empty = valid).

    The TUI renders these inline (red borders + help panel) and
    :func:`validate` flattens them for the one-line errors display.
    """
    from voyage.cli import parse_duration

    errors: dict[str, str] = {}
    if state.backend not in BACKENDS:
        errors["backend"] = f"backend must be one of {', '.join(BACKENDS)}, got {state.backend!r}"
    try:
        parse_duration(state.duration)
    except ValueError as exc:
        errors["duration"] = str(exc)
    if not state.style.strip():
        errors["style"] = "style must be a non-empty human-owned style string"
    if not _flat_folder_name(state.name):
        errors["name"] = f"name must be a flat folder name (no slashes), got {state.name!r}"
    try:
        int(state.seed)
    except ValueError:
        errors["seed"] = f"seed must be an integer, got {state.seed!r}"
    if state.director not in DIRECTORS:
        errors["director"] = (
            f"director must be one of {', '.join(DIRECTORS)}, got {state.director!r}"
        )
    if state.blocks.strip():
        slot: list[str] = []
        _positive_int(state.blocks.strip(), "blocks", slot)
        if slot:
            errors["blocks"] = slot[0]
    if state.take_seconds.strip():
        try:
            take = float(state.take_seconds.strip())
        except ValueError:
            errors["take_seconds"] = (
                f"take-seconds must be a positive number, got {state.take_seconds!r}"
            )
        else:
            if take <= 0:
                errors["take_seconds"] = (
                    f"take-seconds must be positive, got {state.take_seconds!r}"
                )
    if state.quantization not in QUANTIZATIONS:
        errors["quantization"] = (
            f"quantization must be one of {', '.join(QUANTIZATIONS)}, got {state.quantization!r}"
        )
    if state.beats_per_segment.strip():
        beats_slot: list[str] = []
        _positive_int(state.beats_per_segment.strip(), "beats-per-segment", beats_slot)
        if beats_slot:
            errors["beats_per_segment"] = beats_slot[0]
    if state.drift_every_n.strip():
        drift_slot: list[str] = []
        _positive_int(state.drift_every_n.strip(), "drift-every-n", drift_slot)
        if drift_slot:
            errors["drift_every_n"] = drift_slot[0]
    return errors


def validate(state: GenerateFormState) -> list[str]:
    """Field errors in form order (empty = ready to generate)."""
    return list(field_errors(state).values())


def to_generate_namespace(state: GenerateFormState) -> argparse.Namespace:
    """Build the ``cmd_generate`` namespace (raises ValueError if invalid)."""
    from voyage.cli import parse_duration

    errors = validate(state)
    if errors:
        raise ValueError("; ".join(errors))

    def optional_int(raw: str) -> int | None:
        return int(raw.strip()) if raw.strip() else None

    take_raw = state.take_seconds.strip()
    name = state.name.strip()
    output = str(Path("output") / name)
    return argparse.Namespace(
        backend=state.backend,
        duration=parse_duration(state.duration),
        style=state.style.strip(),
        run_id=name,
        output=output,
        seed=int(state.seed),
        force=state.force,
        final_video=str(Path(output) / "final.mp4"),
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


def _toml_string(raw: str) -> str:
    """Quote a string as a TOML basic string."""
    escaped = (
        raw.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def save_last_settings(state: GenerateFormState, path: Path | None = None) -> None:
    """Persist every form field as TOML; silently ignore write failures.

    The file is rewritten wholesale (no merge with previous contents) so a
    fresh launch replays exactly what was last submitted. Any OSError
    (read-only home, path under a file, ...) returns silently — persistence
    must never raise into the UI. The path resolves at call time (not at
    import) so HOME redirection keeps working.
    """
    resolved = path if path is not None else _default_settings_path()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Last TUI form settings (rewritten on every Generate).",
            f"backend = {_toml_string(state.backend)}",
            f"duration = {_toml_string(state.duration)}",
            f"style = {_toml_string(state.style)}",
            f"name = {_toml_string(state.name)}",
            f"seed = {_toml_string(state.seed)}",
            f"force = {'true' if state.force else 'false'}",
            f"skip_bad = {'true' if state.skip_bad else 'false'}",
            f"draft = {'true' if state.draft else 'false'}",
            f"director = {_toml_string(state.director)}",
            f"blocks = {_toml_string(state.blocks)}",
            f"take_seconds = {_toml_string(state.take_seconds)}",
            f"quantization = {_toml_string(state.quantization)}",
            f"beats_per_segment = {_toml_string(state.beats_per_segment)}",
            f"drift_every_n = {_toml_string(state.drift_every_n)}",
            f"verbose = {'true' if state.verbose else 'false'}",
            f"no_color = {'true' if state.no_color else 'false'}",
        ]
        resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return


def _string_field(raw: dict[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    return value if isinstance(value, str) else default


def _choice_field(raw: dict[str, Any], key: str, default: str, choices: tuple[str, ...]) -> str:
    value = raw.get(key, default)
    if isinstance(value, str) and value in choices:
        return value
    return default


def _boolean_field(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    return value if isinstance(value, bool) else default


def load_last_settings(path: Path | None = None) -> GenerateFormState:
    """Reload the last submitted form; defaults when missing/unparseable.

    Unknown keys are ignored; each wrong-typed or invalid field falls back
    to that field's default while valid fields survive — loading never
    raises, so a hand-edited or half-written file cannot break the launch.
    The path resolves at call time (not at import) so HOME redirection
    keeps working.
    """
    defaults = GenerateFormState()
    resolved = path if path is not None else _default_settings_path()
    try:
        parsed: dict[str, Any] = tomllib.loads(resolved.read_bytes().decode("utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return defaults
    # Legacy files (pre-Name-merge) carried run_id/output/final_video: the
    # run_id migrates to name when name is absent; the paths are dropped
    # (runs now always live in output/<name>/ with final.mp4 inside).
    name = _string_field(parsed, "name", "")
    if not name:
        name = _string_field(parsed, "run_id", defaults.name)
    return GenerateFormState(
        backend=_choice_field(parsed, "backend", defaults.backend, BACKENDS),
        duration=_string_field(parsed, "duration", defaults.duration),
        style=_string_field(parsed, "style", defaults.style),
        name=name,
        seed=_string_field(parsed, "seed", defaults.seed),
        force=_boolean_field(parsed, "force", defaults.force),
        skip_bad=_boolean_field(parsed, "skip_bad", defaults.skip_bad),
        draft=_boolean_field(parsed, "draft", defaults.draft),
        director=_choice_field(parsed, "director", defaults.director, DIRECTORS),
        blocks=_string_field(parsed, "blocks", defaults.blocks),
        take_seconds=_string_field(parsed, "take_seconds", defaults.take_seconds),
        quantization=_choice_field(parsed, "quantization", defaults.quantization, QUANTIZATIONS),
        beats_per_segment=_string_field(parsed, "beats_per_segment", defaults.beats_per_segment),
        drift_every_n=_string_field(parsed, "drift_every_n", defaults.drift_every_n),
        verbose=_boolean_field(parsed, "verbose", defaults.verbose),
        no_color=_boolean_field(parsed, "no_color", defaults.no_color),
    )


def gpu_warning(backend: str) -> str:
    """One-line CUDA/GPU notice for CUDA backends, else an empty string."""
    if backend in ("ltxv", "longlive2"):
        return (
            f"{backend} needs the CUDA worker image (VOYAGE_IMAGE=voyage-video) "
            "plus a GPU (--gpus all)."
        )
    return ""
