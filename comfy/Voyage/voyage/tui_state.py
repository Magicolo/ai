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

DESIGN §140 (launcher-TUI as-built): bare ``voyage`` configures the
one-shot ``generate`` command; plan math below stays single-sourced
with ``cli._frames_per_segment`` (issue 024) so TUI predictions can
never drift from CLI truth.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ImportError:  # Python 3.10 worker image (upstream env)
    import tomli as tomllib

from voyage.cli_paths import _RESERVED_FOLDER_NAMES as _RESERVED_FOLDER_NAMES
from voyage.cli_paths import is_flat_folder_name as _shared_flat_folder_name

BACKENDS = ("ltxv", "longlive2", "causvid", "fake")
DIRECTORS = ("qwen", "deterministic")
QUANTIZATIONS = ("fp8", "bf16")


def _fake_row_frames_and_fps() -> tuple[int, int]:
    """(segment_frames, fps) of the fake registry row, else literals.

    Single source (issue 085): unknown backends plan with the fake row's
    geometry. The literals 48/24 below are the unreachable fallback — they
    only fire when `voyage.config` itself cannot import (config needs
    pydantic; this module stays stdlib-only at import time, so the registry
    read is a guarded call, not a top-level import).
    """
    try:
        from voyage.config import BACKEND_REGISTRY

        fake = BACKEND_REGISTRY["fake"]
    except (ImportError, AttributeError, KeyError):
        return 48, 24
    if fake.segment_frames > 0 and fake.fps > 0:
        return fake.segment_frames, fake.fps
    return 48, 24


_FALLBACK_FRAMES_PER_SEGMENT, _FALLBACK_FPS = _fake_row_frames_and_fps()
"""Frames/segment + fps for unknown backends (the fake registry row)."""

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


# Planning math is NOT duplicated here: _planning_frames_and_fps below calls
# cli._frames_per_segment (frames) + the config video preset (fps) — the
# same values cmd_generate plans with. There is no module-level constant
# to drift (issue 024: the old 25/24 ltxv + flat-48 longlive2 math is gone).
# Native frame rate per backend lives in the config video preset; unknown
# backends plan at 24fps (mirrors _frames_per_segment's segment_frames
# default).

FIELD_HELP = {
    "backend": "Video backend preset (geometry + device + audio pairing). "
    "ltxv/longlive2/causvid need the CUDA worker image + a GPU. "
    "longlive2 is deprecated (issue 079, kept for existing runs only) — "
    "new runs should use ltxv.",
    "duration": "Target length, e.g. '5s', '90', '1m30s', '2m', '1h', '1h2m3.5s'. "
    "Rounds up to whole segments, so the video never runs short.",
    "style": "Human-owned style string. Required — baked into the run config and every prompt.",
    "name": "Run + folder name. Required — the run lands in output/<name>/ with final.mp4 inside.",
    "seed": "Base seed (integer). Video/audio takes derive deterministically from it.",
    "director": "Director backend. qwen drifts the story every Nth segment; "
    "deterministic holds the style.",
    "blocks": "Video blocks per segment (empty = preset default 1). More blocks = longer segments.",
    "take_seconds": "Music take length in seconds (empty = 45). Must stay above the "
    "20s audio-ahead window.",
    "quantization": "DiT weight precision. bf16 keeps highlights clean at +~4.5GB VRAM. "
    "Untouched (fp8) inherits the stored run config.",
    "beats_per_segment": "Beats per segment for the rhythm grid (empty = 4, "
    "doubles to hold >=60 BPM).",
    "drift_every_n": "Director drifts every Nth segment (empty = 1); other segments hold.",
    "min_fps": "Floor output fps at finalize (untouched/32 = stored config; 0 disables).",
    "min_resolution": 'Floor output resolution at finalize, WxH e.g. "1280x720" '
    '(untouched/1280x720 = stored config; "0" disables).',
    "no_download": "Fail instead of downloading missing models (verify only). "
    "Off by default — checked runs never touch the network for weights.",
    "no_sfx": "Skip the finalize-time SFX pass even when [sfx] is configured. "
    "Off by default — SFX backend/device overrides stay CLI-only.",
}


# Form defaults that encode "inherit the stored run config" (issue 023).
# The TUI pre-fills these widgets, so a blank field is unreachable without
# the user actively clearing it — the namespace therefore treats the
# default value itself as absent (Unset). Tradeoff, documented: explicitly
# re-selecting the default (e.g. stored bf16 back to fp8) is
# indistinguishable from untouched, so that downgrade needs the CLI flag.
# Non-default values always emit concrete overrides and still win.
_DEFAULT_QUANTIZATION = "fp8"
_DEFAULT_MIN_FPS = "32"
_DEFAULT_MIN_RESOLUTION = "1280x720"


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
    min_fps: str = "32"
    min_resolution: str = "1280x720"
    verbose: bool = False
    no_color: bool = False
    no_download: bool = False
    no_sfx: bool = False


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


# Windows device names can never be photo-folders on any host checkout, so
# the TUI rejects them (case-insensitive, extension-insensitive) alongside
# "." — initializing inside output/ itself would scatter run files among
# every other run (issue 080). The set lives once in `voyage.cli_paths`
# (issue 085 single source — a direct import either way would cycle:
# cli_paths is stdlib-only, so this top-level import is cycle-free while
# `tui_state -> cli` or `cli -> tui_state` would not be).
def _flat_folder_name(raw: str) -> bool:
    """Whether the value is usable as a single output folder name.

    Thin alias over `voyage.cli_paths.is_flat_folder_name` (single source,
    issue 085) — kept under this name for the existing TUI call sites and
    `tests/test_tui_state.py`.
    """
    return _shared_flat_folder_name(raw)


def field_errors(state: GenerateFormState) -> dict[str, str]:
    """Per-field error messages keyed by form field name (empty = valid).

    The TUI renders these inline (invalid highlighting + help panel) and
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
        blocks_errors: list[str] = []
        _positive_int(state.blocks.strip(), "blocks", blocks_errors)
        if blocks_errors:
            errors["blocks"] = blocks_errors[0]
    if state.take_seconds.strip():
        try:
            take = float(state.take_seconds.strip())
        except ValueError:
            errors["take_seconds"] = (
                f"take-seconds must be a positive number, got {state.take_seconds!r}"
            )
        else:
            # nan slips past every comparison (nan <= 0 is False) and inf
            # passes positivity, so finiteness is checked first (issue 062).
            if not math.isfinite(take) or take <= 0:
                errors["take_seconds"] = (
                    f"take-seconds must be a positive finite number, got {state.take_seconds!r}"
                )
            else:
                # Domain floor (issue 111): the runtime requires
                # take_seconds > ahead_seconds (AudioConfig validator) —
                # a field-valid form must never promise a Generate that
                # exits 2. Single-sourced from the model default (issue
                # 024), never a restated literal; blank stays valid.
                from voyage.config import AudioConfig

                ahead_window = AudioConfig.model_fields["ahead_seconds"].default
                ahead_floor = (
                    float(ahead_window) if isinstance(ahead_window, (int, float)) else 20.0
                )
                if take <= ahead_floor:
                    errors["take_seconds"] = (
                        "take-seconds must exceed the audio-ahead window "
                        f"({ahead_floor}s), got {state.take_seconds!r}"
                    )
    if state.quantization not in QUANTIZATIONS:
        errors["quantization"] = (
            f"quantization must be one of {', '.join(QUANTIZATIONS)}, got {state.quantization!r}"
        )
    if state.beats_per_segment.strip():
        beats_errors: list[str] = []
        _positive_int(state.beats_per_segment.strip(), "beats-per-segment", beats_errors)
        if beats_errors:
            errors["beats_per_segment"] = beats_errors[0]
    if state.drift_every_n.strip():
        drift_errors: list[str] = []
        _positive_int(state.drift_every_n.strip(), "drift-every-n", drift_errors)
        if drift_errors:
            errors["drift_every_n"] = drift_errors[0]
    if state.min_fps.strip():
        try:
            min_fps_value = int(state.min_fps.strip())
        except ValueError:
            errors["min_fps"] = f"min-fps must be a non-negative integer, got {state.min_fps!r}"
        else:
            if min_fps_value < 0:
                errors["min_fps"] = f"min-fps must be a non-negative integer, got {state.min_fps!r}"
    if state.min_resolution.strip():
        from voyage.config import AugmentConfig, parse_min_resolution

        try:
            width, height = parse_min_resolution(state.min_resolution.strip())
            AugmentConfig(min_width=width, min_height=height)
        except ValueError as exc:
            errors["min_resolution"] = (
                f"min-resolution must be WxH or 0 to disable, got {state.min_resolution!r} ({exc})"
            )
    return errors


def validate(state: GenerateFormState) -> list[str]:
    """Field errors in form order (empty = ready to generate)."""
    return list(field_errors(state).values())


def to_generate_namespace(state: GenerateFormState) -> argparse.Namespace:
    """Build the ``cmd_generate`` namespace (raises ValueError if invalid)."""
    from voyage.cli import parse_duration
    from voyage.config import Unset, UnsetType

    errors = validate(state)
    if errors:
        raise ValueError("; ".join(errors))

    def optional_int(raw: str) -> int | UnsetType:
        # Absent-encoding (issue 045): blank form fields emit Unset — the
        # single "not provided" value — instead of a second encoding
        # (None) that config consumers would also have to agree on.
        # `apply_draft_overrides`/`resolve_config` tolerate both.
        return int(raw.strip()) if raw.strip() else Unset

    take_raw = state.take_seconds.strip()
    min_fps_raw = state.min_fps.strip()
    min_resolution_raw = state.min_resolution.strip()
    quantization_raw = state.quantization
    name = state.name.strip()
    output = str(Path("output") / name)
    return argparse.Namespace(
        backend=state.backend,
        duration=parse_duration(state.duration),
        style=state.style.strip(),
        run_id=name,
        name=name,
        output=output,
        seed=int(state.seed),
        force=state.force,
        final_video=str(Path(output) / "final.mp4"),
        skip_bad=state.skip_bad,
        draft=state.draft,
        director=state.director,
        blocks=optional_int(state.blocks),
        take_seconds=float(take_raw) if take_raw else Unset,
        quantization=Unset if quantization_raw == _DEFAULT_QUANTIZATION else quantization_raw,
        beats_per_segment=optional_int(state.beats_per_segment),
        drift_every_n=optional_int(state.drift_every_n),
        # Finalize-time augment floors (issue 023): blank AND the prefilled
        # form default both mean "stored TOML wins" (Unset); 0 / "0"
        # explicitly disable a floor, anything else overrides. An untouched
        # form therefore resolves to the stored quantization/floors
        # unchanged; the CLI downgrade to a default value needs --quantization
        # / --min-fps / --min-resolution flags. no_augment stays False —
        # the TUI has no disable-all checkbox (set 0 / "0" explicitly or
        # pass --no-augment on the CLI).
        min_fps=Unset if min_fps_raw in ("", _DEFAULT_MIN_FPS) else int(min_fps_raw),
        min_resolution=(
            Unset if min_resolution_raw in ("", _DEFAULT_MIN_RESOLUTION) else min_resolution_raw
        ),
        no_augment=False,
        verbose=state.verbose,
        no_color=state.no_color,
        # Verify-only + SFX opt-out (issues 145/182): explicit form
        # checkboxes, so the TUI namespace always satisfies cmd_generate
        # AND the user can decline downloads or the heavy SFX pass.
        # SFX backend/device/model overrides stay CLI-only (see help).
        no_download=state.no_download,
        # Finalize-time SFX pass-through (the generate parser defaults;
        # kept explicit so the TUI namespace always satisfies cmd_finalize).
        no_sfx=state.no_sfx,
        sfx_backend=None,
        sfx_caption=None,
        sfx_device=None,
        sfx_model_size=None,
        sfx_workers=1,
    )


def _planning_frames_and_fps(backend: str, blocks: int) -> tuple[int, int]:
    """(frames_per_segment, fps) for TUI planning, from the CLI single source.

    Import-direction note (verified live 2026-09-25): voyage.cli never
    imports voyage.tui_state at module level — its only TUI touch is the
    lazy ``from voyage.tui import run_tui`` inside ``launch_tui`` — so
    these function-level imports cannot cycle (same pattern as the
    existing parse_duration/segments_for_duration imports below).
    Frames come from ``cli._frames_per_segment`` on a planning-only
    ProjectConfig and fps from the config video preset: the same values
    ``cmd_generate`` plans with. Unknown backends fall back to 48 frames
    @ 24fps, mirroring ``_frames_per_segment``'s segment_frames default.
    """
    from voyage.cli import _frames_per_segment
    from voyage.config import ProjectConfig, VideoBackendName, VideoConfig, _video_preset

    if backend not in BACKENDS:
        # Unknown backends fall back to fake geometry (the backend
        # Literal below would otherwise raise ValidationError instead).
        return _FALLBACK_FRAMES_PER_SEGMENT, _FALLBACK_FPS
    try:
        preset = _video_preset(backend)
        preset_fps = preset.get("fps", 24)
        fps = preset_fps if isinstance(preset_fps, int) and preset_fps > 0 else 24
        preset_frames = preset.get("segment_frames", _FALLBACK_FRAMES_PER_SEGMENT)
    except ValueError:
        fps = 24
        preset_frames = _FALLBACK_FRAMES_PER_SEGMENT
    frames = (
        preset_frames
        if isinstance(preset_frames, int) and preset_frames > 0
        else _FALLBACK_FRAMES_PER_SEGMENT
    )
    planning_config = ProjectConfig(
        style="planning",
        video=VideoConfig(
            backend=cast(VideoBackendName, backend),
            blocks_per_segment=blocks,
            fps=fps,
            segment_frames=frames,
        ),
    )
    return _frames_per_segment(planning_config), fps


def _plan_details(state: GenerateFormState) -> tuple[int, int, float, int, int] | None:
    """Full plan struct, or None when duration/blocks do not parse.

    Single source (issue 024): frames/fps come from
    :func:`_planning_frames_and_fps` (CLI truth). Both public planners
    build on this struct and never re-derive frame math. An unknown
    backend (issue 178: a typo, not a future backend) has no plan —
    None, so the plan line agrees with the errors line instead of
    quoting fallback geometry the run will never produce.
    """
    from voyage.cli import parse_duration, segments_for_duration

    if state.backend not in BACKENDS:
        return None
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
    frames_per_segment, fps = _planning_frames_and_fps(state.backend, blocks)
    segments = segments_for_duration(duration_seconds, fps, frames_per_segment)
    planned_frames = segments * frames_per_segment
    return segments, planned_frames, planned_frames / fps, frames_per_segment, fps


def plan_counts(state: GenerateFormState) -> tuple[int, int, float] | None:
    """(segments, frames, seconds) for the current form, or None if invalid."""
    details = _plan_details(state)
    if details is None:
        return None
    segments, planned_frames, seconds, _frames_per_segment, _fps = details
    return segments, planned_frames, seconds


def plan_summary(state: GenerateFormState) -> str:
    """One-line derived plan (segments/frames/seconds) or the first error."""
    from voyage.cli import parse_duration

    if state.backend not in BACKENDS:
        return f"cannot plan: backend must be one of {', '.join(BACKENDS)}, got {state.backend!r}"
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
    details = _plan_details(state)
    if details is None:  # defensive: inputs above already parsed cleanly
        return "cannot plan: check duration/blocks"
    segments, planned_frames, seconds, frames_per_segment, fps = details
    return (
        f"≈{seconds:.1f}s · {segments} segment(s) · {planned_frames} frames "
        f"· {state.backend} {frames_per_segment}f/segment @ {fps}fps"
    )


def _toml_string(raw: str) -> str:
    """Quote a string as a TOML basic string (single shared escaper).

    Thin alias over ``voyage.config._toml_basic_string`` (lazy import so
    this module stays stdlib-only at import time): short escapes cover
    backslash/quote/newline/return/tab and every other C0 control plus
    DEL becomes ``\\uXXXX`` (issue 020 — one stray byte can never corrupt
    tui-last.toml into a total form reset, issue 072).
    """
    from voyage.config import _toml_basic_string

    return _toml_basic_string(raw)


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
            f"min_fps = {_toml_string(state.min_fps)}",
            f"min_resolution = {_toml_string(state.min_resolution)}",
            f"verbose = {'true' if state.verbose else 'false'}",
            f"no_color = {'true' if state.no_color else 'false'}",
            f"no_download = {'true' if state.no_download else 'false'}",
            f"no_sfx = {'true' if state.no_sfx else 'false'}",
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
        min_fps=_string_field(parsed, "min_fps", defaults.min_fps),
        min_resolution=_string_field(parsed, "min_resolution", defaults.min_resolution),
        verbose=_boolean_field(parsed, "verbose", defaults.verbose),
        no_color=_boolean_field(parsed, "no_color", defaults.no_color),
        no_download=_boolean_field(parsed, "no_download", defaults.no_download),
        no_sfx=_boolean_field(parsed, "no_sfx", defaults.no_sfx),
    )


def gpu_warning(backend: str) -> str:
    """One-line CUDA/GPU notice for CUDA backends, else an empty string.

    The CUDA set is cli._CUDA_BACKENDS (single source with the run.sh
    image selection + the torch fast-fail, issue 024): the union covers
    the video/audio/sfx vocabularies, so the SFX ``mmaudio`` backend
    warns here too (issue 021). ``acestep`` membership is inert here
    because the TUI never offers it.
    """
    from voyage.cli import _CUDA_BACKENDS

    if backend in _CUDA_BACKENDS:
        return (
            f"{backend} needs the CUDA worker image (VOYAGE_IMAGE=voyage-video) "
            "plus a GPU (--gpus all)."
        )
    return ""
