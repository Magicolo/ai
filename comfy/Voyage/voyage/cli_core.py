"""Shared CLI core: console factory, run loading, augment overrides.

DESIGN §58 — leaf module of the issue-080 split: the small helpers every verb
needs (`get_console`, `_load_run`, `_augment_overrides`) without any
verb-to-verb edge, so verb modules import them at top level and the
only lazy imports left are the three orchestration call-throughs
(`cmd_stop→cmd_finalize`, `cmd_generate` fan-out, `cmd_benchmark→
cmd_init`, `cmd_inspect→validate_run`). `voyage.cli` re-exports all.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.config import ProjectConfig, is_provided, load_config
from voyage.console import VoyageConsole


def get_console(args: argparse.Namespace) -> VoyageConsole:
    """Console for a subcommand (flags default off for test Namespaces)."""
    return VoyageConsole(
        verbose=bool(getattr(args, "verbose", False)),
        no_color=bool(getattr(args, "no_color", False)),
    )


def _load_run(run: Path) -> tuple[ProjectConfig, str]:
    return load_config(run / paths.CONFIG_FILENAME)


def _augment_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """CLI augment flags → resolve_config kwargs (Track A).

    `--no-augment` wins over explicit floors (both to 0) and forces the
    model pass off. Every read goes through getattr + `is_provided` so
    TUI/hand-built namespaces (Unset blanks, missing attrs) resolve to
    absent, never to a value.
    """
    if bool(getattr(args, "no_augment", False)):
        return {"min_fps": 0, "min_resolution": "0", "use_model_pass": False}
    overrides: dict[str, Any] = {}
    min_fps = getattr(args, "min_fps", None)
    if is_provided(min_fps):
        overrides["min_fps"] = min_fps
    min_resolution = getattr(args, "min_resolution", None)
    if is_provided(min_resolution):
        overrides["min_resolution"] = min_resolution
    use_model_pass = getattr(args, "use_model_pass", None)
    if is_provided(use_model_pass):
        overrides["use_model_pass"] = use_model_pass
    interp_multiplier = getattr(args, "interp_multiplier", None)
    if is_provided(interp_multiplier):
        overrides["interp_multiplier"] = interp_multiplier
    presentation_fps = getattr(args, "presentation_fps", None)
    if is_provided(presentation_fps):
        overrides["presentation_fps"] = presentation_fps
    return overrides
