"""Shared CLI core: console factory, run loading, augment overrides.

DESIGN §58 — leaf module: the small helpers both verbs need
(`get_console`, `_load_run`, `_augment_overrides`) without any
verb-to-verb edge, so `cli_configure`/`cli_generate`/`cli_finalize`
import them at top level.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from voyage.config import ProjectConfig, is_provided
from voyage.console import VoyageConsole
from voyage.persistence import read_effective_config


def get_console(args: argparse.Namespace) -> VoyageConsole:
    """Console for a subcommand (flags default off for test Namespaces)."""
    return VoyageConsole(
        verbose=bool(getattr(args, "verbose", False)),
        no_color=bool(getattr(args, "no_color", False)),
        quiet=bool(getattr(args, "quiet", False)),
    )


def _load_run(run: Path) -> ProjectConfig:
    """Load the run's effective config from its manifest (CLI-is-config)."""
    return read_effective_config(run)


def _augment_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """CLI augment flags → resolve_config kwargs.

    No minimum floors: `--upscale`/`--interpolate` are explicit
    multipliers (1 = no work on that axis). Every read goes through
    getattr + `is_provided` so hand-built namespaces (Unset blanks,
    missing attrs) resolve to absent, never to a value.
    """
    overrides: dict[str, Any] = {}
    upscale = getattr(args, "upscale", None)
    if is_provided(upscale):
        overrides["upscale"] = upscale
    interpolate = getattr(args, "interpolate", None)
    if is_provided(interpolate):
        overrides["interpolate"] = interpolate
    presentation_fps = getattr(args, "presentation_fps", None)
    if is_provided(presentation_fps):
        overrides["presentation_fps"] = presentation_fps
    return overrides
