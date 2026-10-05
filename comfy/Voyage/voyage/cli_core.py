"""Shared CLI core: console factory, run loading, augment + enhancer overrides.

DESIGN §58 — leaf module: the small helpers both verbs need
(`get_console`, `_load_run`, `_augment_overrides`,
`_prompt_enhance_overrides`) without any verb-to-verb edge, so
`cli_configure`/`cli_generate`/`cli_finalize` import them at top level.
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


def _prompt_enhance_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """CLI enhancer flags → resolve_config kwargs (Track B prototype).

    Tri-state like the definition-tier pair: absent (None) means inherit
    the stored manifest value (fresh creates default to True via
    `VideoConfig`); exactly one of --prompt-enhance / --no-prompt-enhance
    wins. Both set raises ValueError — `cmd_configure` pre-checks the
    same conflict with a dedicated message, so this is belt-and-braces
    for direct callers. Every read goes through getattr + `is_provided`
    so hand-built namespaces resolve to absent, never to a value.
    """
    enabled = getattr(args, "prompt_enhance", None)
    disabled = getattr(args, "no_prompt_enhance", None)
    if is_provided(enabled) and is_provided(disabled):
        raise ValueError("pass only one of --prompt-enhance or --no-prompt-enhance")
    if is_provided(enabled):
        return {"prompt_enhance": True}
    if is_provided(disabled):
        return {"prompt_enhance": False}
    return {}


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
