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


def _sfx_dual_pan_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """CLI dual-pan flags → resolve_config kwargs (dual SFX pair).

    Tri-state like the prompt-enhance pair: absent (None) means inherit
    the stored manifest value (fresh creates default to True via
    `SfxConfig`); exactly one of --sfx-dual-pan / --no-sfx-dual-pan
    wins. Both set raises ValueError — `cmd_configure` pre-checks the
    same conflict with a dedicated message, so this is belt-and-braces
    for direct callers. Every read goes through getattr + `is_provided`
    so hand-built namespaces resolve to absent, never to a value.
    """
    enabled = getattr(args, "sfx_dual_pan", None)
    disabled = getattr(args, "no_sfx_dual_pan", None)
    if is_provided(enabled) and is_provided(disabled):
        raise ValueError("pass only one of --sfx-dual-pan or --no-sfx-dual-pan")
    if is_provided(enabled):
        return {"sfx_dual_pan": True}
    if is_provided(disabled):
        return {"sfx_dual_pan": False}
    return {}


def resolve_generate_skips(args: argparse.Namespace) -> dict[str, bool]:
    """Generate-only skip flags → canonical skip tuple (non-persistent).

    Shorthands OR in: --no-audio = --no-music + --no-sfx,
    --no-augment = --no-upscale + --no-interpolate. Every read goes
    through getattr so hand-built namespaces default to off, never crash.
    """
    no_music = bool(getattr(args, "no_music", False))
    no_sfx = bool(getattr(args, "no_sfx", False))
    no_upscale = bool(getattr(args, "no_upscale", False))
    no_interpolate = bool(getattr(args, "no_interpolate", False))
    if bool(getattr(args, "no_audio", False)):
        no_music = True
        no_sfx = True
    if bool(getattr(args, "no_augment", False)):
        no_upscale = True
        no_interpolate = True
    return {
        "skip_music": no_music,
        "skip_sfx": no_sfx,
        "force_upscale_1": no_upscale,
        "force_interpolate_1": no_interpolate,
    }


def generate_skip_key(
    skips: dict[str, bool],
    *,
    manifest_no_sfx: bool,
    stored_upscale: int,
    stored_interpolate: int,
    stored_interp_backend: str = "rife",
    stored_sfx_dual_pan: bool = True,
) -> str:
    """Canonical freshness key for the current generate finalize behavior.

    Combines the generate skips with the stored manifest policy so the
    'nothing to do' gate re-finalizes when the behavior differs from the
    stamped coverage (e.g. same segments but music-only diff). The interp
    backend rides the key because it changes interp pixels: switching
    backends re-finalizes by design (the sidecar weights key misses too).
    The dual-pan flag rides the key because a single-bed final is not
    the spatialized pair: toggling it re-finalizes by design (legacy
    single-bed coverages predate the key segment and miss once, which
    heals them into the pair).
    """
    effective_sfx_off = bool(manifest_no_sfx or skips.get("skip_sfx", False))
    effective_up = 1 if skips.get("force_upscale_1", False) else stored_upscale
    effective_interp = 1 if skips.get("force_interpolate_1", False) else stored_interpolate
    music = int(bool(skips.get("skip_music", False)))
    sfx = int(effective_sfx_off)
    backend = stored_interp_backend if effective_interp > 1 else "-"
    dual = int(bool(stored_sfx_dual_pan) and not effective_sfx_off)
    return (
        f"music={music},sfx={sfx},up={effective_up},"
        f"interp={effective_interp},backend={backend},dual={dual}"
    )


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
    interp_backend = getattr(args, "interp_backend", None)
    if is_provided(interp_backend):
        overrides["interp_backend"] = interp_backend
    return overrides
