"""Run-directory and run-name helpers for the `voyage` CLI (DESIGN §58).

Leaf module of the issue-080 verb-group split: path resolution and
flat-folder validation with zero voyage imports (stdlib only), so the
TUI state layer can share it without an import cycle. `voyage.cli`
re-exports every name below for backward compatibility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run_dir_arg(value: str) -> Path:
    # Absolute: workers spawn with CWD=run_dir, so a relative dir doubles up
    # inside payload paths (qual-longlive2 2026-09-24: generate_blocks
    # circuit-breaker on `output/.../segments/...` missing). Single funnel
    # for every subcommand; mirrors cmd_generate's resolve-once rule.
    return Path(value).resolve()


def resolve_run_dir(value: str) -> Path:
    """Canonical run-dir resolution (issue 008/057: one helper, every verb).

    Absolute + normalized so worker CWD-relative payloads never double up.
    Kept as a named alias of `_run_dir_arg` (which predates it and stays
    for backward-compatible imports) — new code should call this one.
    """
    return _run_dir_arg(value)


_RESERVED_FOLDER_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
"""Windows-reserved basenames, mirrored from the TUI (issue 080)."""


def is_flat_folder_name(value: str) -> bool:
    """Whether the value is usable as a single output folder name (008).

    Mirrors the TUI `_flat_folder_name` check (`tui_state.py`): rejects
    path separators and parent-dotdot so a crafted `--run-id` cannot
    escape `output/` (imported locally here, not from the TUI, because
    the TUI imports this module's `parse_duration` — reverse import
    would be circular). Also rejects `.` and reserved basenames (080).
    """
    text = value.strip()
    if not text or "/" in text or "\\" in text or ".." in text:
        return False
    if text in (".",):
        return False
    return text.split(".")[0].lower() not in _RESERVED_FOLDER_NAMES


def _check_run_id(run_id: str) -> int:
    """Reject traversal run names at the CLI layer (issue 008)."""
    if not is_flat_folder_name(run_id):
        print(
            f"error: --name/--run-id must be a flat folder name (no slashes), got {run_id!r}",
            file=sys.stderr,
        )
        return 2
    return 0


def _effective_run_id(args: argparse.Namespace) -> str:
    """Run name for init/generate: --name wins, --run-id is the legacy alias."""
    named = getattr(args, "name", None)
    if isinstance(named, str) and named != "":
        return named
    return str(args.run_id)
