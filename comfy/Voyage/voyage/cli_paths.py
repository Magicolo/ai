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
    # inside payload paths (workers run with CWD=run_dir: generate_blocks
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
    """Run name for init/generate: --name wins, --run-id is the legacy alias.

    Strips padding (issue 116): `is_flat_folder_name` validates the
    stripped value, and the TUI strips before building its namespace, so
    the accessor strips too — all three agree, and a pasted `" boba "`
    lands in `output/boba/` on both surfaces. A missing/blank `--name`
    falls back to `--run-id` (also stripped); both missing yields ""
    so `_check_run_id` reports exit 2 instead of an AttributeError.
    """
    named = getattr(args, "name", None)
    if isinstance(named, str) and named.strip() != "":
        return named.strip()
    fallback = getattr(args, "run_id", "")
    return fallback.strip() if isinstance(fallback, str) else str(fallback)


def output_root() -> Path:
    """Project output root: `./output/` resolved against the cwd (024).

    The TUI hardcodes `output/<name>` and `generate` defaults to
    `output/<run-id>` — both relative to wherever the user invoked the
    command — so the containment root tracks the cwd, not the package.
    """
    return (Path.cwd() / "output").resolve()


def is_outside_output_dir(path: Path | str) -> bool:
    """Whether a write target escapes the project output tree (024).

    Pure predicate over the resolved path (`Path.relative_to`): True
    when the target is not under `./output/`. Absolute outside-tree
    paths stay legal (documented `/tmp` flows, tmp_path-based suites),
    so callers warn — never reject — on True.
    """
    try:
        Path(path).resolve().relative_to(output_root())
    except ValueError:
        return True
    return False


def warn_if_outside_output_dir(path: Path | str, *, flag: str = "--output") -> bool:
    """Warn on stderr when a write target escapes `./output/` (024).

    Warn-only by design: rejecting would break the documented `/tmp`
    flows and every tmp_path-based test, which init outside the cwd
    tree without `--force`. Returns True when a warning was printed so
    tests can assert the guard fired without parsing stderr.
    """
    if is_outside_output_dir(path):
        print(
            f"warning: {flag} {path} is outside {output_root()} "
            "(writes escape the project output tree)",
            file=sys.stderr,
        )
        return True
    return False
