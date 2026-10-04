"""Run-directory and run-name helpers for the `voyage` CLI (DESIGN §58).

Leaf module: path resolution and flat-folder validation with zero
voyage imports (stdlib only).
"""

from __future__ import annotations

import sys
from pathlib import Path

_RESERVED_FOLDER_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
"""Windows-reserved basenames, mirrored from the TUI (issue 080)."""


def is_flat_folder_name(value: str) -> bool:
    """Whether the value is usable as a single output folder name (008).

    Rejects path separators and parent-dotdot so a crafted name cannot
    escape `output/`. Also rejects `.` and reserved basenames (080).
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


def resolve_run_ref(*, run: str | None, name: str | None) -> Path | None:
    """`--run` vs `--name` → run dir (None + stderr on misuse).

    `--name jango` is the short spelling for the default output path
    (`output/jango`, resolved against the cwd like `generate`'s default
    and the TUI). Passing both flags is exit 2 (ambiguous); passing
    neither is exit 2 (nothing to resolve); a non-flat `--name` is
    exit 2 (traversal guard, same rule as `--run-id`). Empty strings
    read as absent (benchmark defaults `--run` to `""`).
    """
    if run and name:
        print("error: pass only one of --run or --name", file=sys.stderr)
        return None
    if name:
        stripped = name.strip()
        if _check_run_id(stripped) != 0:
            return None
        return (output_root() / stripped).resolve()
    if run:
        # Absolute: workers spawn with CWD=run_dir, so a relative dir
        # doubles up inside payload paths.
        return Path(run).resolve()
    print("error: one of --run or --name is required", file=sys.stderr)
    return None


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
