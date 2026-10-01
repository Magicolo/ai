"""Resolve-gate for `(issue NNN)` citations (issue 165).

Every `(issue NNN)` pointer in the tree (outside `issues/` itself and
generated/vendor dirs) must resolve to an existing `issues/NNN_*.md`
file OR to the removal manifest in `issues/000_INDEX.md`, so a reader
following a pointer lands on a live record or on the manifest entry
that names the git-history text (resolved issues were pruned from the
tree; their full text lives in history). Topic-overlap (pointer text
vs issue title) stays a human review step — this gate catches dangling
numbers only.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ISSUES_DIR = REPO_ROOT / "issues"
_CITE_PATTERN = re.compile(rb"\(issue (\d{3})\)")
_SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".pytest_cache",
        ".venv",
        "output",
        "issues",
        "reports",
    }
)
_SCANNED_SUFFIXES = frozenset({".py", ".sh", ".md", ".toml"})
_SCANNED_NAMES = frozenset({"Dockerfile", ".dockerignore", ".gitignore"})


def _is_scanned(path: Path) -> bool:
    """Only prose/code surfaces carry pointers; skip binaries and vendored trees."""
    if not path.is_file():
        return False
    if any(part in _SKIP_DIRS for part in path.parts):
        return False
    name = path.name
    if name in _SCANNED_NAMES or name.startswith("Dockerfile."):
        return True
    return path.suffix in _SCANNED_SUFFIXES


def _known_issue_numbers() -> set[str]:
    """Open records plus the `000_INDEX.md` removal manifest.

    Open `issues/NNN_*.md` stems resolve directly; every `- NNN` manifest
    line (open section + removed section, 182 pruned numbers with history
    text) resolves via the manifest, so long-standing pointers at resolved
    issues keep passing instead of dangling.
    """
    numbers = {path.stem.split("_")[0] for path in ISSUES_DIR.glob("*_*.md")}
    try:
        index_text = (ISSUES_DIR / "000_INDEX.md").read_text(encoding="utf-8")
    except OSError:
        return numbers
    return numbers | set(re.findall(r"^-\s+(\d{3})\b", index_text, re.MULTILINE))


def _collect_cites() -> list[tuple[Path, str]]:
    """All `(issue NNN)` pointers as `(file, number)` pairs, sorted for stable output."""
    cites: list[tuple[Path, str]] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not _is_scanned(path):
            continue
        for match in _CITE_PATTERN.finditer(path.read_bytes()):
            cites.append((path, match.group(1).decode("ascii")))
    return cites


def test_every_issue_cite_resolves_to_a_file() -> None:
    """Each cited number has a matching record or manifest entry."""
    known = _known_issue_numbers()
    dangling = [
        f"{path.relative_to(REPO_ROOT)} cites missing issue {number}"
        for path, number in _collect_cites()
        if number not in known
    ]
    assert not dangling, "dangling issue cites:\n" + "\n".join(dangling)


def test_cite_scan_finds_known_pointers() -> None:
    """The scan itself sees the tree (guards silent-empty glob/regex rot)."""
    found = {(path.name, number) for path, number in _collect_cites()}
    assert ("gates.sh", "092") in found
    assert ("loop.py", "007") in found
    assert len(found) > 50
