"""Destination mode survives atomic replace (issue 222).

Why this module exists: `tempfile.mkstemp` stages at 0600, so every
`os.replace` silently narrowed scaffolded-0644 manifests to owner-only
and locked out other-uid readers (bind-mounted output/, backup
sidecars, status readers). The writers now fchmod the temp to the
destination's existing mode (0644 for new files) before fsync — these
pins fail on the old code (0600 after every write) and pass on the new.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from voyage.atomic import atomic_copy, atomic_write_bytes, atomic_write_json


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_atomic_write_json_preserves_existing_mode(tmp_path: Path) -> None:
    dest = tmp_path / "manifest.json"
    dest.write_text("{}\n", encoding="utf-8")
    os.chmod(dest, 0o644)
    atomic_write_json(dest, {"a": 1})
    assert _mode(dest) == 0o644
    assert list(tmp_path.glob("*.partial")) == []


def test_atomic_write_json_keeps_restrictive_mode(tmp_path: Path) -> None:
    """Preserve, never widen: a deliberate 0600 stays 0600."""
    dest = tmp_path / "manifest.json"
    dest.write_text("{}\n", encoding="utf-8")
    os.chmod(dest, 0o600)
    atomic_write_json(dest, {"a": 1})
    assert _mode(dest) == 0o600


def test_atomic_write_json_new_file_defaults_0644(tmp_path: Path) -> None:
    """fchmod sets the exact mode regardless of umask — deterministic."""
    dest = tmp_path / "state.json"
    atomic_write_json(dest, {"a": 1})
    assert _mode(dest) == 0o644


def test_atomic_write_bytes_preserves_mode(tmp_path: Path) -> None:
    dest = tmp_path / "state.json"
    dest.write_bytes(b"v1")
    os.chmod(dest, 0o640)
    atomic_write_bytes(dest, b"v2")
    assert dest.read_bytes() == b"v2"
    assert _mode(dest) == 0o640


def test_atomic_copy_preserves_existing_mode(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "dst.bin"
    dest.write_bytes(b"old")
    os.chmod(dest, 0o640)
    atomic_copy(src, dest)
    assert dest.read_bytes() == b"payload"
    assert _mode(dest) == 0o640


def test_atomic_copy_new_file_defaults_0644(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "dst.bin"
    atomic_copy(src, dest)
    assert _mode(dest) == 0o644
