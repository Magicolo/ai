"""Atomic file writes (DESIGN §31). Never write critical state directly
over the previous valid file: write temp + fsync + os.replace."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def fsync_dir(directory: Path) -> None:
    """Persist a directory entry (rename durability, DESIGN §31).

    fsync on the file alone does not guarantee the rename survives a
    power loss on most filesystems — the containing directory entry
    must be synced too.
    """
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=destination.name + ".", suffix=".partial"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
        fsync_dir(destination.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(destination: Path, payload: Any) -> None:
    atomic_write_bytes(destination, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
