"""Atomic file writes (DESIGN §31). Never write critical state directly
over the previous valid file: write temp + fsync + os.replace."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
"""JSON-shaped data: the typed JSON/RPC boundary (issue 036).

Recursive alias (3.10-compatible via string forward references). Use this
instead of bare `Any` for payloads crossing the process boundary so shape
errors surface at the checker, not inside workers at runtime. Callers
holding `Any` (e.g. pydantic wire models) accept it without complaint.
"""


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
        # BaseException (not Exception) on purpose: the temp file must not
        # litter the run dir even on KeyboardInterrupt/SystemExit. The error
        # is always re-raised below — this is cleanup, never swallowing.
        # Best-effort unlink: the replace may already have consumed the temp
        # path, or the directory may be gone — either way there is nothing
        # left worth failing the original error for.
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


COPY_CHUNK_BYTES = 1024 * 1024
"""Chunk size for `atomic_copy`: 1 MiB streams multi-GB finals in constant memory."""


def atomic_copy(source: Path, destination: Path, *, chunk_bytes: int = COPY_CHUNK_BYTES) -> Path:
    """Publish a file atomically without loading it into RAM (issue 043).

    `atomic_write_bytes` takes `bytes`, so publishing a staged final MP4
    through it materializes the whole file in the heap (plus the temp
    copy) — 2-3x transient RAM on hundred-MB finals. This streams
    `source` to a sibling `.partial` temp in `chunk_bytes` pieces, then
    the same flush/fsync/replace/fsync_dir commit, so crash semantics
    match `atomic_write_bytes` at constant memory. Returns `destination`.
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive (got {chunk_bytes})")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=destination.name + ".", suffix=".partial"
    )
    try:
        with os.fdopen(fd, "wb") as out:
            with open(source, "rb") as incoming:
                shutil.copyfileobj(incoming, out, chunk_bytes)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_name, destination)
        fsync_dir(destination.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return destination


def atomic_write_json(destination: Path, payload: Any) -> None:
    """Write JSON atomically. `payload` stays `Any` on purpose (issue 036).

    `json.dumps` serializes anything, so narrowing the write side buys no
    checking — the contract lives on the read side (`read_json` callers
    narrow `JsonValue` with `isinstance`) and on RPC payloads
    (`RpcPayload` in `voyage.rpc`). Widening this to `JsonValue` was tried
    and reverted: manifest/state writers hold `dict[str, object]`, which
    is not a `JsonValue`, and those call sites belong to other passes.
    """
    atomic_write_bytes(destination, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    """Read JSON. Returns `Any` (not `JsonValue`) for the same reason.

    Narrowing here to `JsonValue` was tried and reverted with the write
    side: `isinstance`-narrowed `dict[str, JsonValue]` is not assignable
    to the `dict[str, object]` manifests the readers return. Callers must
    keep validating shape with `isinstance` (see `read_manifest`).
    """
    return json.loads(path.read_text(encoding="utf-8"))
