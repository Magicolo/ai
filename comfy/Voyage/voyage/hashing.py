"""Shared SHA-256 helpers (issue 021).

Single home for every chunked file hash in the tree. Stdlib only, so the
CUDA worker images can import it without dragging the supervisor (torch,
config, ffmpeg probes) or media (ffmpeg) modules across the image
boundary. Chunk size is 8 MiB — identical digests regardless of chunking,
so the old 64 KiB call sites delegate here with zero hash change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE_BYTES = 8 * 1024 * 1024
"""Read chunk for file hashes: constant memory even for multi-GB takes."""


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed in chunks (constant memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Hex SHA-256 of a short string (config content, prompt plans)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
