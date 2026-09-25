"""Deterministic seed derivation (DESIGN §62).

Separate streams per concern via BLAKE2b so a change to director
logging can never silently alter video randomness.
"""

from __future__ import annotations

import hashlib

_MAX_SEED = 2**31 - 1


def derive_seed(run_seed: int, *labels: str | int) -> int:
    """Derive a separated stream seed via BLAKE2b.

    Why length-prefixing: plain colon-joining maps ("a:b", "c") and
    ("a", "b:c") to the identical key string, silently merging two RNG
    streams (issue 070). Prefixing each label with its length keeps the
    encoding unambiguous for any future ":"-containing label.
    """
    encoded_labels = ":".join(f"{len(str(label))}:{label}" for label in labels)
    key = f"{run_seed}:{encoded_labels}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (_MAX_SEED + 1)


def video_seed(run_seed: int, segment: int, block: int) -> int:
    return derive_seed(run_seed, "video", segment, block)


def audio_seed(run_seed: int, segment: int, chunk: int) -> int:
    return derive_seed(run_seed, "audio", segment, chunk)


def director_seed(run_seed: int, decision_index: int) -> int:
    return derive_seed(run_seed, "director", decision_index)
