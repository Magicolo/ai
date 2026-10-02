"""Deterministic seed derivation (DESIGN §62).

Separate streams per concern via BLAKE2b so a change to director
logging can never silently alter video randomness.
"""

from __future__ import annotations

import hashlib
import secrets

#: Top of the derived-seed range: seeds stay in [0, 2**31 - 1] because
#: downstream consumers (sampler RNGs, worker payloads, the seed column in
#: metrics) assume non-negative int32. No range check on `run_seed` itself
#: — config accepts any int and the modulo below already folds every
#: integer input into range, so validation would reject working runs.
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


def random_master_seed() -> int:
    """Fresh random master seed for a new run (omitted `--seed`).

    Non-reproducible by design (user decision 2026-10-02): callers print
    the value so the run can be repeated with an explicit `--seed`.
    `secrets.randbits(31)` keeps it in signed-31-bit range, exactly like
    every `derive_seed` output downstream consumes.
    """
    return secrets.randbits(31)
