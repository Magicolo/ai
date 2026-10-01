"""Tape-tail helpers for the supervisor (DESIGN §§73, 27).

Split from `voyage.supervisor` (issue 081): the recovery-tape
discovery's best-effort taped-tail check as an importable pure function
with no supervisor state. `voyage.supervisor` re-exports the name below
so existing importers keep working; new code imports from here directly.

- `tape_tail_sha_matches`: pure check for one discovery candidate (123).
"""

from __future__ import annotations

import json
from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file


def tape_tail_sha_matches(segment: Path, resolved_tape: Path) -> bool:
    """Best-effort taped-tail check for one discovery candidate (123).

    JSON tapes (ltxv/causvid) may carry `conditioning_tail_sha256` +
    `conditioning_tail_path`: recompute and compare, so a truncated
    tail degrades to an older tape instead of silently anchoring the
    next segment on garbage. Returns True (adopt) whenever the tape
    carries no hash, the tail file is absent (the derive path
    materializes it — absence is not corruption), or the tape is not
    JSON at all (torn JSON — 139/197's territory, never masked
    here): only a clean parse with both keys
    present and a present-but-mismatched tail returns False.
    """
    try:
        raw: JsonValue = json.loads(resolved_tape.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    if not isinstance(raw, dict):
        return True
    digest = raw.get("conditioning_tail_sha256")
    raw_path = raw.get("conditioning_tail_path")
    if not isinstance(digest, str) or not digest or not isinstance(raw_path, str):
        return True
    tail_path = Path(raw_path)
    if not tail_path.is_absolute():
        tail_path = segment / tail_path
    if not tail_path.is_file():
        return True
    try:
        return sha256_file(tail_path) == digest
    except OSError:
        return True
