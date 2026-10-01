"""Tape-tail agreement (issue 081 extraction from `supervisor`).

`tape_tail_sha_matches` is the verbatim best-effort taped-tail check
moved to `voyage.supervisor_tape` so `supervisor.py` shrinks toward the
§12 split signal. Behavior contract: identical to the pre-split
`Supervisor._tape_tail_sha_matches` — True (adopt) on non-JSON, missing
keys, or absent tail; only a clean parse with both keys present and a
present-but-mismatched tail returns False — and the facade re-export is
the same object (single source, not a copy).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import voyage.supervisor as supervisor
import voyage.supervisor_tape as supervisor_tape
from voyage.supervisor_tape import tape_tail_sha_matches


def _write_tail(tmp_path: Path, name: str, data: bytes) -> Path:
    tail = tmp_path / name
    tail.write_bytes(data)
    return tail


def _write_tape(tmp_path: Path, payload: object) -> Path:
    tape = tmp_path / "recovery.pt"
    if isinstance(payload, bytes):
        tape.write_bytes(payload)
    else:
        tape.write_text(json.dumps(payload), encoding="utf-8")
    return tape


def test_facade_reexport_is_single_sourced() -> None:
    """The facade name is the new home object, not a copy (issue 081)."""
    assert supervisor.tape_tail_sha_matches is supervisor_tape.tape_tail_sha_matches


def test_method_agrees_with_moved_function(tmp_path: Path) -> None:
    """The retained `Supervisor` staticmethod delegates (no fork)."""
    tail = _write_tail(tmp_path, "tail.bin", b"conditioning")
    digest = hashlib.sha256(b"conditioning").hexdigest()
    tape = _write_tape(
        tmp_path, {"conditioning_tail_sha256": digest, "conditioning_tail_path": tail.name}
    )
    assert supervisor.Supervisor._tape_tail_sha_matches(tmp_path, tape) is True
    assert tape_tail_sha_matches(tmp_path, tape) is True


def test_non_json_tape_adopts(tmp_path: Path) -> None:
    """Torn/non-JSON tapes never mask (139/197 territory) — adopt."""
    tape = _write_tape(tmp_path, b"\x00\x01not-json")
    assert tape_tail_sha_matches(tmp_path, tape) is True


def test_missing_keys_adopt(tmp_path: Path) -> None:
    """Tapes without both tail keys carry no check — adopt."""
    tape = _write_tape(tmp_path, {"other": "keys"})
    assert tape_tail_sha_matches(tmp_path, tape) is True


def test_absent_tail_adopts(tmp_path: Path) -> None:
    """Absent tail materializes on the derive path — absence is not corruption."""
    tape = _write_tape(
        tmp_path,
        {"conditioning_tail_sha256": "abc", "conditioning_tail_path": "missing.bin"},
    )
    assert tape_tail_sha_matches(tmp_path, tape) is True


def test_mismatched_tail_rejects(tmp_path: Path) -> None:
    """Only a clean parse with both keys and a present-but-mismatched tail rejects."""
    tail = _write_tail(tmp_path, "tail.bin", b"actual-bytes")
    tape = _write_tape(
        tmp_path,
        {
            "conditioning_tail_sha256": hashlib.sha256(b"other-bytes").hexdigest(),
            "conditioning_tail_path": tail.name,
        },
    )
    assert tape_tail_sha_matches(tmp_path, tape) is False
