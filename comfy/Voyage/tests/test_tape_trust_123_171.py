"""Issues 123 + 171: recovery tapes must be verified, not trusted.

123: ltxv/causvid persist `conditioning_tail_sha256` but no resume path
ever recomputes it — a 1-byte-corrupt tail is adopted silently.
171: no gate bounds tape bytes — a sparse multi-GB `.pt` sails through to
`torch.load` and burns the restart budget.

Shared-contract delivery in `voyage.workers.video_common` (the one module
both supervisor and workers can import) + supervisor gates. Worker
call-site wiring (ltxv/causvid `parse_recovery_tape`) is covered by the
live worker tests. CPU-only: sparse files + JSON tapes, no torch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.errors import MediaError
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor
from voyage.workers.video_common import (
    MAX_RECOVERY_TAPE_BYTES,
    check_recovery_tape_size,
    verify_conditioning_tail_sha,
)


def _unstarted_supervisor(run_dir: Path) -> Supervisor:
    config = read_effective_config(run_dir)
    return Supervisor(run_dir, config)


def _done_segment(run_dir: Path, segment_id: str) -> Path:
    segment = paths.segment_dir(run_dir, segment_id)
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_bytes(b"")
    return segment


def _json_tape(segment: Path, tail: Path, digest: str) -> Path:
    tape = segment / "recovery.pt"
    tape.write_text(
        json.dumps(
            {
                "backend": "ltxv",
                "conditioning_tail_path": str(tail),
                "conditioning_tail_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    return tape


# ---------------------------------------------------------------------------
# 123: tail-sha verification helper.
# ---------------------------------------------------------------------------


def test_matching_tail_sha_passes(tmp_path: Path) -> None:
    tail = tmp_path / "tail.mp4"
    tail.write_bytes(b"conditioning-frames")
    digest = hashlib.sha256(b"conditioning-frames").hexdigest()
    verify_conditioning_tail_sha(
        tmp_path, {"conditioning_tail_path": "tail.mp4", "conditioning_tail_sha256": digest}
    )


def test_corrupt_tail_sha_raises(tmp_path: Path) -> None:
    tail = tmp_path / "tail.mp4"
    tail.write_bytes(b"conditioning-frames")
    digest = hashlib.sha256(b"conditioning-frames").hexdigest()
    with tail.open("r+b") as handle:
        handle.seek(4)
        handle.write(b"X")
    with pytest.raises(ValueError, match="tail sha mismatch"):
        verify_conditioning_tail_sha(
            tmp_path,
            {"conditioning_tail_path": "tail.mp4", "conditioning_tail_sha256": digest},
        )


def test_absent_sha_keys_are_noop_for_derive_path(tmp_path: Path) -> None:
    verify_conditioning_tail_sha(tmp_path, {"backend": "ltxv"})
    verify_conditioning_tail_sha(
        tmp_path, {"conditioning_tail_path": "missing.mp4", "conditioning_tail_sha256": "x"}
    )


# ---------------------------------------------------------------------------
# 171: size bounds (shared helper + supervisor gates).
# ---------------------------------------------------------------------------


def test_legitimate_tape_size_passes(tmp_path: Path) -> None:
    tape = tmp_path / "recovery.pt"
    tape.write_bytes(b"tail-latents")
    assert check_recovery_tape_size(tape) == tape
    assert MAX_RECOVERY_TAPE_BYTES == 1024**3


def test_giant_sparse_tape_rejected(tmp_path: Path) -> None:
    tape = tmp_path / "recovery.pt"
    with tape.open("wb") as handle:
        handle.truncate(MAX_RECOVERY_TAPE_BYTES + 1)
    with pytest.raises(ValueError, match="implausible tape size"):
        check_recovery_tape_size(tape)


def test_commit_gate_rejects_giant_tape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    big = run_dir / paths.SEGMENTS_DIRNAME / "big.pt"
    with big.open("wb") as handle:
        handle.truncate(MAX_RECOVERY_TAPE_BYTES + 1)
    supervisor = _unstarted_supervisor(run_dir)
    with pytest.raises(MediaError, match="implausible tape size"):
        supervisor._checked_tape_path("segments/big.pt", "000000")


def test_discovery_skips_giant_tape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    good_segment = _done_segment(run_dir, "000000")
    good_tape = good_segment / "recovery.pt"
    good_tape.write_bytes(b"tape")
    bad_segment = _done_segment(run_dir, "000001")
    with (bad_segment / "recovery.pt").open("wb") as handle:
        handle.truncate(MAX_RECOVERY_TAPE_BYTES + 1)
    supervisor = _unstarted_supervisor(run_dir)
    assert supervisor._latest_recovery_tape() == good_tape
    metrics = (run_dir / paths.LOGS_DIRNAME / "metrics.jsonl").read_text(encoding="utf-8")
    assert "recovery_tape_skipped" in metrics


# ---------------------------------------------------------------------------
# 123: supervisor discovery skips sha-mismatched JSON tapes.
# ---------------------------------------------------------------------------


def test_discovery_skips_sha_mismatched_tape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    good_segment = _done_segment(run_dir, "000000")
    good_tail = good_segment / "tail.mp4"
    good_tail.write_bytes(b"good-frames")
    good_tape = _json_tape(good_segment, good_tail, hashlib.sha256(b"good-frames").hexdigest())
    bad_segment = _done_segment(run_dir, "000001")
    bad_tail = bad_segment / "tail.mp4"
    bad_tail.write_bytes(b"bad-frames1")
    with bad_tail.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"X")
    _json_tape(bad_segment, bad_tail, hashlib.sha256(b"bad-frames1").hexdigest())
    supervisor = _unstarted_supervisor(run_dir)
    assert supervisor._latest_recovery_tape() == good_tape
