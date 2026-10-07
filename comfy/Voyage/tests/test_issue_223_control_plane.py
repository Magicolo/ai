"""Control-plane write-back preserves present-at-read requests (issue 223).

The commit's `_write_state_preserving_control_plane` merges a
`STOP/PAUSE_REQUESTED` observed at read time into the write-back so a
request that landed during the (seconds-long) checksum passes is not
clobbered. Both commit paths run under `_held_run_lock`; no CLI writer
races them (two-verb CLI), and the residual microsecond window vs an
external hand-edit is documented in DESIGN §73, not fixed. These tests
pin the present-at-read behavior on tmp runs, no GPU.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage.persistence import initial_state, read_effective_config, read_state, write_state
from voyage.supervisor import Supervisor


def _supervisor_for(run_dir: Path) -> Supervisor:
    initialize_run_directory(run_dir)
    return Supervisor(run_dir, read_effective_config(run_dir))


def test_writeback_preserves_stop_requested(tmp_path: Path) -> None:
    """A STOP_REQUESTED present at read time wins over the fresh state."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_for(run_dir)
    live = read_state(run_dir)
    live.status = "STOP_REQUESTED"
    write_state(run_dir, live)
    fresh = read_state(run_dir)
    fresh.status = "RUNNING"
    fresh.committed_segments += 1
    supervisor._write_state_preserving_control_plane(fresh)
    settled = read_state(run_dir)
    assert settled.status == "STOP_REQUESTED"
    assert settled.committed_segments == fresh.committed_segments


def test_writeback_preserves_pause_requested(tmp_path: Path) -> None:
    """A PAUSE_REQUESTED present at read time wins over the fresh state."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_for(run_dir)
    live = read_state(run_dir)
    live.status = "PAUSE_REQUESTED"
    write_state(run_dir, live)
    fresh = read_state(run_dir)
    fresh.status = "RUNNING"
    fresh.next_segment_number += 1
    supervisor._write_state_preserving_control_plane(fresh)
    settled = read_state(run_dir)
    assert settled.status == "PAUSE_REQUESTED"
    assert settled.next_segment_number == fresh.next_segment_number


def test_writeback_keeps_fresh_status_without_request(tmp_path: Path) -> None:
    """With no live request the fresh state lands untouched."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_for(run_dir)
    fresh = read_state(run_dir)
    fresh.status = "RUNNING"
    fresh.committed_segments += 1
    supervisor._write_state_preserving_control_plane(fresh)
    settled = read_state(run_dir)
    assert settled.status == "RUNNING"
    assert settled.committed_segments == fresh.committed_segments


def test_writeback_missing_state_file_keeps_fresh(tmp_path: Path) -> None:
    """An unreadable live state falls back to the fresh status (never raw)."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_for(run_dir)
    (run_dir / "state.json").unlink()
    rebuilt = initial_state(read_effective_config(run_dir))
    rebuilt.status = "RUNNING"
    supervisor._write_state_preserving_control_plane(rebuilt)
    assert read_state(run_dir).status == "RUNNING"
