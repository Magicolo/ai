"""Video continuity across separate `run_segments` invocations (swansy bug).

Why this file exists: `output/swansy` proved extensions render fresh when
alone (000000->000001 and 000001->000002 each committed 121f = fresh LTX25
window) but continued when batched (000002->000003 committed 96f = 121f
minus the 25f frozen prefix, via the mid-batch restart+tape resume). The
supervisor resumed the video worker between same-batch commits but never
at startup, so every new process started with an empty session tail and
hard-cut. These tests pin the startup-resume contract: a streaming backend
with a committed tape must resume before the first commit of the new
invocation; no tape (first segment) and non-streaming backends must not
send a resume RPC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.persistence import read_effective_config, read_state, write_state
from voyage.supervisor import Supervisor


def _plant_done_segment(run_dir: Path, segment_id: str) -> Path:
    segment = paths.segment_dir(run_dir, segment_id)
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_bytes(b"done")
    (segment / "recovery.pt").write_bytes(b"tape-bytes")
    # Advance run state past the planted segment so the next commit is the
    # extension (mirrors a first invocation having committed segment_id).
    state = read_state(run_dir)
    state.next_segment_number = int(segment_id) + 1
    state.committed_segments = int(segment_id) + 1
    write_state(run_dir, state)
    return segment / "recovery.pt"


def _stub_run_harness(
    supervisor: Supervisor,
    committed_ids: list[str],
    resume_calls: list[dict[str, Any]],
) -> None:
    """Replace process/worker boundaries so `run_segments` runs CPU-only."""
    supervisor.start_workers = lambda: None  # type: ignore[method-assign]
    supervisor.stop_workers = lambda: None  # type: ignore[method-assign]

    real_resume = supervisor._resume_video_worker

    def _tracking_resume(segment_id: str) -> None:
        resume_calls.append({"segment_id": segment_id})
        real_resume(segment_id)

    supervisor._resume_video_worker = _tracking_resume  # type: ignore[method-assign]

    def _fake_commit() -> str:
        state = read_state(supervisor._run_dir)
        segment_id = paths.format_segment_id(state.next_segment_number)
        committed_ids.append(segment_id)
        return segment_id

    supervisor.commit_one_segment = _fake_commit  # type: ignore[method-assign]


def test_streaming_startup_resumes_from_latest_tape(tmp_path: Path) -> None:
    """Second invocation resumes so the segment continues (not fresh)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, video_backend="ltx25")
    tape = _plant_done_segment(run_dir, "000000")
    config = read_effective_config(run_dir)
    assert config.video.backend == "ltx25"
    supervisor = Supervisor(run_dir, config)

    video_calls: list[tuple[str, dict[str, Any]]] = []

    def _fake_video_call(
        op: str, payload: dict[str, Any], timeout: float = 600.0
    ) -> dict[str, Any]:
        video_calls.append((op, dict(payload)))
        return {"resumed": True}

    supervisor._video.call = _fake_video_call  # type: ignore[assignment]

    committed: list[str] = []
    resume_calls: list[dict[str, Any]] = []
    _stub_run_harness(supervisor, committed, resume_calls)

    assert supervisor.run_segments(1) == ["000001"]
    assert len(resume_calls) == 1
    assert resume_calls[0]["segment_id"] == "000001"
    resume_ops = [op for op, _ in video_calls if op == "resume"]
    assert len(resume_ops) == 1
    _, payload = next(entry for entry in video_calls if entry[0] == "resume")
    assert payload["recovery_path"] == str(tape)


def test_streaming_startup_without_tape_sends_no_resume(tmp_path: Path) -> None:
    """First segment stays fresh: no tape means no resume RPC."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, video_backend="ltx25")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)

    video_calls: list[str] = []

    def _fake_video_call(
        op: str, payload: dict[str, Any], timeout: float = 600.0
    ) -> dict[str, Any]:
        video_calls.append(op)
        return {"resumed": True}

    supervisor._video.call = _fake_video_call  # type: ignore[assignment]

    committed: list[str] = []
    resume_calls: list[dict[str, Any]] = []
    _stub_run_harness(supervisor, committed, resume_calls)

    supervisor.run_segments(1)
    assert "resume" not in video_calls


def test_fake_backend_never_resumes_at_startup(tmp_path: Path) -> None:
    """Stateless fake backend must not pay a resume RPC even with a tape."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, video_backend="fake")
    _plant_done_segment(run_dir, "000000")
    config = read_effective_config(run_dir)
    assert config.video.backend == "fake"
    supervisor = Supervisor(run_dir, config)

    video_calls: list[str] = []

    def _fake_video_call(
        op: str, payload: dict[str, Any], timeout: float = 600.0
    ) -> dict[str, Any]:
        video_calls.append(op)
        return {"resumed": True}

    supervisor._video.call = _fake_video_call  # type: ignore[assignment]

    committed: list[str] = []
    resume_calls: list[dict[str, Any]] = []
    _stub_run_harness(supervisor, committed, resume_calls)

    supervisor.run_segments(1)
    assert "resume" not in video_calls
    assert resume_calls == []
